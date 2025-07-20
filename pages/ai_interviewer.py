import streamlit as st
import os
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
import time
import json
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
from dotenv import load_dotenv
import uuid
import glob
import os
import time

# --- 1. CONFIGURATION & INITIALIZATION ---

# Load environment variables from .env file
load_dotenv()


def get_azure_keys():
    """
    Fetches Azure keys from environment variables and correctly parses the endpoint.
    """
    azure_speech_key = os.getenv("AZURE_SPEECH_API_KEY")
    azure_speech_region = os.getenv("AZURE_SPEECH_REGION")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

    azure_openai_endpoint_full = os.getenv("AZURE_OPENAI_ENDPOINT_GPT4O")

    if not all(
        [
            azure_speech_key,
            azure_speech_region,
            azure_openai_api_key,
            azure_openai_endpoint_full,
        ]
    ):
        st.error(
            "One or more required environment variables are missing from your .env file."
        )
        st.stop()

    try:
        azure_openai_endpoint = (
            "https://" + azure_openai_endpoint_full.split("/")[2] + "/"
        )
        model_deployment_name = azure_openai_endpoint_full.split("/")[5]
    except IndexError:
        st.error(
            "The AZURE_OPENAI_ENDPOINT_GPT4O in your .env file appears to be malformed."
        )
        st.stop()

    return (
        azure_speech_key,
        azure_speech_region,
        azure_openai_api_key,
        azure_openai_endpoint,
        model_deployment_name,
    )


(
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    MODEL_DEPLOYMENT_NAME,
) = get_azure_keys()


# --- 2. HELPER FUNCTIONS (SERVICES) ---


@st.cache_resource
def get_speech_synthesizer(speech_key, speech_region):
    """Initializes Azure Speech Synthesizer and returns speech_config separately."""
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=speech_region
    )
    speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
    audio_config = speechsdk.audio.AudioOutputConfig(filename="output.mp3")
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )
    return speech_config, synthesizer



def get_speech_recognizer(speech_key, speech_region):
    """Initializes Azure Speech Recognizer.
    NOTE: This is NOT cached because it's stateful and causes issues with Streamlit's rerun model.
    """
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=speech_region
    )
    speech_config.speech_recognition_language = "en-US"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    return speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )


@st.cache_resource
def get_openai_client(api_key, endpoint):
    """Initializes Azure OpenAI Client."""
    return AzureOpenAI(
        api_key=api_key, api_version="2024-02-01", azure_endpoint=endpoint
    )

def speak_text(speech_config, text):
    """Converts text to speech using Azure SDK and plays it."""
    if not text:
        return

    # 🔁 Delete all previous .mp3 files (cleanup)
    for file in glob.glob("output_*.mp3"):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

    # 🎤 Create new output file
    unique_filename = f"output_{uuid.uuid4()}.mp3"
    audio_config = speechsdk.audio.AudioOutputConfig(filename=unique_filename)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        with open(unique_filename, "rb") as f:
            st.audio(f.read(), format="audio/mp3", autoplay=True)

        estimated_duration = len(text.split()) * 60 / 150
        time.sleep(estimated_duration)

        # 🗑️ Delete current file after playback
        try:
            os.remove(unique_filename)
        except Exception as e:
            print(f"Failed to delete {unique_filename}: {e}")
    else:
        st.error(f"Error synthesizing speech: {result.reason}")



def recognize_from_microphone(recognizer, status_placeholder):
    """Captures speech from microphone."""
    status_placeholder.info("🎤 Listening... Please speak now.")
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        status_placeholder.success(f"Heard: {result.text}")
        time.sleep(2)
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        status_placeholder.warning("No speech could be recognized.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        st.error("Speech recognition canceled.")
    return None


def get_ai_response(client, model, conversation_history, response_format="text"):
    """Fetches response from OpenAI."""
    response = client.chat.completions.create(
        model=model,
        messages=conversation_history,
        temperature=0.7,
        max_tokens=150,
        response_format=(
            {"type": "json_object"} if response_format == "json" else {"type": "text"}
        ),
    )
    return response.choices[0].message.content


# --- 3. MAIN APPLICATION LOGIC ---

st.set_page_config(page_title="AI HR Interviewer", page_icon="🎙️", layout="wide")
st.title("🎙️ AI HR Screening Interview")

# --- Initialize State & Services ---
if "page_state" not in st.session_state:
    st.session_state.page_state = "SETUP"

speech_config, speech_synthesizer = get_speech_synthesizer(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)
openai_client = get_openai_client(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)


# --- VIEW 1: SETUP ---
if st.session_state.page_state == "SETUP":
    st.header("Interview Setup")
    st.write(
        "Please upload the candidate's CV in PDF format to begin the interview. Camera and microphone access will be requested."
    )

    uploaded_cv = st.file_uploader("Upload CV (PDF)", type="pdf")
    resume_text = None

    if uploaded_cv:
        with st.spinner("Processing CV..."):
            from utils import parse_pdf

            resume_text = parse_pdf(uploaded_cv)
            st.success("CV processed successfully!")
            st.text_area(
                "Parsed CV Preview",
                resume_text[:500] + "...",
                height=150,
                disabled=True,
            )

    st.session_state.candidate_name = "Candidate"

    if st.button(
        "▶️ Start Interview",
        use_container_width=True,
        type="primary",
        disabled=(resume_text is None),
    ):
        st.session_state.resume_text = resume_text
        st.session_state.page_state = "INTERVIEW"
        st.session_state.interview_start_time = time.time()
        st.session_state.messages = []

        system_prompt = f"""
        You are a professional AI HR Recruiter interviewing {st.session_state.candidate_name}.
        - Start with a warm greeting: "Hi, I'm zenople an AI interviewer. Let's start with a quick introduction about yourself."
        - Ask CV-based questions (Work Experience > Skills). One at a time. Ask follow-ups if needed.
        - Conclude politely after ~3 minutes.
        Candidate's Resume:
        {st.session_state.resume_text}
        """
        st.session_state.messages.append({"role": "system", "content": system_prompt})

        with st.spinner("Preparing the first question..."):
            initial_response = get_ai_response(
                openai_client, MODEL_DEPLOYMENT_NAME, st.session_state.messages
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": initial_response}
            )
        st.rerun()

# --- VIEW 2: INTERVIEW ---
elif st.session_state.page_state == "INTERVIEW":
    st.header("Interview in Progress...")

    # Add a button to reset the interview
    if st.button("⬅️ Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    col1 = st.container()
    with col1:
        st.subheader("Conversation")
        chat_container = st.container()
        for message in st.session_state.messages:
            with chat_container.chat_message(message["role"]):
                st.write(message["content"])
        status_indicator = st.empty()

    webrtc_streamer(
        key="audio-stream",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": False, "audio": True},
    )

    # with col2:
    #     st.subheader("Camera Feed")

    #     class VideoStreamTransformer(VideoTransformerBase):
    #         def transform(self, frame):
    #             return frame  # Pass-through frame

    #     webrtc_streamer(
    #         key="interview-cam",
    #         mode=WebRtcMode.SENDONLY,
    #         rtc_configuration={
    #             "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    #         },
    #         media_stream_constraints={"video": True, "audio": True},
    #         video_transformer_factory=VideoStreamTransformer,
    #     )

    total_time = time.time() - st.session_state.interview_start_time
    st.metric(
            "Total Interview Time",
            f"{int(total_time // 60)}m {int(total_time % 60)}s / 8m",
        )

    interview_elapsed = time.time() - st.session_state.interview_start_time
    if interview_elapsed > 480:  # 10 minutes
        st.session_state.page_state = "REPORT"
        st.warning("Interview time limit reached. Generating report.")
        st.rerun()

    # --- Speech Synthesis Logic ---
    if "last_spoken" not in st.session_state:
        st.session_state.last_spoken = None

    # Get the latest assistant message
    assistant_messages = [
        msg for msg in st.session_state.messages if msg["role"] == "assistant"
    ]
    if assistant_messages:  # Check if there are any assistant messages
        latest_assistant_message = assistant_messages[-1]["content"]
        # Speak only if the latest assistant message is different from the last spoken
        if (
            latest_assistant_message
            and st.session_state.last_spoken != latest_assistant_message
        ):
            speak_text(speech_config, latest_assistant_message)
            st.session_state.last_spoken = latest_assistant_message
            # No rerun here to avoid interrupting the flow

        # 🔁 Auto-start recording after AI speaks
        speech_recognizer = get_speech_recognizer(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)
        user_response = recognize_from_microphone(speech_recognizer, status_indicator)

        if user_response:
            st.session_state.messages.append({"role": "user", "content": user_response})

            with st.spinner("AI is thinking..."):
                ai_response = get_ai_response(
                    openai_client, MODEL_DEPLOYMENT_NAME, st.session_state.messages
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_response}
                )
            st.rerun()
        else:
            status_indicator.warning("Could not hear you. Please try again.")
    # --- User Interaction Logic ---
    # if st.button("🎤 Record Your Answer", use_container_width=True, type="primary"):
    #     # Get a fresh recognizer instance
    #     speech_recognizer = get_speech_recognizer(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)
    #     user_response = recognize_from_microphone(speech_recognizer, status_indicator)

    #     if user_response:
    #         st.session_state.messages.append({"role": "user", "content": user_response})

    #         with st.spinner("AI is thinking..."):
    #             ai_response = get_ai_response(
    #                 openai_client, MODEL_DEPLOYMENT_NAME, st.session_state.messages
    #             )
    #             st.session_state.messages.append(
    #                 {"role": "assistant", "content": ai_response}
    #             )
    #         # No immediate rerun; let the next cycle handle the new message
    #         st.rerun()
    #     else:
    #         status_indicator.warning("Could not hear you. Please try again.")

    if st.button("🏁 End Interview and Generate Report"):
        st.session_state.page_state = "REPORT"
        st.rerun()

# --- VIEW 3: FINAL REPORT ---
elif st.session_state.page_state == "REPORT":
    st.header("Interview Concluded")

    if "final_report" not in st.session_state:
        with st.spinner("Generating final report..."):
            conversation_transcript = "\n".join(
                [
                    f"{msg['role']}: {msg['content']}"
                    for msg in st.session_state.messages
                    if msg["role"] != "system"
                ]
            )
            report_prompt = f"""
            You are a senior HR Analyst. Based on the following interview transcript with {st.session_state.candidate_name}, provide a detailed evaluation.

            Interview Transcript:
            ---
            {conversation_transcript}
            ---

            Your Task:
            Provide the final evaluation based on the transcript. Give:
            - Strengths: The candidate's key strengths.
            - Weaknesses: Their potential weaknesses or areas for improvement.
            - Interview Score: A score out of 100.
            - Decision: A final decision ("Shortlisted", "Hold", or "Reject").
            
            Output your response as a single, valid JSON object with the keys: "strengths", "weaknesses", "interview_score", "status".
            """
            final_report_str = get_ai_response(
                openai_client,
                MODEL_DEPLOYMENT_NAME,
                [{"role": "system", "content": report_prompt}],
                response_format="json",
            )
            try:
                st.session_state.final_report = json.loads(final_report_str)
            except json.JSONDecodeError:
                st.session_state.final_report = {"error": final_report_str}

    report = st.session_state.final_report
    if "error" in report:
        st.text(report["error"])
    else:
        st.subheader("Interview Evaluation")
        st.metric("Score", f"{report.get('interview_score', 'N/A')}/100")
        st.info(f"Decision: {report.get('status', 'N/A')}")

        #candidate_name
        st.subheader(f"Candidate Name: {st.session_state.candidate_name}")

        #  strengths 
        st.subheader("Strengths")
        strengths = report.get("strengths", ["N/A"])
        if isinstance(strengths, list):
            strengths_md = "\n".join([f"- {s}" for s in strengths])
            st.markdown(strengths_md)
        else:
            st.markdown(f"- {strengths}")

        #  weaknesses and Areas for Improvement 
        st.subheader("Weaknesses / Areas for Improvement")
        weaknesses = report.get("weaknesses", ["N/A"])
        if isinstance(weaknesses, list):
            weaknesses_md = "\n".join([f"- {w}" for w in weaknesses])
            st.markdown(weaknesses_md)
        else:
            st.markdown(f"- {weaknesses}")

        # --- Export Report as Text File ---
        report_text = f"""
        Interview Evaluation Report
        ===========================
        Candidate Name: {st.session_state.candidate_name}
        Score: {report.get('interview_score', 'N/A')}/100
        Decision: {report.get('status', 'N/A')}

        Strengths:
        {strengths_md if isinstance(strengths, list) else strengths}

        Weaknesses:
        {weaknesses_md if isinstance(weaknesses, list) else weaknesses}
        """
        st.download_button(
            label="📄 Export Report as Text",
            data=report_text,
            file_name=f"{st.session_state.candidate_name}_interview_report.txt",
            mime="text/plain",
        )

    if st.button("🔄 Start New Interview"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()