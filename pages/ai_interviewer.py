import streamlit as st
import os
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
import time
import json
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils import parse_pdf


# --- CONFIGURATION ---
def get_azure_keys():
    """Fetches Azure keys from environment variables."""
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
        st.error("One or more required Azure credentials are not set in the .env file.")
        st.stop()

    azure_openai_endpoint = "https://" + azure_openai_endpoint_full.split("/")[2]
    model_deployment_name = "gpt-4o"

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


# --- AZURE & API SERVICES ---
@st.cache_resource
def get_speech_synthesizer(speech_key, speech_region):
    """Initializes the Azure Speech Synthesizer."""
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=speech_region
    )
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    return speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )


@st.cache_resource
def get_speech_recognizer(speech_key, speech_region):
    """Initializes the Azure Speech Recognizer."""
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=speech_region
    )
    speech_config.speech_recognition_language = "en-US"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    return speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )


def speak_text(synthesizer, text):
    """Converts text to speech."""
    if not text:
        return
    # This function now only handles speaking, not writing to UI
    synthesizer.speak_text_async(text).get()


def recognize_from_microphone(recognizer):
    """Captures audio from the microphone and converts it to text."""
    st.info("Listening...")
    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        st.success(f"Heard: {result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        st.warning("No speech could be recognized. Please try again.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        st.error("Speech recognition canceled.")
    return ""


def get_ai_response(client, model, conversation_history, response_format="text"):
    """Gets a response from the AI model."""
    response = client.chat.completions.create(
        model=model,
        messages=conversation_history,
        temperature=0.7,
        response_format=(
            {"type": "json_object"} if response_format == "json" else {"type": "text"}
        ),
    )
    return response.choices[0].message.content


@st.cache_data
def get_resume_from_api(person_id):
    """Fetches and parses a resume from the API."""
    url = f"https://hackathonapi.aqore.com/PersonResume/DownloadResume?personId={person_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        from io import BytesIO

        resume_file = BytesIO(response.content)
        resume_file.name = f"{person_id}.pdf"
        return parse_pdf(resume_file)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch resume from API: {e}")
        return None


# --- UI & LOGIC ---
st.set_page_config(page_title="AI HR Interview", page_icon="🎙️", layout="wide")
st.title("🎙️ AI HR Screening Interview")

# Initialize services
speech_synthesizer = get_speech_synthesizer(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)
speech_recognizer = get_speech_recognizer(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# --- 1. INTERVIEW SETUP ---
st.header("1. Interview Setup")
setup_cols = st.columns(2)
with setup_cols[0]:
    # Option 1: Select from shortlisted candidates
    if (
        "shortlisted_df" in st.session_state
        and not st.session_state.shortlisted_df.empty
    ):
        shortlisted_candidates = st.session_state.shortlisted_df
        candidate_options = {
            f"{row['name']} ({row['id']})": row["id"]
            for _, row in shortlisted_candidates.iterrows()
        }
        selected_id = st.selectbox(
            "Select a shortlisted candidate:",
            options=list(candidate_options.values()),
            format_func=lambda x: [
                key for key, value in candidate_options.items() if value == x
            ][0],
        )
        st.session_state.resume_text = (
            get_resume_from_api(selected_id) if selected_id else None
        )
        st.session_state.candidate_name = (
            shortlisted_candidates[shortlisted_candidates["id"] == selected_id][
                "name"
            ].iloc[0]
            if selected_id
            else "Candidate"
        )
    else:
        st.info("No shortlisted candidates. Go to the ATS page to process CVs.")

with setup_cols[1]:
    # Option 2: Manual CV upload
    st.write("Or, upload a CV manually for a quick test:")
    manual_cv = st.file_uploader("Upload CV for Interview", type="pdf")
    if manual_cv:
        st.session_state.resume_text = parse_pdf(manual_cv)
        st.session_state.candidate_name = "Candidate"

# --- 2. INTERVIEW ARENA ---
if st.session_state.get("resume_text"):
    st.header("2. Interview Arena")

    arena_cols = st.columns([0.6, 0.4])
    with arena_cols[0]:
        # Chat interface
        st.subheader("Conversation")
        chat_container = st.container(height=400)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with chat_container:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

    with arena_cols[1]:
        # Camera feed
        st.subheader("Camera Feed")
        webrtc_streamer(
            key="interview-cam",
            mode=WebRtcMode.SENDONLY,  # Changed to SENDONLY to prevent audio loopback
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": True},
        )

    # Initialize interview state
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
    if "interview_finished" not in st.session_state:
        st.session_state.interview_finished = False

    # --- Interview Controls ---
    if not st.session_state.interview_started:
        if st.button("▶️ Start Interview", use_container_width=True):
            st.session_state.interview_started = True
            st.session_state.interview_finished = False
            st.session_state.messages = []

            system_prompt = f"""
            You are an AI HR Recruiter conducting a screening interview with {st.session_state.candidate_name}.
            Your task is to assess the candidate based on their resume.
            
            **Resume Context:**
            {st.session_state.resume_text}
            
            **Instructions:**
            1. Greet the candidate by name and ask for a brief self-introduction.
            2. Ask questions based on the resume, prioritizing Experience, then Skills. Ask only one question at a time.
            3. You can ask a maximum of 2 follow-up questions for a specific topic if the candidate's answer is unclear.
            4. After 3-4 main questions, conclude the interview by saying "Thank you for your time. That's all the questions I have."
            """
            st.session_state.messages.append(
                {"role": "system", "content": system_prompt}
            )

            with st.spinner("AI is preparing the first question..."):
                initial_response = get_ai_response(
                    openai_client, MODEL_DEPLOYMENT_NAME, st.session_state.messages
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": initial_response}
                )
                speak_text(speech_synthesizer, initial_response)
            st.rerun()

    if st.session_state.interview_started and not st.session_state.interview_finished:
        if st.button("🎤 Record Answer", use_container_width=True):
            user_response = recognize_from_microphone(speech_recognizer)
            if user_response:
                st.session_state.messages.append(
                    {"role": "user", "content": user_response}
                )

                with st.spinner("AI is thinking..."):
                    ai_response = get_ai_response(
                        openai_client, MODEL_DEPLOYMENT_NAME, st.session_state.messages
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": ai_response}
                    )

                    if "thank you for your time" in ai_response.lower():
                        st.session_state.interview_finished = True

                    speak_text(speech_synthesizer, ai_response)
                st.rerun()

    # --- 3. FINAL REPORT ---
    if st.session_state.interview_finished:
        st.header("3. Final Report")
        st.success("Interview Concluded.")
        if st.button("Generate Final Report", use_container_width=True):
            with st.spinner("Generating final report..."):
                conversation_transcript = "\n".join(
                    [
                        f"{msg['role']}: {msg['content']}"
                        for msg in st.session_state.messages
                        if msg["role"] != "system"
                    ]
                )
                report_prompt = f"""
                You are an AI HR Analyst. Based on the interview transcript with {st.session_state.candidate_name} and their resume, provide a final evaluation.
                
                **Resume:**
                {st.session_state.resume_text}
                
                **Transcript:**
                {conversation_transcript}
                
                **Instructions:**
                Provide a final score out of 100, a concise review summary, and a final decision (Shortlisted / Hold / Reject).
                Output your result in a single, valid JSON object with keys: "interview_score", "review", "status".
                """
                final_report_str = get_ai_response(
                    openai_client,
                    MODEL_DEPLOYMENT_NAME,
                    [{"role": "system", "content": report_prompt}],
                    response_format="json",
                )
                try:
                    final_report = json.loads(final_report_str)
                    st.subheader("Interview Evaluation")
                    st.metric(
                        "Interview Score",
                        f"{final_report.get('interview_score', 'N/A')}/100",
                    )
                    st.text_area(
                        "AI Review", final_report.get("review", "N/A"), height=150
                    )
                    st.info(f"Final Status: **{final_report.get('status', 'N/A')}**")
                except json.JSONDecodeError:
                    st.error("Failed to parse the final report from the AI.")
                    st.text(final_report_str)
else:
    st.info("Select or upload a CV in the setup section to begin an interview.")
