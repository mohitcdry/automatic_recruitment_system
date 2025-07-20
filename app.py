import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from openai import AzureOpenAI
from utils import parse_pdf, get_cv_score
import concurrent.futures


load_dotenv()

# API keys from .env
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT_FULL = os.getenv("AZURE_OPENAI_ENDPOINT_GPT4O_MINI")
GMAIL_PASS = os.getenv("GMAIL_PASS")

# key testing
if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT_FULL, GMAIL_PASS]):
    st.error(
        "One or more required environment variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT_GPT4O_MINI, GMAIL_PASS) are not found. Please set them in your .env file."
    )
    st.stop()

# Sdk base endpoint
AZURE_OPENAI_ENDPOINT = "https://" + AZURE_OPENAI_ENDPOINT_FULL.split("/")[2]

CLIENT = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)
MODEL_DEPLOYMENT_NAME = "gpt-4o-mini"


def process_single_cv(file, client, job_description, model_deployment_name):
    """Helper function to parse and score a single CV."""
    resume_text = parse_pdf(file)
    if "Error" in resume_text:
        return {"error": resume_text, "filename": file.name}

    score_data = get_cv_score(
        client, resume_text, job_description, model_deployment_name
    )

    if "error" not in score_data:
        score_data["id"] = os.path.splitext(file.name)[0]
    else:
        score_data["filename"] = file.name

    return score_data


def main():
    st.set_page_config(page_title="AI-Powered ATS", page_icon=":briefcase:")
    st.title("AI-Powered Applicant Tracking System")
    st.write("Upload CVs and a job description to automatically shortlist candidates.")


    st.header("Configuration")
    job_description = st.text_area("Job Description", height=200)
    uploaded_files = st.file_uploader(
        "Upload CVs (PDF)", type="pdf", accept_multiple_files=True
    )

    if st.button("Process CVs"):
        if uploaded_files and job_description:
            results = []
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Parallel processing for faster data process with 8 max thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_file = {
                    executor.submit(
                        process_single_cv,
                        file,
                        CLIENT,  
                        job_description,
                        MODEL_DEPLOYMENT_NAME,  
                    ): file
                    for file in uploaded_files
                }

                for i, future in enumerate(
                    concurrent.futures.as_completed(future_to_file)
                ):
                    file = future_to_file[future]
                    try:
                        data = future.result()
                        if "error" not in data:
                            results.append(data)
                        else:
                            st.warning(
                                f"Failed to process {data.get('filename', 'a file')}: {data['error']}"
                            )
                    except Exception as exc:
                        st.error(f"{file.name} generated an exception: {exc}")

                    progress = (i + 1) / total_files
                    progress_bar.progress(progress)
                    status_text.text(f"Processing CV {i + 1}/{total_files}")

            status_text.text("Processing complete!")

            if results:
                df = pd.DataFrame(results)
                shortlisted_df = df[df["score"] >= 60].sort_values(
                    by="score", ascending=False
                )

                st.session_state.shortlisted_df = shortlisted_df

                st.success("CVs processed successfully!")

                # Display shortlisted candidates
                st.header("Shortlisted Candidates (Score >= 60)")
                st.dataframe(
                    shortlisted_df[
                        [
                            "id",
                            "name",
                            "email",
                            "score",
                            "domain",
                        ]
                    ]
                )

                # Export to CSV
                csv = shortlisted_df.to_csv(index=False)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.download_button(
                        label="Download Shortlisted Candidates as CSV",
                        data=csv,
                        file_name="shortlisted_candidates.csv",
                        mime="text/csv",
                    )
                with col2:
                    if st.button("📧 Compose Mail to Selected Candidates"):
                        st.session_state.show_mail_form = True

                # Display candidates by domain using expanders
                st.header("Candidates by Domain")
                domains = shortlisted_df["domain"].unique()
                for domain in domains:
                    with st.expander(
                        f"{domain} ({len(shortlisted_df[shortlisted_df['domain'] == domain])} candidates)"
                    ):
                        domain_df = shortlisted_df[shortlisted_df["domain"] == domain]
                        for _, row in domain_df.iterrows():
                            st.markdown(f"**{row['name']}** (Score: {row['score']})")
                            st.write(f"_{row['comment']}_")
                            st.write(f"Email: {row['email']}")
                            st.divider()
            else:
                st.info("No CVs were processed or none met the shortlisting criteria.")
        else:
            st.warning("Please upload CVs and provide a job description.")

##bulk mail
import smtplib
from email.mime.text import MIMEText

def generate_personalized_email(
    client, model_deployment_name, candidate_name, job_title, interview_link
):
    prompt = (
        f"Write a professional email to {candidate_name} informing them they are shortlisted for the position of {job_title}. "
        f"Invite them to attend an AI-powered interview at this link: {interview_link}. "
        "Keep the tone friendly and formal. Sign off as 'HR Team'."
    )
    response = client.chat.completions.create(
        model=model_deployment_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.4,
    )
    return response.choices[0].message.content


def send_bulk_email(emails, subjects, bodies, sender_email, sender_password):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        for email, subject, body in zip(emails, subjects, bodies):
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = sender_email
            msg["To"] = email
            server.sendmail(sender_email, email, msg.as_string())


# Only show Compose Mail button if there are shortlisted candidates
# --- Bulk Mail Section: Only show after shortlisting, at the bottom ---

if "shortlisted_df" in st.session_state and not st.session_state.shortlisted_df.empty:
    st.divider()
    st.subheader("Bulk Email to Shortlisted Candidates")
    if not st.session_state.get("show_mail_form"):
        if st.button("📧 Compose Mail to Selected Candidates"):
            st.session_state.show_mail_form = True

    if st.session_state.get("show_mail_form"):
        shortlisted_df = st.session_state.shortlisted_df
        emails = shortlisted_df["email"].dropna().unique().tolist()
        names = shortlisted_df["name"].fillna("Candidate").tolist()
        job_title = st.session_state.get("job_title", "the position")
        interview_link = "http://192.168.1.84:8502/"

        st.info("Generating personalized emails using GPT-4o-mini...")

        subjects = [f"Invitation: AI Interview for {job_title}"] * len(emails)
        bodies = []
        for name in names:
            body = generate_personalized_email(
                CLIENT, MODEL_DEPLOYMENT_NAME, name, job_title, interview_link
            )
            bodies.append(body)

        sender_email = st.text_input(
            "Sender Gmail address", value="www.michelchaudhary@gmail.com"
        )
        sender_password = st.text_input(
            "Gmail App Password", type="password", value=GMAIL_PASS
        )
        if st.button("Send Bulk Email"):
            send_bulk_email(emails, subjects, bodies, sender_email, sender_password)
            st.success("Emails sent successfully!")
            st.session_state.show_mail_form = False

if __name__ == "__main__":
    main()
