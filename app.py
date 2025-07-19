import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from openai import AzureOpenAI
from utils import parse_pdf, get_cv_score
import concurrent.futures

load_dotenv()


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
        # Include filename in error data for context
        score_data["filename"] = file.name

    return score_data


def main():
    st.set_page_config(page_title="AI-Powered ATS", page_icon=":briefcase:")
    st.title("AI-Powered Applicant Tracking System")
    st.write("Upload CVs and a job description to automatically shortlist candidates.")

    # Get API keys from .env file
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    # Extract base endpoint from the full deployment URL
    azure_openai_endpoint_full = os.getenv("AZURE_OPENAI_ENDPOINT_GPT4O_MINI")

    if not (azure_openai_api_key and azure_openai_endpoint_full):
        st.error(
            "Azure OpenAI API key or endpoint is not found. Please set them in your .env file."
        )
        return

    # The SDK needs the base endpoint, not the full deployment URL
    azure_openai_endpoint = "https://" + azure_openai_endpoint_full.split("/")[2]

    client = AzureOpenAI(
        api_key=azure_openai_api_key,
        api_version="2024-02-01",
        azure_endpoint=azure_openai_endpoint,
    )
    model_deployment_name = "gpt-4o-mini"

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

            # Increase max_workers for I/O-bound tasks and process parsing in the thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                future_to_file = {
                    executor.submit(
                        process_single_cv,
                        file,
                        client,
                        job_description,
                        model_deployment_name,
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

                # Store shortlisted candidates in session state for other pages
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
                st.download_button(
                    label="Download Shortlisted Candidates as CSV",
                    data=csv,
                    file_name="shortlisted_candidates.csv",
                    mime="text/csv",
                )

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


if __name__ == "__main__":
    main()
