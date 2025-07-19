import fitz  # PyMuPDF
from openai import AzureOpenAI
import os
import json
import uuid


def parse_pdf(file):
    """Parses a PDF file and returns its text content."""
    text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        return f"Error parsing PDF: {e}"
    return text


def get_cv_score(
    client: AzureOpenAI,
    resume_text: str,
    job_description: str,
    model_deployment_name: str,
):
    """
    Scores a CV against a job description using the specified GPT model.
    """
    system_prompt = """
You are an AI HR Recruiter and Resume Evaluator. Your task is to analyze a resume against a job description and return a structured JSON output.

== INSTRUCTIONS ==
1.  **Extract Candidate Details**: From the resume text, extract the candidate's full name and email address. If they are not available, use "N/A".
2.  **Score the Resume**: Score the candidate out of 100 based on the following criteria and weights:
    - Domain Experience: 30%
    - Technical Skills Match: 25%
    - Summary/Keyword Density: 15%
    - Job Role Match: 15%
    - Education Relevance: 15%
3.  **Categorize Domain**: Identify the most relevant job domain from this list: ["Light Industry", "Hospitality", "Customer Service", "Manufacturing", "Finance/Accounting", "Information Technology"].
4.  **Provide a Summary**: Write a concise, one-line comment summarizing the candidate's fit for the role.

== RESPONSE FORMAT ==
Provide your response as a single, valid JSON object with the following keys:
- "name": (string) Candidate's full name.
- "email": (string) Candidate's email address.
- "score": (integer) The final score from 0 to 100.
- "domain": (string) The matched job domain.
- "comment": (string) A one-line summary.

Example:
{
  "name": "John Doe",
  "email": "john.doe@example.com",
  "score": 85,
  "domain": "Information Technology",
  "comment": "Strong candidate with relevant experience in cloud technologies."
}
"""

    user_prompt = f"Resume:\n{resume_text}\n\nJob Description:\n{job_description}"

    try:
        response = client.chat.completions.create(
            model=model_deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {
            "error": str(e),
            "name": "Error",
            "email": "N/A",
            "score": 0,
            "domain": "N/A",
            "comment": "Failed to process CV.",
        }
