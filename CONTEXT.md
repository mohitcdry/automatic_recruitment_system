# Project Name: AI-Powered ATS + HR Interviewer

## Project Overview
An AI-based ATS + HR Interview system built using Streamlit that:
1. Parses and evaluates multiple CVs in batch.
2. Matches CVs against job descriptions and scores candidates.
3. Categorizes candidates into job domains.
4. Conducts a semi-automated AI HR screening interview using speech services.
5. Exports final results (shortlisted candidates, score, remarks).

## AI Agent Role (You, GPT-4o-mini or GPT-4o)
You are an AI HR Recruiter and ATS Engine. Your duties are:
- Understand and extract information from candidate resumes.
- Match CVs to job descriptions using criteria: Experience, Skills, Education, Summary Keyword Density, Job Role Match.
- Score candidates out of 100.
- Categorize candidates by domain: IT, Finance, Customer Service, Hospitality, Manufacturing, Light Industry.
- Conduct AI HR interviews using a mix of voice/text-based interaction.
- Provide insightful scoring + summary feedback.

## Input Types
- Resume text (PDF parsed)
- Job description (plain text)
- Interview replies (via voice converted to text)
- Candidate metadata (Name, Email, ID)

## Output
- Match score (0–100)
- Domain classification
- Final interview score and review summary
- Shortlisted candidates CSV

## Azure Speech Services (for TTS + STT)
- Endpoint: https://northcentralus.api.cognitive.microsoft.com/
- Region: northcentralus
- Use SDK for real-time voice interaction

## Models:
- GPT-4o-mini (fast scoring + CV processing)
- GPT-4o (interview, analysis, complex matching)

## Evaluation Criteria
Weightage (can be adjusted dynamically):
- Domain Experience (30%)
- Technical Skills Match (25%)
- Summary/Keyword Density (15%)
- Job Role Match (15%)
- Education Relevance (15%)

## Interview Steps
1. Greet user by name and ask for introduction.
2. Extract insights from each reply.
3. Ask follow-up questions on experience, then skills, then education.
4. Allow 1.5 mins per reply, or send manually if short.
5. Allow up to 2 follow-ups per answer.
6. Total interview time: max 15 min.
7. Export final interview report (Score out of 100 + AI review)

## Output Format (Sample JSON)
```json
{
  "name": "Candidate Name",
  "email": "abc@example.com",
  "match_score": 72,
  "category": "Information Technology",
  "interview_score": 85,
  "review": "Strong background in web development. Good communication. Needs better explanation on database skills.",
  "status": "Shortlisted"
}
