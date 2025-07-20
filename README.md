# AI-Powered Recruitment Platform

Fully automated Recruitment system with minimal human intervention
This project is a comprehensive, AI-driven recruitment platform designed to automate and enhance the hiring process. It features an Applicant Tracking System (ATS) to screen candidates and a conversational AI Interviewer to conduct initial screenings.

## 🚀 Project Idea

The goal of this platform is to streamline the recruitment workflow from initial application to the first interview, saving time for HR professionals and providing a modern experience for candidates.

1.  **Automated Applicant Tracking System (ATS)**:

    -   HR personnel can upload multiple candidate CVs (in PDF format) along with a specific job description.
    -   The system uses **Azure's GPT-4o-mini** to parse each resume, score it against the job description, and categorize candidates based on their domain.
    -   It generates a ranked list of shortlisted candidates, which can be downloaded as a CSV.
    -   A "Compose Mail" feature uses AI to generate personalized bulk emails, inviting shortlisted candidates to an AI-powered interview.

2.  **Conversational AI Interviewer**:
    -   Shortlisted candidates follow a link to a voice-based interview with an AI.
    -   The AI uses **Azure's GPT-4o** and **Azure Speech Services** to conduct a natural, spoken conversation.
    -   It asks questions relevant to the candidate's CV and the job role.
    -   After the interview, it generates a detailed report including a final score, a summary of strengths and weaknesses, and a data-driven hiring recommendation (e.g., "Directly Hire," "HR to Review," "Send to Talent Pool").

---

## 🛠️ Technologies & APIs Used

This project integrates a powerful stack of modern AI services and Python libraries.

### Core Technologies

-   **Backend & Frontend**: Python with **Streamlit** for a fast, interactive web UI.
-   **AI Language Models**:
    -   **Azure OpenAI Service**:
        -   `gpt-4o-mini`: For efficient CV scoring and email generation in the ATS.
        -   `gpt-4o`: For high-quality conversational intelligence in the AI Interviewer.
-   **Speech Services**:
    -   **Azure Speech Services for Speech**:
        -   **Text-to-Speech (TTS)**: Gives the AI Interviewer a natural-sounding voice.
        -   **Speech-to-Text (STT)**: Accurately transcribes the candidate's spoken responses.

### Key SDKs & Libraries

-   **`azure-openai`**: API key to interact with the Azure OpenAI API.
-   **`azure-cognitiveservices-speech`**: The official SDK for Azure's real-time speech services.
-   **`PyMuPDF` (`fitz`)**: A robust library for parsing text and metadata from PDF files.
-   **`pandas`**: Used for organizing and managing candidate data.
-   **`streamlit-webrtc`**: Facilitates requesting microphone permissions from the browser for the voice interview.
-   **`smtplib`**: Python's standard library for sending emails via the Gmail SMTP server.
-   **`python-dotenv`**: For securely managing API keys and environment variables.

---

## ⚙️ How to Run the Project Locally

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

-   Python 3.8 or higher
-   `pip` (Python package installer)

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd auto_requirement
```

### 3. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

-   **On macOS/Linux:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
-   **On Windows:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

### 4. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

You need to provide your API keys and service credentials in an environment file.

1.  Create a new file named `.env` in the root directory of the project.
2.  Copy the content below into your `.env` file and replace the placeholder values with your actual credentials.

    ```properties
    # Azure OpenAI Service
    # Get these from your Azure Portal
    AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_ENDPOINT_GPT4O_MINI="YOUR_GPT4O_MINI_DEPLOYMENT_URL"
    AZURE_OPENAI_ENDPOINT_GPT4O="YOUR_GPT4O_DEPLOYMENT_URL"

    # Azure Speech Service
    # Get these from your Azure Portal
    AZURE_SPEECH_API_KEY="YOUR_AZURE_SPEECH_API_KEY"
    AZURE_SPEECH_REGION="YOUR_AZURE_SPEECH_REGION"

    # Gmail App Password for Bulk Emailing
    # Generate this from your Google Account settings (Security -> App Passwords)
    GMAIL_PASS="YOUR_GMAIL_APP_PASSWORD"
    ```

    **Note**: For `GMAIL_PASS`, do not use your regular Google account password. You must generate a special **App Password** from your Google Account's security settings.

### 6. Run the Streamlit Application

Once the setup is complete, you can run the main application.

```bash
streamlit run app.py
```

The application will open in your web browser, typically at `http://localhost:8501`. You can now start using the ATS and the AI Interviewer.
