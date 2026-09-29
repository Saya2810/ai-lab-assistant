🧪 Local AI Lab Assistant

A local, open-source AI assistant for creating, organizing, and querying scientific laboratory documentation in eLabFTW.

The Lab Assistant combines local language models, speech transcription, image recognition, and the eLabFTW API in a browser-based interface. The goal is to simplify laboratory documentation while keeping scientific data and AI processing local.

Features

eLabFTW Integration

The application communicates directly with eLabFTW through its API.

Currently supported:

* Create new experiments
* Update experiment title and body
* Select experiment templates
* Select categories and statuses
* Add tags
* Link experiments to resources/projects
* Read and write custom fields
* Upload original files as attachments
* Retrieve existing laboratory entries

Organizational fields such as template, category, status, resource, and tags are selected manually by the user rather than inferred automatically by the AI.

Local AI Processing

All AI models are designed to run locally using Ollama.

Current text model:

qwen2.5:7b

The model converts reviewed source material into structured laboratory documentation while being instructed not to invent, correct, or complete scientific information.

Audio Transcription

Audio laboratory notes can be transcribed locally using faster-whisper.

The Whisper model can be stored directly on the laboratory computer so that transcription does not require a connection to the Hugging Face Hub after initial setup.

Handwritten Laboratory Notes

Photos of handwritten laboratory notes can be transcribed using a local vision model.

Current model:

minicpm-v4.6:1b

Large notebook pages are divided into overlapping horizontal sections and processed individually to improve recognition speed and accuracy.

A larger vision model, qwen3-vl:8b, was also tested but was too slow for the intended interactive workflow on the development system.

HEIC images are supported using pillow-heif.

Scientific Formatting

Mathematical expressions extracted from laboratory notes are converted to LaTeX and prepared for Markdown/MathJax rendering in eLabFTW.

The AI is instructed to preserve:

* numerical values
* physical units
* equations
* exponents
* observations
* uncertainties
* unclear or conflicting source information

OCR results should be reviewed by the user before submission.

Laboratory Chat

The project includes an experimental local chatbot that can query existing eLabFTW laboratory entries.

The current pipeline is:

Question
   │
   ▼
eLabFTW experiment retrieval
   │
   ▼
metadata / keyword filtering
   │
   ▼
relevant laboratory entries
   │
   ▼
Qwen 2.5
   │
   ▼
answer + experiment sources

Temporal queries such as:

What happened in the lab today?

can retrieve all laboratory entries associated with the current date.

More specific questions can filter the laboratory notebook by relevant terms.

Semantic retrieval using local embeddings/vector search is planned for a future version.

⸻

## Architecture

```text
                   Browser
                      │
                      ▼
               Streamlit GUI
                      │
      ┌───────────────┼────────────────┐
      │               │                │
      ▼               ▼                ▼
 Text input      Audio files         Images
      │               │                │
      │               ▼                ▼
      │            Whisper          MiniCPM
      │               │                │
      └───────────────┼────────────────┘
                      │
                      ▼
                   Qwen 2.5
                      │
                      ▼
              Structured Lab Entry
                      │
                      ▼
               User Review/Edit
                      │
                      ▼
                 eLabFTW API
                      │
                      ▼
                   eLabFTW
```

Existing eLabFTW entries can also be retrieved through the API and used by the Laboratory Chat.

---

## Project Structure

```text
ai-lab-assistant/
├── app/
│   ├── app.py
│   ├── components/
│   │   ├── chatbot.py
│   │   ├── custom_fields.py
│   │   ├── input.py
│   │   └── lab_form.py
│   └── services/
│       ├── ai.py
│       ├── elabftw.py
│       ├── experiment_service.py
│       ├── file_processing.py
│       ├── rag.py
│       ├── transcription.py
│       └── vision.py
├── models/
│   └── whisper/
│       └── base/
├── elabftw/
│   ├── docker-compose.yml
│   └── data/
└── README.md
```

The exact structure may change as the project is developed.
⸻

Requirements

The current prototype was developed on macOS with Apple Silicon.

Required software:

* Python 3
* Docker Desktop
* eLabFTW
* Ollama
* Streamlit

The final system is intended to run on a central GNU/Linux laboratory machine/server.

⸻

Python Environment

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Upgrade pip:

python -m pip install --upgrade pip

Install the required Python packages:

pip install requests
pip install streamlit
pip install ollama
pip install faster-whisper
pip install pillow pillow-heif

⸻

Ollama Setup

Start Ollama:

ollama serve

Download the text model:

ollama pull qwen2.5:7b

Download the vision model:

ollama pull minicpm-v4.6:1b

Models can be tested directly with:

ollama run qwen2.5:7b

⸻

eLabFTW API Configuration

The Lab Assistant expects an eLabFTW API key in the environment:

export ELABFTW_API_KEY="YOUR_API_KEY"

Do not commit API keys to GitHub.

The current local development instance uses:

https://localhost:8443

The Python API client communicates with:

/api/v2

The local development setup currently uses a self-signed HTTPS certificate. Certificate verification is therefore disabled in the development API client.

This should not be used unchanged for a production deployment.

⸻

Starting the Application

1. Start Docker

On macOS:

open -a Docker

2. Start eLabFTW

cd ~/lab-assistant/elabftw
docker compose up -d

3. Start Ollama

ollama serve

or, when installed as a Homebrew service:

brew services start ollama

4. Activate the Python Environment

cd ~/lab-assistant
source .venv/bin/activate

5. Start the Lab Assistant

streamlit run app/app.py

The interface will then be available in the browser.

⸻

Scientific Integrity

The application is designed as an assistant, not as an autonomous scientific decision-making system.

The language model is instructed to:

* use only supplied source material
* not invent missing measurements
* not repair uncertain OCR results
* not silently correct calculations
* preserve conflicting source information
* preserve units and numerical values
* distinguish experimental measurements from theoretical calculations

Generated laboratory entries should always be reviewed before they are written to eLabFTW.

This is particularly important for handwritten-note recognition, where OCR errors in numbers, signs, exponents, and units can substantially alter scientific meaning.

⸻

Privacy

The project is designed around local processing.

The following components can operate locally:

eLabFTW
Ollama
Qwen
MiniCPM
Whisper
Streamlit

This allows laboratory data to be processed without sending experimental content to external AI APIs.

Actual security and privacy nevertheless depend on the deployment environment, including network configuration, user authentication, operating-system security, backups, eLabFTW permissions, and server configuration.

⸻

Development Status

The project is currently a prototype under active development.

Implemented

* eLabFTW API connection
* experiment creation and updating
* eLabFTW templates
* categories and statuses
* tags
* resources/projects
* custom fields
* file attachments
* local Qwen integration
* local Whisper transcription
* local handwritten-note recognition
* HEIC support
* LaTeX/Markdown preparation
* Streamlit interface
* initial laboratory notebook chatbot
* date-based laboratory entry retrieval
* keyword-based laboratory entry retrieval

Planned

* unified multi-file input workflow
* improved OCR conflict handling
* local embedding model
* semantic/vector search
* improved RAG retrieval
* PDF and document indexing
* querying laboratory attachments
* structured measurement database
* text-to-SQL queries
* multi-user authentication
* per-user eLabFTW credentials
* central laboratory server deployment
* automatic service startup
* production HTTPS configuration

⸻

Intended Deployment

The current prototype runs locally on a development machine.

The planned deployment is:

Laboratory users
       │
       │ HTTPS
       ▼
Central laboratory server
       │
       ├── Lab Assistant / Streamlit
       ├── Ollama
       ├── Qwen
       ├── Vision model
       └── Whisper
               │
               ▼
            eLabFTW

Users should eventually be able to access the Lab Assistant through a browser without manually starting Python or Ollama.

Multi-user authentication and individual eLabFTW identities will be required so that laboratory entries remain attributable to the correct researcher.

⸻

Disclaimer

This software is under active development and should currently be considered experimental.

AI-generated transcriptions and laboratory entries must be reviewed before being used as scientific documentation.