# AI Lab Assistant

Turn raw lab notes into structured [eLabFTW](https://www.elabftw.net) entries, and ask questions about your lab notebook. Runs locally with [Ollama](https://ollama.com).

![Python](https://img.shields.io/badge/python-3.x-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20(local)-black)
![Status](https://img.shields.io/badge/status-prototype-orange)

Type, dictate or photograph your notes. A local LLM drafts an entry, you review it, and only then it is saved to eLabFTW.

> Experimental prototype. Review every generated entry before it is saved.

## Features

- Input as text, audio (local Whisper) or photos of handwritten notes (local MiniCPM)
- Qwen 2.5 drafts the entry: template, category, status, tags, resource, custom fields
- Nothing reaches eLabFTW until you approve the draft
- File attachments, including HEIC
- Lab chatbot for existing entries (date- and keyword-based retrieval)

## How it works

```text
Text / audio / photos
        |
 Whisper, MiniCPM, text input
        |
        v
 Qwen 2.5 (Ollama) --> draft --> you review --> eLabFTW API --> eLabFTW
```

## Quick start

You need Python 3, Ollama, an eLabFTW instance with an API key, and Docker if you run eLabFTW locally.

```bash
# 1. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install requests streamlit ollama faster-whisper pillow pillow-heif

# 2. Models
ollama pull qwen2.5:7b
ollama pull minicpm-v4.6:1b

# 3. Whisper model (optional, audio only)
python -c "from faster_whisper import download_model; download_model('base', output_dir='models/whisper/base')"

# 4. API key
export ELABFTW_API_KEY="YOUR_API_KEY"

# 5. Run
ollama serve
streamlit run app.py
```

Local eLabFTW (optional): `cd elabftw && docker compose up -d`

**Dev setup:** the client accepts the self-signed certificate of `https://localhost:8443` and disables TLS verification (`verify=False`). Do not use this unchanged in production.

## Privacy

Ollama, Qwen, MiniCPM, Whisper, Streamlit and eLabFTW can all run on your own machine, so experiment data does not have to leave it. Security still depends on your setup: network, authentication, backups and eLabFTW permissions.

## Scientific integrity

- The model only uses the material you supply.
- It does not invent measurements, silently fix uncertain OCR or silently correct calculations.
- It keeps conflicting sources, units and numbers, and separates measurements from theoretical values.
- OCR can misread digits, signs, exponents and units. Check every entry before saving.

## Status

**Implemented**

- eLabFTW API: entries, templates, categories, statuses, tags, resources, custom fields, attachments
- Local Qwen, Whisper and handwriting recognition, HEIC support
- Streamlit UI and first version of the lab notebook chatbot

**Planned**

- Multi-file input and better OCR conflict handling
- Local embeddings, semantic search and better RAG
- PDF and attachment indexing
- Structured measurement database and text-to-SQL
- Multi-user authentication and per-user eLabFTW credentials
- Central server deployment with HTTPS and autostart

## Contributing

Helpful areas:

- OCR and transcription quality on real handwritten notes
- eLabFTW integration (templates, custom fields, attachments)
- Retrieval (semantic search, RAG)
- Security: TLS verification, authentication, per-user credentials
- Tests and docs

Open an issue before larger changes. Keep pull requests small and focused.

## License

No license yet. Ask the maintainer before reusing the code. A license (e.g. MIT or Apache-2.0) should be added before accepting outside contributions.
