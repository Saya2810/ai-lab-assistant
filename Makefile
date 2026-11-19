# Convenience targets. Everything here is also documented as plain commands in the README.
PY ?= python

.PHONY: install test e2e doctor demo app chatbot assistant sync

install:            ## install everything incl. test dependencies
	$(PY) -m pip install -r requirements.txt

test:               ## run the whole test suite (offline, no Ollama / eLabFTW needed)
	$(PY) -m pytest

e2e:                ## browser test of the combined app (needs: pip install playwright && playwright install chromium)
	RUN_E2E=1 $(PY) -m pytest tests/e2e -v

doctor:             ## check Ollama, models, eLabFTW and the index (both apps)
	$(PY) -m lab_assistant doctor
	$(PY) -m lab_chatbot doctor

sync:               ## read eLabFTW into the chatbot's local index (incremental)
	$(PY) -m lab_chatbot sync

demo:               ## index a synthetic notebook and run the test questions against it
	$(PY) -m lab_chatbot seed-demo
	$(PY) -m lab_chatbot --db data/chatbot/demo.db eval --full --golden lab_chatbot/golden_demo.json

app:                ## start BOTH in one UI: Lab Assistant + Lab Chatbot (recommended)
	$(PY) -m streamlit run app.py

chatbot:            ## start only the chatbot UI
	$(PY) -m streamlit run lab_chatbot/ui.py

assistant:          ## start only the lab assistant UI
	$(PY) -m streamlit run lab_assistant/app.py
