# AI Lab Assistant

Zwei getrennte Anwendungen für **eLabFTW** in einem Repository:

| | **Lab Assistant** (`lab_assistant/`) | **Lab Chatbot** (`lab_chatbot/`) |
|---|---|---|
| Zweck | Notizen, Fotos handschriftlicher Aufzeichnungen und Audio → geprüfter Entwurf → fertiger eLabFTW-Eintrag (inkl. Tags, Ressource, Custom Fields, Anhänge) | Fragen an das Laborbuch stellen: **Zahlen exakt per SQL**, **Inhalte per semantischer Suche (RAG)** |
| Schreibt in eLabFTW | ja | nein (liest nur) |
| Einzeln starten | `streamlit run lab_assistant/app.py` | `streamlit run lab_chatbot/ui.py` oder `python -m lab_chatbot chat` |

**Beide in einer Oberfläche: `streamlit run app.py`** – links in der Seitenleiste wechselst du zwischen *Lab Assistant* und *Lab Chatbot*. Die beiden Codebasen bleiben getrennt; `app.py` ist nur der gemeinsame Einstieg (Navigation + Seitenkonfiguration).

Beide nutzen nur `shared/` (eLabFTW-Client mit Timeouts, Retries, Paginierung und TLS-Konfiguration). Sie importieren sich **nicht** gegenseitig und haben getrennte Requirements. Alle Modelle laufen lokal über [Ollama](https://ollama.com).

---

## 1. Installation

Voraussetzungen: Python ≥ 3.10 (getestet mit 3.13), [Ollama](https://ollama.com/download), ein laufender eLabFTW-Server (für den Echtbetrieb; zum Ausprobieren reicht die Demo, siehe 3.).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # alles inkl. Tests
# oder nur eine App:  requirements-chatbot.txt / requirements-assistant.txt

ollama pull qwen2.5:7b                     # Chat-/Planner-Modell (beide Apps)
ollama pull embeddinggemma                 # Embeddings (Chatbot)
ollama pull minicpm-v4.6:1b                # Handschrift-OCR (Assistant, optional)
```

Audio-Transkription (Assistant, optional) läuft offline mit einem lokal abgelegten Whisper-Modell. Einmalig herunterladen:

```bash
python -c "from faster_whisper import download_model; download_model('base', output_dir='models/whisper/base')"
```

## 2. Konfiguration

```bash
cp .env.example .env     # dann ELABFTW_API_KEY eintragen
```

Die wichtigsten Variablen (alle in `.env.example` dokumentiert; echte Umgebungsvariablen haben Vorrang vor `.env`):

| Variable | Bedeutung | Default |
|---|---|---|
| `ELABFTW_API_KEY` | API-Key (eLabFTW → Profil → API-Keys). Assistant: Schreibrecht, Chatbot: Leserecht | – (Pflicht) |
| `ELABFTW_URL` | API-URL | `https://localhost:8443/api/v2` |
| `ELABFTW_VERIFY_TLS` | `true` / `false` / Pfad zu CA-Bundle | verifizieren, **außer** für `localhost` (selbstsigniertes Zertifikat) |
| `OLLAMA_HOST` | Ollama-Server | `http://localhost:11434` |
| `LAB_CHAT_MODEL` / `LAB_EMBED_MODEL` | Chatbot-Modelle | `qwen2.5:7b` / `embeddinggemma` |
| `LAB_DB_PATH` | Index-Datenbank des Chatbots | `data/chatbot/labbook.db` |

Ein lokales eLabFTW zum Entwickeln startet `deploy/elabftw/docker-compose.example.yml` (Secrets kommen aus `deploy/elabftw/.env`, siehe dort `.env.example`).

## 3. Schnellstart ohne eLabFTW (Demo)

Der Chatbot bringt ein synthetisches Laborbuch mit (8 Experimente: Raman, PCR, Zellkultur, …). Es landet in einer **separaten** Datenbank und vermischt sich nie mit echten Daten. Benötigt wird nur Ollama mit den beiden Modellen.

```bash
python -m lab_chatbot doctor                     # prüft Ollama, Modelle, Index
python -m lab_chatbot seed-demo                  # → data/chatbot/demo.db
python -m lab_chatbot --db data/chatbot/demo.db chat
LAB_DB_PATH=data/chatbot/demo.db streamlit run app.py     # oder nur: … run lab_chatbot/ui.py
```

Fragen zum Ausprobieren:

* `Wie viele Experimente haben den Tag raman?` → **SQL**, exakt, ohne LLM-Antwort
* `Welche Messungen hatten eine Wellenlänge über 600 nm?` → **SQL** mit Einheiten-Umrechnung (nm/µm/…)
* `Was war die maximale Belichtungszeit?` → **SQL**-Aggregat (200 ms, 5 s, 30 s werden korrekt verglichen: 30 s)
* `Wie viele Experimente gibt es pro Status?` → **SQL** gruppiert, als Tabelle
* `Warum war das Rauschen bei Probe A3 so hoch?` → **RAG**: Hybridsuche, Antwort mit Quellen `[1]`
* `Fasse Experiment 104 zusammen` → ganzer Eintrag als Kontext
* `Wie war das Rauschen bei Laserleistung über 30 mW?` → **beides**: SQL grenzt ein, RAG antwortet

Mit `--explain` (CLI) bzw. dem Schalter „Plan & Zeiten anzeigen“ (UI) sieht man, *wie* die Frage verstanden wurde.

Qualitätsmessung auf der Demo (Golden-Set mit erwarteter Absicht, erwarteten Experimenten und Antwortinhalt):

```bash
python -m lab_chatbot --db data/chatbot/demo.db eval --full --golden lab_chatbot/golden_demo.json
```

## 4. Echtbetrieb

### Lab Chatbot

```bash
python -m lab_chatbot doctor      # was fehlt? (Ollama, Modelle, API-Key, Index)
python -m lab_chatbot sync        # eLabFTW → lokaler Index; inkrementell, beim 2. Lauf nur Änderungen
python -m lab_chatbot chat        # Terminal-Chat; die Oberfläche: streamlit run app.py
```

Weitere Befehle: `ask "…"`, `search "…"` (zeigt nur das Retrieval-Ranking), `stats`, `sync --full` (alles neu einlesen, z. B. nach Modellwechsel), `sync --limit 5` (zum Testen), `eval`. Die UI hat zusätzlich einen Sync-Button. Der Index muss nach neuen Einträgen in eLabFTW wieder synchronisiert werden (kein Hintergrunddienst).

### Beide zusammen (empfohlen)

```bash
streamlit run app.py
```

Startseite ist der **Lab Assistant**, der **Lab Chatbot** ist die zweite Seite in der Seitenleiste. Man kann mitten in einem Eintrag in den Chatbot wechseln (z. B. um nachzusehen, was bei der letzten Probe herauskam) und zurückkehren: Notizen, Titel, Text, Template/Tags/Custom Fields und bereits hochgeladene Dateien samt erkanntem Text bleiben erhalten. Hochgeladene Fotos/Audio-Dateien werden danach mit dem Hinweis „Kept from before you left this page“ weitergeführt (das Upload-Feld selbst leert Streamlit beim Seitenwechsel) und lassen sich mit „Remove this file“ entfernen. Der Chatverlauf und die Schalter in der Seitenleiste des Chatbots bleiben ebenfalls.

### Lab Assistant (einzeln)

```bash
streamlit run lab_assistant/app.py
```

1. Notizen tippen und/oder Dateien hochladen (Foto → OCR, Audio → Transkript, sonstige Dateien als Anhang). Der extrahierte Text ist editierbar.
2. „Generate laboratory entry“ → Entwurf. Das Modell wählt **nie** Template, Kategorie, Status, Tags – das tust du im Formular. Die App meldet Zahlen, die im Entwurf stehen, aber nicht in deinen Quellen (und umgekehrt).
3. Formular prüfen (inkl. Custom Fields des gewählten Templates), „Create lab entry“. Schlägt nach dem Anlegen ein Schritt fehl (Tag, Upload, …), siehst du die Experiment-ID *und* was fehlt.

## 5. Tests

```bash
pytest                            # 292 Tests, rein offline: kein Ollama, kein eLabFTW nötig
pytest tests/test_engine.py -v    # einzelne Datei
```

**Optional: Browser-Test der gemeinsamen Oberfläche.** Startet `app.py` wirklich, öffnet Chromium und spielt eine ganze Sitzung durch (Eintrag schreiben → in den Chatbot wechseln → Frage stellen → zurück → Eintrag anlegen), mit Attrappen für eLabFTW und Ollama. Er prüft, dass beim Seitenwechsel nichts verloren geht.

```bash
pip install playwright && playwright install chromium
RUN_E2E=1 pytest tests/e2e -v     # ~25 s; ohne RUN_E2E wird er übersprungen
```

Die Tests verwenden Fakes für Ollama (Embeddings, Chat) und eLabFTW und starten die Streamlit-Seiten headless (`streamlit.testing`). Abgedeckt sind u. a.: Paginierung/Retries des Clients, Einheiten-Umrechnung, Parser und Chunking, SQLite-Store samt Volltextindex, SQL-Filter und Aggregate (inkl. Injection-Strings), inkrementeller Sync (Änderung, Löschung, Ausfälle), Hybridsuche, Planner (LLM-Pfad, Fallbacks, Validierung), Engine (Routing, Streaming, Quellen, Zahlen-Guard, Prompt-Injection), CLI, Doctor, Eval sowie beide Oberflächen inkl. Teilfehlern beim Anlegen.

**Nicht** automatisch getestet (braucht echte Dienste): Antwortqualität der echten Modelle, das tatsächliche Verhalten deines eLabFTW (Feldnamen, Seitengröße) und die Whisper-/Vision-Modelle. Dafür gibt es den manuellen Ablauf oben (`doctor`, `sync --limit 5`, `eval`).

## 6. Fehlersuche

| Symptom | Ursache / Lösung |
|---|---|
| „Ollama is not reachable …“ | `ollama serve` starten / `OLLAMA_HOST` prüfen |
| „Ollama has no model …“ | `ollama pull <modell>` (Name steht in der Meldung) |
| „ELABFTW_API_KEY is not set“ | `.env` anlegen (siehe 2.) |
| „TLS verification failed“ | `ELABFTW_VERIFY_TLS=false` (nur lokal) oder CA-Pfad angeben |
| „Refusing to delete everything“ beim Sync | eLabFTW lieferte 0 Experimente, der Index ist aber gefüllt (falscher Key/Team?). Nur mit `--force-empty` wird trotzdem geleert |
| Index leer / „Der Index ist noch leer“ | `python -m lab_chatbot sync` bzw. `seed-demo` |
| Chatbot findet zu wenig / zu viel | `search "…"` zeigt Ranking und Scores; Schwelle `LAB_MIN_DENSE` anpassen |
| „Source material is too long …“ (Assistant) | Quelle kürzen oder `LAB_ASSISTANT_NUM_CTX` erhöhen |

## 7. Bekannte Grenzen

* Nicht gegen einen echten eLabFTW-Server getestet; die Feldnamen der API-Antworten werden defensiv geparst (`parse.py`), Seitengröße/`offset` sind angenommen.
* Der Index ist **einbenutzerfähig**: wer den Chatbot bedient, sieht alles, was der verwendete API-Key sehen darf. Keine Rechteprüfung pro Nutzer.
* Aggregate (min/max/Ø) lassen sich nicht mit einer *semantischen* Bedingung kombinieren („maximale Belichtungszeit bei verrauschten Messungen“).
* Die Einheitentabelle (`units.py`) deckt gängige SI-Einheiten ab; unbekannte Einheiten werden nur untereinander (gleicher Text) verglichen, nie geraten.
* Die gemeinsame Oberfläche wurde mit Streamlit 1.67 getestet (`st.navigation`, ab 1.36 vorhanden; ältere Versionen sind ungeprüft). Beim Seitenwechsel leert Streamlit alle Eingabefelder der nicht angezeigten Seite; `app.py` sichert sie vorher (siehe `lab_assistant/state.py`, `lab_chatbot/state.py`). Neue Eingabefelder im Assistant mit eigenem Schlüssel müssen dort eingetragen werden, sonst gehen sie beim Seitenwechsel verloren.
* Die Embedding-Prefixe für `embeddinggemma` stehen in `embeddings.py` (`PREFIXES`); bei anderem Modell dort prüfen.
* Der Schwellwert `LAB_MIN_DENSE=0.30` ist ein begründeter Startwert, kein gemessener – am besten mit eigenen Fragen und `eval` nachjustieren.

## 8. Struktur

```
app.py             gemeinsamer Einstieg: Navigation Lab Assistant | Lab Chatbot (`streamlit run app.py`)
shared/            eLabFTW-Client (Retries, Paginierung, TLS) und Konfiguration
lab_assistant/     Streamlit-App: Eingabe → Entwurf → Formular → Anlegen
lab_chatbot/       Chatbot: store · sync · sql_tools · retrieval · planner · engine · cli · ui
tests/             pytest-Suite mit Fakes; tests/e2e = optionaler Browser-Test
deploy/elabftw/    Beispiel-docker-compose (ohne Secrets)
docs/              Architektur
```

Details zum Aufbau des Chatbots: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Was gegenüber dem Ausgangsstand geändert wurde: [`CHANGES.md`](CHANGES.md).
