# AI Lab Assistant

Zwei lokale Werkzeuge für [eLabFTW](https://www.elabftw.net), das Laborbuch-System. Alle Modelle laufen über [Ollama](https://ollama.com) auf deinem Rechner.

- **Lab Assistant** schreibt Einträge: Notizen, Fotos handschriftlicher Notizen oder Audio werden zu einem Entwurf, du prüfst ihn, dann wird er in eLabFTW angelegt.
- **Lab Chatbot** beantwortet Fragen zum Laborbuch. Zahlen kommen exakt aus SQL, Inhalte aus semantischer Suche (RAG) mit Quellenangaben.

Beide laufen in einer Oberfläche (`streamlit run app.py`) oder einzeln.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20(local)-black)
![Status](https://img.shields.io/badge/status-prototype-orange)

> Experimenteller Prototyp. Prüfe jeden erzeugten Eintrag, bevor du ihn speicherst.

| | Lab Assistant (`lab_assistant/`) | Lab Chatbot (`lab_chatbot/`) |
|---|---|---|
| Zweck | Notizen, Fotos, Audio → Entwurf → eLabFTW-Eintrag | Fragen an das Laborbuch |
| Schreibt in eLabFTW | ja | nein, nur lesen |
| Modelle | Qwen 2.5 (Text), MiniCPM (Handschrift, optional), Whisper (Audio, optional) | Qwen 2.5 (Chat), embeddinggemma (Embeddings) |
| Einzeln starten | `streamlit run lab_assistant/app.py` | `streamlit run lab_chatbot/ui.py` oder `python -m lab_chatbot chat` |

## Installation

Voraussetzungen: Python 3.10 oder neuer (getestet mit 3.13), [Ollama](https://ollama.com/download), ein eLabFTW-Server für den Echtbetrieb. Zum Ausprobieren reicht die Demo (siehe unten).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama pull embeddinggemma
ollama pull minicpm-v4.6:1b
```

Welche Requirements-Datei du brauchst:

- `requirements.txt`: alles inklusive Tests
- `requirements-app.txt`: nur die gemeinsame Oberfläche
- `requirements-assistant.txt` bzw. `requirements-chatbot.txt`: nur eine Hälfte

Audio-Transkription (optional) läuft offline mit einem lokal abgelegten Whisper-Modell:

```bash
python -c "from faster_whisper import download_model; download_model('base', output_dir='models/whisper/base')"
```

## Konfiguration

```bash
cp .env.example .env
```

Trage mindestens `ELABFTW_API_KEY` ein. Ist dein eLabFTW nicht unter `https://localhost:8443/api/v2` erreichbar, setze außerdem `ELABFTW_URL`. Echte Umgebungsvariablen haben Vorrang vor `.env`.

| Variable | Bedeutung | Standard |
|---|---|---|
| `ELABFTW_API_KEY` | API-Key aus eLabFTW (Profil → API-Keys). Assistant braucht Schreibrecht, Chatbot Leserecht | Pflicht |
| `ELABFTW_URL` | API-URL | `https://localhost:8443/api/v2` |
| `ELABFTW_VERIFY_TLS` | `true`, `false` oder Pfad zu einem CA-Bundle | prüfen, außer bei `localhost` |
| `OLLAMA_HOST` | Ollama-Server | `http://localhost:11434` |
| `LAB_CHAT_MODEL` / `LAB_EMBED_MODEL` | Chatbot-Modelle | `qwen2.5:7b` / `embeddinggemma` |
| `LAB_ASSISTANT_MODEL` / `LAB_VISION_MODEL` | Assistant-Modelle | `qwen2.5:7b` / `minicpm-v4.6:1b` |
| `LAB_DB_PATH` | Index-Datenbank des Chatbots | `data/chatbot/labbook.db` |
| `LAB_MIN_DENSE` | Mindest-Ähnlichkeit für Treffer | `0.30` |
| `LAB_OLLAMA_TIMEOUT` | Sekunden, bis eine hängende Ollama-Anfrage abbricht | `300` |
| `LAB_INDEX_IMAGES` | Bildanhänge beim Sync mit dem Bildmodell lesen und durchsuchbar machen | `false` |
| `LAB_MAX_IMAGE_MB` | Größere Bilder werden übersprungen (Sync) oder abgelehnt (Chat) | `20` |
| `LAB_VECTOR_DIMS` | Nur die ersten N Dimensionen im Speicher halten (nur für Matryoshka-Modelle, siehe „Große Datenmengen“). `0` = alle | `0` |
| `LAB_SYNC_INTERVAL` | Abstand der Sync-Runden des Workers in Sekunden (Minimum 60) | `900` |
| `LAB_MAX_CONCURRENT_LLM` | Gleichzeitige Antwort-Generierungen. Nur erhöhen, wenn Ollama mit `OLLAMA_NUM_PARALLEL` > 1 und genug VRAM läuft | `1` |
| `LAB_LLM_QUEUE_TIMEOUT` | Sekunden in der Warteschlange, danach „Zu viele Anfragen warten …“ | `600` |
| `LAB_QUERY_LOG` | Gestellte Fragen samt Zeiten lokal protokollieren (`false` = Fragen nicht speichern) | `true` |
| `LAB_RERANK_MODEL` | Optionaler Reranker, leer = aus (siehe „Große Datenmengen“) | leer |

Alle Variablen stehen in `.env.example`.

## Schnellstart ohne eLabFTW (Demo)

Der Chatbot bringt ein synthetisches Laborbuch mit acht Experimenten mit. Es liegt in einer eigenen Datenbank und vermischt sich nicht mit echten Daten. Benötigt wird nur Ollama.

```bash
python -m lab_chatbot doctor
python -m lab_chatbot seed-demo
LAB_DB_PATH=data/chatbot/demo.db streamlit run app.py
```

Fragen zum Ausprobieren:

- `Wie viele Experimente haben den Tag raman?` → SQL, exakt
- `Welche Messungen hatten eine Wellenlänge über 600 nm?` → SQL mit Einheiten-Umrechnung
- `Was war die maximale Belichtungszeit?` → SQL-Aggregat
- `Warum war das Rauschen bei Probe A3 so hoch?` → RAG mit Quellen `[1]`
- `Wie war das Rauschen bei Laserleistung über 30 mW?` → beides

Mit `--explain` (CLI) oder dem Schalter „Plan & Zeiten anzeigen“ (UI) siehst du, wie die Frage verstanden wurde.

## Echtbetrieb

Die Tests laufen gegen Attrappen. Ob alles mit deinem Ollama und eLabFTW funktioniert, zeigt erst dieser Ablauf:

1. `.env` anlegen und `ELABFTW_API_KEY` eintragen. Ollama muss laufen (`ollama serve` oder die Ollama-App).
2. Assistant prüfen: `python -m lab_assistant doctor`. Liest Templates, Kategorien, Status, Ressourcen und Tags aus eLabFTW, prüft das Textmodell und meldet optionale Teile.
3. Chatbot prüfen: `python -m lab_chatbot doctor`. Holt drei echte Experimente, parst sie und ruft die Modelle einmal auf. Mit `--quick` nur Vorhandensein prüfen.
4. Wenige Experimente einlesen und testen:
   ```bash
   python -m lab_chatbot sync --limit 5
   python -m lab_chatbot search "ein Wort aus einem deiner Einträge"
   ```
5. Alles einlesen: `python -m lab_chatbot sync`
6. Oberfläche starten: `streamlit run app.py`

Der Index wird nicht von selbst aktualisiert. Nach neuen Einträgen in eLabFTW `sync` erneut ausführen, den Sync-Knopf in der UI nutzen (läuft im Hintergrund, der Chat bleibt benutzbar, die Seitenleiste zeigt den Fortschritt) oder den Worker starten (siehe „Große Datenmengen“).

## Lab Assistant

Start: `streamlit run lab_assistant/app.py` (oder als Seite in `app.py`).

1. Notizen tippen und/oder Dateien hochladen. Fotos werden per OCR gelesen, Audio transkribiert, andere Dateien als Anhang angehängt. Der erkannte Text ist editierbar.
2. „Generate laboratory entry“ erzeugt einen Entwurf. Das Modell wählt nie Template, Kategorie, Status oder Tags. Das tust du im Formular.
3. Formular prüfen, inklusive Custom Fields des Templates, dann „Create lab entry“.

Die App markiert Zahlen, die im Entwurf stehen, aber nicht in deinen Quellen vorkommen.

## Lab Chatbot

```bash
python -m lab_chatbot doctor
python -m lab_chatbot sync
python -m lab_chatbot chat
```

Weitere Befehle:

- `ask "…"`: eine Frage
- `search "…"`: nur das Retrieval-Ranking, ohne Antwort
- `stats`: Index-Statistik
- `sync --full`: alles neu einlesen, zum Beispiel nach einem Modellwechsel
- `sync --limit 5`: nur die ersten Experimente, zum Testen
- `eval --golden <datei>`: Testfragen mit erwarteten Ergebnissen laufen lassen
- `seed-demo`: Demo-Laborbuch anlegen
- `worker`: Index regelmäßig im Hintergrund aktualisieren (siehe „Große Datenmengen“)
- `querylog`: Auswertung des Frageprotokolls (`--last N` listet die letzten Fragen)

### Bilder

- **Im Chat:** Bild (Foto, Screenshot, Plot, Gel, Notizseite) an die Frage anhängen. Das Bildmodell (`LAB_VISION_MODEL`) liest Text, Achsen und Werte. Die Antwort zitiert das Bild als eigene Quelle und nennt passende Experimente aus dem Laborbuch getrennt davon. Ohne Fragetext lautet die Frage: „Was zeigt dieses Bild?“
- **Im Index:** Mit `LAB_INDEX_IMAGES=true` liest `sync` auch Bildanhänge der Experimente und hängt den gelesenen Text an den Eintrag. Lesungen werden pro Datei-Hash gespeichert, ein unverändertes Bild geht nie zweimal ans Modell. Der erste Sync nach dem Einschalten liest alle Bilder und dauert entsprechend (grob 2 bis 10 s pro Bild).

Das Bildmodell liest nicht fehlerfrei. Unsicheres markiert es mit `[unclear]`. Werte aus Bildern vor der Verwendung prüfen.

### Große Datenmengen

Der Vektorindex im Speicher wird nicht mehr bei jeder Änderung komplett neu geladen. Neue Abschnitte kommen als Delta-Segmente dazu, gelöschte werden in der Tabelle `chunk_tombstones` vermerkt und als tot markiert. Kleine Segmente werden zusammengeführt, bei mehr als 25 % toten Zeilen wird der Index im Speicher verdichtet. Während ein Thread aktualisiert, nutzen andere Anfragen den bisherigen Stand und warten nicht.

Gemessen auf einem Mac mit 400.000 Abschnitten (768 Dimensionen): Laden 1,4 s, 1,2 GB RAM, Suche 25 ms. Aktualisierung nach 50 geänderten Experimenten 9 ms (vorher: jedes Mal komplett neu laden). Faustregel: rund 3 KB RAM pro Abschnitt bei 768 Dimensionen.

**Weniger RAM mit `LAB_VECTOR_DIMS`:** Bei Matryoshka-Embedding-Modellen (embeddinggemma, nomic-embed-text v1.5, mxbai-embed-large, qwen3-embedding) genügen die ersten N Dimensionen (z. B. `256`) im Speicher. Die besten Kandidaten werden mit den vollständig gespeicherten Vektoren neu bewertet, das endgültige Ranking bleibt also exakt. Ein Neu-Indexieren ist zum Umschalten nicht nötig. Gemessen mit 400.000 Abschnitten: 416 MB, Suche 22 ms (rund 1 KB pro Abschnitt). Bei Modellen ohne Matryoshka-Training nicht verwenden.

**Sync:**

- Das Listing aus eLabFTW wird seitenweise gelesen (konstanter Speicher).
- Es läuft nur ein Sync gleichzeitig, auch über Prozesse hinweg (Sperre in der Datenbank, läuft nach 180 s ab, falls der Halter abstürzt). Andere erhalten „Ein Sync läuft bereits“.
- Jedes Experiment wird einzeln committet. Ein abgebrochener Sync verliert nichts, ein abgebrochener `sync --full` setzt dort fort, wo er stand.
- Experimente werden nur aus dem Index gelöscht, wenn das gesamte Listing gelesen wurde.
- Fortschritt und Status stehen in der Datenbank und erscheinen in der UI, auch bei einem Sync, der woanders gestartet wurde. Nach großen Syncs wird FTS optimiert und das WAL gekürzt.

**Hintergrund-Worker:**

```bash
python -m lab_chatbot worker              # sofort syncen, dann alle LAB_SYNC_INTERVAL Sekunden
python -m lab_chatbot worker --once       # eine Runde, für cron oder systemd-Timer
python -m lab_chatbot worker --full       # erste Runde vollständig
python -m lab_chatbot worker --interval N # Abstand in Sekunden
```

Bei Ausfall von eLabFTW oder Ollama wartet der Worker exponentiell länger (bis 1 h). SIGTERM und Strg-C beenden ihn sauber zwischen zwei Experimenten. Bei großen Laborbüchern und in Gruppen empfiehlt sich der Worker als eigener Dienst (z. B. systemd-Unit oder `nohup`) neben der Streamlit-Oberfläche.

**Planner-Katalog:** Der Planner-Prompt enthält die häufigen Tags, Kategorien und Messgrößen plus die, die zur Frage passen (ohne Beachtung von Akzenten, mit Präfix- und Tippfehler-Toleranz). Planwerte werden gegen alle vorhandenen Werte geprüft. Vorher waren nur die 60 häufigsten Tags und 80 Messgrößen bekannt, seltene Namen fielen in großen Laborbüchern heraus. Gemessen: rund 13 ms pro Frage bei 50.000 Experimenten, 5.000 Tags und 3.000 Messgrößen.

**Optionales Reranking:** `LAB_RERANK_MODEL` (z. B. `BAAI/bge-reranker-v2-m3`, mehrsprachig, oder das kleinere `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) bewertet die besten 40 Kandidaten neu. Braucht `pip install sentence-transformers` (zieht torch nach, groß) bzw. `pip install -e ".[rerank]"`. Bei einem Fehler gilt die normale Reihenfolge. `doctor` prüft das Paket, wenn die Variable gesetzt ist. Noch nicht mit einem echten Modell getestet.

**Grenzen:** Einen echten ANN-Index (HNSW) gibt es nicht: hnswlib hat hier keine Wheels für Python 3.14, und die exakte Suche bleibt bis einige Millionen Abschnitte schnell genug. Darüber oder bei mehreren Laboren bzw. Teams wäre Postgres mit pgvector der nächste Schritt (siehe `docs/ARCHITECTURE.md`).

### Für eine Arbeitsgruppe

Ein Rechner mit Ollama betreibt die Oberfläche, alle anderen öffnen sie im Browser:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Alle teilen sich einen Index. Es läuft immer nur ein Sync gleichzeitig. Antworten laufen über eine faire Warteschlange (FIFO) vor dem Chatmodell: standardmäßig wird eine Antwort gleichzeitig erzeugt (`LAB_MAX_CONCURRENT_LLM`), die UI zeigt „In der Warteschlange: Position N“. Ollama bleibt der Engpass: eine GPU beantwortet eine Frage nach der anderen (vLLM oder ein zweiter GPU-Server würde helfen).

Jede beantwortete Frage schreibt eine Zeile in die Tabelle `query_log` der Index-Datenbank (Zeit, Frage, Route, Quellen, Zeiten je Schritt, Fehler, Modelle). Es bleiben die neuesten 100.000 Zeilen. `python -m lab_chatbot querylog` zeigt p50/p95 je Schritt, Fehlerquote und Anteil der Suchen ohne Quellen. Pro Frage gibt es außerdem eine INFO-Logzeile mit den Zeiten, ohne Fragetext. Die Fragen liegen lokal in der Index-Datenbank. Wer das nicht möchte, setzt `LAB_QUERY_LOG=false`. Streamlit hat keine Anmeldung. Die Oberfläche gehört deshalb nur ins Instituts-Netz oder hinter einen Reverse-Proxy mit Login. Jeder Nutzer sieht alles, was der API-Key des Servers sehen darf.

Intern entscheidet ein Planner pro Frage, ob SQL, Hybridsuche (Vektoren plus Stichwörter) oder beides nötig ist. Details in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Gemeinsame Oberfläche

`streamlit run app.py` zeigt Lab Assistant und Lab Chatbot als zwei Seiten in der Seitenleiste. Beim Wechsel bleiben Eingaben, Chatverlauf und Seitenleisten-Schalter erhalten. Hochgeladene Dateien werden als „detached“ weitergeführt und lassen sich mit „Remove this file“ entfernen.

## Make-Befehle

```bash
make install     # alle Requirements installieren
make test        # Test-Suite, offline
make app         # gemeinsame Oberfläche
make chatbot     # nur Chatbot-Oberfläche
make assistant   # nur Assistant-Oberfläche
make sync        # Chatbot-Index aktualisieren
make doctor      # beide Diagnosen
make demo        # Demo-Laborbuch plus Testfragen
make e2e         # Browser-Test (braucht playwright)
```

## Lokales eLabFTW (optional)

```bash
cd deploy/elabftw
cp .env.example .env
docker compose -f docker-compose.example.yml up -d
```

Die Secrets stehen in `deploy/elabftw/.env`. Die Datei ist nicht im Repo und wird nicht committet. eLabFTW läuft danach unter `https://localhost:8443`.

## Tests

```bash
pytest
```

Die Tests laufen komplett offline, ohne Ollama und ohne eLabFTW. GitHub Actions führt sie bei jedem Push auf `main` und bei jedem Pull Request aus (Python 3.10 und 3.12). Sie nutzen Fakes und prüfen unter anderem Client, Einheiten, Parser, Chunking, SQL-Filter, Sync, Hybridsuche, Planner, Engine und beide Oberflächen.

Browser-Test der gemeinsamen Oberfläche (startet `app.py` mit Chromium, Attrappen für eLabFTW und Ollama):

```bash
pip install playwright
playwright install chromium
RUN_E2E=1 pytest tests/e2e -v
```

## Fehlersuche

| Meldung | Lösung |
|---|---|
| „Ollama is not reachable“ | `ollama serve` starten, `OLLAMA_HOST` prüfen |
| „Ollama has no model …“ | `ollama pull <modell>` mit dem Namen aus der Meldung |
| „ELABFTW_API_KEY is not set“ | `.env` anlegen |
| „TLS verification failed“ | `ELABFTW_VERIFY_TLS=false` (nur lokal) oder CA-Pfad angeben |
| „Refusing to delete everything“ | eLabFTW lieferte 0 Experimente, der Index ist aber gefüllt. Falscher Key oder Team? Nur `--force-empty` leert trotzdem |
| Index leer | `python -m lab_chatbot sync` oder `seed-demo` |
| Zu wenige oder zu viele Treffer | `search` zeigt Scores, dann `LAB_MIN_DENSE` anpassen |
| „Ein Sync läuft bereits“ | Ein anderer Sync oder Worker hält die Sperre. Abwarten, sie läuft nach 180 s ab, wenn der Halter abgestürzt ist |
| „Zu viele Anfragen warten …“ | Warteschlange voll: `LAB_LLM_QUEUE_TIMEOUT` erhöhen, `LAB_MAX_CONCURRENT_LLM` nur bei passender Ollama-Konfiguration |
| Antwort bricht nach Minuten ab | Modell lädt zu langsam: `LAB_OLLAMA_TIMEOUT` erhöhen oder kleineres Modell |
| „Vision model failed“ / Bild wird nicht gelesen | `ollama pull minicpm-v4.6:1b` (oder `LAB_VISION_MODEL`), Format JPEG/PNG, HEIC braucht `pillow-heif` |

## Privacy

Ollama, die Modelle, Whisper, Streamlit und eLabFTW können alle auf deinem Rechner laufen. Experimentdaten müssen dann nicht an externe KI-APIs. Die Sicherheit hängt von deinem Setup ab: Netzwerk, Authentifizierung, Backups und eLabFTW-Rechte.

Der Index ist für einen Benutzer gedacht. Wer den Chatbot bedient, sieht alles, was der API-Key sehen darf. Es gibt keine Rechteprüfung pro Nutzer.

## Wissenschaftliche Integrität

- Die Modelle nutzen nur das Material, das du angibst.
- Sie erfinden keine Messwerte, korrigieren unsichere OCR nicht stillschweigend und rechnen nicht selbst. Zählungen und Aggregate kommen aus SQL.
- Widersprüchliche Quellen, Einheiten und Zahlen bleiben erhalten. Messwerte werden von theoretischen Werten getrennt.
- OCR kann Ziffern, Vorzeichen, Exponenten und Einheiten falsch lesen. Prüfe jeden Eintrag vor dem Speichern.

## Bekannte Grenzen

- Nicht gegen einen echten eLabFTW-Server getestet. Die Feldnamen der API-Antworten werden defensiv geparst.
- Aggregate lassen sich nicht mit einer semantischen Bedingung kombinieren.
- Unbekannte Einheiten werden nur untereinander verglichen, nie geraten.
- Der Schwellwert `LAB_MIN_DENSE=0.30` ist ein Startwert. Mit eigenen Fragen und `eval` nachjustieren.
- Kein ANN-Index, Reranking noch nicht mit echtem Modell getestet (siehe „Große Datenmengen“).
- Die gemeinsame Oberfläche ist mit Streamlit 1.67 getestet. Ältere Versionen sind ungeprüft.

## Struktur

```text
app.py            gemeinsamer Einstieg (Navigation)
shared/           eLabFTW-Client (Retries, Paginierung, TLS) und Konfiguration
lab_assistant/    Streamlit-App: Eingabe, Entwurf, Formular, Anlegen
lab_chatbot/      Chatbot: Store, Sync, SQL, Retrieval, Planner, Engine, CLI, UI
tests/            pytest-Suite, tests/e2e für den Browser-Test
deploy/elabftw/   Beispiel-docker-compose ohne Secrets
docs/             Architektur
```

Lizenz: noch keine. Frag den Maintainer, bevor du den Code wiederverwendest. Vor externen Beiträgen sollte eine Lizenz (z. B. MIT oder Apache-2.0) ergänzt werden.

Beiträge sind willkommen, zum Beispiel bei OCR und Transkription echter handschriftlicher Notizen, eLabFTW-Integration, Retrieval und Sicherheit (TLS, Authentifizierung, Rechte pro Nutzer). Öffne vorher ein Issue für größere Änderungen und halte Pull Requests klein.
