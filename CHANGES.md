# Änderungen gegenüber dem Ausgangsstand

Ausgangsstand: `app/` (Streamlit-Assistant mit eingebettetem Chroma-RAG-Chat), `elabftw/` (docker-compose) und die Vorlage `rag-chatbot-from-scratch`. Dieses Dokument listet, was geändert, hinzugefügt und entfernt wurde – und warum. Alles hier Genannte ist im Code vorhanden; für das meiste gibt es Tests (`pytest`).

## 1. Struktur

| Vorher | Jetzt |
|---|---|
| Ein Paket `app/` (Assistant **und** Chat vermischt: `render_chatbot()` in `app.py`, Chat importiert `MODEL` aus `services.ai`) | `lab_assistant/` und `lab_chatbot/` sind getrennte Codebasen, die sich nicht importieren. Gemeinsam nur `shared/` (eLabFTW-Client + Konfiguration) |
| Imports relativ zum Startordner (`from services.x import …`, nur aus `app/` startbar) | echte Pakete; beide Apps starten aus jedem Verzeichnis (`streamlit run lab_chatbot/ui.py`), zusätzlich `python -m lab_chatbot …` und `pip install -e .` (Befehl `lab-chatbot`) |
| Eine `requirements`-Mischung | `requirements-assistant.txt`, `requirements-chatbot.txt`, `requirements-dev.txt`, `requirements.txt` |
| – | `pyproject.toml`, `Makefile`, `pytest.ini`, `.env.example`, `.gitignore` |

### 1a. Gemeinsame Oberfläche (nachträglich ergänzt)

Anforderung: Lab Assistant **und** Lab Chatbot sollen in derselben Streamlit-Oberfläche erscheinen.

| Was | Wo |
|---|---|
| Neuer Einstieg `streamlit run app.py`: Seitenleisten-Navigation mit zwei Seiten (Start: Lab Assistant). Es ist die einzige Stelle, die beide Pakete kennt; sie importieren sich weiterhin nicht gegenseitig. Seiten werden erst beim Anzeigen importiert, ein fehlendes Paket betrifft nur seine Seite (mit Hinweis auf die passende Requirements-Datei) | `app.py` |
| Beide `render_app(...)` haben `embedded=False/True`: eingebettet rufen sie `st.set_page_config` nicht selbst auf (das gehört dem Launcher; doppelt aufgerufen wäre es ein Streamlit-Fehler) und der Assistant zeigt den Hinweis „starte den Chatbot per CLI“ nicht mehr. Einzelstart (`lab_assistant/app.py`, `lab_chatbot/ui.py`) funktioniert unverändert | `lab_assistant/app.py`, `lab_chatbot/ui.py` |
| **Eingaben überleben den Seitenwechsel.** Streamlit vergisst Widget-Werte aller Seiten, die nicht angezeigt werden (im Experiment verifiziert: Notizen leer nach Hin- und Zurückwechseln). Der Launcher schreibt die Werte vor dem Zeichnen neu in den Session-State („Pinning“). Betroffen: Notizen, Titel, Text, Template, Kategorie, Status, Ressource, Tags, „trotzdem erneut anlegen“, alle Custom Fields; im Chatbot die Schalter der Seitenleiste | `lab_assistant/state.py`, `lab_chatbot/state.py` |
| **Hochgeladene Dateien überleben den Seitenwechsel.** Das Upload-Widget lässt sich nicht pinnen und ist danach leer; ohne Gegenmaßnahme hätte der Assistant die Dateien samt bereits berechnetem OCR-/Transkript-Text stillschweigend verworfen. Jetzt werden sie als „detached“ weitergeführt (Hinweis + Knopf „Remove this file“); erneutes Hochladen derselben Datei übergibt sie wieder dem Upload-Feld, normales Entfernen im Upload-Feld löscht sie wie zuvor | `lab_assistant/components/input.py` |
| Fix, im Browser aufgefallen: die Beispielfragen des Chatbots blieben nach der ersten Frage bis zur nächsten Interaktion stehen (gab es schon vorher) | `lab_chatbot/ui.py` |
| `requirements-app.txt` (beide Hälften), `make app`, `make e2e`, Extra `app` in `pyproject.toml` | Wurzel |
| Tests: 18 neue Tests (Pinning, Launcher, `embedded`, fehlende Pakete, echte `app.py` ohne API-Key, detached Dateien, Beispielfragen) + **Browser-Test** mit echtem Chromium, echtem `streamlit run app.py` und Attrappen für eLabFTW/Ollama | `tests/test_combined_app.py`, `tests/test_assistant_ui.py`, `tests/e2e/` |

### 1b. Anbindung an die echten Dienste (nachträglich ergänzt)

Frage: „Sind sie jetzt mit dem echten Ollama und eLabFTW verbunden?“ Ehrlich: Die Tests laufen gegen Attrappen; eine Prüfung gegen dein System ist von außen nicht möglich. Deshalb gibt es jetzt Werkzeuge, die das bei dir in einem Befehl feststellen, und ich habe die API-Annahmen gegen die offizielle OpenAPI-Spezifikation abgeglichen.

| Was | Wo |
|---|---|
| **Abgleich mit der API-Doku** (OpenAPI v2): `tags` ist ein String (Trennzeichen offen, wird in allen Formen erkannt), Listenfelder `body`, `metadata`, `modified_at`, `category_title`, `status_title`, `fullname` stimmen; `limit` Standard 15, `offset` vorhanden; `Location`-Header nach dem Anlegen stimmt; Upload multipart mit `comment` stimmt | – |
| **Abweichung 1:** Beim Anlegen aus einem Template nennt die aktuelle Doku den Schlüssel `category_id`, der alte Code (und mein Port davon) sendete `template`. Auf einem aktuellen Server (`elabimg:stable`) hätte das Template stillschweigend gefehlt. Jetzt werden **beide** Schlüssel gesendet (unbekannte werden ignoriert) | `lab_assistant/elab_api.py` |
| **Abweichung 2:** Kategorien, Status und Tags holte der alte Code über `teams/current/...`; dokumentiert sind `teams/<id>/...` und `team_tags`. Jetzt: erst der bisherige Pfad, bei 400/404/405 der dokumentierte (Team-ID aus `users/me`). Andere Fehler (401, 500, Netz) werden nicht verdeckt | `lab_assistant/elab_api.py` |
| `python -m lab_assistant doctor` (neu): Struktur-Listen und Template-Felder aus eLabFTW, Textmodell vorhanden **und einmal aufgerufen**, Vision/Audio/HEIC als optionale Hinweise | `lab_assistant/doctor.py`, `__main__.py` |
| `python -m lab_chatbot doctor` erweitert: holt drei echte Experimente und **parst** sie (meldet fehlende Felder: Titel, Datum, Text, Änderungszeit), ruft Embedding- und Chat-Modell je einmal auf (`--quick` lässt das weg). Ist Ollama nicht bereit, gibt es keinen zweiten, verwirrenden Fehler | `lab_chatbot/doctor.py`, `cli.py` |
| Abschlusszeile ehrlich: „All good.“ erscheint nur, wenn auch die nicht-kritischen Prüfungen (z. B. eLabFTW) bestanden haben | `shared/diagnostics.py` (gemeinsame Bausteine beider doctor-Befehle) |
| Browser-Test erweitert: leerer Index → Sync-Knopf → 8 Experimente → Frage; die Attrappen prüfen den API-Key, liefern wie eLabFTW nur 15 Einträge ohne `limit` und zeichnen alle Anfragen auf | `tests/e2e/` |
| README: neue Schritt-für-Schritt-Anleitung „Erstinbetriebnahme“ (4.); **keine `#`-Kommentare mehr in Befehlsblöcken** (macOS-zsh behandelt sie als Argumente: dein Fehler `file or directory not found: #`) | `README.md` |

## 2. Gemeinsame Basis `shared/` (neu)

Ersetzt die zwei getrennten, ungleich fehlerhaften Implementierungen in `services/elabftw.py`:

* **Kein Import-Zeit-Zugriff auf die Umgebung.** Vorher `API_KEY = os.environ["ELABFTW_API_KEY"]` auf Modulebene → `KeyError`-Absturz beim bloßen Import. Jetzt `ElabFTWConfig.from_env()` mit klarer Fehlermeldung.
* **TLS:** vorher überall `verify=False` und global abgeschaltete Warnungen. Jetzt `ELABFTW_VERIFY_TLS` (true/false/CA-Pfad); Standard: verifizieren, nur für `localhost` (selbstsigniert) nicht.
* **Timeouts** (vorher keine → hängende UI) und **Retries** für GET bei Verbindungsfehlern/502–504; POST/PATCH werden bewusst *nicht* wiederholt (Duplikatgefahr).
* **Paginierung** `iter_paginated` (limit/offset). Vorher: ein einziger GET – eLabFTW liefert dabei standardmäßig nur die ersten ~15 Einträge, also fehlten Ressourcen/Tags/Templates im Formular und der RAG-Sync sah nur die ersten Experimente. Schutz gegen Server, die `offset` ignorieren (Endlosschleife) und gegen gekappte Seitengrößen.
* Lesbare Fehler (Status + Antworttext), `created_id()` aus dem `Location`-Header, ein gemeinsamer Upload-Pfad.

## 3. Lab Chatbot

### 3.1 Denkfehler im Ausgangscode → Korrektur

| # | Problem im Ausgangscode | Jetzt |
|---|---|---|
| 1 | Nur semantisch („Zahlen per RAG“): „Wie viele …?“, „Wellenlänge > 600 nm“, „maximale Belichtungszeit“ sind per Vektorsuche nicht beantwortbar; Tags, Kategorie, Status und Custom Fields wurden gar nicht indexiert (nur Titel/Datum/Body) | **SQL für alles Numerische/Strukturierte** (`sql_tools.py`) + **RAG für Inhalte**, vom Planner pro Frage kombiniert. Tags, Kategorie, Status, Autor und Custom Fields sind abfragbare Tabellen |
| 2 | Ganzer Eintrag = ein Embedding. Lange Einträge werden abgeschnitten/verwässert, Details gehen unter | Markdown-bewusstes Chunking mit Kontext-Kopf je Chunk (`chunking.py`); Formeln/Code bleiben ganz |
| 3 | Immer die 5 ähnlichsten Einträge, **ohne Relevanzschwelle** – auch bei Unsinnsfragen bekommt das Modell 5 unpassende Einträge, und die „Quellen“ zeigen alle 5 | Relevanz-Gates (`LAB_MIN_DENSE`, Stichwort-Abdeckung), Cap pro Experiment; **nichts Relevantes ⇒ feste Antwort ohne LLM-Aufruf**; Quellen nur wirklich zitierte |
| 4 | Dokumente und Fragen wurden identisch eingebettet (kein Query-/Dokument-Prefix) | getrennte `embed_documents`/`embed_query` mit modellspezifischen Prefixen |
| 5 | Nur Vektorsuche: Probenbezeichner wie `A3`, `532nm` werden unzuverlässig gefunden | **Hybrid**: Vektoren + BM25 (FTS5), per RRF fusioniert; leichtes Stemming für deutsche Flexion |
| 6 | `sync_labbook()` existierte, wurde aber **nirgends aufgerufen** (kein Button, kein CLI); bei Aufruf: alles jedes Mal neu eingebettet, gelöschte Experimente blieben im Index, nur erste API-Seite, Fehler per `print` | `sync.py` vollständig ausgebaut: CLI `sync`, UI-Button, inkrementell (Zeitstempel + Inhalts-Hash + Index-Signatur), paginiert, Löschungen werden nachgezogen, Fehler pro Experiment gesammelt, **Schutz davor, bei leerem Listing den ganzen Index zu löschen**, atomares Ersetzen pro Experiment |
| 7 | Chroma-Client beim **Import** geöffnet, Pfad aus dem Dateistandort; zweite, getrennte Datenhaltung ohne Bezug zu Filtern | eine SQLite-Datei (Struktur + Chunks + Vektoren + FTS), erst bei Gebrauch geöffnet, Pfad per `LAB_DB_PATH`. Chroma entfällt |
| 8 | LLM-Aufruf ohne `num_ctx` (Ollama kappt zu lange Prompts **am Anfang** – die Regeln im System-Prompt gehen zuerst verloren) | `num_ctx` explizit; Kontextbudget `LAB_MAX_CONTEXT_CHARS` |
| 9 | Notizen unmarkiert in den Prompt verkettet (Prompt-Injection über Laborbuchtext möglich) | `<source>`-Blöcke, System-Prompt erklärt sie zu Daten, Delimiter im Fremdtext werden entschärft |
| 10 | Keine Zitate, keine Prüfung der Antwort; das Modell durfte Zahlen frei „verbessern“ | nummerierte Quellen, `[n]`-Zitate werden geparst und als klickbare Quellen mit eLabFTW-Link gezeigt; **Zahlen-Guard** markiert Zahlen, die nirgends in Quellen/Fakten stehen; Zähl-/Min-/Max-/Ø-Antworten kommen deterministisch aus SQL |
| 11 | Kein Streaming, kein Gesprächsverlauf (Rückfragen wie „und bei Probe B1?“ unmöglich) | Token-Streaming, Verlauf der letzten Runden (`LAB_HISTORY_TURNS`) fließt in Planner und Antwort ein |
| 12 | Fehlerfall in `chatbot.py`: `return` nach dem Anhängen der Nutzerfrage → inkonsistenter Verlauf; rohe Exceptions im UI | Fehler werden als verständliche Meldungen mit Handlungshinweis angezeigt (Ollama aus, Modell fehlt, …) |
| 13 | Chat eingebettet in die Assistant-Seite | eigene App, eigene CLI |

### 3.2 Neu im Chatbot (`lab_chatbot/`)

| Datei | Inhalt |
|---|---|
| `parse.py` | robustes Parsen der API-Antwort (Tags als Liste/`\|`/Komma, Datumsformate, HTML→Markdown, `metadata.extra_fields` → Messwerte) |
| `units.py` | Einheiten → SI inkl. °C/°F-Offsets und deutschem Dezimalkomma; macht „> 600 nm“ über nm/µm hinweg korrekt |
| `chunking.py` | s. o. |
| `store.py` | Schema, Transaktionen, FTS5, Vektor-Cache mit Revisionszähler |
| `sql_tools.py` | parametrisierte Filter-/Zähl-/Gruppier-/Aggregat-Abfragen, Katalog der real vorhandenen Tags/Kategorien/Messgrößen; **kein** freies Text-to-SQL |
| `retrieval.py` | Hybridsuche |
| `planner.py` | eine LLM-Anfrage mit JSON-Schema → Absicht (`search/count/aggregate/list/chitchat`), Filter, Messwert-Bedingung. Jede Angabe wird gegen den Katalog geprüft/„gesnappt“ oder verworfen; bei kaputtem JSON Retry im JSON-Modus, dann **Regel-Planner** (erkennt Tags, Operatoren wie „über/unter/zwischen“, Aggregatwörter, „pro Status“) |
| `engine.py` | Orchestrierung, Streaming-Events, deterministische Antworten, Kontextbudget, Guards |
| `prompts.py` | Antwort-Prompt, zweisprachige feste Texte (de/en) |
| `embeddings.py`, `llm.py`, `factory.py`, `config.py`, `errors.py` | Adapter und Verdrahtung |
| `cli.py`, `__main__.py` | `sync`, `ask`, `chat`, `search`, `stats`, `eval`, `doctor`, `seed-demo` |
| `ui.py` | Streamlit-Chat: Streaming, „Verstanden als …“, Tabellen, Quellen, Warnungen, Sync-Button, Beispielfragen |
| `doctor.py` | prüft FTS5, Ollama, Modelle, Index/Modell-Konsistenz, eLabFTW – mit konkretem Behebungsbefehl |
| `evaluation.py`, `golden_demo.json` | Golden-Set-Messung: Treffer@k, MRR, Absichts-Genauigkeit, Antwortinhalt |
| `demo.py` | synthetisches Laborbuch zum Ausprobieren ohne eLabFTW (eigene DB) |

## 4. Lab Assistant (`lab_assistant/`, portiert aus `app/`)

Fehler im Ausgangscode und ihre Behebung:

| Problem im Ausgangscode | Jetzt |
|---|---|
| `create_complete_experiment` brach bei einem späten Fehler ab; der Zweig „Experiment angelegt, aber später Schritt fehlgeschlagen“ in `lab_form.py` war **unerreichbar** (`experiment_id` wurde erst nach Rückkehr der Funktion gesetzt). Ein fehlgeschlagener Tag/Upload verschwieg das angelegte Experiment | `CreateResult(experiment_id, warnings, done)`: jeder Schritt einzeln abgesichert, Rest läuft weiter, UI zeigt Experiment-ID **und** was fehlt |
| Doppelklick/Rerun legte doppelte Experimente an | Duplikat-Schutz (gleicher Inhalt ⇒ Button gesperrt, bewusstes „trotzdem erneut“) |
| `normalize_math_for_elabftw` entfernte **alle** Code-Fences im Text (zerstörte echte Codeblöcke) und ersetzte Math-Delimiter auch innerhalb von Code; `print()` des Bodys | nur ein umschließender ```markdown-Zaun wird entfernt; Code bleibt unangetastet; kein `print` |
| Custom Fields: leeres Zahlenfeld wurde zu `"0.0"` **erfunden**; `st.number_input` zeigte nur 2 Nachkommastellen; passte die vom Modell genannte Einheit/Option nicht, wurde **still die erste** gewählt | Zahlenfelder als Text mit Validierung (Komma/Punkt/`5.32e-7`, Original-Stellen bleiben); leer bleibt leer; unpassende Einheit/Option ⇒ Warnung und **keine** Vorauswahl; Fehler blockieren das Anlegen |
| Streamlit-Widgets mit festem `key` behielten nach einem neuen KI-Entwurf ihre alten Werte; Custom-Field-Keys kollidierten zwischen Templates | Keys enthalten Template-ID und Entwurfs-Version |
| Neuer KI-Entwurf setzte Template/Kategorie/Status/Tags des Nutzers zurück; identischer Entwurf wurde nicht erneut übernommen (Vergleich über `repr`) | Struktur-Auswahl des Nutzers bleibt; Entwurfs-Zähler statt `repr` |
| Dropdowns über `{titel: id}`: gleiche Titel überschrieben sich | Duplikate werden als `Titel (#id)` unterschieden |
| Uploads nur über den Dateinamen identifiziert (gleicher Name ⇒ Kollision, geänderte Datei blieb alt) | Identität = Inhalts-Hash + Name |
| Strukturdaten (5 API-Aufrufe) bei **jedem** Rerun, Template-Felder bei jedem Tastendruck | 5-Min-Cache + Reload-Button; Template-Felder einmal pro Template |
| `update_experiment_custom_fields` ersetzte `extra_fields` komplett | Merge: nicht angezeigte Template-Felder und übrige Metadaten bleiben erhalten |
| KI-Entwurf: ungeprüftes JSON, `measurements` als beliebiger Typ, kein `num_ctx` (stilles Abschneiden des System-Prompts bei langen Quellen), kein Retry | Validierung/Normalisierung (Reparaturen als Notiz sichtbar), `num_ctx` explizit, zu lange Quellen ⇒ klare Fehlermeldung, ein Retry bei kaputtem JSON |
| Keine Kontrolle, ob das Modell Zahlen erfunden oder weggelassen hat | **Zahlenvergleich Quelle ↔ Entwurf** (`checks.py`), Warnung vor dem Anlegen |
| Fotos: EXIF-Drehung ignoriert; HEIC als harte Abhängigkeit (Import-Absturz ohne `pillow-heif`); Temp-Dateien; `print` | EXIF-Orientierung, `pillow-heif` optional mit Hinweis, Bytes statt Temp-Datei, `logging` |
| Whisper-Pfad fest verdrahtet, Import beim Start | `LAB_WHISPER_DIR`, lazy import, Fehlermeldung inkl. Download-Befehl |
| Ein fehlschlagender Aufruf brachte die Seite zum Absturz; Fehlermeldungen roh | `AIError` mit Handlungshinweis (Ollama aus, Modell fehlt) |

Entfernt (tot bzw. ersetzt): `components/ai_input.py`, `audio_input.py`, `image_input.py` (nirgends importiert), `services/rag.py`, `services/rag_engine/`, `components/chatbot.py`, `generate_text()` und der Wrapper `generate_lab_entry_from_elab()` mit ungenutzten Parametern, `services/elabftw.py` (→ `shared/` + `lab_assistant/elab_api.py`).

Beibehalten, weil gut: die Integritäts-Regeln im Prompt (nichts erfinden/korrigieren, `[unclear]`), das Aufteilen von Fotos in überlappende Streifen, die Trennung von „Entwurf erzeugen“ und „Eintrag anlegen“, das Sperren der organisatorischen Felder für das Modell.

## 5. Deployment

`deploy/elabftw/docker-compose.example.yml`: auf die tatsächlich gesetzten Optionen der eigenen Compose-Datei reduziert; **alle Secrets** (DB-/Root-Passwort, `SECRET_KEY`) durch `${…}`-Pflichtvariablen aus `deploy/elabftw/.env` ersetzt (`.env.example` mit Anleitung zur Erzeugung). OpenCloning ist bewusst nicht enthalten.

## 6. Tests

`tests/` (offline, mit Fakes): Client, Einheiten, Parser/Chunking, Store/SQL, Sync, Retrieval, Planner, Engine, CLI/Doctor/Eval, beide Streamlit-Oberflächen headless, Assistant-Kern. Beim Schreiben der Tests wurden und sind echte Fehler aufgefallen und behoben worden, u. a.: zu große Mathe-/Code-Blöcke wurden mitten in der Formel zerschnitten; ein Absatz direkt vor einem Codeblock wurde mit ihm verschmolzen; „in“ ließ deutsche Kurzfragen als Englisch erscheinen; deutsche Flexion („nachjustierte“ ↔ „Nachjustieren“) fand die Stichwortsuche nicht; der Regel-Planner erkannte keine Messwert-Bedingungen; `stats` umging die Factory.

## 7. Stabilität, große Datenmengen, Bilder

* **Timeouts:** Alle Ollama-Clients entstehen in `shared/ollama.py` mit Timeout (`LAB_OLLAMA_TIMEOUT`, Standard 300 s) und werden wiederverwendet. Vorher konnte ein hängendes Ollama die Oberfläche unbegrenzt blockieren, und der Assistant baute pro Anfrage einen neuen Client.
* **Große Indizes:** Lange ID-Listen (z. B. „alle Experimente einer Kategorie“) liefen ab 32.766 IDs in das Parameterlimit von SQLite. Sie gehen jetzt als ein JSON-Parameter. Vektoren laden ohne Zwischenkopie, die Suche nimmt `argpartition` statt Vollsortierung. Neue Indizes auf Kategorie, Status und Embedding-Modell. SQLite-Pragmas für WAL-Betrieb.
* **Bilder im Chat:** Bilder lassen sich an Fragen anhängen (`engine.ask_stream(..., images=[...])`, Chat-Eingabe mit Datei-Upload, Größenlimit `LAB_MAX_IMAGE_MB`).
* **Bildanhänge im Index:** `LAB_INDEX_IMAGES=true` liest Bildanhänge beim Sync. Ein Cache pro Datei-Hash verhindert doppelte Modellaufrufe, ein kaputtes Bild überspringt nur sich selbst.
* **Gruppenbetrieb:** In der Oberfläche läuft nur ein Sync gleichzeitig.
* **Antwortsprache** wird ausdrücklich im Prompt genannt (das 7B-Modell antwortete sonst teils englisch auf deutsche Fragen).
* **Versionsgrenzen** nach oben für alle Abhängigkeiten, **CI** (`.github/workflows/tests.yml`) mit Python 3.10 und 3.12.
* Emojis aus den Oberflächen entfernt. Der Doctor-Test hing vom lokalen Whisper-Modell ab, Audio ist jetzt immer optional.

## 8. Bewusst nicht getan / offen

* Neue Eingabefelder im Assistant müssen in `lab_assistant/state.py` eingetragen werden, damit sie den Seitenwechsel überleben (Buttons und Upload-Feld ausgenommen).
* `AppTest` kann zwischen `st.Page`-Funktionen nicht wechseln; der echte Seitenwechsel wird nur vom optionalen Browser-Test (`RUN_E2E=1`) abgedeckt.
* Kein Test gegen echten eLabFTW-/Ollama-Betrieb (siehe README „Bekannte Grenzen“).
* Keine Rechteprüfung pro Nutzer im Chatbot-Index.
* Kein Reranker, kein Hintergrund-Sync, keine PDF-Anhänge oder Ressourcen im Index (Bildanhänge optional, siehe 7).
