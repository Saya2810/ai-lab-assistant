# Architektur des Lab Chatbots

Leitidee: **Zahlen sind ein Datenbankproblem, Bedeutung ist ein Suchproblem.** Ein Sprachmodell soll weder zählen noch Einheiten vergleichen noch Messwerte „erinnern“ – dafür gibt es SQL. Ein Datenbank-Filter versteht dagegen kein „warum war das Rauschen hoch?“ – dafür gibt es semantische Suche. Der Chatbot entscheidet pro Frage, was gebraucht wird, und kombiniert beides.

```
Frage ──► Planner (1 LLM-Aufruf, JSON-Schema)  ──validiert gegen──►  Katalog (echte Tags, Kategorien,
              │                                                         Messgrößen aus dem Index)
              │  Fallback bei kaputtem JSON: Regel-Planner (gleiche Validierung)
              ▼
   ┌── count / aggregate / list ──► parametrisierte SQL-Builder ──► Antwort aus Templates (kein LLM)
   │
   ├── search ──► SQL-Filter grenzen Kandidaten ein (optional)
   │              Hybrid-Retrieval: dichte Vektoren (Kosinus) + BM25 (FTS5) ──RRF──► Top-k Chunks
   │              nichts Relevantes? ──► feste „nichts gefunden“-Antwort, **kein** LLM-Aufruf
   │              sonst: LLM antwortet nur aus nummerierten <source>-Blöcken, Token-Streaming
   │              danach: Zitate [n] auswerten, Zahlen-Guard (erfundene Zahlen → Warnung)
   └── chitchat ──► Hilfetext
```

## Einbettung in die gemeinsame Oberfläche

`streamlit run app.py` zeigt Assistant und Chatbot als zwei Seiten (`st.navigation`). `app.py` kennt beide Pakete und sonst nichts: es setzt die Seitenkonfiguration, sichert die Eingabefelder (`state.py` je Paket) und startet die gewählte Seite im Modus `embedded=True`. Die Pakete selbst bleiben unabhängig und weiterhin einzeln startbar.

## Module (`lab_chatbot/`)

| Modul | Aufgabe |
|---|---|
| `parse.py` | eLabFTW-Payload → `ExperimentRecord` (Tags in allen Formen, Datum, HTML→Markdown, `metadata.extra_fields` → typisierte Messwerte, Inhalts-Hash) |
| `units.py` | Einheiten → SI (`value_si` + Dimension), inkl. Offsets (°C/°F) und deutschem Dezimalkomma |
| `chunking.py` | Markdown-bewusstes Chunking; jeder Chunk trägt einen Kopf `[Experiment id \| Titel \| Datum]` + Tags/Kategorie/Status; Code und Formeln bleiben ganz (bis ~2400 Zeichen), eigener Chunk für Messwerte |
| `store.py` | **eine** SQLite-Datei: `experiments`, `exp_tags`, `measurements`, `chunks` (Embedding als float32-BLOB), `chunks_fts` (FTS5). WAL, Verbindung pro Aufruf, `revision`-Zähler invalidiert den Vektor-Cache |
| `sync.py` | inkrementeller Sync: Listing paginiert, `modified_at`- und Inhalts-Hash-Vergleich, atomares Ersetzen pro Experiment, Löschung verschwundener Einträge, Schutz vor „leeres Listing löscht alles“ |
| `sql_tools.py` | `Filters`/`MeasurementCond` → WHERE-Builder (nur gebundene Parameter, kein Text-to-SQL), Liste/Zählen/Gruppieren/Aggregieren (über Einheiten hinweg), `Catalog` |
| `retrieval.py` | Hybridsuche mit RRF, Relevanz-Gates, Cap pro Experiment, leichtes Stemming für die Stichwortsuche |
| `planner.py` | Absicht + Filter + Messwert-Bedingung aus der Frage; jede Angabe wird gegen den Katalog „gesnappt“ oder verworfen (mit Notiz) |
| `engine.py` | orchestriert; `ask_stream` liefert Events `plan → tables → sources → token… → final` |
| `prompts.py` | Antwort-Prompt (Quellen = Daten, nicht Anweisungen), zweisprachige feste Texte |
| `embeddings.py`, `llm.py` | dünne Ollama-Adapter (Query-/Dokument-Prefix, explizites `num_ctx`, verständliche Fehler); Clients aus `shared/ollama.py` mit Timeout |
| `vision.py` | liest Bilder mit dem Bildmodell (Chat-Uploads, Bildanhänge beim Sync); hohe Seiten in 3 überlappenden Streifen, alles auf max. 1600 px verkleinert (`shared/images.py`) |
| `cli.py`, `ui.py`, `doctor.py`, `evaluation.py`, `demo.py`, `factory.py` | Bedienung, Diagnose, Messung, Demo-Daten, Verdrahtung |
| `state.py` | Schlüssel der Seitenleisten-Schalter, die einen Seitenwechsel in der gemeinsamen App überleben müssen |

## Entscheidungen und Begründungen

**Eine SQLite-Datei statt Chroma + Zusatz-DB.** Strukturierte Filter und Vektorsuche müssen *dieselben* Experimente meinen. In einer Datei sind sie transaktional konsistent; Löschen/Aktualisieren kann keine verwaisten Vektoren hinterlassen. Bei Laborbuch-Größe (Tausende Chunks) ist ein Matrixprodukt mit numpy schneller als jede Netzwerk-Abfrage; der Cache hängt am `revision`-Zähler.

**Hybrid statt nur dicht.** Probenbezeichner (`A3`, `P7`), `532nm` oder Gerätenamen findet BM25 zuverlässig, Paraphrasen findet die Vektorsuche. RRF fusioniert nur Ränge, braucht also keine vergleichbaren Scores. Gates: dichter Treffer ab `LAB_MIN_DENSE`; reiner Stichwort-Treffer nur bei ≥ 2 passenden Wörtern oder ≥ 50 % Abdeckung; höchstens 2 Chunks pro Experiment.

**Kein Text-to-SQL.** Das Modell liefert nur strukturierte Parameter (Tag, Zeitraum, Messgröße, Operator, Wert, Einheit). SQL entsteht ausschließlich in `sql_tools.py` mit gebundenen Parametern; Operatoren und `group_by` stammen aus festen Whitelists.

**Einheiten über SI.** Jeder Messwert wird beim Indexieren mit `value_si` und Dimension gespeichert. „> 600 nm“ vergleicht dadurch auch Einträge in µm; „Maximum der Belichtungszeit“ vergleicht ms mit s. Unbekannte Einheiten werden nie geraten, sondern nur mit identischem Einheitentext verglichen.

**Das Modell rechnet nicht.** Zählungen, Min/Max/Ø/Summe und Listen werden deterministisch aus SQL gerendert. Bei Frage-Antwort mit numerischem Filter bekommt das Modell die exakten Messwerte als `<facts>`. Eine nachgelagerte Prüfung markiert Zahlen in der Antwort, die weder in Quellen noch Fakten noch Frage vorkommen.

**Ehrliches Scheitern.** Findet die Suche nichts Relevantes, antwortet die Engine ohne LLM-Aufruf mit „nichts gefunden“ (das Ausgangssystem übergab immer 5 Einträge, egal wie unpassend). Ein Planner-Ausfall degradiert auf Regeln statt auf einen Fehler; ein Ollama-Ausfall wird als handlungsfähige Meldung gezeigt.

**Prompt-Injection.** Laborbuchtext ist Fremdtext. Er steht in `<source>`-Blöcken, der System-Prompt erklärt ihn zu Daten, und Tags wie `</source>` im Text werden entschärft (`prompts.defang`).

**Bilder als Quelle.** Ein angehängtes Bild wird vor der Planung gelesen und als Quelle mit `experiment="uploaded image"` in den Prompt gegeben. Die Laborbuchsuche läuft mit Frage plus Anfang des Bildtexts. Der Prompt verlangt, Bildwerte und Laborbuchwerte getrennt zu nennen. Bildanhänge aus eLabFTW (`LAB_INDEX_IMAGES`) werden beim Sync gelesen, an den Body gehängt und normal gechunkt. Die Lesung ist im Cache `image_texts` pro Datei-Hash und Modell gespeichert. Die Index-Signatur enthält dann das Bildmodell.

**Skalierung.** Vektoren werden in eine vorab allokierte Matrix geladen (eine Kopie statt Liste plus `vstack`), nebenläufige Erstanfragen laden nur einmal. Die dichte Suche nutzt `argpartition` (O(n)) statt vollständigem Sortieren und kopiert bei großen Kandidatenmengen keine Teilmatrix. Lange ID-Listen gehen als ein JSON-Parameter (`json_each`) an SQLite statt als ein Platzhalter pro ID, sonst greift ab 32.766 IDs das Parameterlimit. Pro Verbindung: `synchronous=NORMAL` (sicher mit WAL), 64 MB Cache, 256 MB mmap. Gemessen mit 200.000 Abschnitten: Laden rund 1 s, Suche rund 0,25 s.

**Inkrementeller, sicherer Sync.** Listing wird vollständig paginiert gelesen, *bevor* irgendetwas gelöscht wird. Pro Experiment passiert alles in einer Transaktion (alt raus, neu rein). Index-Signatur = Parser-/Chunker-Version + Embedding-Modell: ändert sich eines, werden alle Experimente neu eingebettet.

## Datenmodell (SQLite)

```
experiments(id, title, date, body, category, status, author, resources, modified_at, content_hash, index_sig, synced_at)
exp_tags(experiment_id, tag, tag_norm)
measurements(experiment_id, name, name_norm, value_text, value_num, unit, unit_norm, dimension, value_si)
chunks(cid, experiment_id, idx, text, embedding BLOB, embed_model, dim)
chunks_fts(text)            -- FTS5, rowid = chunks.cid, unicode61 remove_diacritics 2
meta(key, value)            -- revision, last_sync, embed_model
image_texts(key, model, text, created_at)  -- Cache der Bildlesungen (key = Datei-Hash)
```

## Was aus der Vorlage `rag-chatbot-from-scratch` übernommen wurde

Übernommen als **Idee**, nicht als Code: getrennte Dokument-/Query-Embeddings, Ähnlichkeitsschwelle, Metadaten-Filter, ein Golden-Set zur Messung (`evaluation.py`), Konfiguration über Umgebungsvariablen, CLI + Makefile, Tests mit Fakes. **Bewusst nicht übernommen:** FastAPI, LangGraph, Cross-Encoder-Reranking (Download von Hugging Face, zusätzliche Latenz – stattdessen BM25 + RRF), DeepEval mit externem Judge-Modell (braucht API-Key), Chroma/FAISS, PDF-Ingestion, Cloud-LLM-Provider.

## Erweiterungspunkte

* Anderes Embedding-Modell: `LAB_EMBED_MODEL` + ggf. Prefixe in `embeddings.PREFIXES`, dann `sync --full`.
* Neue Einheit: eine Zeile in `units.py`.
* Reranker: Hook wäre `retrieval.hybrid_search` (nach RRF, vor dem Cap pro Experiment).
* Weitere Quellen (Ressourcen, PDF-Anhänge): zusätzliche Chunk-Typen in `chunking.py`, Abruf in `sync.py` (Bildanhänge: `sync._read_images`).
* Mehr als ~1 Mio. Abschnitte: `Store.vector_data` + dichter Teil von `retrieval.hybrid_search` durch sqlite-vec oder einen ANN-Index ersetzen.
