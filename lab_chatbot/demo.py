"""Synthetic lab notebook for trying the chatbot without an eLabFTW server.

``python -m lab_chatbot seed-demo`` indexes these entries into a *separate*
database (``data/chatbot/demo.db``) so they never mix with real data. They use
the same payload shape as ``GET /api/v2/experiments/{id}``.
"""

from __future__ import annotations

import json

from lab_chatbot.chunking import chunk_experiment
from lab_chatbot.embeddings import Embedder
from lab_chatbot.parse import parse_experiment
from lab_chatbot.store import Store
from lab_chatbot.sync import index_signature


def _fields(**kwargs) -> str:
    extra = {}
    for name, spec in kwargs.items():
        value, unit = spec if isinstance(spec, tuple) else (spec, None)
        entry = {"type": "number", "value": str(value)}
        if unit:
            entry["unit"] = unit
        extra[name.replace("_", " ")] = entry
    return json.dumps({"extra_fields": extra})


def _entry(id_, title, date, tags, category, status, body, **fields):
    return {
        "id": id_, "title": title, "date": date, "tags": "|".join(tags),
        "category_title": category, "status_title": status, "fullname": "Demo User",
        "content_type": 2, "body": body, "metadata": _fields(**fields),
        "modified_at": f"{date} 12:00:00",
    }  # fmt: skip


DEMO_EXPERIMENTS: list[dict] = [
    _entry(
        101, "Spektrometer-Test mit 532 nm Laser", "2026-09-28", ["spektroskopie", "laser"],
        "Messung", "Abgeschlossen",
        "## Aufbau\n\nSpektrometer mit 532 nm Laser getestet. Belichtungszeit 5 s, Laserleistung 20 mW.\n\n"
        "## Beobachtung\n\nDas erste Spektrum zeigte unerwartet starkes Rauschen. Nach dem Nachjustieren "
        "der Faser wurde das Rauschen deutlich geringer.\n\n## Fazit\n\nFaserjustage ist kritisch.",
        Wellenlänge=(532, "nm"), Belichtungszeit=(5, "s"), Laserleistung=(20, "mW"),
    ),
    _entry(
        102, "Raman-Messung Probe A3", "2026-10-02", ["raman", "spektroskopie"],
        "Messung", "Abgeschlossen",
        "## Messung\n\nRaman-Spektrum von Probe A3 bei 785 nm, 10 s Belichtung, 50 mW.\n\n## Ergebnis\n\n"
        "Starker Peak bei 1042 cm^-1. Der Fluoreszenzuntergrund war hoch, deshalb ist das Rauschen "
        "bei Probe A3 erhöht.\n\n$$\n\\Delta\\nu = 1042\\,\\mathrm{cm^{-1}}\n$$",
        Wellenlänge=(785, "nm"), Belichtungszeit=(10, "s"), Laserleistung=(50, "mW"),
    ),
    _entry(
        103, "PCR Optimierung Annealing-Temperatur", "2026-10-05", ["pcr", "molekularbiologie"],
        "Biologie", "In Arbeit",
        "## Ziel\n\nAnnealing-Temperatur für Primerpaar P7 optimieren.\n\n## Durchführung\n\n"
        "Gradient-PCR mit 35 Zyklen. Bei 58 °C war die Bande am saubersten, bei 62 °C gab es keine Ausbeute.",
        Annealing_Temperatur=(58, "°C"), Zyklen=35,
    ),
    _entry(
        104, "Zellkultur HeLa Passage 12", "2026-10-08", ["zellkultur"],
        "Biologie", "Abgeschlossen",
        "## Passage\n\nHeLa-Zellen in Passage 12 gesplittet (1:5). Konfluenz vor dem Split etwa 90 %. "
        "Keine Kontamination im Mikroskop sichtbar.",
        Passage=12,
    ),
    _entry(
        105, "Raman-Messung Probe B1", "2026-10-12", ["raman", "spektroskopie"],
        "Messung", "Abgeschlossen",
        "## Messung\n\nProbe B1 bei 785 nm, 30 s Belichtung, 100 mW.\n\n## Beobachtung\n\n"
        "Sehr geringes Rauschen, aber die Probe wurde bei 100 mW sichtbar verbrannt. "
        "Für B1 sollte die Leistung unter 50 mW bleiben.",
        Wellenlänge=(785, "nm"), Belichtungszeit=(30, "s"), Laserleistung=(100, "mW"),
    ),
    _entry(
        106, "Pufferherstellung Tris-HCl pH 7.4", "2026-10-15", ["puffer"],
        "Labor", "Abgeschlossen",
        "## Rezept\n\n50 mM Tris-HCl, pH mit HCl auf 7.4 eingestellt. 1 L angesetzt und sterilfiltriert.",
        Konzentration=(50, "mM"), pH=7.4,
    ),
    _entry(
        107, "Laser power calibration", "2026-10-20", ["laser", "kalibrierung"],
        "Messung", "Abgeschlossen",
        "## Setup\n\nCalibration of the 532 nm laser with a power meter.\n\n## Result\n\n"
        "The nominal 20 mW setting delivered 18.4 mW at the sample. Power drift over 30 minutes was below 1 %.",
        Wellenlänge=(532, "nm"), Laserleistung=(18.4, "mW"),
    ),
    _entry(
        108, "Fluoreszenzmikroskopie GFP Kontrolle", "2026-11-03", ["mikroskopie", "zellkultur"],
        "Messung", "In Arbeit",
        "## Aufnahme\n\nGFP-Kontrolle bei 488 nm Anregung, Belichtungszeit 200 ms. Signal klar vom "
        "Hintergrund getrennt.",
        Wellenlänge=(488, "nm"), Belichtungszeit=(200, "ms"),
    ),
]


def seed_demo(store: Store, embedder: Embedder, reset: bool = True) -> int:
    """Index the demo notebook into ``store``; returns the number of experiments."""
    if reset:
        store.reset()
    signature = index_signature(embedder)
    for raw in DEMO_EXPERIMENTS:
        record = parse_experiment(raw)
        chunks = chunk_experiment(record)
        vectors = embedder.embed_documents([c.text for c in chunks])
        store.replace_experiment(
            record, chunks, vectors, embed_model=embedder.model,
            index_sig=signature, modified_at=record.modified_at,
        )  # fmt: skip
    store.set_meta("embed_model", embedder.model)
    return len(DEMO_EXPERIMENTS)
