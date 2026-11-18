"""``python -m lab_assistant doctor``: is everything the assistant needs reachable?

Checks what the form is built from (the eLabFTW structure lists), the text model, and
the optional parts (photo OCR, audio, HEIC) so a missing one is explained *before* a
user uploads a photo and gets an error.
"""

from __future__ import annotations

import importlib.util
from typing import Callable

from lab_assistant import elab_api
from lab_assistant.config import AssistantConfig
from shared.config import ConfigError, ElabFTWConfig, env_str
from shared.diagnostics import Check, ollama_checks
from shared.elabftw_client import ElabFTWError

STRUCTURE: list[tuple[str, Callable[[], list[dict]], bool]] = [
    # label, loader, may it be empty?
    ("Templates", elab_api.get_experiment_templates, True),
    ("Categories", elab_api.get_experiment_categories, True),
    ("Statuses", elab_api.get_experiment_statuses, True),
    ("Resources", elab_api.get_resources, True),
    ("Tags", elab_api.get_tags, True),
]


def check_elabftw() -> list[Check]:
    if not env_str("ELABFTW_API_KEY"):
        return [Check("eLabFTW", False, "ELABFTW_API_KEY is not set: copy .env.example to .env and fill it in")]
    try:
        config = ElabFTWConfig.from_env()
    except ConfigError as exc:
        return [Check("eLabFTW", False, str(exc))]
    checks: list[Check] = []
    templates: list[dict] = []
    for label, loader, may_be_empty in STRUCTURE:
        try:
            items = loader()
        except ElabFTWError as exc:
            checks.append(Check(f"eLabFTW {label.lower()}", False, str(exc)[:300]))
            continue
        except Exception as exc:  # noqa: BLE001 - report anything
            checks.append(Check(f"eLabFTW {label.lower()}", False, f"{type(exc).__name__}: {exc}"[:300]))
            continue
        if label == "Templates":
            templates = items
        checks.append(Check(f"eLabFTW {label.lower()}", True if items or may_be_empty else None, f"{len(items)} found"))
    if templates:
        first = templates[0]
        try:
            fields = elab_api.get_custom_fields_from_template(first["id"])
            checks.append(Check("Template fields", True, f"'{first.get('title', first['id'])}': {len(fields)} custom field(s) read"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("Template fields", False, str(exc)[:300]))
    if checks and all(c.ok for c in checks):
        checks.insert(0, Check("eLabFTW", True, f"reachable at {config.api_url}"))
    return checks


def check_ollama(config: AssistantConfig) -> tuple[list[Check], bool]:
    return ollama_checks(
        config.ollama_host,
        [("Text model", config.text_model, True), ("Vision model", config.vision_model, False)],
    )


def check_optional(config: AssistantConfig) -> list[Check]:
    checks = []
    checks.append(
        Check("Photo support", True, "Pillow installed") if importlib.util.find_spec("PIL")
        else Check("Photo support", False, "pip install Pillow")
    )
    checks.append(
        Check("HEIC photos", True, "pillow-heif installed") if importlib.util.find_spec("pillow_heif")
        else Check("HEIC photos", None, "optional, for iPhone photos: pip install pillow-heif", critical=False)
    )
    if importlib.util.find_spec("faster_whisper") is None:
        checks.append(Check("Audio", None, "optional: pip install faster-whisper", critical=False))
    elif not config.whisper_dir.exists():
        checks.append(Check("Audio", None, f"no local Whisper model at {config.whisper_dir} (see README)", critical=False))
    else:
        checks.append(Check("Audio", True, f"Whisper model at {config.whisper_dir}", critical=False))
    return checks


def check_live(config: AssistantConfig) -> list[Check]:
    """Call the text model once, the way the assistant does (JSON mode)."""
    from lab_assistant.ollama_client import AIError, chat

    try:
        reply = chat(config, config.text_model, [{"role": "user", "content": 'Antworte nur mit {"ok": true}'}], json_mode=True)
        return [Check("Text model call", bool(reply.strip()), f"{config.text_model} answered {reply.strip()[:40]!r}")]
    except AIError as exc:
        return [Check("Text model call", False, str(exc)[:300])]


def run_doctor(config: AssistantConfig, live: bool = False) -> list[Check]:
    ollama, ready = check_ollama(config)
    checks = [*check_elabftw(), *ollama, *check_optional(config)]
    if live and ready:
        checks += check_live(config)
    return checks
