"""Building blocks for the ``doctor`` commands of both apps: a check result, model-name
matching, a model listing and the printed report."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Check:
    name: str
    ok: bool | None  # None = warning / not applicable
    detail: str
    critical: bool = True

    @property
    def symbol(self) -> str:
        return "✓" if self.ok else ("!" if self.ok is None else "✗")


def model_present(wanted: str, installed: list[str]) -> bool:
    for name in installed:
        if name == wanted or name == f"{wanted}:latest":
            return True
        if ":" not in wanted and name.split(":")[0] == wanted:
            return True
    return False


def installed_ollama_models(host: str) -> list[str]:
    """Names of the models the Ollama server has. Raises if it cannot be reached."""
    import ollama

    listing = ollama.Client(host=host).list()
    return [(m.get("model") or m.get("name") or "") for m in listing["models"]]


def ollama_checks(host: str, models: list[tuple[str, str, bool]]) -> tuple[list[Check], bool]:
    """``models`` = (label, name, required). Returns the checks and whether every *required* model is there."""
    try:
        installed = installed_ollama_models(host)
    except Exception as exc:  # noqa: BLE001 - report anything
        return [Check("Ollama", False, f"not reachable at {host} ({type(exc).__name__}). Start it with `ollama serve`.")], False
    checks = [Check("Ollama", True, f"reachable at {host}")]
    ready = True
    for label, model, required in models:
        present = model_present(model, installed)
        ready = ready and (present or not required)
        detail = f"{model} installed" if present else f"{model} missing: ollama pull {model}"
        checks.append(Check(label, True if present else (False if required else None), detail, critical=required))
    return checks, ready


def print_report(checks: list[Check]) -> int:
    """Print the checks; exit code 1 if a critical one failed."""
    for check in checks:
        print(f"{check.symbol} {check.name:<18} {check.detail}")
    failed = [c for c in checks if c.ok is False and c.critical]
    soft = [c for c in checks if c.ok is False and not c.critical]
    if failed:
        print(f"\n{len(failed)} problem(s) to fix.")
    elif soft:  # e.g. eLabFTW unreachable: chatting with the existing index still works, `sync` will not
        print("\nUsable, but not everything works: " + ", ".join(c.name for c in soft) + ".")
    else:
        print("\nAll good.")
    return 1 if failed else 0
