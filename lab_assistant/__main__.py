"""``python -m lab_assistant doctor`` — check that the assistant can reach everything it needs."""

from __future__ import annotations

import argparse
import sys

from lab_assistant.config import AssistantConfig


def _run_eval(cases_path: str | None, json_out: str | None) -> int:
    import json
    from pathlib import Path

    from lab_assistant.ai import generate_lab_entry
    from lab_assistant.evaluation import DEFAULT_CASES, evaluate_assistant, format_report, load_cases

    config = AssistantConfig.from_env()
    cases = load_cases(cases_path or DEFAULT_CASES)
    result = evaluate_assistant(cases, lambda notes: generate_lab_entry(notes, config))
    print(format_report(result))
    if json_out:
        Path(json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {json_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab_assistant", description="Lab Assistant (eLabFTW entry writer).")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="check eLabFTW, Ollama and the optional parts")
    doctor.add_argument("--quick", action="store_true", help="do not call the model, only check that it exists")
    ev = sub.add_parser("eval", help="measure the draft generation on labelled example notes (needs Ollama)")
    ev.add_argument("--cases", help="JSON file with cases (default: the bundled demo set)")
    ev.add_argument("--json", dest="json_out", help="write the full per-case result to this file")
    args = parser.parse_args(argv)

    if args.command == "eval":
        return _run_eval(args.cases, args.json_out)

    from lab_assistant.doctor import run_doctor
    from shared.diagnostics import print_report

    if not args.quick:
        print("Calling the model once (the first call loads it into memory, this can take a minute) ...\n")
    return print_report(run_doctor(AssistantConfig.from_env(), live=not args.quick))


if __name__ == "__main__":
    sys.exit(main())
