"""``python -m lab_assistant doctor`` — check that the assistant can reach everything it needs."""

from __future__ import annotations

import argparse
import sys

from lab_assistant.config import AssistantConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab_assistant", description="Lab Assistant (eLabFTW entry writer).")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="check eLabFTW, Ollama and the optional parts")
    doctor.add_argument("--quick", action="store_true", help="do not call the model, only check that it exists")
    args = parser.parse_args(argv)

    from lab_assistant.doctor import run_doctor
    from shared.diagnostics import print_report

    if not args.quick:
        print("Calling the model once (the first call loads it into memory, this can take a minute) ...\n")
    return print_report(run_doctor(AssistantConfig.from_env(), live=not args.quick))


if __name__ == "__main__":
    sys.exit(main())
