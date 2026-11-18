"""Command line interface: ``python -m lab_chatbot <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

from lab_chatbot.config import ChatbotConfig
from lab_chatbot.errors import ChatbotError
from shared.config import ConfigError, data_dir
from shared.elabftw_client import ElabFTWError

HELP_EPILOG = """\
typical first run:
  python -m lab_chatbot doctor      check Ollama, models, eLabFTW, index
  python -m lab_chatbot sync        read eLabFTW into the local index
  python -m lab_chatbot chat        talk to it (or: streamlit run lab_chatbot/ui.py)
no eLabFTW at hand?
  python -m lab_chatbot seed-demo   index a synthetic notebook (separate database)
"""


def format_table(table, show_links: bool = False, max_cell: int = 42) -> str:
    keep = [
        i for i, name in enumerate(table.columns)
        if show_links or name.lower() != "link"
    ]  # fmt: skip
    header = [table.columns[i] for i in keep]
    rows = [
        [str(row[i]).replace("\n", " ")[:max_cell] for i in keep] for row in table.rows
    ]
    widths = [max(len(h), *(len(r[c]) for r in rows)) if rows else len(h) for c, h in enumerate(header)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()  # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths]), *(line(r) for r in rows)])


def _print_answer_events(engine, question: str, history, explain: bool) -> str:
    """Run one question, streaming tokens; returns the final answer text."""
    final = None
    for kind, payload in engine.ask_stream(question, history):
        if kind == "plan" and explain:
            print("── plan ──", file=sys.stderr)
            print(json.dumps(payload.to_dict(), indent=2, ensure_ascii=False, default=str), file=sys.stderr)
        elif kind == "tables":
            for table in payload:
                print(f"\n{table.title}")
                print(format_table(table, show_links=explain))
            print()
        elif kind == "token":
            print(payload, end="", flush=True)
        elif kind == "final":
            final = payload
    assert final is not None
    if final.used_llm:
        print()
    else:
        print(final.text)
    if final.sources:
        shown = final.cited_sources or final.sources
        label = "Sources" if final.cited_sources else "Sources (retrieved, none cited)"
        print(f"\n{label}:")
        for s in shown:
            print(f"  [{s.n}] Experiment {s.experiment_id}: {s.title} ({s.date})  {s.url}")
    for warning in final.warnings:
        print(f"\n! {warning}")
    if explain:
        print(f"\n(latency ms: {final.latencies})", file=sys.stderr)
    return final.text


def image_options(config, build_image_reader) -> dict:
    """Sync keyword arguments for image indexing (empty when LAB_INDEX_IMAGES is off)."""
    if not config.index_images:
        return {}
    return {
        "image_reader": build_image_reader(config),
        "max_image_bytes": int(config.max_image_mb * 1024 * 1024),
        "max_images_per_experiment": config.max_images_per_experiment,
    }


def cmd_sync(args, config) -> int:
    from lab_chatbot.factory import build_embedder, build_image_reader, build_store
    from lab_chatbot.sync import sync_labbook
    from shared.elabftw_client import get_client

    def progress(done: int, total: int, message: str) -> None:
        print(f"\r  {done}/{total} {message:<30}", end="", file=sys.stderr, flush=True)

    report = sync_labbook(
        get_client(), build_store(config), build_embedder(config),
        full=args.full, force_empty=args.force_empty, max_items=args.limit, progress=progress,
        **image_options(config, build_image_reader),
    )  # fmt: skip
    print(file=sys.stderr)
    print(report.summary())
    for experiment_id, error in report.failed:
        print(f"  ✗ experiment {experiment_id}: {error}", file=sys.stderr)
    return 1 if report.failed else 0


def cmd_ask(args, config) -> int:
    from lab_chatbot.factory import build_engine

    _print_answer_events(build_engine(config), args.question, None, args.explain)
    return 0


def cmd_chat(args, config) -> int:
    from lab_chatbot.factory import build_engine

    engine = build_engine(config)
    print("Lab Chatbot. Empty line or 'exit' quits.")
    history: list[dict] = []
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question or question.lower() in ("exit", "quit"):
            return 0
        try:
            answer = _print_answer_events(engine, question, history, args.explain)
        except ChatbotError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def cmd_search(args, config) -> int:
    from lab_chatbot.factory import build_engine

    engine = build_engine(config)
    plan, ids, hits = engine.retrieve_only(args.query)
    print(f"intent={plan.intent} source={plan.source}  query={plan.semantic_query!r}")
    if plan.notes:
        print("notes:", "; ".join(plan.notes))
    print("experiments:", ids)
    for rank, hit in enumerate(hits, start=1):
        dense = "—" if hit.dense is None else f"{hit.dense:.3f}"
        lexical = "—" if hit.lexical_rank is None else str(hit.lexical_rank)
        snippet = hit.text.split("\n", 2)[-1].replace("\n", " ")[:140]
        print(f"{rank:>2}. exp {hit.experiment_id:<5} dense={dense} bm25-rank={lexical}  {snippet}")
    return 0


def cmd_stats(args, config) -> int:
    from lab_chatbot.factory import build_store

    stats = build_store(config).stats()
    print(f"database : {config.db_path}")
    for key, value in stats.items():
        print(f"{key:<22}: {value}")
    return 0


def cmd_eval(args, config) -> int:
    from lab_chatbot.evaluation import evaluate, load_cases
    from lab_chatbot.factory import build_engine

    results, summary = evaluate(build_engine(config), load_cases(args.golden), full=args.full)
    for r in results:
        mark = "✓" if r.passed else "✗"
        extra = f" rank={r.rank}" if r.rank else ""
        error = f"  ERROR {r.error}" if r.error else ""
        print(f"{mark} [{r.intent or '?'}]{extra}  {r.case.question}{error}")
    print()
    for key, value in summary.items():
        print(f"{key:<18}: {'n/a' if value is None else round(value, 3) if isinstance(value, float) else value}")
    return 0 if summary["passed"] == summary["cases"] else 1


def cmd_doctor(args, config) -> int:
    from lab_chatbot.doctor import run_doctor
    from shared.diagnostics import print_report

    if not args.quick:
        print("Calling the models once (the first call loads them into memory, this can take a minute) ...\n")
    return print_report(run_doctor(config, live=not args.quick))


def cmd_seed_demo(args, config) -> int:
    from lab_chatbot.demo import seed_demo
    from lab_chatbot.factory import build_embedder
    from lab_chatbot.store import Store

    path = args.db or data_dir() / "chatbot" / "demo.db"
    count = seed_demo(Store(path), build_embedder(config), reset=True)
    print(f"Indexed {count} demo experiments into {path}")
    print("Try:")
    print(f"  python -m lab_chatbot --db {path} chat")
    print(f"  LAB_DB_PATH={path} streamlit run lab_chatbot/ui.py")
    print(f"  python -m lab_chatbot --db {path} eval --full --golden lab_chatbot/golden_demo.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab_chatbot", description="Ask your eLabFTW lab notebook.",
        epilog=HELP_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
    )  # fmt: skip
    parser.add_argument("--db", help="path of the index database (default: data/chatbot/labbook.db)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sync", help="read eLabFTW into the local index (incremental)")
    p.add_argument("--full", action="store_true", help="re-process every experiment")
    p.add_argument("--force-empty", action="store_true", help="allow an empty listing to clear the index")
    p.add_argument("--limit", type=int, help="only the first N experiments (testing)")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("--explain", action="store_true", help="show the plan and timings")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive chat")
    p.add_argument("--explain", action="store_true")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("search", help="retrieval only (no answer): see what would be found")
    p.add_argument("query")
    p.set_defaults(func=cmd_search)

    sub.add_parser("stats", help="index statistics").set_defaults(func=cmd_stats)
    doctor = sub.add_parser("doctor", help="check the whole setup (calls the models once)")
    doctor.add_argument("--quick", action="store_true", help="do not call the models, only check that they exist")
    doctor.set_defaults(func=cmd_doctor)

    p = sub.add_parser("eval", help="run a golden question set")
    p.add_argument("--golden", required=True, help="JSON file with questions and expectations")
    p.add_argument("--full", action="store_true", help="also run the LLM to check expect_contains")
    p.set_defaults(func=cmd_eval)

    sub.add_parser("seed-demo", help="index a synthetic demo notebook").set_defaults(func=cmd_seed_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        config = ChatbotConfig.from_env(args.db)
        return args.func(args, config)
    except (ChatbotError, ElabFTWError, ConfigError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
