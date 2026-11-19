"""Background sync: keeps the index current without anyone pressing a button.

    python -m lab_chatbot worker              # sync now, then every LAB_SYNC_INTERVAL s
    python -m lab_chatbot worker --once       # one sync, then exit (cron / systemd timer)

Runs in its own process, so a long first sync (hours for a very large notebook)
never blocks the chat UI. Only one sync runs at a time across all processes (a
lease in the database); if the UI or a manual ``sync`` holds it, the worker just
waits for the next round. SIGTERM/Ctrl-C stop it cleanly between two
experiments; everything finished so far stays in the index, and the next run
continues from there.
"""

from __future__ import annotations

import logging
import signal
import threading
from typing import Callable

from lab_chatbot.config import ChatbotConfig
from lab_chatbot.errors import BackendUnavailable, SyncBusy, SyncError

log = logging.getLogger(__name__)

MAX_BACKOFF = 3600.0


def run_worker(
    config: ChatbotConfig,
    *,
    once: bool = False,
    full_first: bool = False,
    interval: float | None = None,
    stop: threading.Event | None = None,
    sync_once: Callable[..., object] | None = None,
) -> int:
    """Sync in a loop until ``stop`` is set. Returns a process exit code."""
    stop = stop or threading.Event()
    interval = float(interval or config.sync_interval)
    sync_once = sync_once or _default_sync(config)
    _install_signal_handlers(stop)

    failures = 0
    full = full_first
    while not stop.is_set():
        wait = interval
        try:
            report = sync_once(full=full, should_stop=stop.is_set)
            log.info("Sync finished: %s", report.summary())
            failures = 0
            full = False
            exit_code = 1 if report.failed else 0
        except SyncBusy as exc:
            log.info("Skipping this round: %s", exc)
            exit_code = 0
        except (BackendUnavailable, SyncError, OSError) as exc:
            failures += 1
            # back off when eLabFTW or Ollama is down, instead of hammering them
            wait = min(MAX_BACKOFF, max(60.0, interval) * 2 ** min(failures - 1, 6))
            log.error("Sync failed (%d in a row), next try in %.0f s: %s", failures, wait, exc)
            exit_code = 1
        except Exception:  # noqa: BLE001 - a worker must survive surprises
            failures += 1
            wait = min(MAX_BACKOFF, max(60.0, interval) * 2 ** min(failures - 1, 6))
            log.exception("Unexpected sync error, next try in %.0f s", wait)
            exit_code = 1
        if once:
            return exit_code
        stop.wait(wait)
    log.info("Worker stopped")
    return 0


def _default_sync(config: ChatbotConfig) -> Callable[..., object]:
    from lab_chatbot.cli import image_options
    from lab_chatbot.factory import build_embedder, build_image_reader, build_store
    from lab_chatbot.sync import sync_labbook
    from shared.elabftw_client import get_client

    store, embedder = build_store(config), build_embedder(config)
    options = image_options(config, build_image_reader)

    def run(full: bool, should_stop: Callable[[], bool]):
        return sync_labbook(get_client(), store, embedder, full=full, should_stop=should_stop, **options)

    return run


def _install_signal_handlers(stop: threading.Event) -> None:
    if threading.current_thread() is not threading.main_thread():
        return  # signals can only be handled in the main thread (e.g. not inside tests' threads)

    def handle(signum, frame):  # noqa: ARG001
        log.info("Signal %s: stopping after the current experiment", signum)
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handle)
