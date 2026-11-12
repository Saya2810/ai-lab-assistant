"""The combined app in a real browser: ``streamlit run app.py`` + Chromium, with stand-ins
for eLabFTW and Ollama (see ``fake_services.py``).

Optional, because it needs Playwright and a Chromium:

    pip install playwright && playwright install chromium
    RUN_E2E=1 pytest tests/e2e -v

It follows one realistic session: write an entry, glance at the chatbot, come back,
create the entry. What it proves that the headless tests cannot: switching pages really
keeps the form (text, template, tags, custom fields, uploaded files) and the chat.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from shared.config import REPO_ROOT

# read at import time: the autouse env isolation in conftest removes LAB_*/ELABFTW_* later
RUN = os.environ.get("RUN_E2E") == "1"
pytestmark = pytest.mark.skipif(not RUN, reason="browser test: set RUN_E2E=1 (needs playwright + chromium)")

QUESTION = "Warum war das Rauschen bei Probe A3 so hoch?"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running_app(tmp_path):
    from lab_chatbot.demo import seed_demo
    from lab_chatbot.store import Store
    from tests.e2e.fake_services import EMBEDDER, Services

    services = Services().start()
    db = tmp_path / "demo.db"
    seed_demo(Store(db), EMBEDDER)
    first, second = tmp_path / "first.txt", tmp_path / "messung.txt"
    first.write_text("removed again right away")
    second.write_text("hello upload")

    env = {k: v for k, v in os.environ.items() if not k.startswith(("ELABFTW_", "LAB_", "OLLAMA_"))}
    env.update(
        LAB_SKIP_DOTENV="1", LAB_DB_PATH=str(db), LAB_DATA_DIR=str(tmp_path / "data"),
        ELABFTW_URL=services.elab_url, ELABFTW_API_KEY="test-key", OLLAMA_HOST=services.ollama_url,
    )  # fmt: skip
    port = free_port()
    log = open(tmp_path / "streamlit.log", "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
    )  # fmt: skip
    url = f"http://localhost:{port}"
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen(f"{url}/_stcore/health", timeout=1).status == 200:
                    break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail("streamlit did not start:\n" + (tmp_path / "streamlit.log").read_text())
        yield url, services, (first, second)
    finally:
        server.terminate()
        server.wait(timeout=15)
        services.stop()


def go(page, label: str) -> None:
    page.locator("[data-testid=stSidebarNavLink]", has_text=label).first.click()
    page.wait_for_timeout(1200)


def choose(page, label: str, option: str, multi: bool = False) -> None:
    """Pick an entry from a Streamlit select (retries: a rerun can close the dropdown)."""
    for _ in range(4):
        page.wait_for_timeout(700)
        page.get_by_label(label, exact=True).first.click()
        try:
            page.locator("[role=option]", has_text=option).first.click(timeout=3000)
            if multi:
                page.keyboard.press("Escape")
            page.wait_for_timeout(900)
            return
        except Exception:
            page.keyboard.press("Escape")
    pytest.fail(f"could not choose {option!r} in {label!r}")


def type_into(page, label: str, text: str) -> None:
    field = page.get_by_label(label, exact=True)
    field.fill(text)
    field.press("Control+Enter" if label in ("Written notes", "Lab notes") else "Enter")
    page.wait_for_timeout(700)


def test_a_session_across_both_pages(running_app):
    from playwright.sync_api import sync_playwright

    url, services, (first, second) = running_app
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(viewport={"width": 1200, "height": 1000}).new_page()
        js_errors: list[str] = []
        page.on("pageerror", lambda e: js_errors.append(str(e)))
        page.goto(url)
        page.wait_for_selector("text=Entry structure", timeout=30000)

        # ---- one app, two pages
        links = " | ".join(page.locator("[data-testid=stSidebarNavLink]").all_inner_texts())
        assert "Lab Assistant" in links and "Lab Chatbot" in links, links
        assert page.get_by_role("heading", name="Lab Assistant").is_visible()
        assert page.get_by_text("streamlit run lab_chatbot").count() == 0  # no "start the other app" hint

        # ---- the uploader: removing a file there removes it; a normal rerun never "detaches" it
        page.locator("input[type=file]").set_input_files(str(first))
        uploaded_heading = page.get_by_role("heading", name="Uploaded files")
        uploaded_heading.wait_for(timeout=15000)
        page.get_by_label("Remove first.txt").click()
        page.wait_for_timeout(2000)
        assert uploaded_heading.count() == 0 and page.get_by_text("Kept from before").count() == 0
        page.locator("input[type=file]").set_input_files(str(second))  # (the same file again is ignored by the dropzone)
        uploaded_heading.wait_for(timeout=15000)

        # ---- write an entry
        type_into(page, "Written notes", "Laser 532 nm, 20 mW, 5 s belichtet")
        assert page.get_by_text("Kept from before you left this page").count() == 0
        page.get_by_role("button", name="Generate laboratory entry").click()
        page.wait_for_selector("text=Draft generated", timeout=30000)
        assert page.get_by_label("Title", exact=True).input_value() == "Raman-Messung Probe B7"
        choose(page, "Template", "Raman Template")
        choose(page, "Tags", "raman", multi=True)
        type_into(page, "Title", "Titel ANGEPASST")
        type_into(page, "Operator", "Dr. Test")

        # ---- glance at the chatbot
        go(page, "Lab Chatbot")
        page.wait_for_selector("text=Fragen an dein eLabFTW-Laborbuch", timeout=30000)
        assert page.get_by_text("Entry structure").count() == 0
        assert page.locator("[data-testid=stMetricValue]").first.inner_text().strip() == "8"
        assert page.get_by_text("Beispielfragen").count() == 1
        page.get_by_test_id("stChatInputTextArea").fill(QUESTION)
        page.keyboard.press("Enter")
        page.wait_for_selector("text=dejustierte Faser", timeout=30000)
        page.wait_for_timeout(1500)
        assert page.get_by_text("Quellen").count() >= 1
        assert page.get_by_text("Beispielfragen").count() == 0  # gone as soon as a question is asked
        page.get_by_text("Plan & Zeiten anzeigen").click()
        page.wait_for_timeout(1000)

        # ---- back: nothing lost
        go(page, "Lab Assistant")
        page.wait_for_selector("text=Entry structure", timeout=30000)
        page.wait_for_timeout(800)
        assert page.get_by_label("Written notes").input_value() == "Laser 532 nm, 20 mW, 5 s belichtet"
        assert page.get_by_label("Title", exact=True).input_value() == "Titel ANGEPASST"
        assert "Signal stabil" in page.get_by_label("Lab notes").input_value()
        assert page.get_by_label("Operator").input_value() == "Dr. Test"
        assert page.get_by_text("Select a template to load its custom fields").count() == 0  # template survived
        assert page.locator("[data-testid=stMain] span", has_text="raman").count() > 0  # tag survived
        assert page.get_by_text("messung.txt").count() > 0
        assert page.get_by_text("Kept from before you left this page").count() == 1  # uploader was reset, file kept

        # ---- create it
        page.get_by_role("button", name="Create lab entry").click()
        page.wait_for_selector("text=Lab entry created in eLabFTW", timeout=30000)

        # ---- the chat is still there, with its sidebar setting
        go(page, "Lab Chatbot")
        page.wait_for_selector("text=Fragen an dein eLabFTW-Laborbuch", timeout=30000)
        page.wait_for_timeout(800)
        assert page.get_by_text("dejustierte Faser").count() > 0
        assert page.get_by_label("Plan & Zeiten anzeigen").is_checked()
        assert not js_errors, js_errors
        browser.close()

    # ---- what eLabFTW actually received
    posts = [(c["path"], c["body"]) for c in services.elab_calls("POST")]
    assert ("experiments", {"template": 5}) in posts
    assert ("experiments/42/tags", {"tag": "raman"}) in posts
    assert any(path == "experiments/42/uploads" for path, _ in posts)
    patches = [c["body"] for c in services.elab_calls("PATCH")]
    assert any(b.get("title") == "Titel ANGEPASST" and "Signal stabil" in b.get("body", "") and b.get("content_type") == 2 for b in patches)
    assert any("Dr. Test" in b.get("metadata", "") and "532" in b.get("metadata", "") for b in patches)
