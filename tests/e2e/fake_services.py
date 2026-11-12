"""Stand-ins for eLabFTW and Ollama, so the real app can be driven in a real browser
without either installed. They answer just enough of the real HTTP APIs and record what
they were asked (``Services.requests``)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from lab_chatbot.demo import DEMO_EXPERIMENTS
from tests.fakes import FakeEmbedder, plan_json

API_KEY = "test-key"
SERVER_DEFAULT_LIMIT = 15  # what eLabFTW returns when no limit is given

EMBEDDER = FakeEmbedder("embeddinggemma")  # the demo index is built with the same function

TEMPLATE_META = {
    "extra_fields": {
        "Wellenlänge": {"type": "number", "value": "", "units": ["nm", "µm"], "unit": "nm"},
        "Operator": {"type": "text", "value": ""},
    }
}
LISTS = {
    "teams/current/experiments_categories": [{"id": 1, "title": "Raman"}],
    "teams/current/experiments_status": [{"id": 1, "title": "Running"}, {"id": 2, "title": "Success"}],
    "experiments_templates": [{"id": 5, "title": "Raman Template"}],
    "items": [{"id": 9, "title": "Laser 532"}],
    "teams/current/tags": [{"id": 1, "tag": "raman"}, {"id": 2, "tag": "pcr"}],
}
DRAFT = {
    "title": "Raman-Messung Probe B7",
    "lab_notes": "Laser bei 532 nm, 20 mW, Belichtung 5 s.\n\n## Ergebnis\nSignal stabil.",
    "measurements": {"Wellenlänge": {"value": "532", "unit": "nm"}},
}
ANSWER = "Das Rauschen war laut Eintrag durch eine dejustierte Faser erhöht [1]."


class Services:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._servers: list[ThreadingHTTPServer] = []
        self.elab_url = self.ollama_url = ""

    def start(self) -> "Services":
        for kind, handler in (("elab", _elab_handler(self)), ("ollama", _ollama_handler(self))):
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self._servers.append(server)
            port = server.server_address[1]
            if kind == "elab":
                self.elab_url = f"http://127.0.0.1:{port}/api/v2"
            else:
                self.ollama_url = f"http://127.0.0.1:{port}"
        return self

    def stop(self) -> None:
        for server in self._servers:
            server.shutdown()

    def elab_calls(self, method: str | None = None) -> list[dict]:
        return [r for r in self.requests if r["svc"] == "elab" and (method is None or r["method"] == method)]


def _json(handler: BaseHTTPRequestHandler, obj, code: int = 200, headers: dict | None = None) -> None:
    body = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    for key, value in (headers or {}).items():
        handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(body)


def _elab_handler(services: Services):
    class Elab(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _path(self):
            url = urlparse(self.path)
            return url.path.split("/api/v2/", 1)[-1].strip("/"), parse_qs(url.query)

        def _body(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            if "json" in self.headers.get("Content-Type", "") and raw:
                try:
                    return json.loads(raw)
                except ValueError:
                    pass
            return {"_bytes": len(raw), "_content_type": self.headers.get("Content-Type", "").split(";")[0]}

        def do_GET(self):
            path, query = self._path()
            if self.headers.get("Authorization") != API_KEY:
                return _json(self, {"description": "invalid API key"}, 401)
            services.requests.append({"svc": "elab", "method": "GET", "path": path, "query": {k: v[0] for k, v in query.items()}})
            if path == "experiments":
                limit = int(query.get("limit", [SERVER_DEFAULT_LIMIT])[0])
                offset = int(query.get("offset", ["0"])[0])
                return _json(self, [dict(e) for e in DEMO_EXPERIMENTS][offset : offset + limit])
            if path.startswith("experiments/") and path.split("/")[1].isdigit():
                found = [e for e in DEMO_EXPERIMENTS if str(e["id"]) == path.split("/")[1]]
                if found:
                    return _json(self, found[0])
            if path == "users/me":
                return _json(self, {"id": 1, "team": 1})
            if path in LISTS:  # eLabFTW paginates: everything is on the first page
                return _json(self, LISTS[path] if int(query.get("offset", ["0"])[0]) == 0 else [])
            if path == "experiments_templates/5":
                return _json(self, {"id": 5, "metadata": json.dumps(TEMPLATE_META)})
            return _json(self, {"id": 42, "metadata": None})

        def do_POST(self):
            path, _ = self._path()
            if self.headers.get("Authorization") != API_KEY:
                return _json(self, {"description": "invalid API key"}, 401)
            services.requests.append({"svc": "elab", "method": "POST", "path": path, "body": self._body()})
            _json(self, {}, 201, {"Location": f"{services.elab_url}/{path}/42"})

        def do_PATCH(self):
            path, _ = self._path()
            if self.headers.get("Authorization") != API_KEY:
                return _json(self, {"description": "invalid API key"}, 401)
            services.requests.append({"svc": "elab", "method": "PATCH", "path": path, "body": self._body()})
            _json(self, {})

    return Elab


def _ollama_handler(services: Services):
    class Ollama(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            _json(self, {"models": [{"name": n, "model": n} for n in ("qwen2.5:7b", "embeddinggemma:latest")]})  # real Ollama sends both keys

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
            if self.path.endswith("/api/embed"):
                texts = request.get("input", [])
                texts = [texts] if isinstance(texts, str) else texts
                return _json(self, {"model": request.get("model"), "embeddings": [EMBEDDER._vec(t).tolist() for t in texts]})
            if not self.path.endswith("/api/chat"):
                return _json(self, {})
            prompt = " ".join(m.get("content", "") for m in request.get("messages", []) if isinstance(m.get("content"), str))
            if "lab_notes" in prompt:  # Lab Assistant: draft
                content = json.dumps(DRAFT, ensure_ascii=False)
            elif request.get("format"):  # Lab Chatbot: planner
                content = json.dumps(plan_json("search", "Rauschen Probe A3 hoch"))
            else:  # Lab Chatbot: grounded answer
                content = ANSWER
            services.requests.append({"svc": "ollama", "path": "chat", "stream": bool(request.get("stream", True))})
            message = lambda text, done: {  # noqa: E731
                "model": request.get("model"), "created_at": "2026-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": text}, "done": done,
                **({"done_reason": "stop"} if done else {}),
            }  # fmt: skip
            if not request.get("stream", True):
                return _json(self, message(content, True))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            words = content.split(" ")
            for i, word in enumerate(words):
                piece = word + (" " if i < len(words) - 1 else "")
                self.wfile.write((json.dumps(message(piece, False)) + "\n").encode())
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write((json.dumps(message("", True)) + "\n").encode())

    return Ollama
