from __future__ import annotations

import pytest
import requests

from shared.config import ConfigError, ElabFTWConfig, experiment_url, site_url_from_api
from shared.elabftw_client import ElabFTWClient, ElabFTWError


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text or (str(payload) if payload is not None else "")
        self.ok = 200 <= status < 300

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, script):
        self.headers = {}
        self.verify = True
        self.script = list(script)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(script, **cfg):
    config = ElabFTWConfig(api_url="https://lab.example/api/v2", api_key="KEY", **cfg)
    session = FakeSession(script)
    return ElabFTWClient(config, session=session, sleep=lambda s: None), session


def test_config_requires_api_key():
    with pytest.raises(ConfigError, match="ELABFTW_API_KEY"):
        ElabFTWConfig.from_env()


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("ELABFTW_API_KEY", "abc")
    monkeypatch.setenv("ELABFTW_URL", "https://localhost:8443/api/v2")
    monkeypatch.setenv("ELABFTW_PAGE_SIZE", "25")
    config = ElabFTWConfig.from_env()
    assert config.api_key == "abc" and config.page_size == 25
    assert config.verify_tls is False  # self-signed localhost default


def test_tls_verification_defaults_on_for_remote_hosts(monkeypatch):
    monkeypatch.setenv("ELABFTW_API_KEY", "abc")
    monkeypatch.setenv("ELABFTW_URL", "https://elab.university.org/api/v2")
    assert ElabFTWConfig.from_env().verify_tls is True
    monkeypatch.setenv("ELABFTW_VERIFY_TLS", "false")
    assert ElabFTWConfig.from_env().verify_tls is False


def test_site_url_and_experiment_url():
    assert site_url_from_api("https://x.org/api/v2") == "https://x.org"
    assert experiment_url(7, "https://x.org") == "https://x.org/experiments.php?mode=view&id=7"


def test_client_sets_auth_header_and_returns_json():
    client, session = make_client([FakeResponse(200, [{"id": 1}])])
    assert client.get("experiments", {"limit": 1}) == [{"id": 1}]
    assert session.headers["Authorization"] == "KEY"
    assert session.calls[0][1] == "https://lab.example/api/v2/experiments"


def test_get_retries_transient_errors():
    client, session = make_client(
        [requests.exceptions.ConnectionError("boom"), FakeResponse(503), FakeResponse(200, {"ok": 1})]
    )
    assert client.get("x") == {"ok": 1}
    assert len(session.calls) == 3


def test_post_is_not_retried():
    client, session = make_client([FakeResponse(503, text="down"), FakeResponse(201)])
    with pytest.raises(ElabFTWError, match="503"):
        client.post("experiments", {"title": "t"})
    assert len(session.calls) == 1  # a retried POST could create duplicates


def test_error_message_contains_status_and_body():
    client, _ = make_client([FakeResponse(403, text="forbidden")])
    with pytest.raises(ElabFTWError) as exc:
        client.get("experiments")
    assert exc.value.status_code == 403 and "forbidden" in str(exc.value)


def test_ssl_error_gives_actionable_message():
    client, _ = make_client([requests.exceptions.SSLError("bad cert")])
    with pytest.raises(ElabFTWError, match="ELABFTW_VERIFY_TLS"):
        client.get("experiments")


def test_created_id_from_location_header():
    response = FakeResponse(201, headers={"Location": "https://lab.example/api/v2/experiments/42"})
    assert ElabFTWClient.created_id(response) == 42
    with pytest.raises(ElabFTWError):
        ElabFTWClient.created_id(FakeResponse(201))


def test_pagination_walks_all_pages_and_stops_on_empty():
    pages = [[{"id": 1}, {"id": 2}], [{"id": 3}], []]
    client, session = make_client([FakeResponse(200, p) for p in pages], page_size=2)
    assert [i["id"] for i in client.iter_paginated("experiments")] == [1, 2, 3]
    offsets = [c[2]["params"]["offset"] for c in session.calls]
    assert offsets == [0, 2, 3]  # offset advances by what was actually received


def test_pagination_does_not_stop_on_short_page():
    # a server may cap `limit` below what was asked: a short page is not the end
    pages = [[{"id": 1}], [{"id": 2}], []]
    client, _ = make_client([FakeResponse(200, p) for p in pages], page_size=100)
    assert [i["id"] for i in client.iter_paginated("experiments")] == [1, 2]


def test_pagination_survives_server_ignoring_offset():
    same = [{"id": 1}, {"id": 2}]
    client, _ = make_client([FakeResponse(200, same), FakeResponse(200, same)], page_size=2)
    assert [i["id"] for i in client.iter_paginated("experiments")] == [1, 2]


def test_pagination_max_items():
    client, _ = make_client([FakeResponse(200, [{"id": i} for i in range(1, 6)])])
    assert len(list(client.iter_paginated("experiments", max_items=3))) == 3


def test_pagination_rejects_non_list():
    client, _ = make_client([FakeResponse(200, {"error": "x"})])
    with pytest.raises(ElabFTWError, match="list"):
        list(client.iter_paginated("experiments"))


def test_the_assistant_writes_with_its_own_key_and_the_chatbot_reads_with_the_default(monkeypatch):
    from shared.config import api_key_variable

    monkeypatch.setenv("ELABFTW_API_KEY", "READ")
    assert ElabFTWConfig.from_env().api_key == "READ"
    assert ElabFTWConfig.from_env(write=True).api_key == "READ"  # no write key: fall back
    monkeypatch.setenv("GENERATE_LABBOOK_ELABFTW_API_KEY", "OLDNAME")
    assert ElabFTWConfig.from_env(write=True).api_key == "OLDNAME"
    monkeypatch.setenv("ELABFTW_WRITE_API_KEY", "WRITE")
    assert ElabFTWConfig.from_env(write=True).api_key == "WRITE"
    assert ElabFTWConfig.from_env().api_key == "READ"  # the chatbot never gets the write key
    assert (api_key_variable(write=True), api_key_variable()) == ("ELABFTW_WRITE_API_KEY", "ELABFTW_API_KEY")
