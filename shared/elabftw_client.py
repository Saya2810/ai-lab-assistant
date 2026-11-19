"""Dependable client for the eLabFTW REST API v2, used by both apps.

Differences to the old ``services/elabftw.py``:

* nothing happens at import time (the API key is read when a client is built)
* TLS verification is configurable instead of hard-coded ``verify=False``
* list endpoints are paginated (``iter_paginated``); the API returns only a
  small page by default, so un-paginated calls silently lose data
* idempotent GET requests are retried on connection errors / 502-504
* errors are one exception type with the server's response attached
"""

from __future__ import annotations

import functools
import json
import logging
import time
from typing import Any, Iterator

import requests

from shared.config import ElabFTWConfig

log = logging.getLogger(__name__)

_RETRY_STATUS = {502, 503, 504}


class ElabFTWError(RuntimeError):
    """Any failure talking to eLabFTW (network, TLS, HTTP error, bad payload)."""

    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ElabFTWClient:
    def __init__(self, config: ElabFTWConfig, session: Any = None, sleep=time.sleep):
        self.config = config
        self.session = session or requests.Session()
        self.session.headers["Authorization"] = config.api_key
        self.session.verify = config.verify_tls
        self._sleep = sleep
        if config.verify_tls is False:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # ------------------------------------------------------------------ core

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict | None = None,
        json_body: Any = None,
        data: dict | None = None,
        files: dict | None = None,
        attempts: int | None = None,
    ) -> requests.Response:
        url = f"{self.config.api_url}/{endpoint.lstrip('/')}"
        tries = attempts if attempts is not None else (3 if method == "GET" else 1)
        for attempt in range(tries):
            last_try = attempt + 1 >= tries
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    data=data,
                    files=files,
                    timeout=self.config.timeout,
                )
            except requests.exceptions.SSLError as exc:
                raise ElabFTWError(
                    "TLS verification failed. For a self-signed local instance set "
                    "ELABFTW_VERIFY_TLS=false, otherwise point it at your CA bundle."
                ) from exc
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if last_try:
                    raise ElabFTWError(
                        f"Cannot reach eLabFTW at {self.config.api_url}: {exc}"
                    ) from exc
                self._sleep(0.5 * 2**attempt)
                continue

            if response.status_code in _RETRY_STATUS and not last_try:
                self._sleep(0.5 * 2**attempt)
                continue
            if not response.ok:
                raise ElabFTWError(
                    f"eLabFTW API error {response.status_code} on {method} {endpoint}: "
                    f"{response.text[:500]}",
                    status_code=response.status_code,
                    body=response.text,
                )
            return response
        raise ElabFTWError(f"eLabFTW request failed: {method} {endpoint}")  # pragma: no cover

    # --------------------------------------------------------------- helpers

    def get(self, endpoint: str, params: dict | None = None) -> Any:
        response = self.request("GET", endpoint, params=params)
        try:
            return response.json()
        except ValueError as exc:
            raise ElabFTWError(f"eLabFTW returned non-JSON data for GET {endpoint}") from exc

    def post(self, endpoint: str, data: Any = None) -> requests.Response:
        return self.request("POST", endpoint, json_body=data)

    def patch(self, endpoint: str, data: dict) -> requests.Response:
        return self.request("PATCH", endpoint, json_body=data)

    def delete(self, endpoint: str) -> requests.Response:
        return self.request("DELETE", endpoint)

    def upload(
        self, endpoint: str, file_name: str, content: bytes, comment: str = ""
    ) -> requests.Response:
        return self.request(
            "POST",
            endpoint,
            files={"file": (file_name, content)},
            data={"comment": comment} if comment else None,
        )

    @staticmethod
    def created_id(response: requests.Response) -> int:
        """eLabFTW returns the new entity's URL in the Location header."""
        location = response.headers.get("Location")
        if not location:
            raise ElabFTWError("eLabFTW did not return a Location header for the new entity.")
        try:
            return int(location.rstrip("/").split("/")[-1])
        except ValueError as exc:
            raise ElabFTWError(f"Unexpected Location header: {location!r}") from exc

    # ------------------------------------------------------------ pagination

    def iter_paginated(
        self,
        endpoint: str,
        params: dict | None = None,
        page_size: int | None = None,
        max_items: int | None = None,
    ) -> Iterator[dict]:
        """Yield every item of a list endpoint, page by page.

        Stops on an empty page, or when a page brings nothing new (a server
        that ignores ``offset`` would otherwise loop forever). It does not
        stop on a short page: some servers cap ``limit`` below what was asked.
        """
        size = page_size or self.config.page_size
        offset = 0
        seen: set[Any] = set()
        yielded = 0
        while True:
            page = self.get(endpoint, {**(params or {}), "limit": size, "offset": offset})
            if not isinstance(page, list):
                raise ElabFTWError(
                    f"Expected a list from {endpoint}, got {type(page).__name__}."
                )
            if not page:
                return
            fresh = 0
            for item in page:
                if isinstance(item, dict) and item.get("id") is not None:
                    key: Any = item["id"]
                else:
                    key = json.dumps(item, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                fresh += 1
                yielded += 1
                yield item
                if max_items is not None and yielded >= max_items:
                    return
            if fresh == 0:
                log.warning("Pagination of %s returned no new items; stopping.", endpoint)
                return
            offset += len(page)


@functools.lru_cache(maxsize=2)
def get_client(write: bool = False) -> ElabFTWClient:
    """Process-wide client, built on first use from the environment.

    ``write=True`` uses the Lab Assistant's key (see ``ElabFTWConfig.from_env``);
    the chatbot uses the default, possibly read-only key.
    """
    return ElabFTWClient(ElabFTWConfig.from_env(write=write))


def reset_client_cache() -> None:
    get_client.cache_clear()
