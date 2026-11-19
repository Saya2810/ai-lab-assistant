"""Environment-driven settings shared by Lab Assistant and Lab Chatbot.

Nothing here reads the environment at import time (except the optional .env
file, loaded lazily), so importing a module never crashes because a variable
is missing. Errors surface when a setting is actually needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_API_URL = "https://localhost:8443/api/v2"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

_env_loaded = False


class ConfigError(RuntimeError):
    """A required setting is missing or invalid."""


def load_env() -> None:
    """Load ``<repo>/.env`` once. Real environment variables always win."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    if os.environ.get("LAB_SKIP_DOTENV"):  # tests: never read a developer's real .env
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def env_str(name: str, default: str | None = None) -> str | None:
    load_env()
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def env_int(name: str, default: int) -> int:
    value = env_str(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = env_str(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {value!r}") from exc


def data_dir() -> Path:
    """Root for all generated data (index, caches). Never committed to git."""
    return Path(env_str("LAB_DATA_DIR") or REPO_ROOT / "data")


# --------------------------------------------------------------------------
# eLabFTW
# --------------------------------------------------------------------------


def elab_api_url() -> str:
    return (env_str("ELABFTW_URL") or DEFAULT_API_URL).rstrip("/")


def site_url_from_api(api_url: str) -> str:
    """``https://host:8443/api/v2`` -> ``https://host:8443``."""
    url = api_url.rstrip("/")
    for suffix in ("/api/v2", "/api/v1"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def elab_site_url() -> str:
    """Base URL of the eLabFTW web UI (used for links; needs no API key)."""
    return (env_str("ELABFTW_SITE_URL") or site_url_from_api(elab_api_url())).rstrip("/")


def experiment_url(experiment_id: int | str, site_url: str | None = None) -> str:
    base = (site_url or elab_site_url()).rstrip("/")
    return f"{base}/experiments.php?mode=view&id={experiment_id}"


def _parse_verify(value: str | None, api_url: str) -> bool | str:
    if value is None:
        host = (urlparse(api_url).hostname or "").lower()
        # Self-signed certificates are normal on a local prototype; anywhere
        # else, verification stays on unless explicitly disabled.
        return host not in _LOCAL_HOSTS
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    path = Path(value).expanduser()
    if not path.exists():
        raise ConfigError(
            f"ELABFTW_VERIFY_TLS={value!r} is neither true/false nor an existing CA bundle path."
        )
    return str(path)


READ_KEY_VAR = "ELABFTW_API_KEY"
WRITE_KEY_VARS = ("ELABFTW_WRITE_API_KEY", "GENERATE_LABBOOK_ELABFTW_API_KEY")


def api_key_variable(write: bool = False) -> str | None:
    """Name of the environment variable whose key is used (None: no key set)."""
    for name in (*WRITE_KEY_VARS, READ_KEY_VAR) if write else (READ_KEY_VAR,):
        if env_str(name):
            return name
    return None


@dataclass(frozen=True)
class ElabFTWConfig:
    api_url: str
    api_key: str
    verify_tls: bool | str = True
    timeout: float = 30.0
    page_size: int = 100

    @property
    def site_url(self) -> str:
        return elab_site_url() if not self.api_url else site_url_from_api(self.api_url)

    @classmethod
    def from_env(cls, write: bool = False) -> "ElabFTWConfig":
        """``write=True``: the key of the Lab Assistant, which creates entries.

        The chatbot only reads, so it can (and should) run with a read-only key in
        ``ELABFTW_API_KEY``; the assistant takes ``ELABFTW_WRITE_API_KEY`` (or the
        older name ``GENERATE_LABBOOK_ELABFTW_API_KEY``) and falls back to ``ELABFTW_API_KEY``.
        """
        name = api_key_variable(write)
        if name is None:
            wanted = " or ".join(WRITE_KEY_VARS + (READ_KEY_VAR,)) if write else READ_KEY_VAR
            raise ConfigError(
                f"{wanted} is not set. Create an API key in eLabFTW "
                "(Settings > API keys) and put it in .env or the environment."
            )
        api_key = env_str(name)
        api_url = elab_api_url()
        return cls(
            api_url=api_url,
            api_key=api_key,
            verify_tls=_parse_verify(env_str("ELABFTW_VERIFY_TLS"), api_url),
            timeout=env_float("ELABFTW_TIMEOUT", 30.0),
            page_size=env_int("ELABFTW_PAGE_SIZE", 100),
        )
