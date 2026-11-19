"""eLabFTW operations used by the Lab Assistant, on top of the shared client.

Differences to the old ``services/elabftw.py``: no key read at import time, TLS and
timeouts come from the shared config, every list endpoint is paginated (eLabFTW
returns only 15 items by default, so resources/tags/templates used to be cut off),
and metadata updates *merge* instead of replacing existing custom fields.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable

from lab_assistant.formatting import normalize_math_for_elabftw
from shared.elabftw_client import ElabFTWClient, ElabFTWError, get_client

log = logging.getLogger(__name__)

MARKDOWN_CONTENT_TYPE = 2


# ------------------------------------------------------------------ write guard
#
# The Lab Assistant must never change an entry that already exists in the notebook.
# It may only fill in an experiment it has just created itself, and only until that
# creation is finished (``close_entry``). Every write below checks this first, so no
# form bug, stale id or future feature can edit a written lab notebook entry.

class ProtectedEntryError(PermissionError):
    """A write to an entry the Lab Assistant did not just create was refused."""


_open_entries: set[int] = set()
_open_lock = threading.Lock()


def _ensure_writable(experiment_id: int) -> None:
    with _open_lock:
        if int(experiment_id) in _open_entries:
            return
    raise ProtectedEntryError(
        f"Experiment {experiment_id} is an existing lab notebook entry. The Lab Assistant only "
        "creates new entries and never changes existing ones."
    )


def close_entry(experiment_id: int) -> None:
    """The creation of ``experiment_id`` is finished: from now on it is read-only for us."""
    with _open_lock:
        _open_entries.discard(int(experiment_id))


def client() -> ElabFTWClient:
    # the assistant creates entries: it uses the write key (ELABFTW_WRITE_API_KEY)
    return get_client(write=True)


def _all(endpoint: str) -> list[dict]:
    return list(client().iter_paginated(endpoint))


# ------------------------------------------------------------------ structure


# The paths the original app used (``teams/current/...``) are not what the API reference
# documents (``teams/<id>/...``, ``team_tags``), and eLabFTW versions differ in what they
# accept. So: try the known-good path first and the documented one when the server says
# "no such path". Any other error (401, 500, network) is real and is raised as is.
_PATH_NOT_FOUND = {400, 404, 405}


def _team_id() -> int:
    me = client().get("users/me")
    team = me.get("team") if isinstance(me, dict) else None
    if team is None:
        raise ElabFTWError("Cannot determine your team: the response of users/me has no 'team'.")
    return int(team)


def _all_first(*endpoints: str | Callable[[], str]) -> list[dict]:
    error: ElabFTWError | None = None
    for endpoint in endpoints:
        try:
            return _all(endpoint() if callable(endpoint) else endpoint)
        except ElabFTWError as exc:
            if exc.status_code not in _PATH_NOT_FOUND:
                raise
            log.info("eLabFTW answered %s for a team endpoint, trying the documented path", exc.status_code)
            error = exc
    assert error is not None
    raise error


def get_experiment_categories() -> list[dict]:
    return _all_first("teams/current/experiments_categories", lambda: f"teams/{_team_id()}/experiments_categories")


def get_experiment_statuses() -> list[dict]:
    return _all_first("teams/current/experiments_status", lambda: f"teams/{_team_id()}/experiments_status")


def get_experiment_templates() -> list[dict]:
    return _all("experiments_templates")


def get_resources() -> list[dict]:
    return _all("items")


def get_resource_categories() -> list[dict]:
    """Categories of resources ("Project", "Antibody", ...); a new resource needs one.

    eLabFTW 6 has them per team; before version 6 the resource categories were
    called ``items_types`` (in 6 that name holds resource *templates*).
    """
    return _all_first(
        "teams/current/resources_categories",
        lambda: f"teams/{_team_id()}/resources_categories",
        "items_types",
    )


def get_tags() -> list[dict]:
    return _all_first("teams/current/tags", "team_tags")


def load_structure() -> dict[str, list[dict]]:
    """Everything the form needs, in one call (the app caches it)."""
    return {
        "categories": get_experiment_categories(),
        "statuses": get_experiment_statuses(),
        "templates": get_experiment_templates(),
        "resources": get_resources(),
        "resource_categories": get_resource_categories(),
        "tags": get_tags(),
    }


# ------------------------------------------------------------------ metadata


def _metadata_dict(entity: dict) -> dict:
    metadata: Any = entity.get("metadata")
    if not metadata:
        return {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            log.warning("Ignoring unparsable metadata JSON")
            return {}
    return metadata if isinstance(metadata, dict) else {}


def get_template(template_id: int) -> dict:
    return client().get(f"experiments_templates/{template_id}")


def get_custom_fields_from_template(template_id: int) -> dict:
    fields = _metadata_dict(get_template(template_id)).get("extra_fields", {})
    return fields if isinstance(fields, dict) else {}


def get_experiment(experiment_id: int) -> dict:
    return client().get(f"experiments/{experiment_id}")


def update_experiment_custom_fields(experiment_id: int, custom_fields: dict) -> None:
    """Write field values, keeping fields the form did not show and other metadata."""
    _ensure_writable(experiment_id)
    metadata = _metadata_dict(get_experiment(experiment_id))
    existing = metadata.get("extra_fields")
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update(custom_fields)
    metadata["extra_fields"] = merged
    client().patch(f"experiments/{experiment_id}", {"metadata": json.dumps(metadata)})


# --------------------------------------------------------------- experiments


def create_experiment(template_id: int | None = None) -> int:
    """Create an empty experiment (optionally from a template) and return its id.

    Title/body/category/status are set in a separate update, because eLabFTW does
    not apply all of them when an experiment is created from a template.
    """
    # Older servers read ``template``, the current API reference calls the same thing
    # ``category_id`` (0 = team default, -1 = empty). Unknown keys are ignored, so send both:
    # otherwise a server that wants the other name would silently create an entry without the template.
    data = {"template": template_id, "category_id": template_id} if template_id is not None else {}
    experiment_id = ElabFTWClient.created_id(client().post("experiments", data))
    with _open_lock:
        _open_entries.add(int(experiment_id))  # writable until close_entry()
    return experiment_id


def update_experiment(
    experiment_id: int,
    title: str | None = None,
    body: str | None = None,
    category_id: int | None = None,
    status_id: int | None = None,
) -> None:
    _ensure_writable(experiment_id)
    data: dict[str, Any] = {}
    if title is not None:
        data["title"] = title
    if body is not None:
        data["body"] = normalize_math_for_elabftw(body)
        data["content_type"] = MARKDOWN_CONTENT_TYPE
    if category_id is not None:
        data["category"] = category_id
    if status_id is not None:
        data["status"] = status_id
    if data:
        client().patch(f"experiments/{experiment_id}", data)


def create_resource(title: str, category_id: int | None = None) -> int:
    """Create a resource (eLabFTW "item", e.g. a project) and return its id.

    The title and category are sent with the creation and set again afterwards:
    older servers ignore fields on POST and would leave an "Untitled" resource.
    """
    data: dict[str, Any] = {"title": title}
    if category_id is not None:
        data["category"] = category_id
    resource_id = ElabFTWClient.created_id(client().post("items", data))
    client().patch(f"items/{resource_id}", data)
    return resource_id


def add_tag_to_experiment(experiment_id: int, tag: str) -> None:
    """Attach a tag; eLabFTW creates it in the team when it does not exist yet."""
    _ensure_writable(experiment_id)
    client().post(f"experiments/{experiment_id}/tags", {"tag": tag})


def link_resource_to_experiment(experiment_id: int, resource_id: int) -> None:
    _ensure_writable(experiment_id)
    # eLabFTW insists on a JSON content type here, hence the empty object
    client().post(f"experiments/{experiment_id}/items_links/{resource_id}", {})


def upload_file_to_experiment(
    experiment_id: int, file_name: str, file_content: bytes, comment: str = ""
) -> None:
    _ensure_writable(experiment_id)
    client().upload(f"experiments/{experiment_id}/uploads", file_name, file_content, comment)

