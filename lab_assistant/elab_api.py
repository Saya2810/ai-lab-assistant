"""eLabFTW operations used by the Lab Assistant, on top of the shared client.

Differences to the old ``services/elabftw.py``: no key read at import time, TLS and
timeouts come from the shared config, every list endpoint is paginated (eLabFTW
returns only 15 items by default, so resources/tags/templates used to be cut off),
and metadata updates *merge* instead of replacing existing custom fields.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from lab_assistant.formatting import normalize_math_for_elabftw
from shared.elabftw_client import ElabFTWClient, get_client

log = logging.getLogger(__name__)

MARKDOWN_CONTENT_TYPE = 2


def client() -> ElabFTWClient:
    return get_client()


def _all(endpoint: str) -> list[dict]:
    return list(client().iter_paginated(endpoint))


# ------------------------------------------------------------------ structure


def get_experiment_categories() -> list[dict]:
    return _all("teams/current/experiments_categories")


def get_experiment_statuses() -> list[dict]:
    return _all("teams/current/experiments_status")


def get_experiment_templates() -> list[dict]:
    return _all("experiments_templates")


def get_resources() -> list[dict]:
    return _all("items")


def get_tags() -> list[dict]:
    return _all("teams/current/tags")


def load_structure() -> dict[str, list[dict]]:
    """Everything the form needs, in one call (the app caches it)."""
    return {
        "categories": get_experiment_categories(),
        "statuses": get_experiment_statuses(),
        "templates": get_experiment_templates(),
        "resources": get_resources(),
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
    data = {"template": template_id} if template_id is not None else {}
    return ElabFTWClient.created_id(client().post("experiments", data))


def update_experiment(
    experiment_id: int,
    title: str | None = None,
    body: str | None = None,
    category_id: int | None = None,
    status_id: int | None = None,
) -> None:
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


def add_tag_to_experiment(experiment_id: int, tag: str) -> None:
    client().post(f"experiments/{experiment_id}/tags", {"tag": tag})


def link_resource_to_experiment(experiment_id: int, resource_id: int) -> None:
    # eLabFTW insists on a JSON content type here, hence the empty object
    client().post(f"experiments/{experiment_id}/items_links/{resource_id}", {})


def upload_file_to_experiment(
    experiment_id: int, file_name: str, file_content: bytes, comment: str = ""
) -> None:
    client().upload(f"experiments/{experiment_id}/uploads", file_name, file_content, comment)

