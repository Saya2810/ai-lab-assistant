"""Create a complete eLabFTW experiment, and tell the truth about partial failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable

from lab_assistant import elab_api


@dataclass
class CreateResult:
    """``experiment_id`` exists as soon as the experiment was created; ``warnings``
    lists every later step that failed (the entry then exists but is incomplete)."""

    experiment_id: int
    warnings: list[str] = field(default_factory=list)
    done: list[str] = field(default_factory=list)
    new_resource_id: int | None = None

    @property
    def complete(self) -> bool:
        return not self.warnings


def create_complete_experiment(
    title: str,
    body: str,
    template_id: int | None = None,
    category_id: int | None = None,
    status_id: int | None = None,
    resource_id: int | None = None,
    new_resource: str | None = None,
    new_resource_category_id: int | None = None,
    tags: list[str] | None = None,
    custom_fields: dict | None = None,
    sources: dict | None = None,
    api: ModuleType = elab_api,
) -> CreateResult:
    """Create the experiment and fill it in step by step.

    Only the creation itself may raise (nothing exists yet then). Every later step
    is isolated: one failing tag or upload no longer hides that the experiment was
    created, and no longer prevents the remaining steps.

    Existing entries are never touched: the eLabFTW layer only allows writes to the
    experiment created here, and closes it again when this function returns.

    ``new_resource`` is the title of a project/resource to create and link. It is
    created only after the experiment exists, so a failed creation never leaves an
    orphaned resource behind.
    """
    experiment_id = api.create_experiment(template_id=template_id)
    try:
        return _fill_in(
            api, experiment_id, title, body, category_id, status_id, resource_id,
            new_resource, new_resource_category_id, tags, custom_fields, sources,
        )  # fmt: skip
    finally:
        api.close_entry(experiment_id)  # written: the assistant may never change it again


def _fill_in(
    api, experiment_id, title, body, category_id, status_id, resource_id,
    new_resource, new_resource_category_id, tags, custom_fields, sources,
) -> CreateResult:  # fmt: skip
    result = CreateResult(experiment_id)

    def step(label: str, action: Callable[[], object]) -> None:
        try:
            action()
            result.done.append(label)
        except Exception as exc:  # report, keep going
            result.warnings.append(f"{label}: {exc}")

    step(
        "title, notes, category, status",
        lambda: api.update_experiment(
            experiment_id, title=title, body=body, category_id=category_id, status_id=status_id
        ),
    )
    for tag in dict.fromkeys(tags or []):
        step(f"tag '{tag}'", lambda tag=tag: api.add_tag_to_experiment(experiment_id, tag))
    if new_resource:
        created: dict[str, int] = {}

        def create_resource() -> None:
            created["id"] = api.create_resource(new_resource, category_id=new_resource_category_id)

        step(f"new resource '{new_resource}'", create_resource)
        result.new_resource_id = created.get("id")
        resource_id = result.new_resource_id
    if resource_id is not None:
        step("resource link", lambda: api.link_resource_to_experiment(experiment_id, resource_id))
    if custom_fields:
        step("custom fields", lambda: api.update_experiment_custom_fields(experiment_id, custom_fields))
    for source in (sources or {}).values():
        if not source.get("attach") or source.get("content") is None:
            continue
        step(
            f"upload '{source['name']}'",
            lambda source=source: api.upload_file_to_experiment(
                experiment_id,
                source["name"],
                source["content"],
                comment="Source file uploaded through Lab Assistant.",
            ),
        )
    return result
