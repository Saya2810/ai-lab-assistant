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
    tags: list[str] | None = None,
    custom_fields: dict | None = None,
    sources: dict | None = None,
    api: ModuleType = elab_api,
) -> CreateResult:
    """Create the experiment and fill it in step by step.

    Only the creation itself may raise (nothing exists yet then). Every later step
    is isolated: one failing tag or upload no longer hides that the experiment was
    created, and no longer prevents the remaining steps.
    """
    experiment_id = api.create_experiment(template_id=template_id)
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
