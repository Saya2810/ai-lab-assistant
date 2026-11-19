"""Step 2: the editable eLabFTW form, and the (safe) creation of the entry."""

from __future__ import annotations

import hashlib
import json
from types import ModuleType
from typing import Callable

import streamlit as st

from lab_assistant import elab_api
from lab_assistant.ai import Draft
from lab_assistant.components.custom_fields import render_custom_field, unmatched_measurements
from lab_assistant.experiment_service import create_complete_experiment
from shared.config import experiment_url


def make_options(items: list[dict], label_key: str = "title") -> dict[str, int]:
    """label -> id. Two entries with the same title used to collapse into one."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item[label_key]] = counts.get(item[label_key], 0) + 1
    return {
        (item[label_key] if counts[item[label_key]] == 1 else f"{item[label_key]} (#{item['id']})"): item["id"]
        for item in items
    }


def tag_names(tags: list[dict]) -> list[str]:
    return list(dict.fromkeys(tag["tag"] for tag in tags if tag.get("tag")))


def template_fields(api: ModuleType, template_id: int) -> dict:
    """Custom-field definitions of a template, fetched once per template and session."""
    cache = st.session_state.setdefault("template_fields", {})
    if template_id not in cache:
        cache[template_id] = api.get_custom_fields_from_template(template_id)
    return cache[template_id]


def apply_draft(draft: Draft) -> None:
    """Copy a freshly generated draft into the form. Structure choices (template,
    category, tags, ...) stay exactly as the user set them."""
    st.session_state["lab_title"] = draft.title
    st.session_state["lab_body"] = draft.lab_notes
    st.session_state["ai_measurements"] = draft.measurements


def render_draft_checks(draft: Draft | None) -> None:
    if draft is None:
        return
    if draft.numbers.invented:
        st.warning(
            "Numbers in the draft that do **not** appear in the sources – please verify: "
            + ", ".join(f"`{n}`" for n in draft.numbers.invented[:12])
        )
    if draft.numbers.missing:
        st.info(
            "Numbers from the sources that are **missing** in the draft: "
            + ", ".join(f"`{n}`" for n in draft.numbers.missing[:12])
        )
    for note in draft.notes:
        st.caption(f"Draft note: {note}")


def render_custom_fields(api: ModuleType, template_id: int | None, key_prefix: str) -> tuple[dict, list[str]]:
    st.subheader("Custom fields")
    if template_id is None:
        st.info("Select a template to load its custom fields.")
        return {}, []
    try:
        fields = template_fields(api, template_id)
    except Exception as error:
        st.warning(f"Could not load custom fields: {error}")
        return {}, []
    if not fields:
        st.info("The selected template has no custom fields.")
        return {}, []

    ai_measurements = st.session_state.get("ai_measurements", {})
    values, errors = {}, []
    for name, data in fields.items():
        values[name], error = render_custom_field(name, data, ai_measurements, key_prefix)
        if error:
            errors.append(f"{name}: {error}")
    leftover = unmatched_measurements(fields, ai_measurements)
    if leftover:
        st.caption("Measurements from the draft that match no field of this template: " + ", ".join(leftover))
    return values, errors


def _signature(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def existing_resource(name: str | None, resources: dict[str, int]) -> int | None:
    """Id of the resource ``name`` refers to, ignoring case and surrounding spaces."""
    if not name:
        return None
    if name in resources:
        return resources[name]
    wanted = " ".join(name.split()).casefold()
    matches = {i for label, i in resources.items() if " ".join(label.split()).casefold() == wanted}
    return matches.pop() if len(matches) == 1 else None


def render_new_resource(
    resource: str | None, resources: dict[str, int], structure: dict
) -> tuple[str | None, int | None]:
    """A typed name that is not an existing resource: ``(new name, category id)``."""
    name = (resource or "").strip()
    if not name or existing_resource(name, resources) is not None:
        return None, None
    categories = make_options(structure.get("resource_categories", []))
    if len(categories) > 1:
        choice = st.selectbox(
            "Category of the new resource", list(categories), index=0, key="lab_resource_category"
        )
        category_id = categories.get(choice)
    else:
        category_id = next(iter(categories.values()), None)
    where = f" in category '{next(iter(categories))}'" if len(categories) == 1 else ""
    st.caption(f"New resource '{name}' will be created{where} when you create the entry.")
    return name, category_id


def render_lab_form(
    structure: dict,
    draft: Draft | None = None,
    api: ModuleType = elab_api,
    refresh_structure: Callable[[], None] | None = None,
) -> None:
    """``refresh_structure`` forgets the cached eLabFTW structure, so a resource or tag
    created with this entry is offered as an existing one next time (not created twice)."""
    templates = make_options(structure["templates"])
    categories = make_options(structure["categories"])
    statuses = make_options(structure["statuses"])
    resources = make_options(structure["resources"], "title")

    version = st.session_state.get("draft_version", 0)
    if draft is not None and st.session_state.get("applied_draft_version") != version:
        apply_draft(draft)
        st.session_state["applied_draft_version"] = version

    st.subheader("Entry structure")
    template = st.selectbox("Template", list(templates), index=None, placeholder="—", key="lab_template")
    category = st.selectbox("Category", list(categories), index=None, placeholder="—", key="lab_category")
    status = st.selectbox("Status", list(statuses), index=None, placeholder="—", key="lab_status")
    resource = st.selectbox(
        "Project / Resource", list(resources), index=None, key="lab_resource",
        accept_new_options=True, placeholder="Choose, or type a new name and press Enter",
        help="Pick an existing project/resource, or type a new name: it is created in eLabFTW "
        "together with the entry and linked to it.",
    )  # fmt: skip
    new_resource, new_resource_category = render_new_resource(resource, resources, structure)
    tags = st.multiselect(
        "Tags", tag_names(structure["tags"]), key="lab_tags", accept_new_options=True,
        placeholder="Choose, or type a new tag and press Enter",
        help="Pick existing tags, or type a new one: eLabFTW creates it with the entry.",
    )  # fmt: skip
    new_tags = [tag for tag in tags if tag not in set(tag_names(structure["tags"]))]

    ids = {
        "template": templates.get(template),
        "category": categories.get(category),
        "status": statuses.get(status),
        "resource": existing_resource(resource, resources),
        "new_resource": new_resource,
        "new_resource_category": new_resource_category,
    }

    st.subheader("Laboratory entry")
    render_draft_checks(draft)
    title = st.text_input("Title", key="lab_title")
    body = st.text_area("Lab notes", height=300, key="lab_body")

    custom_fields, field_errors = render_custom_fields(api, ids["template"], f"t{ids['template']}_v{version}_")

    sources = st.session_state.get("lab_sources", {})
    attachments = sorted(s["name"] for s in sources.values() if s.get("attach"))
    signature = _signature(title, body, ids, tags, custom_fields, attachments)

    last = st.session_state.get("last_created")
    duplicate = bool(last and last["signature"] == signature)
    allow_duplicate = False
    if duplicate:
        st.info(f"This exact entry was already created as experiment {last['id']}.")
        allow_duplicate = st.checkbox("Create it again anyway", key="allow_duplicate")

    if not st.button("Create lab entry", key="create_lab_entry", disabled=duplicate and not allow_duplicate):
        return
    if not title.strip():
        st.warning("Please enter a title.")
        return
    if not body.strip():
        st.warning("Please enter your lab notes.")
        return
    if field_errors:
        st.error("Please fix the custom fields first: " + "; ".join(field_errors))
        return

    try:
        result = create_complete_experiment(
            title=title.strip(), body=body, template_id=ids["template"], category_id=ids["category"],
            status_id=ids["status"], resource_id=ids["resource"], new_resource=ids["new_resource"],
            new_resource_category_id=ids["new_resource_category"], tags=tags,
            custom_fields=custom_fields, sources=sources, api=api,
        )  # fmt: skip
    except Exception as error:  # nothing was created
        hint = ""
        if "read-only key" in str(error):
            hint = ("\n\nThe eLabFTW API key may only read. Create a read/write key in eLabFTW "
                    "(Settings > API keys) and set it as ELABFTW_WRITE_API_KEY.")
        st.error(f"Could not create the lab entry: {error}{hint}")
        return

    st.session_state["last_created"] = {"signature": signature, "id": result.experiment_id}
    if (result.new_resource_id or new_tags) and refresh_structure is not None:
        refresh_structure()
    link = experiment_url(result.experiment_id)
    if result.complete:
        st.success(f"Lab entry created in eLabFTW (experiment {result.experiment_id}): {link}")
        if result.new_resource_id:
            st.success(f"New resource '{new_resource}' created (#{result.new_resource_id}) and linked.")
    else:
        st.warning(
            f"Experiment {result.experiment_id} was created ({link}), but some steps failed – "
            "please complete it in eLabFTW:\n\n" + "\n".join(f"- {w}" for w in result.warnings)
        )
