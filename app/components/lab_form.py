import streamlit as st

from components.custom_fields import (
    find_matching_option,
    render_custom_field,
)

from services.elabftw import (
    get_custom_fields_from_template,
)

from services.experiment_service import (
    create_complete_experiment,
)


def make_options(items):
    return {
        item["title"]: item["id"]
        for item in items
    }


def get_tag_names(tags):
    return [
        tag["tag"]
        for tag in tags
    ]


def convert_lab_notes(lab_notes):
    if not lab_notes:
        return ""

    if isinstance(lab_notes, list):
        return "\n\n".join(
            str(note)
            for note in lab_notes
        )

    return str(lab_notes)


def apply_ai_result(
    ai_result,
    option_names,
):
    """
    Copy a newly generated AI draft into
    the editable form.
    """

    if not ai_result:
        return

    st.session_state["lab_title"] = (
        ai_result.get("title", "")
    )

    st.session_state["lab_body"] = (
        convert_lab_notes(
            ai_result.get(
                "lab_notes",
                [],
            )
        )
    )

    fields = {
        "template": "lab_template",
        "category": "lab_category",
        "status": "lab_status",
        "resource": "lab_resource",
    }

    for ai_field, state_key in fields.items():
        matched = find_matching_option(
            ai_result.get(ai_field),
            option_names[ai_field],
        )

        st.session_state[state_key] = (
            matched or "None"
        )

    matched_tags = []

    for tag in ai_result.get(
        "tags",
        [],
    ):
        matched = find_matching_option(
            tag,
            option_names["tags"],
        )

        if matched:
            matched_tags.append(
                matched
            )

    st.session_state[
        "lab_tags"
    ] = matched_tags

    st.session_state[
        "ai_measurements"
    ] = ai_result.get(
        "measurements",
        {},
    )


def render_structure(
    template_names,
    category_names,
    status_names,
    resource_names,
    tag_names,
):
    st.subheader("Entry structure")

    template = st.selectbox(
        "Template",
        ["None"] + template_names,
        key="lab_template",
    )

    category = st.selectbox(
        "Category",
        ["None"] + category_names,
        key="lab_category",
    )

    status = st.selectbox(
        "Status",
        ["None"] + status_names,
        key="lab_status",
    )

    resource = st.selectbox(
        "Project / Resource",
        ["None"] + resource_names,
        key="lab_resource",
    )

    tags = st.multiselect(
        "Tags",
        tag_names,
        key="lab_tags",
    )

    return (
        template,
        category,
        status,
        resource,
        tags,
    )


def render_custom_fields(
    template_id,
):
    st.subheader("Custom fields")

    if template_id is None:
        st.info(
            "Select a template to load "
            "its custom fields."
        )
        return {}

    try:
        fields = (
            get_custom_fields_from_template(
                template_id
            )
        )

    except Exception as error:
        st.warning(
            "Could not load custom "
            f"fields: {error}"
        )
        return {}

    if not fields:
        st.info(
            "The selected template has "
            "no custom fields."
        )
        return {}

    ai_measurements = (
        st.session_state.get(
            "ai_measurements",
            {},
        )
    )

    values = {}

    for name, data in fields.items():
        values[name] = (
            render_custom_field(
                field_name=name,
                field_data=data,
                ai_measurements=(
                    ai_measurements
                ),
            )
        )

    return values


def get_id(
    selected_name,
    options,
):
    if selected_name == "None":
        return None

    return options[selected_name]


def render_lab_form(
    templates,
    categories,
    statuses,
    resources,
    tags,
    ai_result=None,
):

    # -----------------------------------
    # eLabFTW options
    # -----------------------------------

    options = {
        "template": make_options(
            templates
        ),
        "category": make_options(
            categories
        ),
        "status": make_options(
            statuses
        ),
        "resource": make_options(
            resources
        ),
    }

    names = {
        key: list(value.keys())
        for key, value in options.items()
    }

    names["tags"] = get_tag_names(
        tags
    )

    # -----------------------------------
    # Apply new AI result once
    # -----------------------------------

    if ai_result:
        signature = repr(
            ai_result
        )

        if (
            st.session_state.get(
                "applied_ai_result"
            )
            != signature
        ):
            apply_ai_result(
                ai_result,
                names,
            )

            st.session_state[
                "applied_ai_result"
            ] = signature

    # -----------------------------------
    # Structure
    # -----------------------------------

    (
        selected_template,
        selected_category,
        selected_status,
        selected_resource,
        selected_tags,
    ) = render_structure(
        template_names=names[
            "template"
        ],
        category_names=names[
            "category"
        ],
        status_names=names[
            "status"
        ],
        resource_names=names[
            "resource"
        ],
        tag_names=names[
            "tags"
        ],
    )

    template_id = get_id(
        selected_template,
        options["template"],
    )

    category_id = get_id(
        selected_category,
        options["category"],
    )

    status_id = get_id(
        selected_status,
        options["status"],
    )

    resource_id = get_id(
        selected_resource,
        options["resource"],
    )

    # -----------------------------------
    # Entry
    # -----------------------------------

    st.subheader(
        "Laboratory entry"
    )

    title = st.text_input(
        "Title",
        key="lab_title",
    )

    body = st.text_area(
        "Lab notes",
        height=300,
        key="lab_body",
    )

    # -----------------------------------
    # Custom fields
    # -----------------------------------

    custom_fields = (
        render_custom_fields(
            template_id
        )
    )

    # -----------------------------------
    # Save
    # -----------------------------------

    if not st.button(
        "Create lab entry",
        key="create_lab_entry",
    ):
        return

    if not title.strip():
        st.warning(
            "Please enter a title."
        )
        return

    if not body.strip():
        st.warning(
            "Please enter your lab notes."
        )
        return

    sources = st.session_state.get(
        "lab_sources",
        {},
    )

    experiment_id = None

    try:
        experiment_id = (
            create_complete_experiment(
                title=title,
                body=body,
                template_id=template_id,
                category_id=category_id,
                status_id=status_id,
                resource_id=resource_id,
                tags=selected_tags,
                custom_fields=(
                    custom_fields
                ),
                sources=sources,
            )
        )

        st.success(
            "Lab entry successfully "
            "created in eLabFTW "
            f"(Experiment ID "
            f"{experiment_id})."
        )

    except Exception as error:
        if experiment_id is None:
            st.error(
                "Could not create lab "
                f"entry: {error}"
            )
        else:
            st.error(
                f"Experiment "
                f"{experiment_id} was "
                "created, but a later "
                f"step failed: {error}"
            )