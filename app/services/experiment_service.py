from services.elabftw import (
    create_experiment,
    update_experiment,
    add_tag_to_experiment,
    link_resource_to_experiment,
    upload_file_to_experiment,
    update_experiment_custom_fields,
)


def create_complete_experiment(
    title,
    body,
    template_id=None,
    category_id=None,
    status_id=None,
    resource_id=None,
    tags=None,
    custom_fields=None,
    sources=None,
):
    """
    Create a complete eLabFTW experiment.

    Sources are uploaded only when their
    'attach' value is True.
    """

    tags = tags or []
    custom_fields = custom_fields or {}
    sources = sources or {}

    experiment_id = create_experiment(
        title=title,
        body=body,
        category_id=category_id,
        status_id=status_id,
        template_id=template_id,
    )

    update_experiment(
        experiment_id=experiment_id,
        title=title,
        body=body,
        category_id=category_id,
        status_id=status_id,
    )

    for tag in tags:
        add_tag_to_experiment(
            experiment_id=experiment_id,
            tag=tag,
        )

    if resource_id is not None:
        link_resource_to_experiment(
            experiment_id=experiment_id,
            resource_id=resource_id,
        )

    if custom_fields:
        update_experiment_custom_fields(
            experiment_id=experiment_id,
            custom_fields=custom_fields,
        )

    for source in sources.values():
        if not source.get("attach"):
            continue

        content = source.get("content")

        if content is None:
            continue

        upload_file_to_experiment(
            experiment_id=experiment_id,
            file_name=source["name"],
            file_content=content,
            comment=(
                "Source file uploaded through "
                "Lab Assistant."
            ),
        )

    return experiment_id