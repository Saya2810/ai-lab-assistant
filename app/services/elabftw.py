import os
import json
import requests
import urllib3

# Only needed for our local self-signed HTTPS certificate
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


BASE_URL = "https://localhost:8443/api/v2"
API_KEY = os.environ["ELABFTW_API_KEY"]


class ElabFTWClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": api_key
        }

    def _check_response(self, response):
        """
        Raise a useful error containing the response returned
        by eLabFTW instead of only showing '400 Bad Request'.
        """
        if not response.ok:
            raise RuntimeError(
                f"eLabFTW API error {response.status_code}: "
                f"{response.text}"
            )

    def get(self, endpoint: str):
        response = requests.get(
            f"{self.base_url}/{endpoint}",
            headers=self.headers,
            verify=False
        )

        self._check_response(response)
        return response.json()

    def post(self, endpoint: str, data: dict | None = None):
        """
        Send a POST request to eLabFTW.
        """

        if data is None:
            response = requests.post(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                verify=False
            )

        else:
            response = requests.post(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                json=data,
                verify=False
            )

        self._check_response(response)
        return response

    def patch(self, endpoint: str, data: dict):
        """
        Update an existing eLabFTW entity.
        """

        response = requests.patch(
            f"{self.base_url}/{endpoint}",
            headers=self.headers,
            json=data,
            verify=False
        )

        self._check_response(response)
        return response


# --------------------------------------------------
# Shared eLabFTW client
# --------------------------------------------------

elab = ElabFTWClient(
    base_url=BASE_URL,
    api_key=API_KEY
)


# --------------------------------------------------
# Experiments
# --------------------------------------------------

def get_experiments():
    """
    Return experiments accessible to the current user.
    """

    return elab.get(
        "experiments"
    )


def get_experiment(experiment_id: int):
    """
    Return one complete experiment.
    """

    return elab.get(
        f"experiments/{experiment_id}"
    )


def create_experiment(
    title: str,
    body: str,
    category_id: int | None = None,
    status_id: int | None = None,
    template_id: int | None = None,
):
    """
    Create a new eLabFTW experiment.

    The main experiment properties are updated separately
    after creation because eLabFTW may not apply all fields
    when creating an experiment from a template.
    """

    data = {}

    if template_id is not None:
        data["template"] = template_id

    response = elab.post(
        "experiments",
        data
    )

    # eLabFTW returns the URL of the newly created
    # experiment in the Location header.
    location = response.headers.get(
        "Location"
    )

    if not location:
        raise RuntimeError(
            "eLabFTW did not return a Location header."
        )

    experiment_id = int(
        location.rstrip("/").split("/")[-1]
    )

    return experiment_id


def normalize_math_for_elabftw(text: str) -> str:
    if not text:
        return text

    # Remove Markdown code fences accidentally produced
    # by the LLM.
    text = text.replace("```markdown", "")
    text = text.replace("```md", "")
    text = text.replace("```", "")

    # Display math:
    # \\[ ... \\] or \[ ... \] -> $$ ... $$
    text = text.replace("\\\\[", "$$")
    text = text.replace("\\\\]", "$$")
    text = text.replace("\\[", "$$")
    text = text.replace("\\]", "$$")

    # Inline math:
    # \\( ... \\) or \( ... \) -> $ ... $
    text = text.replace("\\\\(", "$")
    text = text.replace("\\\\)", "$")
    text = text.replace("\\(", "$")
    text = text.replace("\\)", "$")

    # The LLM sometimes returns LaTeX commands with
    # an additional escaping layer.
    text = text.replace("\\\\frac", "\\frac")
    text = text.replace("\\\\sqrt", "\\sqrt")
    text = text.replace("\\\\cdot", "\\cdot")
    text = text.replace("\\\\times", "\\times")
    text = text.replace("\\\\approx", "\\approx")
    text = text.replace("\\\\rightarrow", "\\rightarrow")
    text = text.replace("\\\\Rightarrow", "\\Rightarrow")
    text = text.replace("\\\\left", "\\left")
    text = text.replace("\\\\right", "\\right")
    text = text.replace("\\\\text", "\\text")
    text = text.replace("\\\\mathrm", "\\mathrm")

    return text.strip()

def update_experiment(
    experiment_id: int,
    title: str | None = None,
    body: str | None = None,
    category_id: int | None = None,
    status_id: int | None = None,
):
    """
    Update the main properties of an eLabFTW experiment.

    Laboratory notes are normalized for eLabFTW Markdown
    and MathJax before being sent to the API.
    """

    data = {}

    if title is not None:
        data["title"] = title

    if body is not None:
        body = normalize_math_for_elabftw(
            body
        )

        print()
        print("BODY SENT TO ELABFTW:")
        print(body)
        print()

        data["body"] = body

        # eLabFTW content type:
        # 2 = Markdown
        data["content_type"] = 2

    if category_id is not None:
        data["category"] = category_id

    if status_id is not None:
        data["status"] = status_id

    if not data:
        return None

    return elab.patch(
        f"experiments/{experiment_id}",
        data,
    )

# --------------------------------------------------
# Tags
# --------------------------------------------------

def add_tag_to_experiment(
    experiment_id: int,
    tag: str
):
    """
    Add a tag to an experiment.
    """

    data = {
        "tag": tag
    }

    return elab.post(
        f"experiments/{experiment_id}/tags",
        data
    )


# --------------------------------------------------
# Resource links
# --------------------------------------------------

def link_resource_to_experiment(
    experiment_id: int,
    resource_id: int
):
    """
    Link an eLabFTW Resource/Item to an experiment.

    eLabFTW expects application/json for this endpoint,
    so an empty JSON object is sent.
    """

    return elab.post(
        f"experiments/{experiment_id}/items_links/{resource_id}",
        {}
    )


# --------------------------------------------------
# Categories
# --------------------------------------------------

def get_experiment_categories():
    """
    Return experiment categories of the current team.
    """

    return elab.get(
        "teams/current/experiments_categories"
    )


# --------------------------------------------------
# Statuses
# --------------------------------------------------

def get_experiment_statuses():
    """
    Return experiment statuses of the current team.
    """

    return elab.get(
        "teams/current/experiments_status"
    )


# --------------------------------------------------
# Templates
# --------------------------------------------------

def get_experiment_templates():
    """
    Return experiment templates available to the user.

    This endpoint returns the template overview.
    """

    return elab.get(
        "experiments_templates"
    )


def get_experiment_template(
    template_id: int
):
    """
    Return one complete experiment template,
    including metadata.
    """

    return elab.get(
        f"experiments_templates/{template_id}"
    )


def get_custom_fields_from_template(
    template_id: int
):
    """
    Read custom-field definitions from an
    eLabFTW experiment template.
    """

    template = get_experiment_template(
        template_id
    )

    metadata = template.get(
        "metadata"
    )

    if not metadata:
        return {}

    if isinstance(metadata, str):
        metadata = json.loads(
            metadata
        )

    return metadata.get(
        "extra_fields",
        {}
    )


# --------------------------------------------------
# Resources
# --------------------------------------------------

def get_resources():
    """
    Return eLabFTW Resources / Items.
    """

    return elab.get(
        "items"
    )


# --------------------------------------------------
# Tags
# --------------------------------------------------

def get_tags():
    """
    Return tags of the current team.
    """

    return elab.get(
        "teams/current/tags"
    )


# --------------------------------------------------
# Experiment custom fields / metadata
# --------------------------------------------------

def get_custom_fields_from_experiment(
    experiment_id: int
):
    """
    Read custom fields from an existing experiment.
    """

    experiment = get_experiment(
        experiment_id
    )

    metadata = experiment.get(
        "metadata"
    )

    if not metadata:
        return {}

    if isinstance(metadata, str):
        metadata = json.loads(
            metadata
        )

    return metadata.get(
        "extra_fields",
        {}
    )


def get_experiment_metadata(
    experiment_id: int
):
    """
    Return the complete metadata object
    of an experiment.
    """

    experiment = get_experiment(
        experiment_id
    )

    metadata = experiment.get(
        "metadata"
    )

    if not metadata:
        return {}

    if isinstance(metadata, str):
        metadata = json.loads(
            metadata
        )

    return metadata


def update_experiment_metadata(
    experiment_id: int,
    metadata: dict
):
    """
    Write the complete metadata object
    to an experiment.
    """

    data = {
        "metadata": json.dumps(
            metadata
        )
    }

    return elab.patch(
        f"experiments/{experiment_id}",
        data
    )


def update_experiment_custom_fields(
    experiment_id: int,
    custom_fields: dict
):
    """
    Update the custom fields of an experiment while
    preserving other metadata that may already exist.
    """

    metadata = get_experiment_metadata(
        experiment_id
    )

    metadata["extra_fields"] = (
        custom_fields
    )

    return update_experiment_metadata(
        experiment_id=experiment_id,
        metadata=metadata
    )

# --------------------------------------------------
# Attachments / uploads
# --------------------------------------------------

def upload_file_to_experiment(
    experiment_id: int,
    file_name: str,
    file_content: bytes,
    comment: str = "",
):
    """
    Upload a file as an attachment to an experiment.
    """

    files = {
        "file": (
            file_name,
            file_content
        )
    }

    data = {}

    if comment:
        data["comment"] = comment

    response = requests.post(
        f"{BASE_URL}/experiments/{experiment_id}/uploads",
        headers={
            "Authorization": API_KEY
        },
        files=files,
        data=data,
        verify=False
    )

    elab._check_response(
        response
    )

    return response