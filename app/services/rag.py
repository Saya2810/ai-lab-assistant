from services.elabftw import (
    get_experiments,
    get_experiment,
)
from services.ai import MODEL

import ollama


MAX_EXPERIMENTS = 20


def experiment_to_text(experiment: dict) -> str:
    """
    Convert an eLabFTW experiment into text that can be
    searched and supplied to the language model.
    """

    experiment_id = experiment.get("id", "")
    title = experiment.get("title", "")
    date = experiment.get("date", "")
    body = experiment.get("body", "")

    return (
        f"Experiment ID: {experiment_id}\n"
        f"Title: {title}\n"
        f"Date: {date}\n"
        f"Content:\n{body}"
    )


def load_lab_context() -> list[dict]:
    """
    Load all accessible experiments from eLabFTW.

    The experiments overview does not contain the body,
    so every experiment is loaded individually.
    """

    experiment_overview = get_experiments()

    if not isinstance(experiment_overview, list):
        return []

    experiments = []

    for item in experiment_overview:
        experiment_id = item.get("id")

        if experiment_id is None:
            continue

        try:
            experiment = get_experiment(
                experiment_id
            )

            experiments.append(
                experiment
            )

        except Exception as error:
            print(
                f"Could not load experiment "
                f"{experiment_id}: {error}"
            )

    return experiments


def select_relevant_experiments(
    question: str,
    experiments: list[dict],
) -> list[dict]:
    """
    Select experiments deterministically.

    Rules:
    - Questions about today return ALL experiments from today.
    - Questions about yesterday return ALL experiments from yesterday.
    - Otherwise, filter experiments using words from the question.
    """

    from datetime import date, timedelta

    question_lower = question.lower()

    # --------------------------------------------------
    # TODAY
    # --------------------------------------------------

    today_words = {
        "today",
        "heute",
        "heutige",
        "heutigen",
        "heutiger",
    }

    if any(
        word in question_lower
        for word in today_words
    ):
        today = date.today().isoformat()

        return [
            experiment
            for experiment in experiments
            if str(
                experiment.get("date", "")
            ) == today
        ]

    # --------------------------------------------------
    # YESTERDAY
    # --------------------------------------------------

    yesterday_words = {
        "yesterday",
        "gestern",
    }

    if any(
        word in question_lower
        for word in yesterday_words
    ):
        yesterday = (
            date.today() - timedelta(days=1)
        ).isoformat()

        return [
            experiment
            for experiment in experiments
            if str(
                experiment.get("date", "")
            ) == yesterday
        ]

    # --------------------------------------------------
    # KEYWORD SEARCH
    # --------------------------------------------------

    stop_words = {
        # English
        "the",
        "and",
        "for",
        "with",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "was",
        "were",
        "during",
        "about",
        "from",
        "into",
        "that",
        "this",
        "there",
        "have",
        "has",
        "had",

        # German
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einer",
        "einen",
        "einem",
        "und",
        "oder",
        "was",
        "wie",
        "wann",
        "wo",
        "war",
        "waren",
        "ist",
        "sind",
        "hat",
        "haben",
        "über",
        "im",
        "am",
        "vom",
        "zum",
        "zur",
    }

    question_words = {
        word.lower().strip(
            ".,?!:;()[]{}\"'"
        )
        for word in question.split()
        if (
            len(
                word.strip(
                    ".,?!:;()[]{}\"'"
                )
            ) > 2
            and word.lower().strip(
                ".,?!:;()[]{}\"'"
            ) not in stop_words
        )
    }

    relevant_experiments = []

    for experiment in experiments:

        searchable_text = " ".join(
            [
                str(
                    experiment.get(
                        "title",
                        "",
                    )
                ),
                str(
                    experiment.get(
                        "body",
                        "",
                    )
                ),
                str(
                    experiment.get(
                        "tags",
                        "",
                    )
                ),
                str(
                    experiment.get(
                        "category_title",
                        "",
                    )
                ),
                str(
                    experiment.get(
                        "status_title",
                        "",
                    )
                ),
            ]
        ).lower()

        matching_words = [
            word
            for word in question_words
            if word in searchable_text
        ]

        if matching_words:
            relevant_experiments.append(
                experiment
            )

    return relevant_experiments

def answer_lab_question(question: str) -> dict:
    """
    Answer a question using information retrieved from
    the user's eLabFTW experiments.

    Returns both the generated answer and the experiments
    that were used as sources.
    """

    experiments = load_lab_context()

    relevant_experiments = select_relevant_experiments(
        question=question,
        experiments=experiments,
    )

    if not relevant_experiments:
        return {
            "answer": (
                "I could not find relevant information "
                "in the laboratory notebook."
            ),
            "sources": [],
        }

    context_parts = []

    for experiment in relevant_experiments:
        context_parts.append(
            experiment_to_text(experiment)
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    system_prompt = """
You are a laboratory notebook assistant.

Answer questions using ONLY the laboratory notebook
entries supplied to you.

Rules:

- Do not use outside knowledge.
- Do not invent missing information.
- Do not infer experimental results that are not explicitly
  present in the laboratory notebook.
- Preserve numerical values and units exactly.
- If entries contain conflicting information, mention the
  conflict instead of deciding which value is correct.
- If the supplied entries do not contain enough information
  to answer the question, say so clearly.
- When making a factual statement, identify the relevant
  experiment ID.
- Keep answers concise and scientific.
"""

    user_prompt = f"""
LABORATORY NOTEBOOK ENTRIES:

--- BEGIN LAB NOTEBOOK ---

{context}

--- END LAB NOTEBOOK ---

QUESTION:

{question}

Answer using only the laboratory notebook entries above.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    answer = response[
        "message"
    ][
        "content"
    ].strip()

    sources = [
        {
            "id": experiment.get("id"),
            "title": experiment.get("title"),
            "date": experiment.get("date"),
        }
        for experiment in relevant_experiments
    ]

    return {
        "answer": answer,
        "sources": sources,
    }