from services.elabftw import (
    get_experiments,
    get_experiment,
)

from services.ai import MODEL

from services.rag_engine.vectorstore import (
    index_experiment,
    search_experiments,
)

import ollama


def sync_labbook():
    """
    Synchronize eLabFTW experiments with ChromaDB.
    """

    overview = get_experiments()

    if not isinstance(overview, list):
        raise RuntimeError(
            "Could not retrieve experiments from eLabFTW."
        )

    indexed = 0

    for item in overview:
        experiment_id = item.get("id")

        if experiment_id is None:
            continue

        try:
            experiment = get_experiment(
                experiment_id
            )

            index_experiment(experiment)

            indexed += 1

        except Exception as error:
            print(
                f"Could not index experiment "
                f"{experiment_id}: {error}"
            )

    return indexed


def answer_lab_question(question: str) -> dict:
    """
    Answer laboratory questions using semantic RAG.
    """

    results = search_experiments(
        question,
        top_k=5,
    )

    if not results:
        return {
            "answer": (
                "No indexed laboratory experiments "
                "were found. Synchronize the laboratory "
                "notebook first."
            ),
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        result["text"]
        for result in results
    )

    system_prompt = """
You are a scientific laboratory assistant.

Answer questions using ONLY the retrieved laboratory
notebook entries.

Rules:

- Never invent experimental results.
- Preserve numerical values and units exactly.
- Do not use outside knowledge.
- Mention the relevant experiment IDs.
- If the retrieved information is insufficient,
  say so clearly.
- Keep answers concise and scientific.
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
                "content": (
                    f"LABORATORY NOTEBOOK:\n\n"
                    f"{context}\n\n"
                    f"QUESTION:\n{question}"
                ),
            },
        ],
        options={
            "temperature": 0,
        },
    )

    answer = response["message"]["content"]

    sources = [
        {
            "id": result["metadata"]["experiment_id"],
            "title": result["metadata"]["title"],
            "date": result["metadata"]["date"],
        }
        for result in results
    ]

    return {
        "answer": answer,
        "sources": sources,
    }