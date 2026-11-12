from pathlib import Path

import chromadb
import ollama


EMBEDDING_MODEL = "embeddinggemma"

DB_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "chroma_db"
)

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_or_create_collection(
    name="lab_experiments",
    metadata={"hnsw:space": "cosine"},
)


def embed_text(text: str) -> list[float]:
    """Generate embeddings using local Ollama."""

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response["embeddings"][0]


def index_experiment(experiment: dict):
    """Store an experiment in ChromaDB."""

    experiment_id = str(experiment["id"])

    text = (
        f"Title: {experiment.get('title', '')}\n"
        f"Date: {experiment.get('date', '')}\n"
        f"Content: {experiment.get('body', '')}"
    )

    embedding = embed_text(text)

    collection.upsert(
        ids=[experiment_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[
            {
                "experiment_id": experiment_id,
                "title": str(
                    experiment.get("title", "")
                ),
                "date": str(
                    experiment.get("date", "")
                ),
            }
        ],
    )


def search_experiments(
    question: str,
    top_k: int = 5,
):
    """Retrieve semantically relevant experiments."""

    if collection.count() == 0:
        return []

    embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
    )

    return [
        {
            "text": document,
            "metadata": metadata,
            "distance": distance,
        }
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

