import json
from pathlib import Path

import numpy as np

INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = INDEX_DIR / "vector_index.json"

vector_db = []


def add_vector(embedding, text, metadata):
    vector_db.append(
        {
            "embedding": np.array(embedding),
            "text": text,
            "metadata": metadata,
        }
    )


def clear_vector_db():
    vector_db.clear()


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(query_embedding, top_k):
    query_vector = np.array(query_embedding)
    results = []

    for item in vector_db:
        score = cosine_similarity(query_vector, item["embedding"])
        results.append((score, item))

    results.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in results[:top_k]]


def save_index_to_disk():
    serializable_data = [
        {
            "embedding": item["embedding"].tolist(),
            "text": item["text"],
            "metadata": item["metadata"],
        }
        for item in vector_db
    ]

    INDEX_FILE.write_text(
        json.dumps(serializable_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_index_from_disk():
    if not INDEX_FILE.exists():
        return

    data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))

    vector_db.clear()

    for item in data:
        vector_db.append(
            {
                "embedding": np.array(item["embedding"]),
                "text": item["text"],
                "metadata": item["metadata"],
            }
        )
