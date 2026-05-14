import os
import re
from pathlib import Path

import fitz

from app.embeddings import get_client, get_embedding
from app.vector_store import (
    add_vector,
    clear_vector_db,
    load_index_from_disk,
    save_index_to_disk,
    search,
)

UPLOAD_DIR = Path("data/uploads")


def load_documents():
    documents = []

    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)

        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        elif filename.endswith(".pdf"):
            content = load_pdf(file_path)

        else:
            continue

        documents.append({
            "filename": filename,
            "content": content
        })

    return documents


def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        page_text = page.get_text()

        if page_text:
            text += page_text + "\n"

    text = clean_text(text)

    return text


def clean_text(text):
    text = re.sub(r"\n\d+\n", "\n", text)

    text = re.sub(r"\n{2,}", "\n\n", text)

    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        if len(line) > 0:
            cleaned.append(line)

    return "\n".join(cleaned)


def split_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])

        if len(chunk) > 50:
            chunks.append(chunk)

    return chunks


def index_documents() -> None:
    clear_vector_db()

    documents = load_documents()

    for doc in documents:
        chunks = split_into_chunks(doc["content"])

        for chunk in chunks:
            embedding = get_embedding(chunk)

            add_vector(
                embedding=embedding,
                text=chunk,
                metadata={
                    "filename": doc["filename"]
                },
            )

    save_index_to_disk()


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    client = get_client()

    context_text = "\n\n---\n\n".join(
        f"Source: {item['metadata']['filename']}\n{item['text']}"
        for item in context_chunks
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful RAG assistant. "
                    #"Always answer in Traditional Chinese. "
                    "Answer the user's question only based on the provided context. "
                    "If the answer is not in the context, say you could not find it in the uploaded documents. "
                    "Return the answer in plain text. "
                    "Do NOT use markdown formatting. "
                    "Do NOT use **, bullet points, or special symbols. "
                    "Use simple sentences separated by periods. "
                    "Keep the answer concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context_text}\n\n"
                    "Answer the question based on the context above."
                ),
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content or "No answer generated."


def ask_rag(question: str, debug: bool = False, filename: str | None = None) -> dict:
    if not question.strip():
        return {
            "question": question,
            "answer": "Question cannot be empty.",
            "sources": [],
            "retrieved_chunks": [],
        }

    query_embedding = get_embedding(question)

    comparison_keywords = {
        "compare",
        "comparison",
        "difference",
        "differences",
        "trade-off",
        "trade-offs",
    }

    question_lower = question.lower()

    top_k = 4 if any(
        word in question_lower
        for word in comparison_keywords
    ) else 2

    raw_results = search(
        query_embedding,
        top_k=top_k,
        filename=filename,
    )

    similarity_threshold = 0.15

    raw_results = [
        item for item in raw_results
        if item["score"] >= similarity_threshold
    ]

    if not raw_results:
        return {
            "question": question,
            "answer": "I could not find relevant content in the uploaded documents.",
            "sources": [],
            "retrieved_chunks": [],
        }

    selected_results = []
    seen_sources = set()
    selected_ids = set()

    for item in raw_results:
        source = item["metadata"]["filename"]
        item_id = id(item)

        if source not in seen_sources:
            selected_results.append(item)
            seen_sources.add(source)
            selected_ids.add(item_id)

    for item in raw_results:
        if len(selected_results) >= top_k:
            break

        item_id = id(item)

        if item_id not in selected_ids:
            selected_results.append(item)
            selected_ids.add(item_id)

    answer = generate_answer(question, selected_results)

    sources = sorted({
        item["metadata"]["filename"]
        for item in selected_results
    })

    result = {
        "question": question,
        "answer": answer,
        "sources": sources,
    }

    if debug:
        result["retrieved_chunks"] = [
            {
                "source": item["metadata"]["filename"],
                "content": item["text"],
                "score": item["score"],
            }
            for item in selected_results
        ]

    return result
