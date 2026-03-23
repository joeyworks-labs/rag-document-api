# RAG Document API

A backend service for document question answering using Retrieval-Augmented Generation (RAG).

This project enables users to upload documents and ask questions, with answers generated based on relevant retrieved content.

---

## Features

* Document upload (`.txt`, `.md`)
* Text chunking (paragraph-based)
* Embedding-based semantic retrieval
* Multi-document support
* LLM-generated answers
* Source attribution
* Debug mode (`retrieved_chunks`)
* Persistent vector index (no re-embedding on restart)

---

## Architecture

```
Client
↓
FastAPI (/ask, /upload)
↓
RAG Pipeline
├─ Chunking
├─ Embedding (OpenAI)
├─ Vector Search (cosine similarity)
├─ Retrieval (top-k + multi-source)
└─ LLM Answer Generation
```

---

## API Endpoints

### POST /upload

Upload a document for indexing.

**Supported formats**

* .txt
* .md

---

### POST /ask

Ask a question based on uploaded documents.

#### Request

```json
{
  "question": "What are the advantages of microservices?"
}
```

#### Query Params

* `debug` (optional): return retrieved chunks

---

## Response

### Normal mode

```json
{
  "question": "...",
  "answer": "...",
  "sources": ["file.md"]
}
```

### Debug mode

```json
{
  "question": "...",
  "answer": "...",
  "sources": ["file.md"],
  "retrieved_chunks": [
    {
      "source": "file.md",
      "content": "..."
    }
  ]
}
```

---

## How It Works

1. Upload document
2. Split into chunks (paragraph-based)
3. Generate embeddings
4. Store in vector index (JSON)

On query:

1. Embed question
2. Perform similarity search
3. Retrieve relevant chunks
4. Generate answer using LLM

---

## Tech Stack

* Python
* FastAPI
* OpenAI API
* NumPy
* WSL (recommended dev environment)

---

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:

```
OPENAI_API_KEY=your_api_key
```

Run server:

```bash
python3 -m uvicorn app.main:app --reload
```

---

## Example

### Question

Compare monolith and microservices

### Answer

Monolith systems are simpler to develop and deploy with lower operational complexity. Microservices provide better scalability and flexibility but introduce higher system complexity and operational overhead.

---

## Current Status

This project is a RAG MVP with production-oriented design, including:

* Semantic retrieval (embedding-based)
* Multi-document reasoning
* Persistent index
* Debug visibility

---

## Future Improvements

* PDF support
* Incremental indexing (no full rebuild on upload)
* Vector database integration (e.g. FAISS, Pinecone)
* Structured output (JSON responses)
* Frontend UI (e.g. Streamlit)
* Go-based service layer (gateway / orchestration)

---

## Notes

* `retrieved_chunks` is intended for debugging and evaluation, not end-user display
* LLM output is constrained to plain text (no markdown formatting)
