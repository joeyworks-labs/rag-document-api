# RAG Document API

A Retrieval-Augmented Generation (RAG) backend service for document-based question answering.

Users can upload documents and ask questions through semantic retrieval and LLM-generated responses.

---

## Features

- Document upload (`.txt`, `.md`, `.pdf`)
- Semantic retrieval using embeddings
- Multi-document retrieval
- Source attribution
- Debug retrieval mode
- Persistent vector index
- Frontend UI support

---

## Architecture

```text
Client
↓
FastAPI API Layer
↓
RAG Pipeline
├─ Document Parsing
├─ Chunking
├─ Embedding Generation
├─ Vector Similarity Search
├─ Context Retrieval
└─ LLM Response Generation
```

---

## API Endpoints

### POST `/upload`

Upload and index documents.

Supported formats:

- `.txt`
- `.md`
- `.pdf`

---

### POST `/ask`

Ask questions based on indexed documents.

Request:

```json
{
  "question": "What are the advantages of microservices?"
}
```

Optional query params:

- `debug=true`

---

## Example Response

```json
{
  "question": "...",
  "answer": "...",
  "sources": ["architecture.pdf"]
}
```

Debug mode:

```json
{
  "question": "...",
  "answer": "...",
  "sources": ["architecture.pdf"],
  "retrieved_chunks": [
    {
      "source": "architecture.pdf",
      "content": "..."
    }
  ]
}
```

---

## Tech Stack

- Python
- FastAPI
- OpenAI API
- NumPy
- PyMuPDF
- WSL

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=your_api_key
```

Run server:

```bash
python3 -m uvicorn app.main:app --reload
```

---

## Current Capabilities

- Embedding-based semantic retrieval
- PDF document parsing
- Persistent vector storage
- Multi-document reasoning
- Retrieval debugging support
- Basic frontend UI integration

---

## Future Improvements

- Chunking Strategy Optimization
- Similarity Threshold Control
- PDF Parsing Quality Enhancement
- Vector Database Upgrade
- Metadata Filtering Support
- Retrieval Reranking Mechanism
- Hybrid Search Architecture

---

## Notes

- `retrieved_chunks` is intended for debugging and evaluation
- Responses are generated using retrieved document context