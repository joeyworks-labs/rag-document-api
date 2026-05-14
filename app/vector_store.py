import chromadb
import uuid

client = chromadb.PersistentClient(path="data/chroma_db")

collection = client.get_or_create_collection(
    name="rag_documents"
)


def add_vector(embedding, text, metadata):
    collection.add(
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
        ids=[str(uuid.uuid4())]
    )


def clear_vector_db():
    all_items = collection.get()

    if all_items["ids"]:
        collection.delete(ids=all_items["ids"])


def search(query_embedding, top_k, filename=None):
    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }

    if filename:
        query_args["where"] = {
            "filename": filename
        }

    results = collection.query(**query_args)

    retrieved = []

    for i in range(len(results["documents"][0])):
        retrieved.append({
            "score": 1 - results["distances"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    return retrieved


def save_index_to_disk():
    pass


def load_index_from_disk():
    pass
