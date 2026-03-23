import os
from openai import OpenAI

client = None


def get_client():
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        client = OpenAI(api_key=api_key)
    return client


def get_embedding(text: str) -> list[float]:
    client = get_client()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding
