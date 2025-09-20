import os, math
from openai import OpenAI

"""
Helpers de embeddings y coseno (por si luego usas búsqueda semántica).
"""

_client = None
def _client_once():
    """Singleton de OpenAI client para no reconstruir por llamada."""

    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def embed_text(text: str, model: str = "text-embedding-3-large"):
    """Retorna vector embedding para `text`."""

    client = _client_once()
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def cosine(a, b):
    """Similitud de coseno entre dos vectores."""

    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb + 1e-12)