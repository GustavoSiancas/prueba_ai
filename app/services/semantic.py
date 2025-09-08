import os, math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str, model: str = "text-embedding-3-large"):
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb + 1e-12)

def summarize_video_textual(resumen_visual: str | None, transcript: str | None):
    """Combina resumen visual + transcripción en un texto único compacto para embed."""
    parts = []
    if resumen_visual: parts.append(resumen_visual)
    if transcript: parts.append("TRANSCRIPCIÓN: " + transcript)
    if not parts: return ""
    
    text = "\n".join(parts)
    return text[:6000]