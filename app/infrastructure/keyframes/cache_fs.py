import os, base64
from typing import List

"""
Cache de keyframes en sistema de archivos.

- Cada video se guarda en `KEYFRAMES_DIR/<video_id>/NNN.jpg`.
- Se almacena/lee JPEG crudo (sin prefijo data-URL), y en memoria se expone como base64.
"""

def _dir_for(video_id: str, base_dir: str) -> str:
    """Crea/retorna el directorio para un `video_id` en el cache."""

    d = os.path.join(base_dir, video_id)
    os.makedirs(d, exist_ok=True)
    return d

def save_keyframes_from_b64(video_id: str, frames_b64: List[str], base_dir: str) -> None:
    """Persiste una lista de frames (base64 JPEG) a disco, numerados 000.jpg, 001.jpg, ..."""

    d = _dir_for(video_id, base_dir)
    for i, b64 in enumerate(frames_b64):
        with open(os.path.join(d, f"{i:03d}.jpg"), "wb") as f:
            f.write(base64.b64decode(b64))

def load_keyframes_b64(video_id: str, base_dir: str, limit: int | None = None) -> List[str]:
    """Carga frames desde disco y los devuelve como lista de base64 (sin data-URL)."""

    d = os.path.join(base_dir, video_id)
    if not os.path.isdir(d):
        return []
    files = sorted([p for p in os.listdir(d) if p.endswith(".jpg")])
    if limit is not None:
        files = files[:limit]
    out = []
    for name in files:
        with open(os.path.join(d, name), "rb") as f:
            out.append(base64.b64encode(f.read()).decode("utf-8"))
    return out
