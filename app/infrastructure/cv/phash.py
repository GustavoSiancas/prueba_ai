import cv2
import numpy as np

"""
Huella visual global basada en dHash "mayoritario".
Se muestrean N frames espaciados y se vota bit a bit (64 bits).
"""

def _dhash(image_bgr, hash_size=8):
    """Calcula dHash (64 bits) de una imagen BGR."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Distancia Hamming entre dos vectores binarios."""

    return int(np.count_nonzero(a != b))

def video_fingerprint(path: str, seconds_interval: float = 5.0, max_frames: int = 20):
    """
    Calcula pHash "mayoritario" de un video.

    Args:
        path: ruta al .mp4
        seconds_interval: separación entre frames muestreados.
        max_frames: máximo de frames a votar.

    Returns:
        np.ndarray (uint8) de 64 bits (0/1) o None si falla.
    """

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cv2.CAP_PROP_FRAME_COUNT and cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total / fps) if fps > 0 else 0

    hashes = []
    t = 0.0
    while len(hashes) < max_frames and (duration == 0 or t <= duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        hashes.append(_dhash(frame))
        t += seconds_interval

    cap.release()
    if not hashes:
        return None

    arr = np.stack(hashes).astype(np.int8)
    votes = arr.sum(axis=0) >= (arr.shape[0] / 2.0)
    return votes.astype(np.uint8)

def similarity_percent(fp1, fp2) -> float:
    """
    Similitud (100 - Hamming%) entre dos pHash64.

    Regla:
      - Se recorta a la longitud mínima si difieren.
    """

    if fp1 is None or fp2 is None:
        return 0.0
    if fp1.shape[0] != fp2.shape[0]:
        n = min(fp1.shape[0], fp2.shape[0])
        fp1 = fp1[:n]
        fp2 = fp2[:n]
    dist = _hamming(fp1, fp2)
    return round(100.0 * (1.0 - dist / float(fp1.shape[0])), 2)