import cv2
import numpy as np

"""
Huella de secuencia: dHash por frame muestreado uniformemente.
Sirve para near-duplicates con trims/speedups usando ventana de alineación.
"""

def get_duration_s(path: str) -> float:
    """Duración del video en segundos (0 si FPS inválido)."""

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return (frames / fps) if fps > 0 else 0.0

def frame_hash_sequence(path: str, seconds_interval: float = 2.0, max_frames: int = 60, hash_size=8):
    """
    Secuencia de hashes (bool[frames, 64]) muestreando cada `seconds_interval`.

    Returns:
        np.ndarray dtype=bool con forma (M, 64). Vacío si falla.
    """

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return np.array([], dtype=np.bool_)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total / fps) if fps > 0 else 0

    seq = []
    t = 0.0
    while len(seq) < max_frames and (duration == 0 or t <= duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        seq.append(diff.flatten())
        t += seconds_interval
    cap.release()
    return np.array(seq, dtype=np.bool_)

def sequence_match_percent(seqA: np.ndarray, seqB: np.ndarray, bit_tolerance: int = 5, window: int = 2):
    """
    % de frames de A que encuentran "mejor match" en B dentro de una ventana temporal.

    Args:
        bit_tolerance: #bits distintos máximos para considerar match (0..64).
        window: desfase temporal permitido +/-window (para trims/speed).

    Returns:
        Porcentaje 0..100.
    """

    if seqA.size == 0 or seqB.size == 0:
        return 0.0
    matches = 0
    for i, hA in enumerate(seqA):
        start = max(0, i - window)
        end = min(len(seqB), i + window + 1)
        best = 65

        for j in range(start, end):
            dist = int(np.count_nonzero(hA != seqB[j]))
            if dist < best:
                best = dist

        if best <= bit_tolerance:
            matches += 1

    return round(100.0 * matches / len(seqA), 2)
