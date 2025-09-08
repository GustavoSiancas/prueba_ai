import cv2
import numpy as np

def _dhash(image_bgr, hash_size=8):
    """Devuelve vector booleano de 64 bits (8x8) con diferencias horizontales."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))

def video_fingerprint(path: str, seconds_interval: float = 5.0, max_frames: int = 20):
    """Toma N frames separados 'seconds_interval' y hace un 'voto mayoritario' bit a bit."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
    if fp1 is None or fp2 is None:
        return 0.0
    dist = _hamming(fp1, fp2)
    return round(100.0 * (1.0 - dist / float(fp1.shape[0])), 2)

def frame_hash_sequence(path: str, seconds_interval: float = 2.0, max_frames: int = 60, hash_size=8):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
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
    Compara dos secuencias de hashes (dHash por frame).
    - bit_tolerance: cuántos bits distintos permito para llamar "match" de 2 frames (0..64).
    - window: tolero desalineación temporal +/- window frames (para trims/speed).
    Retorna % de frames de A que encuentran match en B.
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