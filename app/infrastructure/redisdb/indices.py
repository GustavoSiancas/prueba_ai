import json, time, hashlib
import numpy as np
from .client import redis

"""
Índices y helpers de persistencia en Redis:

Claves:
- campaign:{campaign_id}:vids (ZSET) -> members: video_id; score: timestamp
- campaign:{campaign_id}:url2vid (HASH) -> sha1(url) : video_id
- video:{video_id}:features (HASH) -> {url, phash64(JSON), seq_sig(JSON), duration_s, ts}
"""

def _b(s: bytes | str) -> bytes:
    return s if isinstance(s, bytes) else s.encode()

def _d(b: bytes | str) -> str:
    return b.decode() if isinstance(b, bytes) else b

def _sha1(text: str) -> str:
    return hashlib.sha1(_b(text)).hexdigest()

def _vid_key(video_id: str) -> str:
    return f"video:{video_id}:features"

def _camp_index(campaign_id: str) -> str:
    return f"campaign:{campaign_id}:vids"

def _url_index(campaign_id: str) -> str:
    return f"campaign:{campaign_id}:url2vid"

def save_video_features(
    campaign_id: str,
    video_url: str,
    phash64_bits: np.ndarray,
    seq_bits: np.ndarray,
    duration_s: float
) -> str:
    """
    Persiste huellas del video y las asocia a la campaña.

    Efectos:
      - HSET video:{video_id}:features
      - ZADD campaign:{campaign_id}:vids
      - HSET campaign:{campaign_id}:url2vid

    Returns:
      video_id (sha1 de la URL).
    """

    video_id = _sha1(video_url)

    phash_list = phash64_bits.astype(int).tolist()
    seq_list = seq_bits.astype(int).tolist()

    redis.hset(_vid_key(video_id), mapping={
        "campaign_id": campaign_id,
        "url": video_url,
        "phash64": json.dumps(phash_list, ensure_ascii=False),
        "seq_sig": json.dumps(seq_list, ensure_ascii=False),
        "duration_s": str(float(duration_s)),
        "ts": str(int(time.time()*1000)),
    })

    redis.zadd(_camp_index(campaign_id), {video_id: time.time()})
    redis.hset(_url_index(campaign_id), _sha1(video_url), video_id)
    return video_id

def get_features_by_url(campaign_id: str, video_url: str) -> dict | None:
    """
    Obtiene features del video a partir de campaign_id + URL.

    Returns:
      dict con {video_id, url, phash64(list[int]), seq_sig(list[list[int]]), duration_s}
      o None si no existe.
    """

    vid = redis.hget(_url_index(campaign_id), _sha1(video_url))
    if not vid:
        return None
    h = redis.hgetall(_vid_key(_d(vid)))
    if not h:
        return None
    return {
        "video_id": _d(vid),
        "url": _d(h.get(b"url", b"")),
        "phash64": json.loads(_d(h.get(b"phash64", b"[]"))),
        "seq_sig": json.loads(_d(h.get(b"seq_sig", b"[]"))),
        "duration_s": float(_d(h.get(b"duration_s", b"0"))),
    }

def recent_candidates(campaign_id: str, k: int = 50) -> list[dict]:
    """
    Devuelve hasta k últimos videos de la campaña (más recientes primero),
    incluyendo huellas completas para comparación rápida.

    Returns:
      lista de dicts con {video_id, url, phash64, seq_sig, duration_s}
    """

    ids = [ _d(i) for i in redis.zrevrange(_camp_index(campaign_id), 0, k-1) ]
    out = []
    for vid in ids:
        h = redis.hgetall(_vid_key(vid))
        if not h:
            continue
        out.append({
            "video_id": vid,
            "url": _d(h.get(b"url", b"")),
            "phash64": json.loads(_d(h.get(b"phash64", b"[]"))),
            "seq_sig": json.loads(_d(h.get(b"seq_sig", b"[]"))),
            "duration_s": float(_d(h.get(b"duration_s", b"0"))),
        })
    return out
