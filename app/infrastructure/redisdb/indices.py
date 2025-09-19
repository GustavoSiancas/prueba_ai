import json, time, hashlib
import numpy as np
from .client import redis
from app.infrastructure.bitpack import (
    pack_phash64, unpack_phash64, pack_bool_bits, unpack_bool_bits
)

"""
Índices y almacenamiento en Redis para huellas de video.

Estructura:
- campaign:{campaign_id}:url2vid (HASH) -> sha1(url) -> video_id
- video:{video_id}:features (HASH) -> {
    url, phash64_b (8 bytes), seq_b (N*8 bytes), seq_rows, seq_cols=64,
    duration_s, ts (ms)
  }
- campaign:{campaign_id}:vids (ZSET) -> video_id scoreado por timestamp

Nota: guardamos **binarios empaquetados**; al leer, convertimos a np.ndarray.
"""


def _b(s): return s if isinstance(s, bytes) else s.encode()
def _d(b): return b.decode() if isinstance(b, bytes) else b
def _sha1(text: str) -> str: return hashlib.sha1(_b(text)).hexdigest()

def _vid_key(video_id: str) -> str:         return f"video:{video_id}:features"
def _camp_index(campaign_id: str) -> str:   return f"campaign:{campaign_id}:vids"
def _url_index(campaign_id: str) -> str:    return f"campaign:{campaign_id}:url2vid"

def save_video_features(
    campaign_id: str,
    video_url: str,
    phash64_bits: np.ndarray,
    seq_bits: np.ndarray,
    duration_s: float
) -> str:
    video_id = _sha1(video_url)

    phash_b = pack_phash64(phash64_bits)
    rows, cols = seq_bits.shape
    seq_b = pack_bool_bits(seq_bits)

    pipe = redis.pipeline()
    pipe.hset(_vid_key(video_id), mapping={
        b"url":         _b(video_url),
        b"phash64_b":   phash_b,
        b"seq_b":       seq_b,
        b"seq_rows":    _b(str(rows)),
        b"seq_cols":    _b(str(cols)),
        b"duration_s":  _b(str(float(duration_s))),
        b"ts":          _b(str(int(time.time() * 1000))),
    })
    pipe.zadd(_camp_index(campaign_id), {video_id: time.time()})
    pipe.hset(_url_index(campaign_id), _b(_sha1(video_url)), _b(video_id))
    pipe.execute()
    return video_id

def get_features_by_url(campaign_id: str, video_url: str):
    vid = redis.hget(_url_index(campaign_id), _b(_sha1(video_url)))
    if not vid:
        return None
    h = redis.hgetall(_vid_key(_d(vid)))
    if not h:
        return None

    # Campos en bytes
    phash_b = h.get(b"phash64_b")
    seq_b   = h.get(b"seq_b")
    rows    = int(_d(h.get(b"seq_rows", b"0")))
    cols    = int(_d(h.get(b"seq_cols", b"0")))

    if not phash_b or not seq_b or rows <= 0 or cols <= 0:
        return None

    return {
        "video_id":   _d(vid),
        "url":        _d(h.get(b"url", b"")),
        "phash64":    unpack_phash64(phash_b),
        "seq_sig":    unpack_bool_bits(seq_b, rows, cols),
        "duration_s": float(_d(h.get(b"duration_s", b"0"))),
    }

def recent_candidates(campaign_id: str, k: int = 50) -> list[dict]:
    ids = [ _d(i) for i in redis.zrevrange(_camp_index(campaign_id), 0, k-1) ]
    out = []
    for vid in ids:
        h = redis.hgetall(_vid_key(vid))
        if not h: continue
        phash_b = h.get(b"phash64_b")
        seq_b   = h.get(b"seq_b")
        rows    = int(_d(h.get(b"seq_rows", b"0")))
        cols    = int(_d(h.get(b"seq_cols", b"0")))
        if not phash_b or not seq_b or rows <= 0 or cols <= 0:
            continue
        out.append({
            "video_id":   vid,
            "url":        _d(h.get(b"url", b"")),
            "phash64":    unpack_phash64(phash_b),
            "seq_sig":    unpack_bool_bits(seq_b, rows, cols),
            "duration_s": float(_d(h.get(b"duration_s", b"0"))),
        })
    return out