import numpy as np
from app.infrastructure.pg.client import get_pool
from app.infrastructure.bitpack import pack_phash64, pack_bool_bits, unpack_phash64, unpack_bool_bits

"""
DAO de Postgres para `video_features`.
- Esquema en /app/infrastructure/pg/migrations/001_init.sql
- Se usa como write-through y como fallback para repoblar Redis.
"""

def pg_save_video_features(video_id: str, campaign_id: str, url: str,
                           phash64_bits: np.ndarray, seq_bits: np.ndarray,
                           duration_s: float) -> None:
    """Inserta (idempotente por URL) las huellas del video."""

    rows, cols = seq_bits.shape
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO video_features (video_id, campaign_id, url, phash64, seq_sig, seq_rows, seq_cols, duration_s)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (url) DO NOTHING
            """,
            (
                video_id, campaign_id, url,
                pack_phash64(phash64_bits),
                pack_bool_bits(seq_bits),
                rows, cols, float(duration_s),
            )
        )

def pg_get_by_url(url: str):
    """Recupera un registro por URL y decodifica phash/seq a np.ndarray."""

    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT video_id, campaign_id, url, phash64, seq_sig, seq_rows, seq_cols, duration_s
            FROM video_features WHERE url = %s
            """,
            (url,)
        )
        row = cur.fetchone()
    if not row:
        return None
    video_id, campaign_id, url, phash_b, seq_b, rows, cols, duration_s = row
    return {
        "video_id": video_id,
        "campaign_id": campaign_id,
        "url": url,
        "phash64": unpack_phash64(bytes(phash_b)),
        "seq_sig": unpack_bool_bits(bytes(seq_b), rows, cols),
        "duration_s": float(duration_s),
    }
