from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from app.infrastructure.redisdb.indices import get_features_by_url, recent_candidates

router = APIRouter(tags=["_dev"])

"""
Endpoints de soporte para desarrollo/QA.
Permiten inspeccionar rápidamente lo que está almacenado en Redis
para una campaña y/o una URL específica.

Claves de Redis relevantes:
- campaign:{campaign_id}:url2vid (HASH): sha1(url) -> video_id
- video:{video_id}:features (HASH): {url, phash64, seq_sig, duration_s, ts}
- campaign:{campaign_id}:vids (ZSET): video_id scoreado por ts
"""

@router.get("/_dev/features")
def dev_features(
    campaign_id: str = Query(..., description="ID de campaña"),
    url: Optional[str] = Query(None, description="URL exacta del video"),
    limit: int = Query(10, ge=1, le=100, description="Cuántos recientes listar si no pasas URL")
):
    """
        Devuelve features guardadas en Redis.

        - Si se pasa `url`: retorna la huella completa (pHash64 y seq_sig) de esa URL en la campaña.
        - Si NO se pasa `url`: lista hasta `limit` videos recientes (sin los arrays grandes).

        Ejemplos:
            GET /api/_dev/features?campaign_id=cmp-1&url=https://tiktok...
            GET /api/_dev/features?campaign_id=cmp-1&limit=5
        """

    if url:
        res = get_features_by_url(campaign_id, url)

        if not res:
            raise HTTPException(status_code=404, detail="No hay mapping URL→video_id para esa campaña/URL.")

        video_id, feat = res["video_id"], res

        if not feat:
            raise HTTPException(status_code=404, detail="No se encontraron features para esa URL.")

        phash_len = len(feat.get("phash64", []))
        seq_shape = [len(feat.get("seq_sig", [])), len(feat.get("seq_sig", [ [] ])[0]) if feat.get("seq_sig") else 0]

        return {
            "video_id": video_id,
            "url": feat.get("url"),
            "duration_s": feat.get("duration_s"),
            "phash_bits": phash_len,
            "seq_bits_shape": seq_shape,
            "raw": {
                "phash64": feat.get("phash64"),
                "seq_sig": feat.get("seq_sig"),
            }
        }

    # Listing de recientes (sin arrays para respuesta ligera)
    items = recent_candidates(campaign_id, k=limit)
    for it in items:
        it["phash_bits"] = len(it.get("phash64", []))
        seq = it.get("seq_sig", [])
        it["seq_bits_shape"] = [len(seq), len(seq[0]) if seq else 0]

        it.pop("phash64", None)
        it.pop("seq_sig", None)

    return {"count": len(items), "items": items}
