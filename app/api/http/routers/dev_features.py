from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from pydantic import HttpUrl

from app.infrastructure.pg.dao import pg_get_by_url, pg_recent_candidates

router = APIRouter(tags=["_dev"])

"""
Endpoints de soporte (DEV/QA) – PG-only:
- Inspecciona lo guardado en Postgres (source of truth).
- Útil para validar persistencia, tamaños y formas sin devolver arrays gigantes por defecto.
"""

@router.get(
    "/_dev/features",
    summary="Inspección de features en Postgres",
    description=(
        "Lee features por `url` o lista `limit` recientes de una campaña, **desde Postgres**.\n\n"
        "**Cuando pasas `url`:**\n"
        "- Devuelve tamaños/shape y, en `raw`, los arrays ya decodificados (¡sólo para QA!).\n\n"
        "**Cuando NO pasas `url`:**\n"
        "- Lista hasta `limit` recientes (sin arrays) para payloads livianos."
    ),
    responses={
        200: {"description": "OK – Resultado de inspección."},
        404: {"description": "No existe entrada para la URL indicada en Postgres."},
    },
)
def dev_features(
    campaign_id: str = Query(..., description="ID de campaña"),
    url: Optional[HttpUrl] = Query(None, description="URL exacta del video"),
    limit: int = Query(10, ge=1, le=100, description="Cuántos recientes listar si no pasas URL")
):
    """
    Devuelve features guardadas en Postgres.
    - Si `url` existe: retorna huella completa (tamaños + `raw` con arrays decodificados).
    - Si NO hay `url`: lista recientes (sin arrays grandes).
    """
    if url:
        rec = pg_get_by_url(str(url))
        if not rec:
            raise HTTPException(status_code=404, detail="No hay features en Postgres para esa URL.")

        ph = rec.get("phash64", [])
        seq = rec.get("seq_sig", [])
        phash_len = len(ph)
        seq_shape = [len(seq), len(seq[0]) if len(seq) > 0 else 0]

        ph_json = ph.tolist() if hasattr(ph, "tolist") else ph
        seq_json = seq.tolist() if hasattr(seq, "tolist") else seq

        return {
            "video_id": rec.get("video_id"),
            "url": rec.get("url"),
            "duration_s": rec.get("duration_s"),
            "phash_bits": phash_len,
            "seq_bits_shape": seq_shape,
            "raw": {"phash64": ph_json, "seq_sig": seq_json},
        }

        # Recientes (sin blobs)
    items = pg_recent_candidates(campaign_id, k=limit)
    out = []
    for it in items:
        out.append({
            "video_id": it["video_id"],
            "url": it["url"],
            "duration_s": it["duration_s"],
            "phash_bits": it.get("phash_bits", 64),
            "seq_bits_shape": [it.get("seq_rows", 0), it.get("seq_cols", 0)],
            "created_at": it.get("created_at"),
        })
    return {"count": len(out), "items": out}