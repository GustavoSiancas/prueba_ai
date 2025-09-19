from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional
from app.infrastructure.settings import get_settings, Settings
from app.infrastructure.redisdb.indices import (
    get_features_by_url, recent_candidates, save_video_features as redis_save_features
)
from app.infrastructure.pg.dao import pg_get_by_url

router = APIRouter(tags=["_dev"])

"""
Endpoints de soporte (DEV/QA):
- Exploran features guardadas en Redis (binarios empaquetados) y PG (write-through).
- Útiles para validar persistencia, tamaños y formas sin traer arrays gigantes por defecto.
"""

@router.get(
    "/_dev/features",
    summary="Inspección de features (Redis/PG)",
    description=(
        "Lee features por `url` o lista `k` recientes de una campaña.\n\n"
        "**Cuando pasas `url`:**\n"
        "- Busca en Redis; si MISS y `PG_ENABLED=true`, trae de PG y repuebla Redis.\n"
        "- Devuelve tamaños/shape y, en `raw`, los arrays ya decodificados (¡sólo para QA!).\n\n"
        "**Cuando NO pasas `url`:**\n"
        "- Lista hasta `limit` recientes (sin arrays) para payloads livianos."
    ),
    responses={
        200: {"description": "OK – Resultado de inspección."},
        404: {"description": "No existe entrada para la URL indicada en Redis/PG."},
    },
)
def dev_features(
    campaign_id: str = Query(..., description="ID de campaña"),
    url: Optional[str] = Query(None, description="URL exacta del video"),
    limit: int = Query(10, ge=1, le=100, description="Cuántos recientes listar si no pasas URL"),
    settings: Settings = Depends(get_settings),
):
    """
    Devuelve features guardadas.
    - Si `url` existe: intenta Redis; si MISS y hay PG, intenta PG y repuebla Redis.
    - Si NO hay `url`: lista recientes (sin arrays grandes).
    """
    if url:
        res = get_features_by_url(campaign_id, url)

        # Fallback a PG + repoblado de Redis
        if not res and getattr(settings, "PG_ENABLED", False):
            rec = pg_get_by_url(url)
            if rec:
                try:
                    redis_save_features(
                        campaign_id=campaign_id,
                        video_url=rec["url"],
                        phash64_bits=rec["phash64"],
                        seq_bits=rec["seq_sig"],
                        duration_s=rec["duration_s"],
                    )
                except Exception as e:
                    print(f"[warn] repoblar Redis desde PG: {e}")
                res = rec

        if not res:
            raise HTTPException(status_code=404, detail="No hay features en Redis/PG para esa URL.")

        video_id, feat = res["video_id"], res

        # Formas
        phash_len = len(feat.get("phash64", []))
        seq = feat.get("seq_sig", [])
        seq_shape = [len(seq), len(seq[0]) if len(seq) > 0 else 0]

        # JSON-friendly (np.ndarray -> list)
        phash_json = feat.get("phash64")
        if hasattr(phash_json, "tolist"):
            phash_json = phash_json.tolist()
        seq_json = feat.get("seq_sig")
        if hasattr(seq_json, "tolist"):
            seq_json = seq_json.tolist()

        return {
            "video_id": video_id,
            "url": feat.get("url"),
            "duration_s": feat.get("duration_s"),
            "phash_bits": phash_len,
            "seq_bits_shape": seq_shape,
            "raw": {"phash64": phash_json, "seq_sig": seq_json},
        }

    # Recientes (sin arrays para payload ligero)
    items = recent_candidates(campaign_id, k=limit)
    for it in items:
        ph = it.get("phash64", [])
        it["phash_bits"] = len(ph)
        seq = it.get("seq_sig", [])
        it["seq_bits_shape"] = [len(seq), len(seq[0]) if len(seq) > 0 else 0]
        it.pop("phash64", None)
        it.pop("seq_sig", None)

    return {"count": len(items), "items": items}
