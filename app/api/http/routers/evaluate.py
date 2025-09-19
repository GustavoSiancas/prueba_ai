from fastapi import APIRouter, Depends, HTTPException
from app.api.http.schemas.requests import EvaluateRequest
from app.api.http.schemas.responses import EvaluateResponse
from app.application.services.evaluate_service import EvaluateService
from app.infrastructure.settings import get_settings, Settings

router = APIRouter(tags=["evaluate"])

"""
Router público de evaluación.
- POST /api/evaluate: Detecta duplicados (HASH/SEQ) y evalúa alineación con el brief.
"""

@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluar video (duplicados + alineación con campaña)",
    description=(
        "Orquesta el flujo completo:\n"
        "1) Dedupe visual ligero (pHash64 + huella de secuencia) contra recientes y `candidates`.\n"
        "2) Si NO es duplicado: genera resumen con VLM usando keyframes cacheados (o extraídos) "
        "y (opcional) transcripción de audio para validar requisitos del brief.\n\n"
        "**Notas**\n"
        "- No re-descarga el video si ya se tienen huellas; usa cache, y sólo cae a descarga si faltan insumos.\n"
        "- Persiste huellas en Redis (bits) y, si `PG_ENABLED=true`, también en Postgres (write-through)."
    ),
    responses={
        200: {"description": "OK – Resultado de deduplicación y alineación."},
        500: {"description": "Error interno en la orquestación (descarga/VLM/IO)."},
    },
)
def evaluate(req: EvaluateRequest, settings: Settings = Depends(get_settings)):
    """
    Ejecuta EvaluateService con la configuración inyectada.

    Retorna:
        EvaluateResponse:
            - duplicated=True si HASH/SEQ fue positivo (corta el flujo).
            - alignment=AlignmentResult si NO es duplicado.

    Errores:
        500: cualquier excepción no controlada se propaga como HTTP 500.
    """
    service = EvaluateService(settings=settings)
    try:
        return service.evaluate(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))