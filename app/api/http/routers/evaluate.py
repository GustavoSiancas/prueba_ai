from fastapi import APIRouter, Depends, HTTPException
from app.api.http.schemas.requests import EvaluateRequest
from app.api.http.schemas.responses import EvaluateResponse, AlignmentResult
from app.application.services.evaluate_service import EvaluateService
from app.infrastructure.settings import get_settings, Settings

router = APIRouter(tags=["evaluate"])

"""
Router público de evaluación:
- POST /api/evaluate: inspecciona duplicados y alineación con campaña.

Las reglas de negocio viven en EvaluateService.
"""

@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest, settings: Settings = Depends(get_settings)):
    """
        Evalúa un video contra la campaña:
          1) Detecta duplicados (pHash/seq) contra recientes y candidates[].
          2) Si no hay duplicado, resume el video (VLM) y juzga la alineación
             con la descripción de campaña.

        Retorna:
          EvaluateResponse con campos de dedupe y alignment.

        Errores:
          500 si ocurre un fallo no controlado en el servicio.
        """

    service = EvaluateService(settings=settings)

    try:
        result = service.evaluate(req)
        return result
    except Exception as e:
        # Propaga como 500 para que el cliente pueda distinguir errores vs. 200 válidos
        raise HTTPException(status_code=500, detail=str(e))