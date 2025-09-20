from fastapi import APIRouter, Depends, HTTPException
from app.api.http.schemas.requests import EvaluateRequest
from app.api.http.schemas.responses import EvaluateResponse
from app.application.services.evaluate_service import EvaluateService
from app.infrastructure.settings import get_settings, Settings

router = APIRouter(tags=["evaluate"])

"""
Router público de evaluación de videos.

Este endpoint orquesta:
  1) Detección de duplicados visuales (ligero y rápido):
     - HASH gate: pHash64 "mayoritario" (64 bits).
     - SEQ gate: huella de secuencia (bool[M,64]) con tolerancia temporal.
  2) Si NO es duplicado:
     - Obtiene/usa keyframes (cacheados en FS o extraídos en caliente) y, opcionalmente,
       una transcripción ASR de audio (si está habilitado).
     - Genera un resumen con el VLM a partir de los frames base64 (+ texto ASR si existe).
     - Juzga el alineamiento con el brief de campaña y devuelve un AlignmentResult.

Notas:
  - Minimiza redescargas: sólo baja el .mp4 si faltan insumos para dedupe o VLM.
  - Persistencia de huellas: depende de la política del servicio (p. ej., escribir sólo cuando no es duplicado
    y/o cuando el resultado fue aprobado). Si usas Redis + Postgres, el servicio puede hacer write-through.
"""

@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluar video: duplicados + alineación con campaña",
    description=(
        "Evalúa un video dentro de una campaña.\n\n"
        "Flujo:\n"
        "1) Dedupe visual rápido contra recientes y `candidates` (HASH/SEQ).\n"
        "2) Si NO es duplicado: se generan keyframes (o se leen del cache FS) y, "
        "   si está habilitado, se transcribe el audio (ASR) para enriquecer el contexto.\n"
        "3) Con esos insumos, el VLM produce un resumen y luego se compara con el brief para "
        "   calcular `match_percent` y `aproved`.\n\n"
        "**Detalles**\n"
        "- Evita re-descargas cuando ya hay huellas o keyframes cacheados.\n"
        "- La política de persistencia (p. ej., no guardar rechazados) vive en `EvaluateService`.\n"
        "- Si activas límites de presupuesto (guardrails), el servicio puede degradar el camino semántico."
    ),
    responses={
        200: {
            "description": "OK – Resultado de deduplicación y/o alineación.",
        },
        500: {
            "description": "Error interno (descarga, VLM, ASR o IO).",
        },
    },
)
def evaluate(req: EvaluateRequest, settings: Settings = Depends(get_settings)) -> EvaluateResponse:
    """
    Ejecuta la evaluación de un video contra una campaña.

    Entradas:
        - `campaign_id` (str): Identificador lógico de la campaña.
        - `video_url` (HttpUrl): URL del video a evaluar.
        - `candidates` (List[HttpUrl]): URLs explícitas a comparar como posibles duplicados.
        - `descripcion` (str): Brief/reglas de la campaña para juzgar alineación.

    Salidas:
        EvaluateResponse:
            - `duplicated` (bool): True si se detectó duplicado (HASH/SEQ) y se corta el flujo.
            - `duplicate_reason` (str|None): "HASH", "SEQ" (u otra razón si el servicio la define).
            - `duplicate_candidate_url` (str|None): URL contra la que se detectó el duplicado.
            - `alignment` (AlignmentResult|None): Solo cuando NO hubo duplicado.
            - `cost` (dict): Métricas de uso (llm_calls, embedding_calls, etc.).

    Errores:
        - 500: Se propaga cualquier excepción no controlada (descarga, VLM, ASR, persistencia).
    """
    service = EvaluateService(settings=settings)
    try:
        return service.evaluate(req)
    except Exception as e:
        # Propaga como 500 para que el cliente distinga error de una respuesta 200 válida
        raise HTTPException(status_code=500, detail=str(e))
