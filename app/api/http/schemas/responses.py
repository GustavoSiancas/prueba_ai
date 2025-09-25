from pydantic import BaseModel
from typing import Optional, Literal

class AlignmentResult(BaseModel):
    """
    Resultado del juicio de alineación con el brief de la campaña.
    - aproved: bool de aprobación.
    - match_percent: 0..100 (% de cumplimiento).
    - reasons: explicación breve (2–3 oraciones).
    """
    aproved: bool
    match_percent: float
    reasons: str

class EvaluateResponse(BaseModel):
    """
    Respuesta de /api/evaluate:
    - duplicated: True si se detecta copia.
    - duplicate_reason: "URL" (misma URL), "HASH" (pHash), "SEQ" (secuencia),
                        "AUDIO" (audio), "SEMANTIC" (resumen/semántica).
    - duplicate_candidate_url: URL contra la que se detectó el duplicado.
    - alignment: AlignmentResult si no hubo duplicado.
    - cost: métricas de uso (llm_calls, embedding_calls, etc.)
    """
    duplicated: bool
    duplicate_reason: Optional[Literal["HASH", "SEQ", "AUDIO", "SEMANTIC", "URL"]] = None
    duplicate_candidate_url: Optional[str] = None
    alignment: Optional[AlignmentResult] = None
    cost: dict

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "duplicated": False,
                "duplicate_reason": None,
                "duplicate_candidate_url": None,
                "alignment": {
                    "aproved": True,
                    "match_percent": 94.5,
                    "reasons": "Se observa facecam superior, gameplay inferior y subtítulos bilingües claramente visibles."
                },
                "cost": {"llm_calls": 1, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
            }]
        }
    }

class ScriptResponse(BaseModel):
    """
    Respuesta de generación de guion (stateless).

    Campos:
      - format: formato del campo `script` (por ahora "markdown").
      - has_script: bandera de éxito.
      - script: contenido en el formato declarado.
    """
    format: str
    has_script: bool
    script: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "format": "markdown",
                "has_script": True,
                "script": "### Hook\n...\n### Desarrollo\n...\n### CTA\n..."
            }]
        }
    }