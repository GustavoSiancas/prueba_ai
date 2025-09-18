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
        - duplicated: True si se detecta copia (HASH/SEQ/...).
        - duplicate_reason: cuál gate disparó (HASH, SEQ, AUDIO, SEMANTIC).
        - duplicate_candidate_url: URL contra la que se detectó el duplicado.
        - alignment: AlignmentResult si no hubo duplicado.
        - cost: métricas de uso (llm_calls, embedding_calls, etc.)
        """

    duplicated: bool
    duplicate_reason: Optional[Literal["HASH", "SEQ", "AUDIO", "SEMANTIC"]] = None
    duplicate_candidate_url: Optional[str] = None
    alignment: Optional[AlignmentResult] = None
    cost: dict
