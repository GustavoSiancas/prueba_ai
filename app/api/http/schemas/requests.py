from pydantic import BaseModel, HttpUrl, Field, AliasChoices
from typing import List

class EvaluateRequest(BaseModel):
    """
        Payload de evaluación de un video dentro de una campaña.

        Campos:
          - campaign_id: Identificador lógico de la campaña.
          - video_url: URL del video a evaluar.
          - candidates: URLs explícitas a comparar como posibles duplicados.
          - descripcion: Reglas/brief de la campaña para el juez de alineación.
        """

    campaign_id: str = Field(..., description="ID de campaña")
    video_url: HttpUrl = Field(..., validation_alias=AliasChoices("video_url", "url"))
    candidates: List[HttpUrl] = Field(default_factory=list, validation_alias=AliasChoices("candidates", "urls", "List<urls>"))
    descripcion: str = Field(..., validation_alias=AliasChoices("descripcion", "DescripcionCampaña", "DescripcionCampana", "descripcionCampaña"))
