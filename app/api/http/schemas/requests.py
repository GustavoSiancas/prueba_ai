from pydantic import BaseModel, HttpUrl, Field, AliasChoices
from pydantic import ConfigDict
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
    video_url: HttpUrl = Field(..., description="URL del video base", validation_alias=AliasChoices("video_url", "url"))
    candidates: List[HttpUrl] = Field(
        default_factory=list,
        description="Posibles duplicados explícitos (opcional)",
        validation_alias=AliasChoices("candidates", "urls", "List<urls>")
    )
    descripcion: str = Field(
        ...,
        description="Brief de campaña con requisitos a validar",
        validation_alias=AliasChoices("descripcion", "DescripcionCampaña", "DescripcionCampana", "descripcionCampaña"),
    )

    # Ejemplos para OpenAPI
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "campaign_id": "cmp-1",
                "video_url": "https://www.tiktok.com/@user/video/1234567890",
                "candidates": ["https://www.tiktok.com/@user/video/987654321"],
                "descripcion": "Buscamos clips con facecam arriba, gameplay abajo y subtítulos ES/EN."
            }
        ]
    })
