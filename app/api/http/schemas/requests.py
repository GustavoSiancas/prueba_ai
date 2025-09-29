from pydantic import BaseModel, HttpUrl, Field, AliasChoices, ConfigDict
from typing import List, Optional
from datetime import date

class EvaluateRequest(BaseModel):
    """
    Payload de evaluación de un video dentro de una campaña.

    Campos:
      - campaign_id: Identificador lógico de la campaña.
      - video_url: URL del video a evaluar.
      - candidates: URLs explícitas a comparar como posibles duplicados.
      - descripcion: Reglas/brief de la campaña para el juez de alineación.
      - end_date: Fecha fin de la campaña (YYYY-MM-DD).
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "campaign_id": "cmp-1",
                "end_date": "2025-12-31",
                "video_url": "https://www.tiktok.com/@user/video/1234567890",
                "candidates": ["https://www.tiktok.com/@user/video/987654321"],
                "descripcion": "Buscamos clips con facecam arriba, gameplay abajo y subtítulos ES/EN."
            }
        }
    )

    campaign_id: str = Field(..., description="ID de campaña")

    end_date: date = Field(
        ...,
        alias="endDate",
        description="Fecha fin de la campaña (YYYY-MM-DD).",
        validation_alias=AliasChoices("end_date", "endDate"),
    )

    video_url: HttpUrl = Field(
        ...,
        description="URL del video base",
        validation_alias=AliasChoices("video_url", "url"),
    )

    candidates: List[HttpUrl] = Field(
        default_factory=list,
        description="Posibles duplicados explícitos (opcional)",
        validation_alias=AliasChoices("candidates", "urls", "List<urls>"),
    )

    descripcion: str = Field(
        ...,
        description="Brief de campaña con requisitos a validar",
        validation_alias=AliasChoices("descripcion", "DescripcionCampaña", "DescripcionCampana", "descripcionCampaña"),
    )

class GenerateScriptRequest(BaseModel):
    """
    Entrada directa para generar guion.
    """
    description: str = Field(..., description="Descripción/brief de la campaña")
    category: str = Field(..., description="Categoría de la campaña (ej. GAMING, BEAUTY, FOOD, etc.)")
    creator_type: str = Field(..., description="Tipo de creador (UGC | CLIPPER)")
    requirements: Optional[str] = Field(
        None,
        description="Requisitos de la campaña (texto consolidado)",
        validation_alias=AliasChoices("requirements", "prompt", "extra_prompt", "guidelines"),
    )

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "description": "Bebida energética enfocada a gamers; destacar sabor y energía sin crash.",
                "category": "GAMING",
                "creator_type": "UGC",
                "requirements": "Evitar claims médicos; mencionar envío gratis; CTA a link en bio."
            }
        ]
    })