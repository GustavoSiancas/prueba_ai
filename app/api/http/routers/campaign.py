from fastapi import APIRouter, HTTPException

from app.api.http.schemas.requests import GenerateScriptRequest
from app.api.http.schemas.responses import ScriptResponse
from app.application.services.script_service import ScriptGeneratorService

router = APIRouter(tags=["campaign"])

@router.post(
    "/campaign/script",
    response_model=ScriptResponse,
    summary="Generar guion de campaña",
    description=(
        "Genera un guion ideal de video en base a descripción, categoría y tipo de creador, "
        "más los requisitos de la campaña. Devuelve Markdown crudo"
    ),
)
def generate_script_stateless(payload: GenerateScriptRequest):
    """
    Handler stateless para generación de guiones.

    Flujo:
      1) Valida/normaliza el payload (Pydantic ya hace parte).
      2) Construye prompt y llama a ScriptGeneratorService.
      3) Devuelve el guion en Markdown.

    Respuestas:
      200: { has_script: true, script: "..." }
      500: error en llamada al modelo o validación adicional.
    """
    try:
        svc = ScriptGeneratorService()
        script_md = svc.generate_script(
            description=payload.description,
            category=payload.category,
            creator_type=payload.creator_type,
            requirements=payload.requirements,
        )
        return ScriptResponse(format="markdown", has_script=True, script=script_md)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo generar el guion: {e}")