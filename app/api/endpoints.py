from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from app.services.video_analyzer import analyze_video_with_chatgpt, comparar_descripcion_con_resumen_ia
from app.services.downloader import descargar_video
import os
import json

router = APIRouter()

@router.post("/analyze/")
async def analyze_video(
    video_url: str = Form(...),
    descripcion: str = Form(...)
):
    video_path = descargar_video(video_url)
    if not video_path or not os.path.exists(video_path):
        return JSONResponse(status_code=500, content={"error": "No se pudo descargar el video."})
    resumen = analyze_video_with_chatgpt(video_path)
    if not resumen:
        return JSONResponse(status_code=500, content={"error": "No se pudo analizar el video."})
    resultado_raw = comparar_descripcion_con_resumen_ia(descripcion, resumen)
    if not resultado_raw:
        return JSONResponse(status_code=500, content={"error": "No se pudo comparar la descripción con el resumen del video."})
    try:
        resultado_json = json.loads(resultado_raw)
        return resultado_json
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=500, content={"error": f"No se pudo interpretar la respuesta JSON de la IA: {e}", "raw": resultado_raw})
