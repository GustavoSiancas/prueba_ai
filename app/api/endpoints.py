def detectar_red_social(url: str) -> str:
    url = url.lower()
    if "instagram.com" in url:
        return "INSTAGRAM"
    elif "tiktok.com" in url:
        return "TIKTOK"
    elif "youtube.com" in url or "youtu.be" in url:
        return "YOUTUBE"
    return "DESCONOCIDO"
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from enum import Enum
from app.services.video_analyzer import analyze_video_with_chatgpt, comparar_descripcion_con_resumen_ia
from app.services.downloader import descargar_video
import os
import json

router = APIRouter()

class SocialMediaEnum(str, Enum):
    INSTAGRAM = "INSTAGRAM"
    TIKTOK = "TIKTOK"
    YOUTUBE = "YOUTUBE"

class ComparacionRequest(BaseModel):
    video_url: str
    descripcion: str
    social_media: SocialMediaEnum

@router.post("/analyze/")
async def analyze_video(data: ComparacionRequest):
    descripcion = data.descripcion
    video_url = data.video_url
    social_media = data.social_media
    red_detectada = detectar_red_social(video_url)
    if red_detectada != social_media:
        return JSONResponse(status_code=400, content={"error": f"El link proporcionado pertenece a {red_detectada}, pero seleccionaste {social_media}."})
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
