def detectar_red_social(url: str) -> str:
    url = url.lower()
    if "instagram.com" in url:
        return "INSTAGRAM"
    elif "tiktok.com" in url:
        return "TIKTOK"
    elif "youtube.com" in url or "youtu.be" in url:
        return "YOUTUBE"
    return "DESCONOCIDO"

import cv2
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from starlette.concurrency import run_in_threadpool
from typing import List
from enum import Enum

from app.services.video_analyzer import analyze_video_with_chatgpt, comparar_descripcion_con_resumen_ia, extract_audio, transcribe_audio
from app.services.downloader import descargar_video
from app.services.similarity import (
    video_fingerprint, similarity_percent,
    frame_hash_sequence, sequence_match_percent
)
from app.services.semantic import embed_text, summarize_video_textual, cosine
import os
import json

router = APIRouter()

class SocialMediaEnum(str, Enum):
    INSTAGRAM = "INSTAGRAM"
    TIKTOK = "TIKTOK"
    YOUTUBE = "YOUTUBE"

class ComparacionRequest(BaseModel):
    video_url: HttpUrl
    descripcion: str
    social_media: SocialMediaEnum

@router.post("/analyze/")
async def analyze_video(data: ComparacionRequest):
    descripcion = data.descripcion
    video_url = data.video_url
    social_media = data.social_media

    red_detectada = detectar_red_social(video_url)
    if red_detectada == "DESCONOCIDO":
        return JSONResponse(status_code=400, content={"error":"URL de red social no reconocida."})
    if red_detectada != social_media:
        return JSONResponse(status_code=400, content={"error": f"El link proporcionado pertenece a {red_detectada}, pero seleccionaste {social_media}."})
    
    video_path = await run_in_threadpool(descargar_video, str(video_url))
    if not video_path or not os.path.exists(video_path):
        return JSONResponse(status_code=500, content={"error": "No se pudo descargar el video."})
    
    resumen = await run_in_threadpool(analyze_video_with_chatgpt, video_path)
    if not resumen:
        return JSONResponse(status_code=500, content={"error": "No se pudo analizar el video."})
    
    resultado_raw = await run_in_threadpool(comparar_descripcion_con_resumen_ia, descripcion, resumen)
    if not resultado_raw:
        return JSONResponse(status_code=500, content={"error": "No se pudo comparar la descripción con el resumen del video."})
    
    try:
        resultado_json = json.loads(resultado_raw)
        return resultado_json
    
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=500, content={"error": f"No se pudo interpretar la respuesta JSON de la IA: {e}", "raw": resultado_raw})


class DupCheckRequest(BaseModel):
    video_url: HttpUrl
    candidates: List[HttpUrl]

HASH_DUP_THRESHOLD = 95.0    # %
SEQ_DUP_THRESHOLD  = 85.0    # %
SEM_STRICT_THRESHOLD = 92.0  # %
MIN_SEQ_FOR_SEM     = 40.0   # %
AUDIO_DUP_THRESHOLD  = 88.0  # %
MAX_DURATION_DRIFT   = 0.15  # 15%

def get_duration_s(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return (frames / fps) if fps > 0 else 0.0

@router.post("/dup-check/")
async def dup_check(req: DupCheckRequest):
    """
    Compara un video (video_url) contra una lista (candidates).
    Declara duplicado apenas uno cumpla el umbral (hash -> secuencia -> semántico).
    Retorna y corta en el primer match.
    """

    # Descargar el video base a comparar con los demás candidatos
    base_path = await run_in_threadpool(descargar_video, str(req.video_url))
    if not base_path:
        return JSONResponse(status_code=500, content={"error": "No se pudo descargar el video base."})

    # Precalculados del base (se hacen bajo demanda para ahorrar costo)
    base_fp = None         # fingerprint global (dHash promedio)
    base_seq = None        # secuencia de dHash por frame
    base_summary = None    # resumen visual (gpt-4o)
    base_embed = None      # embedding del texto (resumen [+ transcripción opcional])

    # Utilidades internas
    async def ensure_base_fp():
        nonlocal base_fp
        if base_fp is None:
            base_fp = await run_in_threadpool(video_fingerprint, base_path)

    async def ensure_base_seq():
        nonlocal base_seq
        if base_seq is None:
            base_seq = await run_in_threadpool(frame_hash_sequence, base_path, 2.0, 60)

    async def ensure_base_embed():
        nonlocal base_summary, base_embed
        if base_embed is None:
            if base_summary is None:
                base_summary = await run_in_threadpool(analyze_video_with_chatgpt, base_path)
                if not base_summary:
                    raise RuntimeError("No se pudo generar el resumen del video base.")
            base_text = summarize_video_textual(base_summary, None)
            base_embed = await run_in_threadpool(embed_text, base_text)

    # Iteraramos candidatos
    for idx, url in enumerate(req.candidates, start=1):
        cand_url = str(url)

        # Se descarga
        cand_path = await run_in_threadpool(descargar_video, cand_url)
        if not cand_path:
            
            continue

        # Fase 1: hash perceptual global
        await ensure_base_fp()
        cand_fp = await run_in_threadpool(video_fingerprint, cand_path)
        hash_percent = similarity_percent(base_fp, cand_fp)

        if hash_percent >= HASH_DUP_THRESHOLD:
            return {
                "duplicate": True,
                "matched_by": "hash",
                "video_url": str(req.video_url),
                "match_url": cand_url,
                "scores": {"hash_percent": hash_percent}
            }

        # Fase 2: secuencia de frames (near-dup)
        await ensure_base_seq()
        cand_seq = await run_in_threadpool(frame_hash_sequence, cand_path, 2.0, 60)
        seq_percent = sequence_match_percent(base_seq, cand_seq, bit_tolerance=5, window=2)

        if seq_percent >= SEQ_DUP_THRESHOLD:
            return {
                "duplicate": True,
                "matched_by": "sequence",
                "video_url": str(req.video_url),
                "match_url": cand_url,
                "scores": {"hash_percent": hash_percent, "seq_percent": seq_percent}
            }

        # Fase 3: semántico (resumen -> embedding)
        try:
            await ensure_base_embed()
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Falló resumen/embedding del video base: {e}"})

        cand_summary = await run_in_threadpool(analyze_video_with_chatgpt, cand_path)
        if not cand_summary:
            # no pudo resumir -> seguir con el siguiente
            continue

        cand_text = summarize_video_textual(cand_summary, None)
        cand_embed = await run_in_threadpool(embed_text, cand_text)
        sem_sim = round(100.0 * cosine(base_embed, cand_embed), 2)

        # Audio (solo si lo necesitamos para decidir)
        audio_percent = 0.0
        if sem_sim >= SEM_STRICT_THRESHOLD:
            a1 = await run_in_threadpool(extract_audio, base_path, "dup_base.mp3")
            a2 = await run_in_threadpool(extract_audio, cand_path, "dup_cand.mp3")
            t1 = await run_in_threadpool(transcribe_audio, a1) if a1 else None
            t2 = await run_in_threadpool(transcribe_audio, a2) if a2 else None
            if t1 and t2:
                e_a1 = await run_in_threadpool(embed_text, t1)
                e_a2 = await run_in_threadpool(embed_text, t2)
                audio_percent = round(100.0 * cosine(e_a1, e_a2), 2)
        # Duración (como pista adicional)
        d_base = await run_in_threadpool(get_duration_s, base_path)
        d_cand = await run_in_threadpool(get_duration_s, cand_path)
        duration_ratio = round(abs(d_base - d_cand) / max(d_base, d_cand or 1.0), 4)

        # Regla final: semántica y señales extras
        if sem_sim >= SEM_STRICT_THRESHOLD and (
            seq_percent >= MIN_SEQ_FOR_SEM or
            audio_percent >= AUDIO_DUP_THRESHOLD or
            duration_ratio <= MAX_DURATION_DRIFT
        ):
            return {
                "duplicate": True,
                "matched_by": "semantic+gate",
                "video_url": str(req.video_url),
                "match_url": cand_url,
                "scores": {
                    "hash_percent": hash_percent,
                    "seq_percent": seq_percent,
                    "semantic_percent": sem_sim,
                    "audio_percent": audio_percent,
                    "duration_ratio": duration_ratio
                }
            }

    # Si llegamos aquí significa que ninguno cumplió los umbrales, por lo tanto no hubo ningún duplicado
    return {
        "duplicate": False,
        "matched_by": "none",
        "video_url": str(req.video_url),
        "checked": len(req.candidates)
    }