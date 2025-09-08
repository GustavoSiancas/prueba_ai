import cv2
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field, AliasChoices
from starlette.concurrency import run_in_threadpool
from typing import List
from enum import Enum

from app.services.video_analyzer import (
    analyze_video_with_chatgpt, comparar_descripcion_con_resumen_ia,
    extract_audio, transcribe_audio
)
from app.services.downloader import descargar_video
from app.services.similarity import (
    video_fingerprint, similarity_percent,
    frame_hash_sequence, sequence_match_percent
)
from app.services.semantic import embed_text, summarize_video_textual, cosine
import os
import json

router = APIRouter()

# --- UMBRALES ---
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

class EvaluateRequest(BaseModel):
    video_url: HttpUrl = Field(validation_alias=AliasChoices("video_url", "url"))
    candidates: List[HttpUrl] = Field(validation_alias=AliasChoices("candidates", "urls", "List<urls>"))
    descripcion: str = Field(validation_alias=AliasChoices("descripcion", "DescripcionCampaña", "DescripcionCampana", "descripcionCampaña"))

@router.post("/evaluate/")
async def evaluate(req: EvaluateRequest):
    """
    1) Verifica duplicado: base vs lista (hash→secuencia→semántico con compuerta).
       Si hay duplicado => respuesta de copia + porcentajeAprobacion=0.
    2) Si no hay duplicado => analiza video y compara con la descripción.
       Devuelve razones de aceptación/rechazo y porcentajeAprobacion (1..100).
    """
    # --------- DESCARGA BASE ---------
    base_path = await run_in_threadpool(descargar_video, str(req.video_url))
    if not base_path:
        return JSONResponse(status_code=500, content={"error": "No se pudo descargar el video base."})

    base_fp = None
    base_seq = None
    base_summary = None
    base_embed = None

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
                    return JSONResponse(status_code=500, content={"error": "No se pudo generar resumen del video base."})
            base_text = summarize_video_textual(base_summary, None)
            base_embed = await run_in_threadpool(embed_text, base_text)

    # --------- ETAPA: DUPLICADOS (corta en el 1º match) ---------
    for url in req.candidates:
        cand_url = str(url)
        cand_path = await run_in_threadpool(descargar_video, cand_url)
        if not cand_path:
            continue

        # Hash global
        await ensure_base_fp()
        cand_fp = await run_in_threadpool(video_fingerprint, cand_path)
        hash_percent = similarity_percent(base_fp, cand_fp)
        if hash_percent >= HASH_DUP_THRESHOLD:
            return {
                "respuesta": f"Se detectó copia del video (hash perceptual alto) contra: {cand_url}",
                "porcentajeAprobacion": 0
            }

        # Secuencia de frames
        await ensure_base_seq()
        cand_seq = await run_in_threadpool(frame_hash_sequence, cand_path, 2.0, 60)
        seq_percent = sequence_match_percent(base_seq, cand_seq, bit_tolerance=5, window=2)
        if seq_percent >= SEQ_DUP_THRESHOLD:
            return {
                "respuesta": f"Se detectó copia (near-duplicate por secuencia de frames) contra: {cand_url}",
                "porcentajeAprobacion": 0
            }

        # Semántico con compuerta
        resp = await ensure_base_embed()
        if isinstance(resp, JSONResponse):  # error al generar resumen base
            return resp

        cand_summary = await run_in_threadpool(analyze_video_with_chatgpt, cand_path)
        if not cand_summary:
            continue
        cand_text = summarize_video_textual(cand_summary, None)
        cand_embed = await run_in_threadpool(embed_text, cand_text)
        sem_sim = round(100.0 * cosine(base_embed, cand_embed), 2)

        audio_percent = 0.0
        if sem_sim >= SEM_STRICT_THRESHOLD:
            a1 = await run_in_threadpool(extract_audio, base_path, "dup_base.mp3")
            a2 = await run_in_threadpool(extract_audio, cand_path, "dup_cand.mp3")
            t1 = await run_in_threadpool(transcribe_audio, a1) if a1 else None
            t2 = await run_in_threadpool(transcribe_audio, a2) if a2 else None
            if t1 and t2:
                e1 = await run_in_threadpool(embed_text, t1)
                e2 = await run_in_threadpool(embed_text, t2)
                audio_percent = round(100.0 * cosine(e1, e2), 2)

        d_base = await run_in_threadpool(get_duration_s, base_path)
        d_cand = await run_in_threadpool(get_duration_s, cand_path)
        duration_ratio = round(abs(d_base - d_cand) / max(d_base, d_cand or 1.0), 4)

        if sem_sim >= SEM_STRICT_THRESHOLD and (
            seq_percent >= MIN_SEQ_FOR_SEM or
            audio_percent >= AUDIO_DUP_THRESHOLD or
            duration_ratio <= MAX_DURATION_DRIFT
        ):
            return {
                "respuesta": f"Se detectó copia (equivalencia semántica fuerte) contra: {cand_url}",
                "porcentajeAprobacion": 0
            }

    # --------- SI NO HAY COPIA: EVALÚA CAMPAÑA ---------
    # Usa el resumen ya generado; si no existe aún, se crea.
    if base_summary is None:
        base_summary = await run_in_threadpool(analyze_video_with_chatgpt, base_path)
        if not base_summary:
            return JSONResponse(status_code=500, content={"error": "No se pudo analizar el video base."})

    cmp_raw = await run_in_threadpool(comparar_descripcion_con_resumen_ia, req.descripcion, base_summary)
    if not cmp_raw:
        return JSONResponse(status_code=500, content={"error": "No se pudo comparar la descripción con el resumen."})

    try:
        cmp_json = json.loads(cmp_raw)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=500, content={"error": f"Respuesta IA inválida: {e}", "raw": cmp_raw})

    # Normalizar % (si vino 0..1 lo pasamos a 0..100)
    match_percent = cmp_json.get("match_percent", 0)
    try:
        match_percent = float(match_percent)
        if match_percent <= 1.0:  # asume 0..1
            match_percent = round(match_percent * 100.0, 2)
        else:
            match_percent = round(match_percent, 2)
    except Exception:
        match_percent = 0.0

    reasons = cmp_json.get("reasons", "Sin motivos especificados.")
    aproved = cmp_json.get("aproved", False)

    respuesta_txt = f"{'Aprobado' if aproved else 'No aprobado'}: {reasons}"
    return {
        "respuesta": respuesta_txt,
        "porcentajeAprobacion": match_percent
    }