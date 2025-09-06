from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import os
import json
from Prueba import analyze_video_with_chatgpt, comparar_descripcion_con_resumen_ia
from download import descargar_video

app = FastAPI()

@app.post("/analyze/")
async def analyze_video(
    video_url: str = Form(...),
    descripcion: str = Form(...)
):
    # Descargar el video y obtener el path local
    video_path = descargar_video(video_url)
    if not video_path or not os.path.exists(video_path):
        return JSONResponse(
            status_code=500,
            content={"error": "No se pudo descargar el video."}
        )

    # Analizar video y extraer resumen
    resumen = analyze_video_with_chatgpt(video_path)
    if not resumen:
        return JSONResponse(
            status_code=500,
            content={"error": "No se pudo analizar el video."}
        )

    # Comparar con la descripción
    resultado_raw = comparar_descripcion_con_resumen_ia(descripcion, resumen)
    if not resultado_raw:
        return JSONResponse(
            status_code=500,
            content={"error": "No se pudo comparar la descripción con el resumen del video."}
        )

    # Parsear JSON devuelto por la IA
    try:
        resultado_json = json.loads(resultado_raw)
        return resultado_json
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"No se pudo interpretar la respuesta JSON de la IA: {e}", "raw": resultado_raw}
        )
