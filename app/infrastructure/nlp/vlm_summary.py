import os, base64, json, subprocess, tempfile
import cv2
from openai import OpenAI

"""
Resumen visual del video (sin ASR):
- Modo 'free': devuelve narrativa en texto.
- Modo 'hybrid': devuelve JSON con narrative + listas + layout_hints.
"""

def _uniform_keyframes(video_path: str, max_frames: int = 16, scale=640):
    """
    Extrae frames uniformemente espaciados, reescalados y comprimidos a JPEG base64.

    Returns:
        lista[str base64] de hasta `max_frames` frames.
    """

    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max_frames) or 1
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok: break
        if idx % step == 0:
            h, w = frame.shape[:2]
            new_w = scale
            new_h = int(h * (scale / w))
            resized = cv2.resize(frame, (new_w, new_h))
            _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            frames.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
        idx += 1
    cap.release()
    return frames

def analyze_video_free_narrative(video_path: str, transcript_text: str | None = None, max_frames: int = 16) -> str:
    """
    Devuelve un texto descriptivo (120–200 palabras) basado SOLO en lo visible
    en los frames (y en transcript_text si se provee).

    No inventar idiomas de subtítulos ni layout si no es evidente.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    frames = _uniform_keyframes(video_path, max_frames=max_frames)

    messages = [{
        "role":"user",
        "content":[
            {"type":"text","text":
             ("Describe con detalle (120–200 palabras) lo que ocurre en el video, en español, "
              "basándote SOLO en lo que se ve en los fotogramas y, si se provee, lo que se oye en el texto. "
              "Menciona disposición visual si es evidente (ej.: 'persona en la parte superior' y 'gameplay abajo'). "
              "Si aparecen subtítulos, indícalo y el idioma solo si se percibe con claridad. No inventes nada.")
            },
            {"type":"text","text":"Fotogramas representativos:"}
        ]
    }]

    for f in frames:
        messages[0]["content"].append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{f}"}})

    if transcript_text:
        messages[0]["content"].append({"type":"text","text":f"TEXTO/TRANSCRIPCIÓN:\n{transcript_text[:8000]}"})

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1200,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def analyze_video_hybrid(video_path: str, transcript_text: str | None = None, max_frames: int = 16) -> dict:
    """
    Devuelve JSON con:
      - narrative (120–200 palabras),
      - events/people/objects/locations/topics/heard_phrases (listas),
      - layout_hints: {facecam_top, gameplay_bottom, subtitles:{present, language}}.

    Reglas: usar sólo evidencia clara; si no es visible/oyible, 'desconocido'.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    frames = _uniform_keyframes(video_path, max_frames=max_frames)

    messages = [{
        "role": "user",
        "content": [
            {"type":"text","text":
             ("Devuélveme SOLO JSON con claves EXACTAS: "
              "{narrative, events, people, objects, locations, topics, layout_hints, heard_phrases}. "
              "Reglas: evidencia solo de imágenes (y texto si lo incluyo); si no se ve/oye, usa 'desconocido'. "
              "'narrative': 120–200 palabras. "
              "'layout_hints': {facecam_top: bool|desconocido, gameplay_bottom: bool|desconocido, subtitles:{present: bool, language: 'es'|'en'|'desconocido'}}. "
              "Idioma: español.")
            },
            {"type":"text","text":"Fotogramas representativos:"}
        ]
    }]

    for f in frames:
        messages[0]["content"].append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{f}"}})

    if transcript_text:
        messages[0]["content"].append({"type":"text","text":f"TEXTO/TRANSCRIPCIÓN (opcional):\n{transcript_text[:8000]}"})

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type":"json_object"},
        max_tokens=1400,
        temperature=0.1
    )
    data = json.loads(resp.choices[0].message.content)
    data.setdefault("narrative","")
    for k in ["events","people","objects","locations","topics","heard_phrases"]:
        if k not in data or not isinstance(data[k], list):
            data[k] = []
    if "layout_hints" not in data or not isinstance(data["layout_hints"], dict):
        data["layout_hints"] = {}
    return data

def summarize_video_textual(summary) -> str:
    """
    Normaliza el resumen a texto compacto (<= 6000 chars):
      - Si summary es dict (hybrid), combina narrative + layout_hints.
      - Si summary es str (free), lo devuelve truncado.
    """

    if summary is None:
        return ""
    if isinstance(summary, str):
        return summary[:6000]
    parts = []
    if summary.get("narrative"):
        parts.append(summary["narrative"])
    lh = summary.get("layout_hints", {})
    if lh:
        parts.append(f"LAYOUT: facecam_top={lh.get('facecam_top','desconocido')}, gameplay_bottom={lh.get('gameplay_bottom','desconocido')}, "
                     f"subtitles.present={lh.get('subtitles',{}).get('present','desconocido')}, "
                     f"subtitles.lang={lh.get('subtitles',{}).get('language','desconocido')}")
    return "\n".join(parts)[:6000]