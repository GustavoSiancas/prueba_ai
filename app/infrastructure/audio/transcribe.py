import os
from typing import Optional

"""
Selector de ASR (no usado en el flujo actual).
- Si hay OPENAI_API_KEY, usa Whisper API remota.
- Si no, retorna None (placeholder para ASR local).
"""

def transcribe_audio(audio_path: str) -> Optional[str]:
    """
    Transcribe audio a texto (opcional, fuera del camino principal).

    Args:
        audio_path: ruta a .mp3/.wav.

    Returns:
        str con la transcripción, o None si falla/no configurado.
    """

    use_openai = os.getenv("OPENAI_API_KEY") is not None
    if not os.path.exists(audio_path):
        return None

    if use_openai:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
            return transcript
        except Exception as e:
            print(f"OpenAI Whisper error: {e}")
            return None
    else:
        try:
            # Aquí podría ir faster-whisper local.
            return None
        except Exception as e:
            print(f"ASR local error: {e}")
            return None
