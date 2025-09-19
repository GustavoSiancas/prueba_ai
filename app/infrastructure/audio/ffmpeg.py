import subprocess, os

"""
Extracción de audio:
- Convierte el track de un .mp4 a WAV mono 16 kHz (por defecto), ideal para ASR.
"""

def extract_wav_mono16k(video_path: str, out_wav_path: str, sr: int = 16000) -> bool:
    """
    Extrae audio a WAV mono con sample rate `sr`.

    Returns:
        True si el archivo existe y tiene tamaño > 0, False en fallo.
    """

    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", out_wav_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 0
    except Exception:
        return False
