import yt_dlp
import os

"""
Descarga de videos (TikTok u otros) a MP4 con yt_dlp.
Incluye fallback a cookies de navegador para TikTok si es necesario.
"""

MOBILE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 15_5 like Mac OS X) "
             "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 "
             "Mobile/15E148 Safari/604.1")

def descargar_video(url, output_folder="videos", size_mb_limit=200, timeout_s=30):
    """
    Descarga un video a `output_folder` usando yt_dlp.

    Args:
        url: URL del video.
        output_folder: carpeta destino.
        size_mb_limit: límite duro de tamaño.
        timeout_s: timeout de socket.

    Returns:
        ruta absoluta del archivo descargado o None si falla.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    outtmpl = os.path.join(output_folder, "%(id)s.%(ext)s")
    is_tiktok = "tiktok.com" in url.lower()

    base_opts = {
        "outtmpl": outtmpl,
        "format": "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 1,
        "http_headers": {
            "User-Agent": MOBILE_UA if is_tiktok else None,
            "Referer": "https://www.tiktok.com/" if is_tiktok else None,
        },
        "file_size_limit": size_mb_limit * 1024 * 1024,
        "socket_timeout": timeout_s,
        "verbose": False,
    }

    browsers = [None]
    if is_tiktok:
        browsers += [("edge",), ("chrome",), ("brave",)]

    last_err = None
    for cookies_spec in browsers:
        opts = dict(base_opts)
        if cookies_spec is not None:
            opts["cookiesfrombrowser"] = cookies_spec
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                return filename
        except Exception as e:
            last_err = e

    print(f"Error al descargar el video: {last_err}")
    return None
