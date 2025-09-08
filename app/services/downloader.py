import yt_dlp
import os

MOBILE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 15_5 like Mac OS X) "
             "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 "
             "Mobile/15E148 Safari/604.1")

def descargar_video(url, output_folder="videos"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    outtmpl = os.path.join(output_folder, "%(title)s.%(ext)s")
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
                print(f"⬇️ Descargando video desde: {url} {f'con cookies {cookies_spec[0]}' if cookies_spec else '(sin cookies)'}")
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                print(f"✅ Video guardado como: {filename}")
                return filename
        except Exception as e:
            last_err = e
            print(f"⚠️ Intento {'con '+cookies_spec[0] if cookies_spec else 'sin cookies'} falló: {e}")

    print(f"❌ Error al descargar el video: {last_err}")
    return None
