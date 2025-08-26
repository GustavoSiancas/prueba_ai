import yt_dlp
import os

def descargar_video(url, output_folder="videos"):
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	output_path = os.path.join(output_folder, "%(title)s.%(ext)s")
	ydl_opts = {
		'outtmpl': output_path,
		'format': 'mp4/bestvideo+bestaudio/best',
		'merge_output_format': 'mp4',
		'quiet': False
	}
	try:
		with yt_dlp.YoutubeDL(ydl_opts) as ydl:
			print(f"⬇️ Descargando video desde: {url}")
			info = ydl.extract_info(url, download=True)
			filename = ydl.prepare_filename(info)
			print(f"✅ Video guardado como: {filename}")
			return filename
	except Exception as e:
		print(f"❌ Error al descargar el video: {e}")
		return None
