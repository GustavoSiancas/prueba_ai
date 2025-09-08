import cv2
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv
import moviepy.editor as mp
import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
	raise ValueError("Error: OPENAI_API_KEY no se encontró en las variables de entorno o en el archivo .env.")

client = OpenAI(api_key=OPENAI_API_KEY)

output_frames_dir = "video_frames"
audio_output_path = "video_audio.mp3"
seconds_per_frame = 5
max_frames = 20

def _cleanup():
    if os.path.exists(output_frames_dir):
        for f in os.listdir(output_frames_dir):
            try: os.remove(os.path.join(output_frames_dir, f))
            except: pass
        try: os.rmdir(output_frames_dir)
        except: pass

    if os.path.exists(audio_output_path):
        try: os.remove(audio_output_path)
        except: pass

def extract_frames(video_path, output_dir, seconds_interval=None, max_frames_limit=None):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	vidcap = cv2.VideoCapture(video_path)
	fps = vidcap.get(cv2.CAP_PROP_FPS)
	total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	duration_seconds = total_frames / fps
	extracted_frames = []
	current_frame = 0
	frame_count = 0

	while True:
		success, image = vidcap.read()
		if not success:
			break

		if seconds_interval is not None:
			current_second = current_frame / fps
			if int(current_second * 10) % int(seconds_interval * 10) == 0:
				if frame_count < max_frames_limit:
					_, buffer = cv2.imencode(".jpg", image)
					extracted_frames.append(base64.b64encode(buffer).decode("utf-8"))
					frame_count += 1
					vidcap.set(cv2.CAP_PROP_POS_MSEC, (current_second + seconds_interval) * 1000)
					current_frame = int((current_second + seconds_interval) * fps)
				else:
					break

		current_frame += 1
	vidcap.release()

	print(f"Extraídos {frame_count} fotogramas del video.")
	return extracted_frames

def extract_audio(video_path, output_audio_path):
	try:
		video = mp.VideoFileClip(video_path)
		video.audio.write_audiofile(output_audio_path)

		print(f"Audio extraído a: {output_audio_path}")
		return output_audio_path
	
	except Exception as e:
		print(f"Error al extraer audio: {e}")
		return None

def transcribe_audio(audio_path):
	if not audio_path or not os.path.exists(audio_path):
		return None
	
	try:
		with open(audio_path, "rb") as audio_file:
			transcript = client.audio.transcriptions.create(
				model="whisper-1",
				file=audio_file,
				response_format="text"
			)

		print("Transcripción de audio completa.")

		return transcript
	
	except Exception as e:
		print(f"Error al transcribir audio con Whisper: {e}")
		return None

def analyze_video_with_chatgpt(video_path):
	print(f"Iniciando análisis de video '{video_path}' con la API de OpenAI (GPT-4o)...")

	base64_frames = extract_frames(video_path, output_frames_dir, seconds_per_frame, max_frames)

	if not base64_frames:
		print("No se pudieron extraer fotogramas del video. Abortando análisis visual.")

	transcription = None

	if os.path.exists(video_path):
		audio_file_path = extract_audio(video_path, audio_output_path)

		if audio_file_path:
			transcription = transcribe_audio(audio_file_path)

	messages = [
		{
			"role": "user",
			"content": [
				{"type": "text", "text": "Describe a detalle qué ocurre en este video. Presta atención a eventos, personas, objetos y contexto."},
				{"type": "text", "text": "Aquí están algunos fotogramas clave del video:"}
			]
		}
	]
	for frame in base64_frames:
		messages[0]["content"].append({
			"type": "image_url", 
			"image_url": {"url": f"data:image/jpeg;base64,{frame}"}
			})
		
	if transcription:
		messages[0]["content"].append({
			"type": "text", 
			"text": f"\n\nTambién, aquí está la transcripción de audio del video:\n{transcription}"
			})
		
	print("\nEnviando fotogramas y transcripción a GPT-4o...")

	try:
		response = client.chat.completions.create(
			model="gpt-4o",
			messages=messages,
			max_tokens=2000
		)
		print("\n--- Respuesta de GPT-4o ---")
		return response.choices[0].message.content
	
	except Exception as e:
		print(f"Error al llamar a la API de GPT-4o: {e}")
		print("Asegúrate de tener acceso al modelo GPT-4o y de no haber excedido los límites de tokens o requests.")
		return None

	finally:
		_cleanup()


def comparar_descripcion_con_resumen_ia(descripcion, resumen):
	prompt = (
		f"A continuación, compara una descripción de campaña con un resumen generado desde un video. "
		f"Quiero que me devuelvas la comparación en formato JSON estricto, con tres campos: \n"
		'{"match_percent": "...","aproved": "...", "reasons": "..."}\n\n'
		f'El campo "aproved" debe ser true si el porcentaje de match es mayor o igual a 70, y false en caso contrario.\n\n'
		f"Descripción de campaña:\n\"{descripcion}\"\n\n"
		f"Resumen del video:\n\"{resumen}\"\n\n"
		f"Evalúa qué tanto se alinean ambos textos considerando tono, contenido, duración, objetivos y estilo."
	)
	messages = [
		{"role": "user", "content": prompt}
	]
	try:
		response = client.chat.completions.create(
			model="gpt-4o",
			messages=messages,
			response_format={"type": "json_object"}, 
		)
		return response.choices[0].message.content
	except Exception as e:
		print(f"Error al comparar descripción y resumen con la IA: {e}")
		return None
