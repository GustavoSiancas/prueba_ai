# Usa una imagen oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos necesarios al contenedor

COPY requirements.txt ./
COPY . ./

# Instala dependencias del sistema para OpenCV
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Instala Uvicorn si no está en requirements.txt
RUN pip install --no-cache-dir uvicorn

# Expón el puerto (Render lo ignora, pero es útil localmente)
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
