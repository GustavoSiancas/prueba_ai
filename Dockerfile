# Usa una imagen oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY requirements.txt .
COPY main.py .
COPY Prueba.py .
COPY download.py .

# Instala dependencias del sistema para OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Expón el puerto si es necesario (opcional)
# EXPOSE 8000

# Ejecuta el script principal
CMD ["python", "main.py"]
