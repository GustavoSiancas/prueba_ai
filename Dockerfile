# Usa una imagen oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY requirements.txt .
COPY main.py .
COPY Prueba.py .
COPY download.py .

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Expón el puerto si es necesario (opcional)
# EXPOSE 8000

# Ejecuta el script principal
CMD ["python", "main.py"]
