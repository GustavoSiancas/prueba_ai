FROM python:3.11-slim

# Evita .pyc y asegura logs flush
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Crea usuario no-root
RUN adduser --disabled-password --gecos "" appuser

# Dependencias del sistema necesarias (ffmpeg + build tools mínimos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala primero dependencias para cacheo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código
COPY . .

# Baja privilegios
USER appuser

# Railway inyecta $PORT; deja fallback local a 8000
ENV PORT=8000
# Control opcional del nro de workers via env
ENV UVICORN_WORKERS=1

# Comando de arranque (no hardcodees el puerto)
CMD ["sh", "-c", "python -m app.scripts.run_sql_migrations && uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}"]