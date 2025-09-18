from pydantic_settings import BaseSettings
from functools import lru_cache

"""
Configuración centralizada (se carga de .env si existe).
"""

class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    REDIS_URL: str = "redis://localhost:6379/0"

    # Descarga
    VIDEO_MAX_MB: int = 200
    DL_TIMEOUT_S: int = 30

    # Resumen VLM
    FRAMES_MAX: int = 20
    FRAME_SCENE_THRESHOLD: float = 0.25

    # Umbrales de dedupe
    HASH_DUP_THRESHOLD: float = 95.0
    SEQ_DUP_THRESHOLD: float = 85.0
    AUDIO_DUP_THRESHOLD: float = 88.0
    SEM_STRICT_THRESHOLD: float = 92.0
    MIN_SEQ_FOR_SEM: float = 40.0
    MAX_DURATION_DRIFT: float = 0.15

    # Presupuesto diario
    USD_DAY_CAP: float = 5.0
    MAX_LLM_CALLS: int = 50
    MAX_EMB_CALLS: int = 200

    # Modo de resumen
    SUMMARY_MODE: str = "free"  # "free" | "hybrid"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    """Singleton de Settings para inyectar en FastAPI."""

    return Settings()
