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

    # Activar/desactivar BudgetGuardRails
    BUDGET_GUARDRAILS_ENABLED: bool = False

    # PostgreSQL
    PG_ENABLED: bool = False
    PG_DSN: str | None = None

    # Keyframes cache (FS)
    KEYFRAME_CACHE_ENABLED: bool = True
    KEYFRAMES_DIR: str = "/data/keyframes"

    # Audio / ASR
    AUDIO_ASR_ENABLED: bool = True  # usa Whisper si hay OPENAI_API_KEY
    AUDIO_TARGET_SR: int = 16000

    class Config:
        env_file = ".env"

    # Borrado Automático de huellas (Postgres)
    CLEANUP_INTERVAL_MIN: int = 60  # corre cada 60 min por defecto

@lru_cache
def get_settings() -> Settings:
    """Singleton de Settings para inyectar en FastAPI."""

    return Settings()
