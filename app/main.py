import asyncio
from contextlib import asynccontextmanager, suppress
from fastapi import FastAPI
from app.api.http.routers.evaluate import router as evaluate_router
from app.api.http.routers.dev_features import router as dev_features_router
from app.api.http.routers.campaign import router as campaign_router
from app.infrastructure.settings import get_settings
from app.application.services.retention_cleanup import RetentionCleaner

"""
Punto de entrada de la app FastAPI.
Incluye routers públicos y de desarrollo.
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Reemplaza a @app.on_event('startup'/'shutdown').

    - En startup: inicia el cleaner (si PG_ENABLED=True).
    - En shutdown: cancela y espera el task del cleaner.
    """
    settings = get_settings()
    task = None

    if settings.PG_ENABLED:
        cleaner = RetentionCleaner(settings=settings)
        task = asyncio.create_task(cleaner.run_forever())
        # Guarda referencia por si quieres inspeccionar en runtime:
        app.state.retention_cleaner_task = task

    try:
        yield
    finally:
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

app = FastAPI(title="Inklop IA Service", version="1.0.0", lifespan=lifespan)

# Rutas públicas y de soporte
app.include_router(evaluate_router, prefix="/api")
app.include_router(dev_features_router, prefix="/api")
app.include_router(campaign_router, prefix="/api")

@app.get("/health", tags=["health"])
def health():
    """Healthcheck simple para liveness/readiness."""

    return {"status": "ok"}
