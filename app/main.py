from fastapi import FastAPI
from app.api.http.routers.evaluate import router as evaluate_router
from app.api.http.routers.dev_features import router as dev_features_router
from app.api.http.routers.campaign import router as campaign_router

"""
Punto de entrada de la app FastAPI.
Incluye routers públicos y de desarrollo.
"""

app = FastAPI(title="Inklop IA Service", version="1.0.0")

# Rutas públicas y de soporte
app.include_router(evaluate_router, prefix="/api")
app.include_router(dev_features_router, prefix="/api")
app.include_router(campaign_router, prefix="/api")

@app.get("/health", tags=["health"])
def health():
    """Healthcheck simple para liveness/readiness."""

    return {"status": "ok"}
