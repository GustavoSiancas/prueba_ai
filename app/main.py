from fastapi import FastAPI
from app.api.http.routers.evaluate import router as evaluate_router
from app.api.http.routers.dev_features import router as dev_features_router

"""
Punto de entrada de la app FastAPI.
Incluye routers públicos y de desarrollo.
"""

app = FastAPI(title="Inklop Video Inspector", version="0.1.0")

# Rutas públicas y de soporte
app.include_router(evaluate_router, prefix="/api")
app.include_router(dev_features_router, prefix="/api")

@app.get("/health")
def health():
    """Healthcheck simple para liveness/readiness."""

    return {"ok": True}
