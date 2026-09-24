"""Main application entry point for OceanEmbed API.

Configures FastAPI, CORS origins for the React dashboard,
lifespan events, and mounts API routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routes import (
    health_router,
    reconstruct_router,
    embedding_router,
    argo_router,
    history_router,
    report_router,
    transect_router,
    departure_router,
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers under /api
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(reconstruct_router, prefix=settings.API_V1_STR)
app.include_router(transect_router, prefix=settings.API_V1_STR)
app.include_router(departure_router, prefix=settings.API_V1_STR)
app.include_router(embedding_router, prefix=settings.API_V1_STR)
app.include_router(argo_router, prefix=settings.API_V1_STR)
app.include_router(history_router, prefix=settings.API_V1_STR)
app.include_router(report_router, prefix=settings.API_V1_STR)




@app.get("/", tags=["Root"])
def root():
    """Root entry point with documentation links and service status."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "domain": "North Indian Ocean (5°N–30°N, 45°E–105°E)",
        "documentation": "/docs",
        "api_prefix": settings.API_V1_STR,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
