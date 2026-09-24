"""API routes package."""

from api.routes.health import router as health_router
from api.routes.reconstruct import router as reconstruct_router
from api.routes.embedding import router as embedding_router
from api.routes.argo import router as argo_router
from api.routes.history import router as history_router
from api.routes.report import router as report_router
from api.routes.transect import router as transect_router
from api.routes.departure import router as departure_router

__all__ = [
    "health_router",
    "reconstruct_router",
    "embedding_router",
    "argo_router",
    "history_router",
    "report_router",
    "transect_router",
    "departure_router",
]



