"""API routes package."""

from api.routes.health import router as health_router
from api.routes.reconstruct import router as reconstruct_router
from api.routes.embedding import router as embedding_router
from api.routes.argo import router as argo_router

__all__ = [
    "health_router",
    "reconstruct_router",
    "embedding_router",
    "argo_router",
]
