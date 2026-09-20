"""Health check and service status route."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List

from api.config import settings
from api.services.inference_service import InferenceService, get_inference_service

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    active_provider: str
    is_mock: bool
    domain: dict
    depths_m: List[int]


@router.get("/health", response_model=HealthResponse)
def get_health(
    service: InferenceService = Depends(get_inference_service),
) -> HealthResponse:
    """Return service health, active provider, and domain metadata."""
    provider = service.get_provider()
    return HealthResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        version=settings.VERSION,
        active_provider=provider.provider_name,
        is_mock=provider.is_mock,
        domain={
            "lat_min": settings.LAT_MIN,
            "lat_max": settings.LAT_MAX,
            "lon_min": settings.LON_MIN,
            "lon_max": settings.LON_MAX,
        },
        depths_m=settings.STANDARD_DEPTHS,
    )
