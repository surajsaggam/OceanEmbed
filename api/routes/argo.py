"""In-situ Argo float matchup routes."""

from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import Optional
from api.schemas.argo import ArgoObservation
from api.services.inference_service import InferenceService, get_inference_service
from api.config import settings

router = APIRouter(tags=["Argo"])


@router.get(
    "/argo/nearby",
    response_model=Optional[ArgoObservation],
    status_code=status.HTTP_200_OK,
    summary="Find Nearest Independent In-Situ Argo Float Observation",
    description="Searches for independent Argo float profiles within temporal/spatial search radius.",
)
def get_nearby_argo(
    date: str = Query(..., description="Query date in YYYY-MM-DD format", examples=["2023-06-15"]),
    latitude: float = Query(..., ge=settings.LAT_MIN, le=settings.LAT_MAX, description="Query latitude (°N)"),
    longitude: float = Query(..., ge=settings.LON_MIN, le=settings.LON_MAX, description="Query longitude (°E)"),
    service: InferenceService = Depends(get_inference_service),
) -> Optional[ArgoObservation]:
    try:
        provider = service.get_provider()
        return provider.find_nearby_argo(date, latitude, longitude)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error finding nearby Argo float: {str(e)}",
        )
