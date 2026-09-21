"""Reconstruction history endpoint."""

from typing import List
from fastapi import APIRouter, Depends, Query, status

from api.schemas.history import ReconstructionHistoryItem
from api.services.history_service import HistoryService, get_history_service

router = APIRouter(tags=["History"])


@router.get(
    "/history",
    response_model=List[ReconstructionHistoryItem],
    status_code=status.HTTP_200_OK,
    summary="Get Recent Reconstruction History",
    description=(
        "Retrieves recent persistent reconstruction queries from the local database, "
        "ordered newest first. Includes observation date, coordinates, oceanographic regime, "
        "and execution timestamp."
    ),
)
def get_reconstruction_history(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of historical reconstruction items to retrieve",
    ),
    history_service: HistoryService = Depends(get_history_service),
) -> List[ReconstructionHistoryItem]:
    return history_service.get_recent_history(limit=limit)
