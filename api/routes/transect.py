"""Vertical Subsurface Transect endpoint."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas.transect import TransectRequest, TransectResponse
from api.services.inference_service import InferenceService, get_inference_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Transect"])


@router.post(
    "/transect",
    response_model=TransectResponse,
    status_code=status.HTTP_200_OK,
    summary="Reconstruct 2D Vertical Subsurface Temperature Transect",
    description=(
        "Reconstructs 2D vertical temperature cross-section across multiple geographic waypoints "
        "at all 15 authoritative standard depths (0 to 1000m) for a given date in the North Indian Ocean."
    ),
)
def reconstruct_transect(
    request: TransectRequest,
    service: InferenceService = Depends(get_inference_service),
) -> TransectResponse:
    try:
        provider = service.get_provider()
        return provider.predict_transect(request)
    except NotImplementedError as nie:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(nie),
        )
    except RuntimeError as re:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(re),
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Transect reconstruction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transect inference error: {str(e)}",
        )
