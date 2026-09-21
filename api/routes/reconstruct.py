"""Subsurface temperature reconstruction endpoint."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas.reconstruction import ReconstructionRequest, ReconstructionResponse
from api.services.inference_service import InferenceService, get_inference_service
from api.services.history_service import HistoryService, get_history_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Reconstruction"])


@router.post(
    "/reconstruct",
    response_model=ReconstructionResponse,
    status_code=status.HTTP_200_OK,
    summary="Reconstruct 15-Depth Subsurface Ocean Temperature",
    description=(
        "Reconstructs the vertical temperature profile at 15 authoritative standard depths "
        "(0 to 1000m) for a given date and location in the North Indian Ocean."
    ),
)
def reconstruct_profile(
    request: ReconstructionRequest,
    service: InferenceService = Depends(get_inference_service),
    history_service: HistoryService = Depends(get_history_service),
) -> ReconstructionResponse:
    try:
        provider = service.get_provider()
        response = provider.predict_profile(request)

        # Automatically record successful reconstruction to persistent history
        try:
            regime = response.embedding.regime_label if response.embedding else "North Indian Ocean"
            history_service.save_reconstruction(
                date=response.date,
                latitude=response.latitude,
                longitude=response.longitude,
                regime=regime,
                timestamp=response.timestamp,
            )
        except Exception as hist_err:
            logger.warning(f"Failed to record reconstruction to persistent history: {hist_err}")

        return response
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )

