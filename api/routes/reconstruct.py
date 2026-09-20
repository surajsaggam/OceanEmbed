"""Subsurface temperature reconstruction endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas.reconstruction import ReconstructionRequest, ReconstructionResponse
from api.services.inference_service import InferenceService, get_inference_service

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
) -> ReconstructionResponse:
    try:
        provider = service.get_provider()
        return provider.predict_profile(request)
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
