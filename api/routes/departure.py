"""Subsurface Reconstruction Departure endpoint.

Evaluates: Departure = OceanIQ Reconstruction - GLORYS12V1 Reference.
Strict scientific framing: Model/reconstruction departure from GLORYS reanalysis
reference on the held-out test date. NOT a climatological anomaly or marine heatwave.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas.departure import DepartureRequest, DepartureResponse
from api.services.inference_service import InferenceService, get_inference_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Departure"])


@router.post(
    "/departure",
    response_model=DepartureResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute Subsurface Reconstruction Departure from GLORYS12V1 Reference",
    description=(
        "Calculates Subsurface Reconstruction Departure (OceanIQ reconstructed temperature minus "
        "GLORYS12V1 reanalysis reference) across standard depths for the held-out test date. "
        "Strictly an evaluation departure, NOT a climatological anomaly or marine heatwave."
    ),
)
def get_reconstruction_departure(
    request: DepartureRequest,
    service: InferenceService = Depends(get_inference_service),
) -> DepartureResponse:
    try:
        provider = service.get_provider()
        return provider.get_reconstruction_departure(request)
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
        logger.error(f"Reconstruction departure calculation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Departure calculation error: {str(e)}",
        )
