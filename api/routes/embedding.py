"""Ocean Embedding visualization routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas.embedding import EmbeddingScatterResponse
from api.services.inference_service import InferenceService, get_inference_service

router = APIRouter(tags=["Embedding"])


@router.get(
    "/embedding",
    response_model=EmbeddingScatterResponse,
    status_code=status.HTTP_200_OK,
    summary="Get 2D PCA Ocean Embedding Scatter Data",
    description=(
        "Returns reference/historical 2D PCA projection points representing "
        "oceanographic regimes across the North Indian Ocean."
    ),
)
def get_embedding_scatter(
    service: InferenceService = Depends(get_inference_service),
) -> EmbeddingScatterResponse:
    try:
        provider = service.get_provider()
        return provider.get_embedding_scatter()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching embedding data: {str(e)}",
        )
