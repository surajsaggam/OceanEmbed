"""Schemas for 128-D Ocean Embedding visualization (2D PCA/UMAP projections)."""

from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingCoordinates(BaseModel):
    """2D projection of the 128-D latent embedding for the active reconstruction."""
    pca_1: float = Field(..., description="First principal component / dimension 1")
    pca_2: float = Field(..., description="Second principal component / dimension 2")
    regime_label: str = Field(..., description="Oceanographic regime label (e.g. Bay of Bengal Plume)")
    vector_dim: int = Field(128, description="Original bottleneck embedding dimensionality")


class EmbeddingScatterPoint(BaseModel):
    """Historical/reference point in the 2D embedding scatter plot."""
    id: str
    pca_1: float
    pca_2: float
    regime: str
    region: str = Field(..., description="Basin region (Arabian Sea, Bay of Bengal, Equatorial)")
    season: str = Field(..., description="Monsoon season (SW Monsoon, NE Monsoon, Pre-Monsoon, Post-Monsoon)")
    latitude: float
    longitude: float


class EmbeddingScatterResponse(BaseModel):
    """Response containing reference scatter points for the embedding map."""
    points: List[EmbeddingScatterPoint]
    regimes: List[str]
    total_points: int
    is_mock: bool = Field(True, description="Whether embedding points are synthetic or derived from ML checkpoint")
