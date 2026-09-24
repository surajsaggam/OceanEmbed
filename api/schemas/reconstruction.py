"""Schemas for subsurface temperature profile reconstruction request and response."""

from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from api.config import settings
from api.schemas.surface import SurfaceContext
from api.schemas.argo import ArgoObservation
from api.schemas.embedding import EmbeddingCoordinates


class ReconstructionRequest(BaseModel):
    """Request payload for subsurface ocean temperature reconstruction."""
    date: str = Field(
        ...,
        description="Observation date in YYYY-MM-DD format (e.g. 2023-06-15)",
        json_schema_extra={"example": "2023-06-15"},
    )
    latitude: float = Field(
        ...,
        ge=settings.LAT_MIN,
        le=settings.LAT_MAX,
        description=f"Latitude within North Indian Ocean domain ({settings.LAT_MIN}°N to {settings.LAT_MAX}°N)",
        json_schema_extra={"example": 18.5},
    )
    longitude: float = Field(
        ...,
        ge=settings.LON_MIN,
        le=settings.LON_MAX,
        description=f"Longitude within North Indian Ocean domain ({settings.LON_MIN}°E to {settings.LON_MAX}°E)",
        json_schema_extra={"example": 88.25},
    )

    @field_validator("date")
    def validate_date(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("Date must be formatted as YYYY-MM-DD")


class ModelMetadata(BaseModel):
    """Metadata describing model version and inference execution."""
    name: str = "OceanEmbed"
    version: str = "0.1.0-prototype"
    provider_type: str = Field(
        ...,
        description="Provider executing inference (e.g. 'mock_climatology' or 'pytorch_checkpoint')",
        json_schema_extra={"example": "mock_climatology"},
    )
    checkpoint_hash: Optional[str] = Field(None, description="SHA256 hash of loaded checkpoint")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")


class ReconstructionResponse(BaseModel):
    """Authoritative response payload for a subsurface ocean temperature reconstruction."""
    request_id: str = Field(..., description="Unique query identifier")
    date: str = Field(..., description="Observation date (YYYY-MM-DD)")
    latitude: float = Field(..., description="Query latitude (°N)")
    longitude: float = Field(..., description="Query longitude (°E)")
    depths_m: List[int] = Field(
        default=settings.STANDARD_DEPTHS,
        description="The 15 authoritative standard output depths (meters)",
    )
    temperature_c: List[float] = Field(
        ...,
        description="Reconstructed subsurface temperatures at the 15 standard depths (°C)",
    )
    surface_context: SurfaceContext = Field(..., description="Multi-source surface observation context")
    d26_depth_m: Optional[float] = Field(
        None,
        description="Depth of the 26°C isotherm in meters (proxy for Tropical Cyclone Heat Potential)",
    )
    mixed_layer_depth_m: Optional[float] = Field(
        None,
        description="Estimated Mixed Layer Depth in meters (temperature drop threshold)",
    )
    tchp_kj_cm2: Optional[float] = Field(
        None,
        description="Tropical Cyclone Heat Potential in kJ/cm² integrated from surface to D26 isotherm (excess heat > 26°C)",
    )
    argo_comparison: Optional[ArgoObservation] = Field(
        None,
        description="Collocated in-situ Argo float profile for independent blind validation",
    )
    embedding: EmbeddingCoordinates = Field(
        ...,
        description="2D projection of the 128-D latent Ocean Embedding for this location",
    )
    model: ModelMetadata = Field(..., description="Model provenance and inference metadata")
    is_mock: bool = Field(
        ...,
        description="CRITICAL: True if synthetic/mock data; False only for real model inference",
    )
    provenance: str = Field(
        ...,
        description="Human-readable provenance and data lineage statement",
    )
    timestamp: str = Field(..., description="ISO 8601 server generation timestamp")
