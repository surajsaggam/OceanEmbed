"""Schemas for Vertical Subsurface Transect / Cross-Section."""

from typing import List, Optional
from pydantic import BaseModel, Field


class TransectPoint(BaseModel):
    """Geographic coordinate point along a transect."""
    latitude: float = Field(..., ge=5.0, le=30.0, description="Latitude in degrees North")
    longitude: float = Field(..., ge=45.0, le=105.0, description="Longitude in degrees East")


class TransectRequest(BaseModel):
    """Request specification for vertical subsurface transect."""
    date: str = Field(..., description="Observation date (YYYY-MM-DD)")
    points: List[TransectPoint] = Field(..., min_length=2, description="At least start and end points")
    num_samples: Optional[int] = Field(25, ge=3, le=80, description="Target number of sampling stations along transect")


class TransectStation(BaseModel):
    """Discrete vertical sounding station along the transect."""
    index: int
    latitude: float
    longitude: float
    distance_km: float
    is_valid_ocean: bool
    temperature_c: Optional[List[float]] = None
    d26_depth_m: Optional[float] = None
    mixed_layer_depth_m: Optional[float] = None
    sst_c: Optional[float] = None


class TransectResponse(BaseModel):
    """Authoritative 2D vertical cross-section reconstruction output."""
    date: str
    depths_m: List[int]
    total_distance_km: float
    stations: List[TransectStation]
    model_name: str
    is_mock: bool
    data_source: str
