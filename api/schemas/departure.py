"""Schemas for Subsurface Reconstruction Departure analysis."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DepthDepartureMetrics(BaseModel):
    """Aggregate statistical evaluation metrics for a specific standard depth."""
    depth_m: int
    rmse: float = Field(..., description="Root Mean Squared Error (°C)")
    mae: float = Field(..., description="Mean Absolute Error (°C)")
    mean_bias: float = Field(..., description="Mean Signed Bias (°C): Pred - Ref")
    min_departure_c: float
    max_departure_c: float
    valid_cells: int


class StationDepartureProfile(BaseModel):
    """Vertical sounding departure schedule at the active coordinate station."""
    latitude: float
    longitude: float
    depths_m: List[int]
    reconstructed_c: List[float]
    reference_c: List[float]
    departure_c: List[float]
    mean_absolute_departure_c: float


class DepartureRequest(BaseModel):
    """Specification for departure analysis query."""
    date: str = Field("2019-01-01", description="Observation date (YYYY-MM-DD)")
    depth_m: int = Field(100, description="Standard depth level in meters (0..1000)")
    latitude: Optional[float] = Field(None, ge=5.0, le=30.0)
    longitude: Optional[float] = Field(None, ge=45.0, le=105.0)


class DepartureResponse(BaseModel):
    """Authoritative Reconstruction Departure response."""
    date: str
    selected_depth_m: int
    depths_m: List[int]
    reference_name: str
    result_label: str
    scientific_note: str
    depth_metrics: DepthDepartureMetrics
    all_depth_metrics: List[DepthDepartureMetrics]
    station_profile: Optional[StationDepartureProfile] = None
    grid_lat: List[float]
    grid_lon: List[float]
    grid_departure: List[List[Optional[float]]]
    is_mock: bool
