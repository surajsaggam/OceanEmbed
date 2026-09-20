"""Schemas for independent Argo float observations and comparisons."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ArgoObservation(BaseModel):
    """Independent in-situ Argo float observation profile."""
    float_id: str = Field(..., description="WMO float identifier or profile ID (e.g. INCOIS-2902145)")
    date: str = Field(..., description="Observation date (YYYY-MM-DD)")
    latitude: float = Field(..., description="Float observation latitude")
    longitude: float = Field(..., description="Float observation longitude")
    distance_km: float = Field(..., description="Distance between query coordinate and float in km")
    depths_m: List[int] = Field(..., description="Standard observation depths in meters")
    temperature_c: List[float] = Field(..., description="Measured in-situ temperature profile in °C")
    rmse: Optional[float] = Field(None, description="Profile Root Mean Square Error vs reconstruction (°C)")
    mae: Optional[float] = Field(None, description="Profile Mean Absolute Error vs reconstruction (°C)")
    bias: Optional[float] = Field(None, description="Profile Mean Bias (Reconstructed - Argo) in °C")
    is_mock: bool = Field(True, description="Whether this Argo observation is mock or verified in-situ")
