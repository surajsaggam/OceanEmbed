"""Schemas for reconstruction history."""

from pydantic import BaseModel, Field


class ReconstructionHistoryItem(BaseModel):
    """A persistent record of a completed reconstruction query."""

    id: int = Field(..., description="Unique database ID of the history record")
    date: str = Field(..., description="Observation date (YYYY-MM-DD)")
    latitude: float = Field(..., description="Target latitude (°N)")
    longitude: float = Field(..., description="Target longitude (°E)")
    regime: str = Field(..., description="Identified oceanographic regime")
    timestamp: str = Field(..., description="ISO 8601 timestamp when reconstruction occurred")
