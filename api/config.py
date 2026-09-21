"""Configuration settings for OceanEmbed API.

Defines spatial domain bounds for the North Indian Ocean,
authoritative 15 standard depths, and application configuration.
"""

from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    # API Metadata
    PROJECT_NAME: str = "OceanEmbed API"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"
    DESCRIPTION: str = (
        "Reconstruction of Subsurface Ocean Temperature from Surface Observations "
        "across the North Indian Ocean (5°N–30°N, 45°E–105°E)."
    )

    # North Indian Ocean Spatial Domain
    LAT_MIN: float = 5.0
    LAT_MAX: float = 30.0
    LON_MIN: float = 45.0
    LON_MAX: float = 105.0

    # Authoritative 15 Output Depths (meters)
    STANDARD_DEPTHS: List[int] = [
        0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
    ]

    # CORS settings for local dashboard
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Provider Mode: "mock" or "real"
    DEFAULT_PROVIDER: str = "real"


settings = Settings()
