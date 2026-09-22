"""Schemas for surface observations context (SST, SSS, SSH, currents, winds)."""

from pydantic import BaseModel, Field


class SurfaceContext(BaseModel):
    """The 7 surface observation physical variables used by OceanEmbed."""
    sst_c: float = Field(..., description="Sea Surface Temperature in °C (OceanIQ observation archive)")
    sss_psu: float = Field(..., description="Sea Surface Salinity in PSU (OceanIQ observation archive)")
    ssh_m: float = Field(..., description="Sea Surface Height / SLA in meters (OceanIQ observation archive)")
    current_u_ms: float = Field(..., description="Zonal surface current velocity in m/s (OceanIQ observation archive)")
    current_v_ms: float = Field(..., description="Meridional surface current velocity in m/s (OceanIQ observation archive)")
    wind_u_ms: float = Field(..., description="Zonal 10m surface wind in m/s (OceanIQ observation archive)")
    wind_v_ms: float = Field(..., description="Meridional 10m surface wind in m/s (OceanIQ observation archive)")
