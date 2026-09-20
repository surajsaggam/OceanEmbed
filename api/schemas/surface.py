"""Schemas for surface observations context (SST, SSS, SSH, currents, winds)."""

from pydantic import BaseModel, Field


class SurfaceContext(BaseModel):
    """The 7 surface observation physical variables used by OceanEmbed."""
    sst_c: float = Field(..., description="Sea Surface Temperature in °C (OSTIA)")
    sss_psu: float = Field(..., description="Sea Surface Salinity in PSU (SMAP/SMOS)")
    ssh_m: float = Field(..., description="Sea Surface Height / SLA in meters (DUACS)")
    current_u_ms: float = Field(..., description="Zonal surface current velocity in m/s (OSCAR)")
    current_v_ms: float = Field(..., description="Meridional surface current velocity in m/s (OSCAR)")
    wind_u_ms: float = Field(..., description="Zonal 10m surface wind in m/s (CCMP / ASCAT)")
    wind_v_ms: float = Field(..., description="Meridional 10m surface wind in m/s (CCMP / ASCAT)")
