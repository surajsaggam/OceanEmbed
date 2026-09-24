"""Mock inference provider for OceanEmbed prototype.

Generates physically plausible, oceanographically consistent synthetic
profiles and multi-source telemetry for the North Indian Ocean.
Clearly labeled as MOCK / SYNTHETIC to preserve scientific integrity.
"""

import os
import json
import math
import time
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import numpy as np

from api.config import settings
from api.schemas.reconstruction import (
    ReconstructionRequest,
    ReconstructionResponse,
    ModelMetadata,
)
from api.schemas.surface import SurfaceContext
from api.schemas.argo import ArgoObservation
from api.schemas.embedding import (
    EmbeddingCoordinates,
    EmbeddingScatterPoint,
    EmbeddingScatterResponse,
)
from api.schemas.transect import (
    TransectPoint,
    TransectRequest,
    TransectResponse,
    TransectStation,
)
from api.schemas.departure import (
    DepartureRequest,
    DepartureResponse,
    DepthDepartureMetrics,
    StationDepartureProfile,
)
from ml_interface.base_provider import AbstractOceanEmbedProvider


class MockInferenceProvider(AbstractOceanEmbedProvider):
    """Synthetic oceanographic profile generator."""

    def __init__(self) -> None:
        self._provider_name = "mock_climatology"
        self._mock_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mock_data"
        )
        self._argo_catalog: List[Dict[str, Any]] = self._load_argo_catalog()
        self._embedding_scatter: List[Dict[str, Any]] = self._load_embedding_scatter()

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def is_mock(self) -> bool:
        return True

    def _load_argo_catalog(self) -> List[Dict[str, Any]]:
        path = os.path.join(self._mock_data_dir, "mock_argo_floats.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _load_embedding_scatter(self) -> List[Dict[str, Any]]:
        path = os.path.join(self._mock_data_dir, "mock_embeddings_scatter.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _compute_haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Great-circle distance in kilometers."""
        r = 6371.0  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2.0) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(r * c, 1)

    def _determine_regime(
        self, lat: float, lon: float, month: int
    ) -> Dict[str, Any]:
        """Determine characteristic physical regime and baseline parameters."""
        is_summer_monsoon = month in [6, 7, 8, 9]

        if lon <= 62.0 and lat <= 18.0 and is_summer_monsoon:
            # Somali / Western Arabian Sea Upwelling
            return {
                "regime": "Somali / Oman Upwelling Zone",
                "region": "Arabian Sea",
                "sst": 25.8,
                "sss": 35.7,
                "ssh": -0.12,
                "u_curr": -0.35,
                "v_curr": 0.85,
                "u_wind": -3.5,
                "v_wind": 9.2,
                "z_therm": 52.0,
                "t_deep": 5.4,
                "pca_1": -2.8,
                "pca_2": -2.0,
            }
        elif lon < 77.0:
            # Arabian Sea High Salinity Water
            return {
                "regime": "Arabian Sea High-Salinity Water",
                "region": "Arabian Sea",
                "sst": 29.2,
                "sss": 36.4,
                "ssh": 0.08,
                "u_curr": 0.15,
                "v_curr": -0.10,
                "u_wind": -2.1,
                "v_wind": 3.8,
                "z_therm": 72.0,
                "t_deep": 5.6,
                "pca_1": -1.6,
                "pca_2": 1.2,
            }
        elif lon >= 82.0 and lat >= 16.0:
            # Bay of Bengal Freshwater Plume (River runoff cap)
            return {
                "regime": "Bay of Bengal Freshwater Plume",
                "region": "Bay of Bengal",
                "sst": 29.8,
                "sss": 32.2,
                "ssh": 0.19,
                "u_curr": 0.22,
                "v_curr": -0.16,
                "u_wind": -4.2,
                "v_wind": 5.5,
                "z_therm": 88.0,
                "t_deep": 5.3,
                "pca_1": 2.4,
                "pca_2": -1.2,
            }
        elif lon >= 77.0 and lat > 8.0:
            # Central Bay of Bengal
            return {
                "regime": "Central Bay of Bengal",
                "region": "Bay of Bengal",
                "sst": 29.5,
                "sss": 33.6,
                "ssh": 0.14,
                "u_curr": 0.18,
                "v_curr": -0.08,
                "u_wind": -3.0,
                "v_wind": 4.2,
                "z_therm": 82.0,
                "t_deep": 5.4,
                "pca_1": 1.7,
                "pca_2": -0.3,
            }
        else:
            # Equatorial Indian Ocean
            return {
                "regime": "Equatorial Warm Pool",
                "region": "Equatorial",
                "sst": 30.1,
                "sss": 34.5,
                "ssh": 0.10,
                "u_curr": 0.45,
                "v_curr": 0.05,
                "u_wind": 1.2,
                "v_wind": 1.8,
                "z_therm": 95.0,
                "t_deep": 5.5,
                "pca_1": 0.5,
                "pca_2": 0.2,
            }

    def _generate_profile(
        self, sst: float, t_deep: float, z_therm: float, scale: float = 38.0
    ) -> List[float]:
        """Generate vertical temperature curve using oceanographic sigmoidal thermocline."""
        temps = []
        for z in settings.STANDARD_DEPTHS:
            # Sigmoidal thermocline transition
            sig = 1.0 / (1.0 + math.exp((z - z_therm) / scale))
            t = t_deep + (sst - t_deep) * sig
            # Add subtle near-surface mixed-layer preservation
            if z <= 20:
                t = sst - (0.015 * z)
            # Round for precision consistency
            temps.append(round(t, 2))
        return temps

    def _calculate_d26(self, depths: List[int], temps: List[float]) -> Optional[float]:
        """Calculate D26 isotherm depth (meters) via linear interpolation.
        Returns None if maximum profile temperature never reaches 26.0°C.
        """
        target = 26.0
        if not temps or max(temps) < target or min(temps) >= target:
            return None
        for i in range(len(temps) - 1):
            t1, t2 = temps[i], temps[i + 1]
            z1, z2 = depths[i], depths[i + 1]
            if t1 >= target >= t2:
                if t1 == t2:
                    return float(z1)
                fraction = (t1 - target) / (t1 - t2)
                return round(z1 + fraction * (z2 - z1), 1)
        return None

    def _calculate_mld(
        self, depths: List[int], temps: List[float], threshold: float = 0.5
    ) -> Optional[float]:
        """Calculate Mixed Layer Depth (meters) where temperature drops by threshold from surface."""
        target = temps[0] - threshold
        for i in range(len(temps) - 1):
            t1, t2 = temps[i], temps[i + 1]
            z1, z2 = depths[i], depths[i + 1]
            if t1 >= target >= t2:
                if t1 == t2:
                    return float(z1)
                fraction = (t1 - target) / (t1 - t2)
                return round(z1 + fraction * (z2 - z1), 1)
        return float(depths[-1])

    def predict_profile(self, request: ReconstructionRequest) -> ReconstructionResponse:
        start_time = time.perf_counter()
        req_date = date.fromisoformat(request.date)
        regime_info = self._determine_regime(
            request.latitude, request.longitude, req_date.month
        )

        # Reconstructed temperature profile
        temps = self._generate_profile(
            sst=regime_info["sst"],
            t_deep=regime_info["t_deep"],
            z_therm=regime_info["z_therm"],
        )

        d26 = self._calculate_d26(settings.STANDARD_DEPTHS, temps)
        mld = self._calculate_mld(settings.STANDARD_DEPTHS, temps)

        # Check for nearby Argo float matchup
        argo_match = self.find_nearby_argo(
            request.date, request.latitude, request.longitude
        )
        if argo_match:
            # Compute actual RMSE, MAE, bias against this specific profile
            errors = [
                pred - obs
                for pred, obs in zip(temps, argo_match.temperature_c)
            ]
            argo_match.rmse = round(
                math.sqrt(sum(e**2 for e in errors) / len(errors)), 3
            )
            argo_match.mae = round(sum(abs(e) for e in errors) / len(errors), 3)
            argo_match.bias = round(sum(errors) / len(errors), 3)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        request_id = f"rec_{req_date.strftime('%Y%m%d')}_{int(request.latitude * 100):04d}_{int(request.longitude * 100):05d}"

        return ReconstructionResponse(
            request_id=request_id,
            date=request.date,
            latitude=request.latitude,
            longitude=request.longitude,
            depths_m=settings.STANDARD_DEPTHS,
            temperature_c=temps,
            surface_context=SurfaceContext(
                sst_c=regime_info["sst"],
                sss_psu=regime_info["sss"],
                ssh_m=regime_info["ssh"],
                current_u_ms=regime_info["u_curr"],
                current_v_ms=regime_info["v_curr"],
                wind_u_ms=regime_info["u_wind"],
                wind_v_ms=regime_info["v_wind"],
            ),
            d26_depth_m=d26,
            mixed_layer_depth_m=mld,
            argo_comparison=argo_match,
            embedding=EmbeddingCoordinates(
                pca_1=round(regime_info["pca_1"] + (request.longitude - 80) * 0.02, 2),
                pca_2=round(regime_info["pca_2"] + (request.latitude - 15) * 0.03, 2),
                regime_label=regime_info["regime"],
                vector_dim=128,
            ),
            model=ModelMetadata(
                name="OceanEmbed",
                version="0.1.0-mock",
                provider_type=self._provider_name,
                checkpoint_hash=None,
                inference_time_ms=elapsed_ms,
            ),
            is_mock=True,
            provenance=(
                "Synthetic Oceanographic Climatology Generator (Mock Provider). "
                "Values are physically modeled for prototype development and testing. "
                "Argo-like comparisons and error metrics are computed exclusively between synthetic demo profiles."
            ),
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def get_embedding_scatter(self) -> EmbeddingScatterResponse:
        points = [EmbeddingScatterPoint(**p) for p in self._embedding_scatter]
        regimes = sorted(list(set(p.regime for p in points)))
        return EmbeddingScatterResponse(
            points=points,
            regimes=regimes,
            total_points=len(points),
            is_mock=True,
        )

    def find_nearby_argo(
        self, date_str: str, latitude: float, longitude: float
    ) -> Optional[ArgoObservation]:
        """Look for closest catalog float within spatial threshold."""
        closest_float: Optional[Dict[str, Any]] = None
        min_distance = 150.0  # Max search radius in km

        for argo in self._argo_catalog:
            dist = self._compute_haversine_distance(
                latitude, longitude, argo["latitude"], argo["longitude"]
            )
            if dist < min_distance:
                min_distance = dist
                closest_float = argo

        if closest_float:
            return ArgoObservation(
                float_id=closest_float["float_id"],
                date=closest_float["date"],
                latitude=closest_float["latitude"],
                longitude=closest_float["longitude"],
                distance_km=min_distance,
                depths_m=closest_float["depths_m"],
                temperature_c=closest_float["temperature_c"],
                is_mock=True,
            )
        return None

    def predict_transect(self, request: TransectRequest) -> TransectResponse:
        """Generates synthetic temperature profile stations along a transect."""
        waypoints = [(p.latitude, p.longitude) for p in request.points]

        seg_lengths = []
        total_dist = 0.0
        for i in range(len(waypoints) - 1):
            d = self._compute_haversine_distance(
                waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1]
            )
            seg_lengths.append(d)
            total_dist += d

        target_samples = request.num_samples or max(15, min(60, int(total_dist / 25.0) + 1))
        sample_dists = np.linspace(0.0, total_dist, target_samples).tolist() if total_dist > 0 else [0.0] * target_samples
        cum_segs = [0.0] + list(np.cumsum(seg_lengths))

        stations: List[TransectStation] = []
        depths = list(settings.STANDARD_DEPTHS)

        for idx, s_dist in enumerate(sample_dists):
            seg_idx = 0
            while seg_idx < len(seg_lengths) - 1 and s_dist > cum_segs[seg_idx + 1]:
                seg_idx += 1

            seg_len = seg_lengths[seg_idx]
            frac = (s_dist - cum_segs[seg_idx]) / seg_len if seg_len > 0 else 0.0
            frac = min(1.0, max(0.0, frac))

            p_start = waypoints[seg_idx]
            p_end = waypoints[seg_idx + 1]
            s_lat = p_start[0] + frac * (p_end[0] - p_start[0])
            s_lon = p_start[1] + frac * (p_end[1] - p_start[1])

            # Determine whether point is valid ocean (simple NIO check)
            is_ocean = not (72.5 <= s_lon <= 85.0 and 8.0 <= s_lat <= 22.0)  # Land mask for India

            if is_ocean:
                st_recon = self.predict_profile(
                    ReconstructionRequest(date=request.date, latitude=s_lat, longitude=s_lon)
                )
                temps = st_recon.temperature_c
                d26 = st_recon.d26_depth_m
                mld = st_recon.mixed_layer_depth_m
                sst = temps[0]
            else:
                temps = None
                d26 = None
                mld = None
                sst = None

            stations.append(
                TransectStation(
                    index=idx,
                    latitude=round(s_lat, 3),
                    longitude=round(s_lon, 3),
                    distance_km=round(s_dist, 1),
                    is_valid_ocean=is_ocean,
                    temperature_c=temps,
                    d26_depth_m=d26,
                    mixed_layer_depth_m=mld,
                    sst_c=sst,
                )
            )

        return TransectResponse(
            date=request.date,
            depths_m=depths,
            total_distance_km=round(total_dist, 1),
            stations=stations,
            model_name="OceanIQ Mock Transect Generator",
            is_mock=True,
            data_source="Synthetic Climatology Baseline",
        )

    def get_reconstruction_departure(
        self, request: DepartureRequest
    ) -> DepartureResponse:
        """Synthetic Subsurface Reconstruction Departure calculation for UI development."""
        depths = list(settings.STANDARD_DEPTHS)
        req_depth = request.depth_m if request.depth_m in depths else 100

        all_depth_metrics: List[DepthDepartureMetrics] = []
        for d in depths:
            # Thermocline (~100m) naturally has larger departures than surface or abyss
            factor = 1.0 + 1.2 * math.exp(-0.5 * ((d - 100.0) / 60.0) ** 2)
            rmse = round(0.45 * factor, 3)
            mae = round(0.32 * factor, 3)
            bias = round(0.05 * math.sin(d / 150.0), 3)
            all_depth_metrics.append(
                DepthDepartureMetrics(
                    depth_m=d,
                    rmse=rmse,
                    mae=mae,
                    mean_bias=bias,
                    min_departure_c=round(-1.8 * factor, 3),
                    max_departure_c=round(1.9 * factor, 3),
                    valid_cells=14200,
                )
            )

        sel_metric = next((m for m in all_depth_metrics if m.depth_m == req_depth), all_depth_metrics[7])

        station_profile = None
        if request.latitude is not None and request.longitude is not None:
            recon = self.predict_profile(
                ReconstructionRequest(
                    date=request.date,
                    latitude=request.latitude,
                    longitude=request.longitude,
                )
            )
            recon_t = recon.temperature_c
            # Add synthetic baseline departures
            dep_t = [
                round(0.4 * math.sin(d / 80.0) * math.cos(request.latitude / 10.0), 3)
                for d in depths
            ]
            ref_t = [round(r - dep, 3) for r, dep in zip(recon_t, dep_t)]
            mean_abs_dep = round(float(np.mean(np.abs(dep_t))), 3)

            station_profile = StationDepartureProfile(
                latitude=round(request.latitude, 3),
                longitude=round(request.longitude, 3),
                depths_m=depths,
                reconstructed_c=recon_t,
                reference_c=ref_t,
                departure_c=dep_t,
                mean_absolute_departure_c=mean_abs_dep,
            )

        # Generate downsampled grid: 26 lats, 31 lons (step 4)
        lats = [round(5.0 + i * 1.0, 2) for i in range(26)]
        lons = [round(45.0 + j * 2.0, 2) for j in range(31)]
        grid_dep: List[List[Optional[float]]] = []

        for lat in lats:
            row: List[Optional[float]] = []
            for lon in lons:
                # Mask out Indian subcontinent
                if 72.0 <= lon <= 85.0 and 8.0 <= lat <= 24.0:
                    row.append(None)
                else:
                    val = 0.5 * math.sin(lat / 5.0) * math.cos(lon / 8.0)
                    row.append(round(val, 3))
            grid_dep.append(row)

        return DepartureResponse(
            date=request.date,
            selected_depth_m=req_depth,
            depths_m=depths,
            reference_name="GLORYS12V1 Reanalysis Reference (Synthetic Mock Mode)",
            result_label="OceanIQ Reconstruction Departure",
            scientific_note="Departure = OceanIQ reconstruction − GLORYS12V1 reference (Demonstration Mock Mode)",
            depth_metrics=sel_metric,
            all_depth_metrics=all_depth_metrics,
            station_profile=station_profile,
            grid_lat=lats,
            grid_lon=lons,
            grid_departure=grid_dep,
            is_mock=True,
        )


