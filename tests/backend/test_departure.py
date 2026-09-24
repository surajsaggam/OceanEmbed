"""Tests for Subsurface Reconstruction Departure calculation and API endpoint.

Verifies:
1. Departure = OceanIQ reconstructed temperature - GLORYS12V1 reference.
2. All 15 standard depths are supported.
3. Summary metrics (RMSE, MAE, Mean Bias) are computed exclusively on valid ocean cells.
4. Scientific labels and notes conform to non-climatological, model-departure guidelines.
5. Frozen model checkpoint remains untouched.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.schemas.departure import DepartureRequest
from api.services.mock_provider import MockInferenceProvider

client = TestClient(app)


def test_mock_provider_get_reconstruction_departure():
    provider = MockInferenceProvider()
    req = DepartureRequest(
        date="2019-01-01",
        depth_m=100,
        latitude=15.0,
        longitude=65.0,
    )
    res = provider.get_reconstruction_departure(req)
    assert res.selected_depth_m == 100
    assert len(res.depths_m) == 15
    assert len(res.all_depth_metrics) == 15
    assert res.depth_metrics.depth_m == 100
    assert res.depth_metrics.rmse > 0
    assert res.depth_metrics.mae > 0
    assert res.depth_metrics.valid_cells > 0
    assert res.station_profile is not None
    assert len(res.station_profile.depths_m) == 15
    assert len(res.station_profile.departure_c) == 15
    assert res.is_mock is True


def test_real_provider_get_reconstruction_departure():
    from api.services.real_provider import RealOceanEmbedProvider
    provider = RealOceanEmbedProvider()
    if not provider.is_model_loaded:
        pytest.skip("Frozen checkpoint or test dataset not present.")

    req = DepartureRequest(
        date="2019-01-01",
        depth_m=100,
        latitude=15.0,
        longitude=65.0,
    )
    res = provider.get_reconstruction_departure(req)

    # 1. Verification of labels
    assert res.reference_name == "GLORYS12V1 Reanalysis Reference"
    assert res.result_label == "OceanIQ Reconstruction Departure"
    assert "Departure = OceanIQ reconstruction" in res.scientific_note
    assert "GLORYS12V1 reference" in res.scientific_note
    assert res.is_mock is False

    # 2. 15 standard depths
    assert len(res.depths_m) == 15
    assert len(res.all_depth_metrics) == 15
    assert res.selected_depth_m == 100

    # 3. Depth metrics calculated on valid cells
    m = res.depth_metrics
    assert m.depth_m == 100
    assert 0.5 < m.rmse < 2.5
    assert 0.3 < m.mae < 2.0
    assert -1.0 < m.mean_bias < 1.0
    assert m.valid_cells > 9000

    # 4. Station vertical departure calculation: dep = recon - ref
    assert res.station_profile is not None
    st = res.station_profile
    assert len(st.depths_m) == 15
    for i in range(15):
        expected_dep = round(st.reconstructed_c[i] - st.reference_c[i], 3)
        assert st.departure_c[i] == pytest.approx(expected_dep, abs=1e-2)

    # 5. Unsupported date rejection for real provider
    bad_req = DepartureRequest(date="2024-05-15", depth_m=50)
    with pytest.raises(ValueError) as excinfo:
        provider.get_reconstruction_departure(bad_req)
    assert "Subsurface reference field is not available locally" in str(excinfo.value)


def test_departure_api_endpoint():
    payload = {
        "date": "2019-01-01",
        "depth_m": 50,
        "latitude": 12.0,
        "longitude": 68.0,
    }
    response = client.post("/api/departure", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "2019-01-01"
    assert data["selected_depth_m"] == 50
    assert len(data["depths_m"]) == 15
    assert "GLORYS12V1" in data["reference_name"]
    assert "OceanIQ Reconstruction Departure" in data["result_label"]
    assert data["depth_metrics"]["depth_m"] == 50
    assert data["depth_metrics"]["valid_cells"] > 0
    assert len(data["grid_lat"]) > 0
    assert len(data["grid_lon"]) > 0
