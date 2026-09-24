"""Tests for 2D Vertical Subsurface Transect endpoint and provider logic."""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.schemas.transect import TransectRequest, TransectPoint
from api.services.mock_provider import MockInferenceProvider

client = TestClient(app)


def test_mock_provider_predict_transect():
    provider = MockInferenceProvider()
    req = TransectRequest(
        date="2019-01-01",
        points=[
            TransectPoint(latitude=15.0, longitude=60.0),
            TransectPoint(latitude=15.0, longitude=72.0),
        ],
        num_samples=10,
    )
    res = provider.predict_transect(req)
    assert res.date == "2019-01-01"
    assert res.total_distance_km > 0
    assert len(res.stations) == 10
    assert len(res.depths_m) == 15
    assert res.stations[0].distance_km == 0.0
    assert res.stations[-1].distance_km == pytest.approx(res.total_distance_km, abs=0.5)

    # Ocean station check
    ocean_stations = [s for s in res.stations if s.is_valid_ocean]
    assert len(ocean_stations) > 0
    for s in ocean_stations:
        assert len(s.temperature_c) == 15
        assert s.sst_c is not None


def test_transect_api_endpoint():
    payload = {
        "date": "2019-01-01",
        "points": [
            {"latitude": 12.0, "longitude": 65.0},
            {"latitude": 18.0, "longitude": 70.0},
        ],
        "num_samples": 8,
    }
    response = client.post("/api/transect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "2019-01-01"
    assert len(data["depths_m"]) == 15
    assert len(data["stations"]) == 8
    assert data["total_distance_km"] > 0


def test_transect_validation_min_points():
    payload = {
        "date": "2019-01-01",
        "points": [{"latitude": 15.0, "longitude": 60.0}],
    }
    response = client.post("/api/transect", json=payload)
    assert response.status_code == 422  # Pydantic validation error: min_length=2


def test_real_provider_predict_transect():
    from api.services.real_provider import RealOceanEmbedProvider
    provider = RealOceanEmbedProvider()
    if not provider.is_model_loaded:
        pytest.skip("Frozen checkpoint not present in test environment.")

    req = TransectRequest(
        date="2019-01-01",
        points=[
            TransectPoint(latitude=15.0, longitude=65.0),
            TransectPoint(latitude=15.0, longitude=75.0),
        ],
        num_samples=12,
    )
    res = provider.predict_transect(req)
    assert res.date == "2019-01-01"
    assert res.is_mock is False
    assert len(res.stations) == 12
    assert len(res.depths_m) == 15
    assert res.total_distance_km > 500

    # Ensure real ocean stations have 15 temperatures within plausible NIO range
    ocean_stations = [s for s in res.stations if s.is_valid_ocean]
    assert len(ocean_stations) > 0
    for s in ocean_stations:
        assert len(s.temperature_c) == 15
        assert 0.0 < s.temperature_c[0] < 35.0
        assert 0.0 < s.temperature_c[-1] < 15.0

