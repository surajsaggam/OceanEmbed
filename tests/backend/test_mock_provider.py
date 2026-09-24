"""Unit tests for the MockInferenceProvider."""

import pytest
from api.services.mock_provider import MockInferenceProvider
from api.schemas.reconstruction import ReconstructionRequest
from api.config import settings


@pytest.fixture
def mock_provider():
    return MockInferenceProvider()


def test_mock_provider_properties(mock_provider):
    assert mock_provider.provider_name == "mock_climatology"
    assert mock_provider.is_mock is True


def test_predict_profile_structure(mock_provider):
    req = ReconstructionRequest(
        date="2023-06-15",
        latitude=18.5,
        longitude=88.25,
    )
    resp = mock_provider.predict_profile(req)

    assert resp.date == "2023-06-15"
    assert resp.latitude == 18.5
    assert resp.longitude == 88.25
    assert resp.depths_m == settings.STANDARD_DEPTHS
    assert len(resp.temperature_c) == 15

    # Check realistic oceanographic stratification (decreasing temperature with depth)
    assert resp.temperature_c[0] > resp.temperature_c[5]  # Surface warmer than 50m
    assert resp.temperature_c[5] > resp.temperature_c[10] # 50m warmer than 200m
    assert resp.temperature_c[10] > resp.temperature_c[14] # 200m warmer than 1000m
    assert 4.0 <= resp.temperature_c[14] <= 8.0           # 1000m deep water ~5-7°C

    # Surface context matches
    assert resp.surface_context.sst_c == pytest.approx(resp.temperature_c[0], abs=0.1)

    # Derived metrics
    assert resp.d26_depth_m is not None
    assert resp.mixed_layer_depth_m is not None
    assert resp.tchp_kj_cm2 is not None

    # Provenance
    assert resp.is_mock is True
    assert "Mock Provider" in resp.provenance


def test_argo_matchup_and_metrics(mock_provider):
    # Coordinate corresponding to our cataloged Bay of Bengal float (18.5N, 88.25E)
    req = ReconstructionRequest(
        date="2023-06-15",
        latitude=18.5,
        longitude=88.25,
    )
    resp = mock_provider.predict_profile(req)

    assert resp.argo_comparison is not None
    assert resp.argo_comparison.float_id == "SYNTHETIC-ARGO-DEMO-01"
    assert resp.argo_comparison.rmse is not None
    assert resp.argo_comparison.mae is not None
    assert resp.argo_comparison.bias is not None
    assert resp.argo_comparison.rmse >= 0.0


def test_embedding_scatter(mock_provider):
    scatter_resp = mock_provider.get_embedding_scatter()
    assert scatter_resp.total_points > 0
    assert len(scatter_resp.points) == scatter_resp.total_points
    assert len(scatter_resp.regimes) > 0
    assert scatter_resp.is_mock is True
