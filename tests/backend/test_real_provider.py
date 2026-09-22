"""Unit tests for RealOceanEmbedProvider."""

import pytest
from api.schemas.reconstruction import ReconstructionRequest
from api.services.real_provider import RealOceanEmbedProvider


@pytest.fixture(scope="module")
def provider():
    return RealOceanEmbedProvider()


def test_provider_initialization(provider):
    assert provider.provider_name == "oceanembed_frozen"
    assert provider.is_mock is False
    assert provider.is_model_loaded is True


def test_real_inference_execution(provider):
    req = ReconstructionRequest(date="2019-01-01", latitude=18.5, longitude=88.25)
    resp = provider.predict_profile(req)

    assert resp.date == "2019-01-01"
    assert resp.latitude == 18.5
    assert resp.longitude == 88.25
    assert len(resp.depths_m) == 15
    assert len(resp.temperature_c) == 15
    assert resp.is_mock is False
    assert resp.model.name == "OceanEmbed Phase-1 Primary Model"
    assert resp.model.version == "phase1-frozen"
    assert resp.model.provider_type == "pytorch_checkpoint"
    assert resp.model.inference_time_ms > 0
    assert resp.embedding.vector_dim == 128
    assert isinstance(resp.embedding.pca_1, float)
    assert isinstance(resp.embedding.pca_2, float)
    assert resp.surface_context.sst_c > 0


def test_unsupported_date_raises_error(provider):
    # Enforces scientific integrity: synthetic data is NEVER fabricated for unobserved dates
    req = ReconstructionRequest(date="2023-06-15", latitude=18.5, longitude=88.25)
    with pytest.raises(ValueError) as excinfo:
        provider.predict_profile(req)
    assert "No preprocessed surface observations found" in str(excinfo.value)
    assert "synthetic surface data is not fabricated" in str(excinfo.value)


def test_derived_oceanographic_indices(provider):
    # Test D26 and MLD derivation
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    temps = [29.0, 28.8, 28.5, 27.5, 26.5, 24.0, 21.0, 18.0, 15.0, 13.0, 11.0, 9.0, 7.5, 6.0, 5.0]

    d26 = provider._compute_d26(depths, temps)
    assert d26 is not None
    assert 30.0 <= d26 <= 50.0

    mld = provider._compute_mld(depths, temps, threshold=0.5)
    assert mld is not None
    assert 5.0 <= mld <= 20.0


def test_real_argo_lookup_behavior(provider):
    # Real Argo float exists on 2019-01-01 near 17.88°N, 66.24°E
    argo = provider.find_nearby_argo("2019-01-01", 17.88, 66.24)
    assert argo is not None
    assert argo.is_mock is False
    assert "WMO-2902201" in argo.float_id
    assert len(argo.temperature_c) == 15

    # Coordinates far from any in-situ float return None (no fake fallback)
    argo_none = provider.find_nearby_argo("2019-01-01", 18.5, 88.25)
    assert argo_none is None



def test_embedding_scatter_points(provider):
    scatter = provider.get_embedding_scatter()
    assert scatter.is_mock is False
    assert scatter.total_points > 0
    assert len(scatter.points) == scatter.total_points
    assert len(scatter.regimes) > 0
