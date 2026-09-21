"""Tests for reconstruction history service and API endpoint."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.history_service import HistoryService, get_history_service

client = TestClient(app)


@pytest.fixture
def test_history_service(tmp_path):
    """Provide an isolated history service backed by a temporary database."""
    test_db = tmp_path / "test_history.db"
    service = HistoryService(db_path=str(test_db))
    app.dependency_overrides[get_history_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_history_service, None)


@pytest.fixture(autouse=True)
def setup_history(test_history_service):
    """Ensure history service is initialized and clean before each test."""
    test_history_service.clear_history()
    yield test_history_service
    test_history_service.clear_history()



def test_history_service_save_and_retrieve(setup_history):
    service = setup_history
    item1 = service.save_reconstruction(
        date="2023-06-15",
        latitude=18.5,
        longitude=88.25,
        regime="Bay of Bengal (BoB)",
    )
    assert item1.id > 0
    assert item1.date == "2023-06-15"
    assert item1.latitude == 18.5
    assert item1.longitude == 88.25
    assert item1.regime == "Bay of Bengal (BoB)"
    assert item1.timestamp is not None

    item2 = service.save_reconstruction(
        date="2023-07-20",
        latitude=10.0,
        longitude=53.0,
        regime="Somali Upwelling Zone",
    )

    history = service.get_recent_history(limit=10)
    assert len(history) == 2
    # Newest item first (item2, then item1)
    assert history[0].id == item2.id
    assert history[0].regime == "Somali Upwelling Zone"
    assert history[1].id == item1.id
    assert history[1].regime == "Bay of Bengal (BoB)"


def test_history_api_endpoint(setup_history):
    service = setup_history
    service.save_reconstruction(
        date="2023-05-10",
        latitude=6.0,
        longitude=80.0,
        regime="Equatorial Warm Pool",
    )

    response = client.get("/api/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["regime"] == "Equatorial Warm Pool"
    assert data[0]["latitude"] == 6.0
    assert data[0]["longitude"] == 80.0
    assert data[0]["date"] == "2023-05-10"


def test_reconstruct_endpoint_auto_saves_to_history(setup_history):
    service = setup_history
    assert len(service.get_recent_history()) == 0

    payload = {
        "date": "2023-06-15",
        "latitude": 18.5,
        "longitude": 88.25,
    }
    reconstruct_resp = client.post("/api/reconstruct", json=payload)
    assert reconstruct_resp.status_code == 200
    recon_data = reconstruct_resp.json()

    # Verify history now contains the saved reconstruction automatically
    history_resp = client.get("/api/history")
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert len(history) == 1
    assert history[0]["date"] == "2023-06-15"
    assert history[0]["latitude"] == 18.5
    assert history[0]["longitude"] == 88.25
    assert history[0]["regime"] == recon_data["embedding"]["regime_label"]
    assert history[0]["timestamp"] == recon_data["timestamp"]
