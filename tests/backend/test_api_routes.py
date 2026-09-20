"""Integration tests for FastAPI REST routes."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "OceanEmbed API"
    assert "North Indian Ocean" in data["domain"]


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["active_provider"] == "mock_climatology"
    assert data["is_mock"] is True
    assert len(data["depths_m"]) == 15


def test_reconstruct_endpoint_valid():
    payload = {
        "date": "2023-06-15",
        "latitude": 18.5,
        "longitude": 88.25,
    }
    response = client.post("/api/reconstruct", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "2023-06-15"
    assert data["latitude"] == 18.5
    assert data["longitude"] == 88.25
    assert len(data["depths_m"]) == 15
    assert len(data["temperature_c"]) == 15
    assert data["is_mock"] is True
    assert "surface_context" in data
    assert "embedding" in data


def test_reconstruct_endpoint_out_of_bounds():
    # Lat 35 is out of North Indian Ocean domain (max 30.0)
    payload = {
        "date": "2023-06-15",
        "latitude": 35.0,
        "longitude": 88.25,
    }
    response = client.post("/api/reconstruct", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (validation error)


def test_embedding_endpoint():
    response = client.get("/api/embedding")
    assert response.status_code == 200
    data = response.json()
    assert data["total_points"] > 0
    assert len(data["points"]) > 0
    assert len(data["regimes"]) > 0


def test_argo_nearby_endpoint():
    response = client.get(
        "/api/argo/nearby",
        params={"date": "2023-06-15", "latitude": 18.5, "longitude": 88.25},
    )
    assert response.status_code == 200
    data = response.json()
    assert data is not None
    assert data["float_id"] == "SYNTHETIC-ARGO-DEMO-01"
    assert len(data["depths_m"]) == 15
