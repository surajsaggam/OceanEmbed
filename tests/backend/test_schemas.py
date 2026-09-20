"""Unit tests for Pydantic API schemas."""

import pytest
from pydantic import ValidationError
from api.schemas.reconstruction import ReconstructionRequest, ReconstructionResponse
from api.config import settings


def test_valid_reconstruction_request():
    req = ReconstructionRequest(
        date="2023-06-15",
        latitude=18.5,
        longitude=88.25,
    )
    assert req.date == "2023-06-15"
    assert req.latitude == 18.5
    assert req.longitude == 88.25


def test_invalid_date_format():
    with pytest.raises(ValidationError):
        ReconstructionRequest(
            date="15-06-2023",  # Non ISO format
            latitude=18.5,
            longitude=88.25,
        )


def test_latitude_out_of_bounds():
    # Below LAT_MIN (5.0)
    with pytest.raises(ValidationError):
        ReconstructionRequest(
            date="2023-06-15",
            latitude=4.9,
            longitude=88.25,
        )
    # Above LAT_MAX (30.0)
    with pytest.raises(ValidationError):
        ReconstructionRequest(
            date="2023-06-15",
            latitude=30.1,
            longitude=88.25,
        )


def test_longitude_out_of_bounds():
    # Below LON_MIN (45.0)
    with pytest.raises(ValidationError):
        ReconstructionRequest(
            date="2023-06-15",
            latitude=15.0,
            longitude=44.9,
        )
    # Above LON_MAX (105.0)
    with pytest.raises(ValidationError):
        ReconstructionRequest(
            date="2023-06-15",
            latitude=15.0,
            longitude=105.1,
        )


def test_standard_depths_count():
    assert len(settings.STANDARD_DEPTHS) == 15
    assert settings.STANDARD_DEPTHS == [
        0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
    ]
