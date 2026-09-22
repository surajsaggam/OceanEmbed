"""Tests for OceanIQ PDF report generation endpoint and service."""

import re
from fastapi.testclient import TestClient
from api.main import app
from api.schemas.reconstruction import ReconstructionResponse
from api.services.pdf_report_service import generate_reconstruction_pdf

client = TestClient(app)


def test_generate_report_pdf_endpoint():
    """Test generating a technical PDF report from a real reconstruction."""
    recon_res = client.post(
        "/api/reconstruct",
        json={"date": "2019-01-01", "latitude": 18.5, "longitude": 88.25},
    )
    assert recon_res.status_code == 200
    recon_data = recon_res.json()

    # Request PDF generation using the exact reconstruction data
    pdf_res = client.post("/api/report/pdf", json=recon_data)
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert "OceanIQ_Report_2019-01-01" in pdf_res.headers["content-disposition"]

    # Verify valid PDF signature
    content = pdf_res.content
    assert content.startswith(b"%PDF-")
    assert len(content) > 10_000


def test_generate_report_pdf_direct_service():
    """Directly test the pdf_report_service with both None Argo and active Argo comparison."""
    recon_res = client.post(
        "/api/reconstruct",
        json={"date": "2019-01-01", "latitude": 18.5, "longitude": 88.25},
    )
    assert recon_res.status_code == 200
    recon = ReconstructionResponse(**recon_res.json())

    # Generate without Argo
    pdf_bytes_no_argo = generate_reconstruction_pdf(recon)
    assert pdf_bytes_no_argo.startswith(b"%PDF-")
    assert len(pdf_bytes_no_argo) > 10_000

    # Add mock Argo to test Argo comparison rendering path
    from api.schemas.argo import ArgoObservation
    recon.argo_comparison = ArgoObservation(
        float_id="WMO-2902145",
        date="2019-01-01",
        latitude=18.6,
        longitude=88.3,
        distance_km=14.2,
        depths_m=recon.depths_m,
        temperature_c=[t + 0.15 for t in recon.temperature_c],
        rmse=0.28,
        mae=0.22,
        bias=0.08,
        is_mock=True,
    )

    pdf_bytes_with_argo = generate_reconstruction_pdf(recon)
    assert pdf_bytes_with_argo.startswith(b"%PDF-")
    assert len(pdf_bytes_with_argo) > 10_000


def test_pdf_content_corrections():
    """Verify that the generated PDF contains all updated wording and no hardcoded mission names."""
    recon_res = client.post(
        "/api/reconstruct",
        json={"date": "2019-01-01", "latitude": 18.5, "longitude": 88.25},
    )
    assert recon_res.status_code == 200
    recon = ReconstructionResponse(**recon_res.json())
    pdf_bytes = generate_reconstruction_pdf(recon)
    pdf_text = pdf_bytes.decode("latin-1", errors="ignore")

    # 1. Depth matrix section renamed
    assert "Subsurface Temperature Structure" in pdf_text
    assert "Depth Matrix Analysis" not in pdf_text

    # 2. Embedding conservative wording
    assert "128-D Latent Representation" in pdf_text
    assert "Contextual Reference Points" in pdf_text

    # 3. Provenance and neutral sources
    assert "OceanIQ observation archive" in pdf_text
    for forbidden in ["OSTIA", "SMAP", "CMEMS", "OSCAR", "ERA5", "ASCAT"]:
        assert forbidden not in pdf_text, f"Forbidden source {forbidden} found in PDF!"

    # 4. Strict 4-page structure
    page_count = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
    assert page_count == 4, f"Expected 4 pages, got {page_count}"


def test_cold_profile_d26_not_reached():
    """Verify that when temperature never reaches 26C, D26 is NOT reported as 0.0 m."""
    recon_res = client.post(
        "/api/reconstruct",
        json={"date": "2019-01-01", "latitude": 18.5, "longitude": 88.25},
    )
    assert recon_res.status_code == 200
    r_dict = recon_res.json()

    # Create cold profile where max temp < 26.0 C
    r_dict["temperature_c"] = [
        24.0, 23.5, 23.0, 22.0, 21.0, 19.0, 17.0, 15.0, 13.0, 11.0, 9.0, 8.0, 7.0, 6.0, 5.0
    ]
    r_dict["d26_depth_m"] = None

    cold_recon = ReconstructionResponse(**r_dict)
    pdf_bytes = generate_reconstruction_pdf(cold_recon)
    pdf_text = pdf_bytes.decode("latin-1", errors="ignore")

    assert "Not reached" in pdf_text, "Expected 'Not reached' for cold profile D26"
    assert "0.0 m" not in pdf_text, "D26 must not be reported as 0.0 m"
