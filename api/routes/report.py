"""API route for generating high-resolution PDF reconstruction reports."""

import logging
from fastapi import APIRouter, Depends, HTTPException, Response

from api.schemas.report import ReportPdfRequest
from api.schemas.departure import DepartureRequest
from api.services.pdf_report_service import generate_reconstruction_pdf
from api.services.inference_service import InferenceService, get_inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/report", tags=["Report Generation"])


@router.post(
    "/pdf",
    summary="Generate OceanIQ Technical PDF Report",
    description=(
        "Generates an authoritative, publication-quality technical PDF report for the current "
        "reconstruction using the exact data already returned. Does not run a second inference "
        "and does not fabricate any values."
    ),
    response_class=Response,
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Publication-quality multi-page PDF report file.",
        },
        500: {"description": "Internal error during PDF document generation."},
    },
)
def generate_report_pdf(
    payload: ReportPdfRequest,
    service: InferenceService = Depends(get_inference_service),
) -> Response:
    """Generate and return PDF report binary stream for the provided reconstruction."""
    try:
        # If departure was not provided in request, attempt lookup if date is the verified held-out evaluation date
        if payload.departure is None and payload.date == "2019-01-01":
            try:
                provider = service.get_provider()
                payload.departure = provider.get_reconstruction_departure(
                    DepartureRequest(
                        date=payload.date,
                        depth_m=100,
                        latitude=payload.latitude,
                        longitude=payload.longitude,
                    )
                )
            except Exception as dep_err:
                logger.debug(f"Departure context not available for report: {dep_err}")

        pdf_bytes = generate_reconstruction_pdf(payload)
        filename = f"OceanIQ_Report_{payload.date}_{payload.latitude:.2f}N_{payload.longitude:.2f}E.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "application/pdf",
            },
        )
    except Exception as exc:
        logger.error(f"Failed to generate PDF report: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assemble reconstruction PDF report: {str(exc)}",
        )
