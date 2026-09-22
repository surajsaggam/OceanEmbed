"""API route for generating high-resolution PDF reconstruction reports."""

import logging
from fastapi import APIRouter, HTTPException, Response

from api.schemas.reconstruction import ReconstructionResponse
from api.services.pdf_report_service import generate_reconstruction_pdf

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
def generate_report_pdf(reconstruction: ReconstructionResponse) -> Response:
    """Generate and return PDF report binary stream for the provided reconstruction."""
    try:
        pdf_bytes = generate_reconstruction_pdf(reconstruction)
        filename = f"OceanIQ_Report_{reconstruction.date}_{reconstruction.latitude:.2f}N_{reconstruction.longitude:.2f}E.pdf"

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
