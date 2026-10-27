"""PDF report generation."""

from __future__ import annotations

from typing import Any

from fpdf import FPDF

from config import DISCLAIMER


def build_report(
    doctor_name: str,
    patient: dict[str, Any],
    recording_date: str,
    prediction: dict[str, Any],
) -> bytes:
    """Generate a PDF report and return its bytes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AeroVita Local - Lung Sound Screening Report", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Doctor: {doctor_name}", ln=True)
    pdf.cell(0, 8, f"Date: {recording_date}", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Patient", ln=True)
    pdf.set_font("Helvetica", size=11)
    for label in ("name", "national_id", "gender", "birth_date", "hometown", "phone"):
        pdf.cell(0, 7, f"{label.replace('_', ' ').title()}: {patient.get(label, '')}", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    status = "Healthy" if prediction["is_healthy"] else "Abnormal"
    pdf.cell(0, 8, f"Result: {status}", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, f"Healthy score: {prediction['healthy_score']:.2%}", ln=True)
    for label, probability in prediction.get("top_predictions", []):
        pdf.cell(0, 7, f"{label}: {probability:.2%}", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, DISCLAIMER)
    return bytes(pdf.output())
