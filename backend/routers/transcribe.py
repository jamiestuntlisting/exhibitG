"""
API endpoints for Exhibit G transcription.
"""

import os
import io
import csv
import logging
import uuid
import tempfile
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db
from models.exhibit_g import ExhibitG, ExhibitGRow, TranscriptionRun, CellDisagreement
from services.pipeline import process_exhibit_g
from services.claude_corrector import iterative_grid_correction

logger = logging.getLogger(__name__)

router = APIRouter()

# ──────────────────────────────────────────────
# Pydantic schemas for API responses
# ──────────────────────────────────────────────


class ExhibitGRowSchema(BaseModel):
    id: int
    row_number: int
    cast_number: str
    cast_name: str
    character: str
    status_code: str
    makeup_time: str
    report_on_set: str
    dismiss_on_set: str
    dismiss_mu_hair: str
    nd_breakfast_out: str
    nd_breakfast_in: str
    first_meal_in: str
    first_meal_out: str
    second_meal_in: str
    second_meal_out: str
    leave_for_loc: str
    arrive_on_loc: str
    leave_loc: str
    arrive_hotel: str
    stunt_adjust: str
    mileage: str
    mpv_fc: str

    class Config:
        from_attributes = True


class DisagreementSchema(BaseModel):
    id: int
    row_number: int
    column_name: str
    values: dict
    resolved_value: Optional[str]
    is_resolved: bool

    class Config:
        from_attributes = True


class ExhibitGSummarySchema(BaseModel):
    id: int
    original_filename: str
    production_co: str
    series_title: str
    date: str
    status: str
    created_at: str
    row_count: int
    disagreement_count: int

    class Config:
        from_attributes = True


class ExhibitGDetailSchema(BaseModel):
    id: int
    original_filename: str
    corrected_image_url: str
    original_image_url: str
    skew_angle: Optional[float]
    production_co: str
    series_title: str
    episode_name: str
    prod_number: str
    date: str
    day_x_of_y: str
    contact: str
    phone: str
    status: str
    created_at: str
    rows: list[ExhibitGRowSchema]
    disagreements: list[DisagreementSchema]

    class Config:
        from_attributes = True


class RowUpdateSchema(BaseModel):
    cast_number: Optional[str] = None
    cast_name: Optional[str] = None
    character: Optional[str] = None
    status_code: Optional[str] = None
    makeup_time: Optional[str] = None
    report_on_set: Optional[str] = None
    dismiss_on_set: Optional[str] = None
    dismiss_mu_hair: Optional[str] = None
    nd_breakfast_out: Optional[str] = None
    nd_breakfast_in: Optional[str] = None
    first_meal_in: Optional[str] = None
    first_meal_out: Optional[str] = None
    second_meal_in: Optional[str] = None
    second_meal_out: Optional[str] = None
    leave_for_loc: Optional[str] = None
    arrive_on_loc: Optional[str] = None
    leave_loc: Optional[str] = None
    arrive_hotel: Optional[str] = None
    stunt_adjust: Optional[str] = None
    mileage: Optional[str] = None
    mpv_fc: Optional[str] = None


class HeaderUpdateSchema(BaseModel):
    production_co: Optional[str] = None
    series_title: Optional[str] = None
    episode_name: Optional[str] = None
    prod_number: Optional[str] = None
    date: Optional[str] = None
    day_x_of_y: Optional[str] = None
    contact: Optional[str] = None
    phone: Optional[str] = None


class ResolveDisagreementSchema(BaseModel):
    resolved_value: str


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────


@router.post("/upload")
async def upload_and_transcribe(
    file: UploadFile = File(...),
    use_paddle: bool = Query(True, description="Use PaddleOCR"),
    use_easyocr: bool = Query(True, description="Use EasyOCR"),
    use_claude: bool = Query(True, description="Use Claude Vision"),
    db: Session = Depends(get_db),
):
    """Upload an Exhibit G image and process it."""
    # Validate file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    # Read file content
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Process
    try:
        exhibit = process_exhibit_g(
            file_content=content,
            original_filename=file.filename or "unknown",
            db=db,
            use_paddle=use_paddle,
            use_easyocr=use_easyocr,
            use_claude=use_claude,
        )

        return _exhibit_to_detail(exhibit)

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("")
def list_transcriptions(db: Session = Depends(get_db)):
    """List all transcriptions."""
    exhibits = db.query(ExhibitG).order_by(ExhibitG.created_at.desc()).all()
    result = []
    for ex in exhibits:
        row_count = db.query(ExhibitGRow).filter_by(exhibit_g_id=ex.id).count()
        disagreement_count = db.query(CellDisagreement).filter_by(
            exhibit_g_id=ex.id, is_resolved=False
        ).count()
        result.append({
            "id": ex.id,
            "original_filename": ex.original_filename,
            "production_co": ex.production_co or "",
            "series_title": ex.series_title or "",
            "date": ex.date or "",
            "status": ex.status,
            "created_at": ex.created_at.isoformat() if ex.created_at else "",
            "row_count": row_count,
            "disagreement_count": disagreement_count,
        })
    return result


@router.get("/{exhibit_id}")
def get_transcription(exhibit_id: int, db: Session = Depends(get_db)):
    """Get full transcription detail."""
    exhibit = db.query(ExhibitG).filter_by(id=exhibit_id).first()
    if not exhibit:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return _exhibit_to_detail(exhibit)


@router.get("/{exhibit_id}/corrected-image")
def get_corrected_image(exhibit_id: int, db: Session = Depends(get_db)):
    """Serve the corrected/deskewed image."""
    exhibit = db.query(ExhibitG).filter_by(id=exhibit_id).first()
    if not exhibit or not exhibit.corrected_image_path:
        raise HTTPException(status_code=404, detail="Image not found")
    if not os.path.exists(exhibit.corrected_image_path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    return FileResponse(exhibit.corrected_image_path, media_type="image/png")


@router.get("/{exhibit_id}/original-image")
def get_original_image(exhibit_id: int, db: Session = Depends(get_db)):
    """Serve the original uploaded image."""
    exhibit = db.query(ExhibitG).filter_by(id=exhibit_id).first()
    if not exhibit or not exhibit.original_image_path:
        raise HTTPException(status_code=404, detail="Image not found")
    if not os.path.exists(exhibit.original_image_path):
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    ext = os.path.splitext(exhibit.original_image_path)[1].lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    media_type = media_map.get(ext, "application/octet-stream")
    return FileResponse(exhibit.original_image_path, media_type=media_type)


@router.put("/{exhibit_id}/rows/{row_id}")
def update_row(
    exhibit_id: int,
    row_id: int,
    update: RowUpdateSchema,
    db: Session = Depends(get_db),
):
    """Update a row (human edit)."""
    row = db.query(ExhibitGRow).filter_by(id=row_id, exhibit_g_id=exhibit_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Row not found")

    update_data = update.model_dump(exclude_none=True)
    for key, value in update_data.items():
        setattr(row, key, value)

    db.commit()
    db.refresh(row)
    return ExhibitGRowSchema.model_validate(row)


@router.put("/{exhibit_id}/header")
def update_header(
    exhibit_id: int,
    update: HeaderUpdateSchema,
    db: Session = Depends(get_db),
):
    """Update header fields."""
    exhibit = db.query(ExhibitG).filter_by(id=exhibit_id).first()
    if not exhibit:
        raise HTTPException(status_code=404, detail="Transcription not found")

    update_data = update.model_dump(exclude_none=True)
    for key, value in update_data.items():
        setattr(exhibit, key, value)

    db.commit()
    db.refresh(exhibit)
    return {"status": "updated"}


@router.put("/{exhibit_id}/resolve/{disagreement_id}")
def resolve_disagreement(
    exhibit_id: int,
    disagreement_id: int,
    body: ResolveDisagreementSchema,
    db: Session = Depends(get_db),
):
    """Resolve a cell disagreement with the chosen value."""
    disagreement = db.query(CellDisagreement).filter_by(
        id=disagreement_id, exhibit_g_id=exhibit_id
    ).first()
    if not disagreement:
        raise HTTPException(status_code=404, detail="Disagreement not found")

    disagreement.resolved_value = body.resolved_value
    disagreement.is_resolved = True

    # Also update the corresponding cell in the ExhibitGRow
    row = db.query(ExhibitGRow).filter_by(
        exhibit_g_id=exhibit_id,
        row_number=disagreement.row_number
    ).first()
    if row and hasattr(row, disagreement.column_name):
        setattr(row, disagreement.column_name, body.resolved_value)

    # Check if all disagreements are resolved
    remaining = db.query(CellDisagreement).filter_by(
        exhibit_g_id=exhibit_id, is_resolved=False
    ).count()

    if remaining == 0:
        exhibit = db.query(ExhibitG).filter_by(id=exhibit_id).first()
        if exhibit:
            exhibit.status = "complete"

    db.commit()
    return {"status": "resolved", "remaining_disagreements": remaining}


@router.get("/{exhibit_id}/export/csv")
def export_csv(exhibit_id: int, db: Session = Depends(get_db)):
    """Export transcription as CSV."""
    exhibit = db.query(ExhibitG).filter_by(id=exhibit_id).first()
    if not exhibit:
        raise HTTPException(status_code=404, detail="Transcription not found")

    rows = db.query(ExhibitGRow).filter_by(exhibit_g_id=exhibit_id).order_by(ExhibitGRow.row_number).all()

    # Build CSV
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    columns = [
        "#", "CAST", "CHARACTER", "STATUS",
        "MAKEUP/HAIR/WRDRBE", "REPORT ON SET", "DISMISS ON SET", "DISMISS MU/HAIR",
        "ND BRKFST OUT", "ND BRKFST IN",
        "1ST MEAL IN", "1ST MEAL OUT",
        "2ND MEAL IN", "2ND MEAL OUT",
        "LEAVE FOR LOC", "ARRIVE ON LOC", "LEAVE LOC", "ARRIVE AT HOTEL",
        "STUNT ADJ", "MILEAGE", "MPV/FC",
    ]
    writer.writerow(columns)

    for row in rows:
        writer.writerow([
            row.cast_number, row.cast_name, row.character, row.status_code,
            row.makeup_time, row.report_on_set, row.dismiss_on_set, row.dismiss_mu_hair,
            row.nd_breakfast_out, row.nd_breakfast_in,
            row.first_meal_in, row.first_meal_out,
            row.second_meal_in, row.second_meal_out,
            row.leave_for_loc, row.arrive_on_loc, row.leave_loc, row.arrive_hotel,
            row.stunt_adjust, row.mileage, row.mpv_fc,
        ])

    output.seek(0)

    filename = f"exhibit_g_{exhibit_id}_{exhibit.series_title or 'export'}.csv"
    filename = filename.replace(" ", "_")

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{exhibit_id}/engines/{row_number}/{column_name}")
def get_engine_readings(
    exhibit_id: int,
    row_number: int,
    column_name: str,
    db: Session = Depends(get_db),
):
    """Get all engine readings for a specific cell (for review modal)."""
    runs = db.query(TranscriptionRun).filter_by(
        exhibit_g_id=exhibit_id,
        row_number=row_number,
        column_name=column_name,
    ).all()

    return [
        {
            "engine": run.engine,
            "value": run.value,
            "confidence": run.confidence,
        }
        for run in runs
    ]


# ──────────────────────────────────────────────
# Grid Correction Testing Endpoints
# ──────────────────────────────────────────────

# Store for grid correction sessions (in-memory for simplicity)
_grid_correction_sessions: dict = {}


@router.post("/test-grid-correction")
async def test_grid_correction(file: UploadFile = File(...)):
    """Test the iterative grid correction algorithm.

    Runs the new grid-based correction and returns all intermediate steps
    for debugging and visualization.
    """
    # Validate file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    # Read file content
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Decode image
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Generate session ID
    session_id = str(uuid.uuid4())[:8]

    # Create output directory for this session
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "processed",
        f"grid_correction_{session_id}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Run iterative grid correction
    try:
        corrected_image, steps = iterative_grid_correction(
            image,
            save_intermediates=True,
            output_dir=output_dir
        )
    except Exception as e:
        logger.error(f"Grid correction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Grid correction failed: {str(e)}")

    # Store session info
    _grid_correction_sessions[session_id] = {
        "output_dir": output_dir,
        "steps": steps,
        "filename": file.filename,
    }

    # Convert steps to API response format
    response_steps = []
    for step in steps:
        response_steps.append({
            "step": step["step"],
            "name": step["name"],
            "description": step["description"],
            "image_url": f"/api/transcribe/grid-steps/{session_id}/{step['filename']}",
            "data": step["data"],
        })

    return {
        "session_id": session_id,
        "filename": file.filename,
        "steps": response_steps,
        "final_image_url": f"/api/transcribe/grid-steps/{session_id}/{steps[-1]['filename']}" if steps else None,
    }


@router.get("/grid-steps/{session_id}/{filename}")
def get_grid_step_image(session_id: str, filename: str):
    """Serve an intermediate step image from the grid correction process."""
    session = _grid_correction_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    filepath = os.path.join(session["output_dir"], filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(filepath, media_type="image/png")


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _exhibit_to_detail(exhibit: ExhibitG) -> dict:
    """Convert an ExhibitG ORM object to a detail response dict."""
    return {
        "id": exhibit.id,
        "original_filename": exhibit.original_filename,
        "corrected_image_url": f"/api/transcribe/{exhibit.id}/corrected-image",
        "original_image_url": f"/api/transcribe/{exhibit.id}/original-image",
        "skew_angle": exhibit.skew_angle,
        "production_co": exhibit.production_co or "",
        "series_title": exhibit.series_title or "",
        "episode_name": exhibit.episode_name or "",
        "prod_number": exhibit.prod_number or "",
        "date": exhibit.date or "",
        "day_x_of_y": exhibit.day_x_of_y or "",
        "contact": exhibit.contact or "",
        "phone": exhibit.phone or "",
        "status": exhibit.status,
        "created_at": exhibit.created_at.isoformat() if exhibit.created_at else "",
        "rows": [
            {
                "id": row.id,
                "row_number": row.row_number,
                "cast_number": row.cast_number or "",
                "cast_name": row.cast_name or "",
                "character": row.character or "",
                "status_code": row.status_code or "",
                "makeup_time": row.makeup_time or "",
                "report_on_set": row.report_on_set or "",
                "dismiss_on_set": row.dismiss_on_set or "",
                "dismiss_mu_hair": row.dismiss_mu_hair or "",
                "nd_breakfast_out": row.nd_breakfast_out or "",
                "nd_breakfast_in": row.nd_breakfast_in or "",
                "first_meal_in": row.first_meal_in or "",
                "first_meal_out": row.first_meal_out or "",
                "second_meal_in": row.second_meal_in or "",
                "second_meal_out": row.second_meal_out or "",
                "leave_for_loc": row.leave_for_loc or "",
                "arrive_on_loc": row.arrive_on_loc or "",
                "leave_loc": row.leave_loc or "",
                "arrive_hotel": row.arrive_hotel or "",
                "stunt_adjust": row.stunt_adjust or "",
                "mileage": row.mileage or "",
                "mpv_fc": row.mpv_fc or "",
            }
            for row in exhibit.rows
        ],
        "disagreements": [
            {
                "id": d.id,
                "row_number": d.row_number,
                "column_name": d.column_name,
                "values": d.values or {},
                "resolved_value": d.resolved_value,
                "is_resolved": d.is_resolved,
            }
            for d in exhibit.disagreements
        ],
    }
