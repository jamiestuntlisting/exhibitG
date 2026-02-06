"""
Processing pipeline orchestrator for Exhibit G transcription.
Wires together: preprocessing → template detection → triple OCR → comparison → DB storage.
"""

import os
import uuid
import logging
from typing import Dict, Optional

from sqlalchemy.orm import Session

from services.image_preprocessor import preprocess_image
from services.template_detector import detect_table, EXHIBIT_G_COLUMNS
from services.ocr_engine import read_all_cells
from services.claude_ocr import read_with_claude, claude_results_to_cell_readings
from services.comparison_engine import compare_all_cells
from models.exhibit_g import ExhibitG, ExhibitGRow, TranscriptionRun, CellDisagreement

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

# Column names in order (matching the ExhibitGRow model fields)
ROW_COLUMNS = [
    "cast_number", "cast_name", "character", "status_code",
    "makeup_time", "report_on_set", "dismiss_on_set", "dismiss_mu_hair",
    "nd_breakfast_out", "nd_breakfast_in",
    "first_meal_in", "first_meal_out",
    "second_meal_in", "second_meal_out",
    "leave_for_loc", "arrive_on_loc", "leave_loc", "arrive_hotel",
    "stunt_adjust", "mileage", "mpv_fc",
]


def save_uploaded_file(file_content: bytes, original_filename: str) -> str:
    """Save an uploaded file to the uploads directory."""
    ext = os.path.splitext(original_filename)[1].lower()
    file_id = str(uuid.uuid4())[:8]
    saved_filename = f"{file_id}{ext}"
    saved_path = os.path.join(UPLOADS_DIR, saved_filename)

    with open(saved_path, "wb") as f:
        f.write(file_content)

    return saved_path


def process_exhibit_g(
    file_content: bytes,
    original_filename: str,
    db: Session,
    use_paddle: bool = True,
    use_easyocr: bool = True,
    use_claude: bool = True,
) -> ExhibitG:
    """
    Full processing pipeline for an Exhibit G image.

    Steps:
    1. Save uploaded file
    2. Preprocess (HEIC conversion, deskew, binarize, denoise)
    3. Detect table grid (template detection)
    4. Run OCR engines on detected cells
    5. Run Claude Vision on full image
    6. Compare all readings
    7. Store results in database

    Args:
        file_content: Raw file bytes
        original_filename: Original filename
        db: SQLAlchemy session
        use_paddle: Whether to use PaddleOCR
        use_easyocr: Whether to use EasyOCR
        use_claude: Whether to use Claude Vision

    Returns:
        ExhibitG database record with all relationships populated
    """
    # Step 1: Save uploaded file
    logger.info(f"Processing: {original_filename}")
    upload_path = save_uploaded_file(file_content, original_filename)

    # Step 2: Preprocess image
    logger.info("Step 2: Preprocessing image...")
    preprocess_result = preprocess_image(upload_path)
    corrected_path = preprocess_result["corrected_image_path"]
    color_corrected_path = preprocess_result["color_corrected_path"]
    skew_angle = preprocess_result["skew_angle"]
    gray_image = preprocess_result["gray_image"]

    # Create the ExhibitG record
    exhibit = ExhibitG(
        original_filename=original_filename,
        original_image_path=upload_path,
        corrected_image_path=color_corrected_path,
        skew_angle=skew_angle,
        status="processing",
    )
    db.add(exhibit)
    db.flush()  # Get the ID

    # Step 3: Detect table
    logger.info("Step 3: Detecting table structure...")
    table_info = detect_table(gray_image)
    cells = table_info["cells"]
    data_rows = table_info["data_rows"]

    # Store cell grid info
    cell_grid_data = {
        "table_bounds": table_info["table_bounds"],
        "h_lines": table_info["h_lines"],
        "v_lines": table_info["v_lines"],
        "data_rows": data_rows,
        "column_info": [
            {"name": ci["name"], "label": ci["label"],
             "x_start": ci["x_start"], "x_end": ci["x_end"]}
            for ci in table_info["column_info"]
        ] if table_info["column_info"] else [],
    }
    exhibit.cell_grid = cell_grid_data

    if not cells:
        logger.warning("No table cells detected. Falling back to Claude-only mode.")
        use_paddle = False
        use_easyocr = False

    # Step 4 & 5: Run OCR engines
    paddle_results = []
    easy_results = []
    claude_results = []

    # Flatten cells for OCR engines
    flat_cells = [cell for row in cells for cell in row]

    if use_paddle and flat_cells:
        logger.info(f"Step 4a: Running PaddleOCR on {len(flat_cells)} cells...")
        try:
            paddle_results = read_all_cells(cells, engine="paddleocr")
            # Store transcription runs
            for r in paddle_results:
                run = TranscriptionRun(
                    exhibit_g_id=exhibit.id,
                    engine="paddleocr",
                    row_number=r["row_index"],
                    column_name=r["column_name"],
                    value=r["text"],
                    confidence=r["confidence"],
                )
                db.add(run)
            logger.info(f"PaddleOCR: {len(paddle_results)} cell readings.")
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")

    if use_easyocr and flat_cells:
        logger.info(f"Step 4b: Running EasyOCR on {len(flat_cells)} cells...")
        try:
            easy_results = read_all_cells(cells, engine="easyocr")
            for r in easy_results:
                run = TranscriptionRun(
                    exhibit_g_id=exhibit.id,
                    engine="easyocr",
                    row_number=r["row_index"],
                    column_name=r["column_name"],
                    value=r["text"],
                    confidence=r["confidence"],
                )
                db.add(run)
            logger.info(f"EasyOCR: {len(easy_results)} cell readings.")
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")

    if use_claude:
        logger.info("Step 5: Running Claude Vision OCR...")
        try:
            claude_raw = read_with_claude(color_corrected_path)
            claude_results = claude_results_to_cell_readings(claude_raw)

            # Store header data from Claude
            header = claude_raw.get("header", {})
            exhibit.production_co = header.get("production_co", "")
            exhibit.series_title = header.get("series_title", "")
            exhibit.episode_name = header.get("episode_name", "")
            exhibit.prod_number = header.get("prod_number", "")
            exhibit.date = header.get("date", "")
            exhibit.day_x_of_y = header.get("day_x_of_y", "")
            exhibit.contact = header.get("contact", "")
            exhibit.phone = header.get("phone", "")

            for r in claude_results:
                run = TranscriptionRun(
                    exhibit_g_id=exhibit.id,
                    engine="claude",
                    row_number=r["row_index"],
                    column_name=r["column_name"],
                    value=r["text"],
                    confidence=r["confidence"],
                )
                db.add(run)
            logger.info(f"Claude: {len(claude_results)} cell readings.")
        except Exception as e:
            logger.error(f"Claude OCR failed: {e}")

    # Step 6: Compare readings
    logger.info("Step 6: Comparing OCR results...")
    comparisons = compare_all_cells(paddle_results, easy_results, claude_results)

    # Step 7: Build ExhibitGRow records from comparisons
    # Group comparisons by row
    rows_data: Dict[int, Dict[str, str]] = {}
    flagged_cells = []

    for comp in comparisons:
        row_idx = comp["row_index"]
        col_name = comp["column_name"]

        if row_idx not in rows_data:
            rows_data[row_idx] = {}

        rows_data[row_idx][col_name] = comp["final_value"]

        if comp["is_flagged"]:
            flagged_cells.append(comp)

    # Create ExhibitGRow records
    for row_idx in sorted(rows_data.keys()):
        row_vals = rows_data[row_idx]

        # Skip completely empty rows
        has_content = any(v.strip() for v in row_vals.values() if v)
        if not has_content:
            continue

        row = ExhibitGRow(
            exhibit_g_id=exhibit.id,
            row_number=row_idx,
            cast_number=row_vals.get("cast_number", ""),
            cast_name=row_vals.get("cast_name", ""),
            character=row_vals.get("character", ""),
            status_code=row_vals.get("status_code", ""),
            makeup_time=row_vals.get("makeup_time", ""),
            report_on_set=row_vals.get("report_on_set", ""),
            dismiss_on_set=row_vals.get("dismiss_on_set", ""),
            dismiss_mu_hair=row_vals.get("dismiss_mu_hair", ""),
            nd_breakfast_out=row_vals.get("nd_breakfast_out", ""),
            nd_breakfast_in=row_vals.get("nd_breakfast_in", ""),
            first_meal_in=row_vals.get("first_meal_in", ""),
            first_meal_out=row_vals.get("first_meal_out", ""),
            second_meal_in=row_vals.get("second_meal_in", ""),
            second_meal_out=row_vals.get("second_meal_out", ""),
            leave_for_loc=row_vals.get("leave_for_loc", ""),
            arrive_on_loc=row_vals.get("arrive_on_loc", ""),
            leave_loc=row_vals.get("leave_loc", ""),
            arrive_hotel=row_vals.get("arrive_hotel", ""),
            stunt_adjust=row_vals.get("stunt_adjust", ""),
            mileage=row_vals.get("mileage", ""),
            mpv_fc=row_vals.get("mpv_fc", ""),
        )
        db.add(row)

    # Create CellDisagreement records for flagged cells
    for comp in flagged_cells:
        disagreement = CellDisagreement(
            exhibit_g_id=exhibit.id,
            row_number=comp["row_index"],
            column_name=comp["column_name"],
            values=comp.get("all_values", comp.get("raw_values", {})),
            is_resolved=False,
        )
        db.add(disagreement)

    # Update status
    if flagged_cells:
        exhibit.status = "review"
        logger.info(f"Completed with {len(flagged_cells)} disagreements requiring review.")
    else:
        exhibit.status = "complete"
        logger.info("Completed with no disagreements.")

    db.commit()
    db.refresh(exhibit)

    return exhibit
