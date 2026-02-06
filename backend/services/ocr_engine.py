"""
OCR engine wrappers for PaddleOCR and EasyOCR.
Each engine reads individual cell images and returns text + confidence.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded engine instances
_paddle_reader = None
_easy_reader = None


def _get_paddle_reader():
    """Lazy-initialize PaddleOCR reader."""
    global _paddle_reader
    if _paddle_reader is None:
        from paddleocr import PaddleOCR
        _paddle_reader = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False,
            use_gpu=False,
        )
        logger.info("PaddleOCR initialized.")
    return _paddle_reader


def _get_easy_reader():
    """Lazy-initialize EasyOCR reader."""
    global _easy_reader
    if _easy_reader is None:
        import easyocr
        _easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        logger.info("EasyOCR initialized.")
    return _easy_reader


def _preprocess_cell(cell_image: np.ndarray) -> np.ndarray:
    """Preprocess a cell image for better OCR accuracy."""
    if cell_image is None or cell_image.size == 0:
        return cell_image

    # Ensure grayscale
    if len(cell_image.shape) == 3:
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_image.copy()

    # Upscale small cells for better OCR
    h, w = gray.shape[:2]
    if h < 30 or w < 50:
        scale = max(30 / max(h, 1), 50 / max(w, 1), 2.0)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Enhance contrast
    gray = cv2.equalizeHist(gray)

    # Slight blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def read_cell_paddle(cell_image: np.ndarray) -> Tuple[str, float]:
    """
    Read a single cell image using PaddleOCR.

    Returns:
        Tuple of (text, confidence)
    """
    if cell_image is None or cell_image.size == 0:
        return ("", 0.0)

    try:
        reader = _get_paddle_reader()
        processed = _preprocess_cell(cell_image)

        # PaddleOCR expects BGR or grayscale
        result = reader.ocr(processed, cls=True)

        if result and result[0]:
            texts = []
            confidences = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    conf = line[1][1]
                    texts.append(text)
                    confidences.append(conf)
            if texts:
                combined_text = " ".join(texts).strip()
                avg_confidence = sum(confidences) / len(confidences)
                return (combined_text, avg_confidence)

        return ("", 0.0)

    except Exception as e:
        logger.warning(f"PaddleOCR error: {e}")
        return ("", 0.0)


def read_cell_easyocr(cell_image: np.ndarray) -> Tuple[str, float]:
    """
    Read a single cell image using EasyOCR.

    Returns:
        Tuple of (text, confidence)
    """
    if cell_image is None or cell_image.size == 0:
        return ("", 0.0)

    try:
        reader = _get_easy_reader()
        processed = _preprocess_cell(cell_image)

        result = reader.readtext(processed)

        if result:
            texts = []
            confidences = []
            for bbox, text, conf in result:
                texts.append(text)
                confidences.append(conf)
            if texts:
                combined_text = " ".join(texts).strip()
                avg_confidence = sum(confidences) / len(confidences)
                return (combined_text, avg_confidence)

        return ("", 0.0)

    except Exception as e:
        logger.warning(f"EasyOCR error: {e}")
        return ("", 0.0)


def read_cells_batch(cells: List, engine: str = "paddleocr") -> List[Dict]:
    """
    Read a batch of cell images using the specified engine.

    Args:
        cells: List of CellRegion objects
        engine: "paddleocr" or "easyocr"

    Returns:
        List of dicts with keys: row_index, col_index, column_name, text, confidence
    """
    read_fn = read_cell_paddle if engine == "paddleocr" else read_cell_easyocr
    results = []

    for cell in cells:
        text, confidence = read_fn(cell.image)
        results.append({
            "row_index": cell.row_index,
            "col_index": cell.col_index,
            "column_name": cell.column_name,
            "text": text,
            "confidence": confidence,
            "engine": engine,
        })

    return results


def read_all_cells(all_rows_cells: List[List], engine: str = "paddleocr") -> List[Dict]:
    """
    Read all cells from all rows using the specified engine.

    Args:
        all_rows_cells: List of lists of CellRegion objects
        engine: "paddleocr" or "easyocr"

    Returns:
        Flat list of cell reading results
    """
    all_results = []
    for row_cells in all_rows_cells:
        results = read_cells_batch(row_cells, engine)
        all_results.extend(results)

    return all_results
