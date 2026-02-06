"""
Template detector for SAG-AFTRA Exhibit G forms.
Uses morphological operations to detect the table grid,
then maps detected columns to known Exhibit G column positions.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class CellRegion:
    """Represents a detected cell in the table."""
    row_index: int
    col_index: int
    column_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    image: Optional[np.ndarray] = field(default=None, repr=False)


# Known Exhibit G column layout
# These are approximate ratios of column boundaries within the table width
# Derived from the standard SAG-AFTRA Exhibit G form
EXHIBIT_G_COLUMNS = [
    {"name": "cast_number", "label": "#", "ratio_start": 0.000, "ratio_end": 0.030},
    {"name": "cast_name", "label": "CAST", "ratio_start": 0.030, "ratio_end": 0.130},
    {"name": "character", "label": "CHARACTER", "ratio_start": 0.130, "ratio_end": 0.240},
    {"name": "status_code", "label": "W/S/H/F/TR", "ratio_start": 0.240, "ratio_end": 0.290},
    {"name": "makeup_time", "label": "MAKE-UP HAIR WRDRBE", "ratio_start": 0.290, "ratio_end": 0.345},
    {"name": "report_on_set", "label": "REPORT ON SET", "ratio_start": 0.345, "ratio_end": 0.395},
    {"name": "dismiss_on_set", "label": "DISMISS ON SET", "ratio_start": 0.395, "ratio_end": 0.445},
    {"name": "dismiss_mu_hair", "label": "DISMISS MU/HAIR WRDRBE", "ratio_start": 0.445, "ratio_end": 0.505},
    {"name": "nd_breakfast_out", "label": "N.D. BRKFST OUT", "ratio_start": 0.505, "ratio_end": 0.540},
    {"name": "nd_breakfast_in", "label": "N.D. BRKFST IN", "ratio_start": 0.540, "ratio_end": 0.575},
    {"name": "first_meal_in", "label": "1ST MEAL IN", "ratio_start": 0.575, "ratio_end": 0.612},
    {"name": "first_meal_out", "label": "1ST MEAL OUT", "ratio_start": 0.612, "ratio_end": 0.650},
    {"name": "second_meal_in", "label": "2ND MEAL IN", "ratio_start": 0.650, "ratio_end": 0.688},
    {"name": "second_meal_out", "label": "2ND MEAL OUT", "ratio_start": 0.688, "ratio_end": 0.725},
    {"name": "leave_for_loc", "label": "LEAVE FOR LOC", "ratio_start": 0.725, "ratio_end": 0.770},
    {"name": "arrive_on_loc", "label": "ARRIVE ON LOC", "ratio_start": 0.770, "ratio_end": 0.815},
    {"name": "leave_loc", "label": "LEAVE LOC", "ratio_start": 0.815, "ratio_end": 0.855},
    {"name": "arrive_hotel", "label": "ARRIVE AT HOTEL", "ratio_start": 0.855, "ratio_end": 0.900},
    {"name": "stunt_adjust", "label": "STUNT ADJUST", "ratio_start": 0.900, "ratio_end": 0.935},
    {"name": "mileage", "label": "Mileage", "ratio_start": 0.935, "ratio_end": 0.960},
    {"name": "mpv_fc", "label": "MPV/FC", "ratio_start": 0.960, "ratio_end": 1.000},
]


def detect_lines(binary_image: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Detect horizontal and vertical lines in the binary image.
    Returns lists of y-coordinates (horizontal lines) and x-coordinates (vertical lines).
    """
    h, w = binary_image.shape[:2]

    # Invert if needed (we need black lines on white background to be white on black)
    if np.mean(binary_image) > 127:
        inverted = cv2.bitwise_not(binary_image)
    else:
        inverted = binary_image.copy()

    # Detect horizontal lines
    horiz_kernel_len = max(w // 30, 40)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
    horiz_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

    # Detect vertical lines
    vert_kernel_len = max(h // 30, 40)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
    vert_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vert_kernel, iterations=2)

    # Find horizontal line y-positions
    horiz_cnts, _ = cv2.findContours(horiz_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_positions = []
    for cnt in horiz_cnts:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Only consider lines that span a significant portion of the width
        if cw > w * 0.3:
            h_positions.append(y + ch // 2)

    # Find vertical line x-positions
    vert_cnts, _ = cv2.findContours(vert_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_positions = []
    for cnt in vert_cnts:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Only consider lines that span a significant portion of the height
        if ch > h * 0.1:
            v_positions.append(x + cw // 2)

    # Sort and deduplicate (merge lines within 10px of each other)
    h_positions = _merge_close_values(sorted(h_positions), threshold=10)
    v_positions = _merge_close_values(sorted(v_positions), threshold=10)

    return h_positions, v_positions


def _merge_close_values(values: List[int], threshold: int = 10) -> List[int]:
    """Merge values that are within threshold of each other."""
    if not values:
        return []

    merged = [values[0]]
    for v in values[1:]:
        if v - merged[-1] > threshold:
            merged.append(v)
        else:
            # Average with the last merged value
            merged[-1] = (merged[-1] + v) // 2
    return merged


def find_table_bounds(h_lines: List[int], v_lines: List[int],
                      image_h: int, image_w: int) -> Tuple[int, int, int, int]:
    """
    Find the bounding box of the main data table.
    Returns (x1, y1, x2, y2).
    """
    if not h_lines or not v_lines:
        # Fallback: use the whole image
        return (0, 0, image_w, image_h)

    # The table is typically the region with the most horizontal lines
    # Find the cluster of horizontal lines that are evenly spaced (the data rows)
    # The first few h_lines are header rows, the rest are data rows

    # Table x-bounds: leftmost and rightmost vertical lines
    x1 = v_lines[0]
    x2 = v_lines[-1]

    # Table y-bounds: we need to find where the data rows start
    # Typically the header area has 3-4 lines close together, then evenly spaced data rows
    y1 = h_lines[0]
    y2 = h_lines[-1]

    return (x1, y1, x2, y2)


def find_header_region(h_lines: List[int], table_y1: int,
                       image_w: int) -> Tuple[int, int, int, int]:
    """
    Find the region above the table that contains header fields.
    Returns (x1, y1, x2, y2).
    """
    # Header is everything above the first table horizontal line
    return (0, 0, image_w, max(0, table_y1 - 5))


def identify_data_rows(h_lines: List[int], table_bounds: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    """
    Identify the y-boundaries for each data row within the table.
    The header rows (column labels) are skipped.
    Returns list of (y_start, y_end) tuples for each data row.
    """
    _, table_y1, _, table_y2 = table_bounds

    # Filter horizontal lines within the table
    table_h_lines = [y for y in h_lines if table_y1 <= y <= table_y2]

    if len(table_h_lines) < 3:
        return []

    # Calculate row heights to find the header vs data boundary
    heights = [table_h_lines[i + 1] - table_h_lines[i] for i in range(len(table_h_lines) - 1)]

    if not heights:
        return []

    # The header typically has a taller row (for the column labels)
    # Data rows are more uniform in height
    median_height = sorted(heights)[len(heights) // 2]

    # Find where the uniform data rows start
    # Skip rows that are significantly different from median (likely headers)
    data_start_idx = 0
    for i, h in enumerate(heights):
        if abs(h - median_height) < median_height * 0.5:
            data_start_idx = i
            break

    # The header usually takes 2-3 rows at the top
    # We want to skip those and start from the data rows
    # Look for the first row that has typical data row height
    header_row_count = 0
    for i, h in enumerate(heights):
        if h > median_height * 1.3:
            header_row_count = i + 1
        else:
            break

    # If we couldn't determine header count, default to 3
    if header_row_count == 0:
        header_row_count = min(3, len(table_h_lines) - 1)

    # Build data row boundaries
    data_rows = []
    for i in range(header_row_count, len(table_h_lines) - 1):
        y_start = table_h_lines[i]
        y_end = table_h_lines[i + 1]
        # Only include rows with reasonable height
        if y_end - y_start > 10:
            data_rows.append((y_start, y_end))

    return data_rows


def extract_cells(image: np.ndarray, data_rows: List[Tuple[int, int]],
                  v_lines: List[int], table_bounds: Tuple[int, int, int, int]) -> List[List[CellRegion]]:
    """
    Extract individual cell images from the detected grid.
    Maps detected columns to known Exhibit G column names.
    """
    table_x1, _, table_x2, _ = table_bounds
    table_width = table_x2 - table_x1

    # Map detected vertical lines to column boundaries
    # Use the known Exhibit G column ratios as a guide
    column_boundaries = _map_columns_to_ratios(v_lines, table_x1, table_width)

    all_rows = []
    for row_idx, (y_start, y_end) in enumerate(data_rows):
        row_cells = []
        for col_idx, col_info in enumerate(column_boundaries):
            x1 = col_info["x_start"]
            x2 = col_info["x_end"]

            # Add small padding
            pad = 2
            cx1 = max(0, x1 + pad)
            cy1 = max(0, y_start + pad)
            cx2 = min(image.shape[1], x2 - pad)
            cy2 = min(image.shape[0], y_end - pad)

            cell_img = image[cy1:cy2, cx1:cx2]

            cell = CellRegion(
                row_index=row_idx,
                col_index=col_idx,
                column_name=col_info["name"],
                x1=cx1, y1=cy1, x2=cx2, y2=cy2,
                image=cell_img if cell_img.size > 0 else None
            )
            row_cells.append(cell)

        all_rows.append(row_cells)

    return all_rows


def _map_columns_to_ratios(v_lines: List[int], table_x1: int,
                            table_width: int) -> List[Dict]:
    """
    Map detected vertical lines to known Exhibit G column positions.
    Uses a greedy matching approach: for each expected column boundary,
    find the nearest detected vertical line.
    """
    if not v_lines or table_width <= 0:
        # Fallback: use pure ratio-based columns
        return [
            {
                "name": col["name"],
                "label": col["label"],
                "x_start": table_x1 + int(col["ratio_start"] * table_width),
                "x_end": table_x1 + int(col["ratio_end"] * table_width),
            }
            for col in EXHIBIT_G_COLUMNS
        ]

    # Calculate expected x-positions for all column boundaries
    all_boundaries = set()
    for col in EXHIBIT_G_COLUMNS:
        all_boundaries.add(col["ratio_start"])
        all_boundaries.add(col["ratio_end"])
    expected_positions = sorted(all_boundaries)
    expected_x = [table_x1 + int(r * table_width) for r in expected_positions]

    # For each expected boundary, snap to nearest detected vertical line
    snapped = []
    for ex in expected_x:
        closest = min(v_lines, key=lambda vl: abs(vl - ex))
        # Only snap if within 5% of table width
        if abs(closest - ex) < table_width * 0.05:
            snapped.append(closest)
        else:
            snapped.append(ex)

    # Build column boundaries from snapped positions
    # Map each column to its start and end from the sorted snapped positions
    result = []
    for col in EXHIBIT_G_COLUMNS:
        start_x = table_x1 + int(col["ratio_start"] * table_width)
        end_x = table_x1 + int(col["ratio_end"] * table_width)

        # Snap to nearest detected lines
        snap_start = min(v_lines, key=lambda vl: abs(vl - start_x), default=start_x)
        snap_end = min(v_lines, key=lambda vl: abs(vl - end_x), default=end_x)

        if abs(snap_start - start_x) < table_width * 0.05:
            start_x = snap_start
        if abs(snap_end - end_x) < table_width * 0.05:
            end_x = snap_end

        result.append({
            "name": col["name"],
            "label": col["label"],
            "x_start": start_x,
            "x_end": end_x,
        })

    return result


def detect_table(image: np.ndarray) -> Dict:
    """
    Main entry point: detect the Exhibit G table structure.

    Args:
        image: Grayscale or binary preprocessed image

    Returns:
        Dict with:
        - header_region: (x1, y1, x2, y2) for the header area
        - table_bounds: (x1, y1, x2, y2) for the main table
        - data_rows: list of (y_start, y_end) for each data row
        - cells: list of lists of CellRegion
        - h_lines: detected horizontal line positions
        - v_lines: detected vertical line positions
        - column_info: column boundary mappings
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Binarize for line detection
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    h, w = binary.shape[:2]

    # Detect lines
    h_lines, v_lines = detect_lines(binary)

    if not h_lines or not v_lines:
        return {
            "header_region": (0, 0, w, 0),
            "table_bounds": (0, 0, w, h),
            "data_rows": [],
            "cells": [],
            "h_lines": [],
            "v_lines": [],
            "column_info": [],
        }

    # Find table bounds
    table_bounds = find_table_bounds(h_lines, v_lines, h, w)

    # Find header region
    header_region = find_header_region(h_lines, table_bounds[1], w)

    # Identify data rows
    data_rows = identify_data_rows(h_lines, table_bounds)

    # Extract cells
    table_x1 = table_bounds[0]
    table_width = table_bounds[2] - table_bounds[0]
    column_info = _map_columns_to_ratios(v_lines, table_x1, table_width)

    cells = extract_cells(gray, data_rows, v_lines, table_bounds)

    return {
        "header_region": header_region,
        "table_bounds": table_bounds,
        "data_rows": data_rows,
        "cells": cells,
        "h_lines": h_lines,
        "v_lines": v_lines,
        "column_info": column_info,
    }
