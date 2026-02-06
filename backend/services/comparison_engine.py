"""
Three-way comparison engine for OCR results.
Compares readings from PaddleOCR, EasyOCR, and Claude Vision.
Uses majority vote with Levenshtein distance for fuzzy matching.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Fallback Levenshtein distance implementation."""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]


# Time normalization patterns
TIME_PATTERNS = [
    # "730A" -> "7:30A"
    (r'^(\d{1,2})(\d{2})\s*([AaPp])[Mm]?$', r'\1:\2\3'),
    # "7:30 AM" -> "7:30A"
    (r'^(\d{1,2}:\d{2})\s*([AaPp])[Mm]?$', r'\1\2'),
    # "7:30am" -> "7:30A"
    (r'^(\d{1,2}:\d{2})\s*([AaPp])[Mm]$', r'\1\2'),
    # "12p" -> "12P"
    (r'^(\d{1,2})\s*([AaPp])[Mm]?$', r'\1\2'),
]


def normalize_time(value: str) -> str:
    """Normalize time strings for comparison."""
    if not value or not value.strip():
        return ""

    v = value.strip()

    for pattern, replacement in TIME_PATTERNS:
        match = re.match(pattern, v)
        if match:
            v = re.sub(pattern, replacement, v)
            break

    # Uppercase AM/PM indicator
    v = v.upper()

    return v


def normalize_value(value: str, column_name: str = "") -> str:
    """
    Normalize a cell value for comparison.

    Args:
        value: Raw OCR value
        column_name: Column name to determine normalization type

    Returns:
        Normalized string
    """
    if not value:
        return ""

    v = value.strip()

    # Time columns
    time_columns = {
        "makeup_time", "report_on_set", "dismiss_on_set", "dismiss_mu_hair",
        "nd_breakfast_out", "nd_breakfast_in",
        "first_meal_in", "first_meal_out",
        "second_meal_in", "second_meal_out",
        "leave_for_loc", "arrive_on_loc", "leave_loc", "arrive_hotel",
    }

    if column_name in time_columns:
        return normalize_time(v)

    # Name columns - normalize case
    name_columns = {"cast_name", "character"}
    if column_name in name_columns:
        # Capitalize first letter of each word
        return v.title()

    # Status code - uppercase
    if column_name == "status_code":
        return v.upper()

    return v


def are_values_similar(v1: str, v2: str, threshold: int = 1) -> bool:
    """
    Check if two normalized values are similar enough to be considered agreement.
    Uses Levenshtein distance with a threshold.
    """
    if v1 == v2:
        return True

    # Both empty
    if not v1 and not v2:
        return True

    # One empty, one not
    if not v1 or not v2:
        return False

    dist = levenshtein_distance(v1, v2)
    return dist <= threshold


def compare_three_way(readings: Dict[str, Dict]) -> Dict:
    """
    Compare readings from three engines for a single cell.

    Args:
        readings: Dict mapping engine name -> {"text": str, "confidence": float}

    Returns:
        Dict with:
        - final_value: the agreed-upon or best value
        - is_flagged: whether this cell has a disagreement
        - all_values: dict of all engine values
        - agreement_type: "unanimous", "majority", "high_confidence", "disagreement"
    """
    engines = list(readings.keys())
    values = {eng: readings[eng].get("text", "") for eng in engines}
    confidences = {eng: readings[eng].get("confidence", 0.0) for eng in engines}

    # Normalize all values
    # (column_name would be passed in a real call, but for now we normalize generically)
    norm_values = {eng: values[eng].strip() for eng in engines}

    # Check for unanimous agreement
    unique_normalized = set(norm_values.values())
    if len(unique_normalized) == 1:
        return {
            "final_value": values[engines[0]],
            "is_flagged": False,
            "all_values": values,
            "agreement_type": "unanimous",
        }

    # Check for majority (2 of 3 agree)
    if len(engines) >= 3:
        for i in range(len(engines)):
            for j in range(i + 1, len(engines)):
                if are_values_similar(norm_values[engines[i]], norm_values[engines[j]]):
                    # These two agree - use the one with higher confidence
                    winner = engines[i] if confidences[engines[i]] >= confidences[engines[j]] else engines[j]
                    return {
                        "final_value": values[winner],
                        "is_flagged": False,
                        "all_values": values,
                        "agreement_type": "majority",
                    }

    # Check for 2-engine case
    if len(engines) == 2:
        if are_values_similar(norm_values[engines[0]], norm_values[engines[1]]):
            winner = engines[0] if confidences[engines[0]] >= confidences[engines[1]] else engines[1]
            return {
                "final_value": values[winner],
                "is_flagged": False,
                "all_values": values,
                "agreement_type": "majority",
            }

    # No agreement - use the highest confidence value but flag for review
    best_engine = max(engines, key=lambda e: confidences[e])
    return {
        "final_value": values[best_engine],
        "is_flagged": True,
        "all_values": values,
        "agreement_type": "disagreement",
    }


def compare_all_cells(
    paddle_results: List[Dict],
    easy_results: List[Dict],
    claude_results: List[Dict],
    column_names: Optional[List[str]] = None
) -> List[Dict]:
    """
    Compare all cell readings from three engines.

    Args:
        paddle_results: List of cell readings from PaddleOCR
        easy_results: List of cell readings from EasyOCR
        claude_results: List of cell readings from Claude

    Returns:
        List of comparison results, one per cell
    """
    # Index results by (row_index, column_name)
    def index_results(results: List[Dict]) -> Dict:
        indexed = {}
        for r in results:
            key = (r["row_index"], r["column_name"])
            indexed[key] = {"text": r["text"], "confidence": r["confidence"]}
        return indexed

    paddle_idx = index_results(paddle_results)
    easy_idx = index_results(easy_results)
    claude_idx = index_results(claude_results)

    # Get all unique (row, column) keys
    all_keys = set()
    all_keys.update(paddle_idx.keys())
    all_keys.update(easy_idx.keys())
    all_keys.update(claude_idx.keys())

    comparisons = []
    for key in sorted(all_keys):
        row_index, column_name = key

        readings = {}
        if key in paddle_idx:
            readings["paddleocr"] = paddle_idx[key]
        if key in easy_idx:
            readings["easyocr"] = easy_idx[key]
        if key in claude_idx:
            readings["claude"] = claude_idx[key]

        if not readings:
            continue

        # Normalize before comparison
        norm_readings = {}
        for eng, rd in readings.items():
            norm_readings[eng] = {
                "text": normalize_value(rd["text"], column_name),
                "confidence": rd["confidence"],
            }

        result = compare_three_way(norm_readings)
        result["row_index"] = row_index
        result["column_name"] = column_name
        result["raw_values"] = {eng: readings[eng]["text"] for eng in readings}

        comparisons.append(result)

    return comparisons
