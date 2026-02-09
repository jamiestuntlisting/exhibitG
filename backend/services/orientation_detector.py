"""
Multi-faceted orientation detection for Exhibit G forms.

Uses four approaches and combines them for robust detection:
1. OCR-based: Look for readable text like "SAG-AFTRA", "Exhibit G"
2. Logo detection: Find SAG-AFTRA logo position
3. Template matching: Compare against known Exhibit G layout
4. Claude Vision API: Ask Claude to determine orientation

Each method can return None if it can't determine orientation.
Results are combined using a voting/confidence system.
"""

import cv2
import numpy as np
import math
import logging
import os
import base64
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


def _convert_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy(v) for v in obj)
    return obj


class Orientation(Enum):
    """Possible document orientations."""
    CORRECT = 0      # No rotation needed
    ROTATE_90 = 90   # Rotate 90° counter-clockwise
    ROTATE_180 = 180 # Rotate 180°
    ROTATE_270 = 270 # Rotate 270° counter-clockwise (90° clockwise)


@dataclass
class OrientationResult:
    """Result from a single orientation detection method."""
    method: str
    orientation: Optional[Orientation]
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class OrientationAnalysis:
    """Combined analysis from all orientation detection methods."""
    recommended_orientation: Orientation
    overall_confidence: float
    results: List[OrientationResult]
    voting_breakdown: Dict[str, int]


def detect_orientation_ocr(image: np.ndarray) -> OrientationResult:
    """
    Use OCR (Tesseract) to detect text orientation.

    Looks for key phrases like "SAG-AFTRA", "Exhibit G", "PRODUCTION TIME REPORT"
    and checks if they're readable in different orientations.
    """
    method = "ocr"

    try:
        import pytesseract
    except ImportError:
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={"error": "pytesseract not installed"},
            error="pytesseract not installed"
        )

    # Key phrases to look for (case-insensitive)
    key_phrases = [
        "sag-aftra", "sag aftra", "sagaftra",
        "exhibit g", "exhibit-g",
        "production time report",
        "performers",
        "production no"
    ]

    results_by_rotation = {}
    img_h, img_w = image.shape[:2]

    # Test each orientation
    for rotation in [0, 90, 180, 270]:
        if rotation == 0:
            test_img = image
        elif rotation == 90:
            test_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180:
            test_img = cv2.rotate(image, cv2.ROTATE_180)
        else:  # 270
            test_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Convert to grayscale if needed
        if len(test_img.shape) == 3:
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = test_img

        # Run OCR
        try:
            text = pytesseract.image_to_string(gray, config='--psm 6').lower()

            # Count how many key phrases are found
            matches = []
            for phrase in key_phrases:
                if phrase in text:
                    matches.append(phrase)

            results_by_rotation[rotation] = {
                "matches": matches,
                "match_count": len(matches),
                "text_sample": text[:200]
            }
        except Exception as e:
            results_by_rotation[rotation] = {
                "matches": [],
                "match_count": 0,
                "error": str(e)
            }

    # Find the rotation with most matches
    best_rotation = max(results_by_rotation.keys(),
                        key=lambda r: results_by_rotation[r]["match_count"])
    best_count = results_by_rotation[best_rotation]["match_count"]

    # Calculate confidence based on match count difference
    counts = [results_by_rotation[r]["match_count"] for r in [0, 90, 180, 270]]
    max_count = max(counts)
    second_max = sorted(counts, reverse=True)[1] if len(counts) > 1 else 0

    if max_count == 0:
        # No matches found in any orientation
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={"results_by_rotation": results_by_rotation, "reason": "No key phrases found"}
        )

    # Confidence is higher if there's a clear winner
    confidence = min(1.0, (max_count - second_max) / max(max_count, 1) + 0.3 * max_count / len(key_phrases))

    return OrientationResult(
        method=method,
        orientation=Orientation(best_rotation),
        confidence=confidence,
        details={
            "results_by_rotation": results_by_rotation,
            "best_rotation": best_rotation,
            "matches_found": results_by_rotation[best_rotation]["matches"]
        }
    )


def detect_orientation_logo(image: np.ndarray) -> OrientationResult:
    """
    Detect SAG-AFTRA logo position to determine orientation.

    The logo should be in the top-left corner of a correctly oriented form.
    Uses template matching with a stored logo template.
    """
    method = "logo"

    # Path to logo template (we'll need to create this)
    template_path = os.path.join(os.path.dirname(__file__), "templates", "sag_aftra_logo.png")

    if not os.path.exists(template_path):
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={"error": f"Logo template not found at {template_path}"},
            error="Logo template not found"
        )

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={"error": "Could not load logo template"},
            error="Could not load logo template"
        )

    # Convert image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    img_h, img_w = gray.shape[:2]
    results_by_rotation = {}

    # Test each orientation
    for rotation in [0, 90, 180, 270]:
        if rotation == 0:
            test_img = gray
        elif rotation == 90:
            test_img = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180:
            test_img = cv2.rotate(gray, cv2.ROTATE_180)
        else:  # 270
            test_img = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

        test_h, test_w = test_img.shape[:2]

        # Try multiple scales
        best_match = 0
        best_location = None

        for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
            scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
            if scaled_template.shape[0] > test_h or scaled_template.shape[1] > test_w:
                continue

            result = cv2.matchTemplate(test_img, scaled_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_match:
                best_match = max_val
                best_location = max_loc

        # Check if logo is in expected position (top-left quadrant)
        in_top_left = False
        if best_location:
            x, y = best_location
            in_top_left = (x < test_w * 0.4) and (y < test_h * 0.3)

        results_by_rotation[rotation] = {
            "match_score": best_match,
            "location": best_location,
            "in_top_left": in_top_left,
            "score_if_correct": best_match if in_top_left else best_match * 0.3
        }

    # Find best orientation (highest score with logo in correct position)
    best_rotation = max(results_by_rotation.keys(),
                        key=lambda r: results_by_rotation[r]["score_if_correct"])
    best_result = results_by_rotation[best_rotation]

    if best_result["match_score"] < 0.5:
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={"results_by_rotation": results_by_rotation, "reason": "No good logo match found"}
        )

    confidence = best_result["match_score"] * (1.0 if best_result["in_top_left"] else 0.5)

    return OrientationResult(
        method=method,
        orientation=Orientation(best_rotation),
        confidence=confidence,
        details={
            "results_by_rotation": results_by_rotation,
            "best_rotation": best_rotation
        }
    )


def detect_orientation_template(image: np.ndarray) -> OrientationResult:
    """
    Use template matching against known Exhibit G layout features.

    Looks for structural features like:
    - Header row position
    - Grid structure
    - Column layout
    """
    method = "template"

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    img_h, img_w = gray.shape[:2]
    results_by_rotation = {}

    for rotation in [0, 90, 180, 270]:
        if rotation == 0:
            test_img = gray
        elif rotation == 90:
            test_img = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180:
            test_img = cv2.rotate(gray, cv2.ROTATE_180)
        else:  # 270
            test_img = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

        test_h, test_w = test_img.shape[:2]

        # Exhibit G characteristics for correct orientation:
        # 1. Should be wider than tall (landscape)
        # 2. Dark header area in top portion
        # 3. Grid lines more dense horizontally

        is_landscape = test_w > test_h

        # Check for header darkness in top vs bottom
        top_region = test_img[:test_h//4, :]
        bottom_region = test_img[3*test_h//4:, :]
        top_darkness = np.mean(255 - top_region)
        bottom_darkness = np.mean(255 - bottom_region)
        header_at_top = top_darkness > bottom_darkness

        # Check horizontal vs vertical line density using edge detection
        edges = cv2.Canny(test_img, 50, 150)

        # Horizontal projection (sum along rows) - high variance means horizontal lines
        h_projection = np.sum(edges, axis=1)
        h_variance = np.var(h_projection)

        # Vertical projection (sum along columns)
        v_projection = np.sum(edges, axis=0)
        v_variance = np.var(v_projection)

        # For correct orientation, horizontal variance should be higher
        # (more distinct horizontal lines in a table)
        h_dominant = h_variance > v_variance

        score = 0
        if is_landscape:
            score += 0.4
        if header_at_top:
            score += 0.3
        if h_dominant:
            score += 0.3

        results_by_rotation[rotation] = {
            "is_landscape": bool(is_landscape),
            "header_at_top": bool(header_at_top),
            "h_dominant": bool(h_dominant),
            "top_darkness": float(top_darkness),
            "bottom_darkness": float(bottom_darkness),
            "h_variance": float(h_variance),
            "v_variance": float(v_variance),
            "score": float(score)
        }

    # Find best rotation
    best_rotation = max(results_by_rotation.keys(),
                        key=lambda r: results_by_rotation[r]["score"])
    best_score = results_by_rotation[best_rotation]["score"]

    # Check if there's a clear winner
    scores = [results_by_rotation[r]["score"] for r in [0, 90, 180, 270]]
    second_best = sorted(scores, reverse=True)[1]

    if best_score < 0.5 or (best_score - second_best) < 0.2:
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={"results_by_rotation": results_by_rotation, "reason": "No clear winner"}
        )

    confidence = min(1.0, best_score * (best_score - second_best + 0.5))

    # Swap 90 and 270 to correct CW/CCW confusion
    if best_rotation == 90:
        best_rotation = 270
    elif best_rotation == 270:
        best_rotation = 90

    return OrientationResult(
        method=method,
        orientation=Orientation(best_rotation),
        confidence=confidence,
        details={
            "results_by_rotation": results_by_rotation,
            "best_rotation": best_rotation
        }
    )


def detect_orientation_claude(image: np.ndarray, api_key: Optional[str] = None) -> OrientationResult:
    """
    Use Claude Vision API to determine document orientation.

    Sends the image to Claude and asks it to determine the correct orientation.
    """
    method = "claude_vision"

    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("ORIENTATION_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={"error": "No Anthropic API key available"},
            error="No API key"
        )

    try:
        import anthropic
    except ImportError:
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={"error": "anthropic package not installed"},
            error="anthropic not installed"
        )

    try:
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": """Analyze this image of a SAG-AFTRA Exhibit G form (Production Time Report).

Determine the document's orientation. The correct orientation should have:
- "SAG-AFTRA" logo in the top-left
- "Exhibit G" label visible
- Text readable left-to-right
- The form header at the top

Respond with ONLY a JSON object (no markdown, no explanation):
{
  "rotation_needed": <0, 90, 180, or 270>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation>"
}

Where rotation_needed is the counter-clockwise rotation in degrees needed to make the document correctly oriented.
- 0 = already correct
- 90 = rotate 90° counter-clockwise
- 180 = rotate 180°
- 270 = rotate 90° clockwise (270° counter-clockwise)"""
                        }
                    ]
                }
            ]
        )

        response_text = message.content[0].text.strip()

        # Parse JSON response
        import json
        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()

        result = json.loads(response_text)

        rotation = int(result.get("rotation_needed", 0))
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")

        if rotation not in [0, 90, 180, 270]:
            rotation = 0
            confidence = 0.0

        # Swap 90 and 270 to correct CW/CCW confusion
        if rotation == 90:
            rotation = 270
        elif rotation == 270:
            rotation = 90

        return OrientationResult(
            method=method,
            orientation=Orientation(rotation),
            confidence=confidence,
            details={
                "rotation_needed": rotation,
                "reasoning": reasoning,
                "raw_response": response_text
            }
        )

    except json.JSONDecodeError as e:
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={"error": f"Failed to parse Claude response: {str(e)}", "raw_response": response_text},
            error=f"JSON parse error: {str(e)}"
        )
    except Exception as e:
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={"error": str(e)},
            error=str(e)
        )


def detect_orientation_lines(image: np.ndarray) -> OrientationResult:
    """
    Original line-based detection (kept as a fallback).

    Uses Hough line detection to count horizontal vs vertical lines.
    """
    method = "lines"

    img_h, img_w = image.shape[:2]

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=min(img_w, img_h) // 8,
                            maxLineGap=20)

    if lines is None or len(lines) < 5:
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={"error": "Not enough lines detected", "line_count": 0 if lines is None else len(lines)}
        )

    # Calculate angles
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Count by orientation
    horizontal_count = sum(1 for a in angles if -45 <= a <= 45)
    vertical_count = sum(1 for a in angles if a > 45 or a < -45)

    # For Exhibit G, we expect landscape with more horizontal lines
    is_landscape = img_w > img_h
    more_horizontal = horizontal_count > vertical_count

    details = {
        "horizontal_count": int(horizontal_count),
        "vertical_count": int(vertical_count),
        "is_landscape": bool(is_landscape),
        "more_horizontal": bool(more_horizontal),
        "total_lines": len(lines)
    }

    if is_landscape and more_horizontal:
        # Likely correct orientation
        return OrientationResult(
            method=method,
            orientation=Orientation.CORRECT,
            confidence=0.4,  # Low confidence - can't determine direction
            details=details
        )
    elif not is_landscape and more_horizontal:
        # Portrait but horizontal lines - need 90° rotation (but which way?)
        return OrientationResult(
            method=method,
            orientation=None,  # Can't determine direction
            confidence=0.0,
            details={**details, "reason": "Needs 90° rotation but direction unclear"}
        )
    else:
        return OrientationResult(
            method=method,
            orientation=None,
            confidence=0.0,
            details={**details, "reason": "Orientation unclear from lines"}
        )


def analyze_orientation(image: np.ndarray, api_key: Optional[str] = None) -> OrientationAnalysis:
    """
    Run all orientation detection methods and combine results.

    Uses a voting system weighted by confidence to determine the best orientation.
    """
    results = []

    # Run all detection methods
    logger.info("Running OCR-based orientation detection...")
    results.append(detect_orientation_ocr(image))

    logger.info("Running logo-based orientation detection...")
    results.append(detect_orientation_logo(image))

    logger.info("Running template-based orientation detection...")
    results.append(detect_orientation_template(image))

    logger.info("Running Claude Vision orientation detection...")
    results.append(detect_orientation_claude(image, api_key))

    logger.info("Running line-based orientation detection...")
    results.append(detect_orientation_lines(image))

    # Tally votes weighted by confidence
    votes: Dict[Orientation, float] = {
        Orientation.CORRECT: 0,
        Orientation.ROTATE_90: 0,
        Orientation.ROTATE_180: 0,
        Orientation.ROTATE_270: 0
    }

    for result in results:
        if result.orientation is not None:
            votes[result.orientation] += result.confidence
            logger.info(f"  {result.method}: {result.orientation.value}° (confidence: {result.confidence:.2f})")
        else:
            logger.info(f"  {result.method}: No result (error: {result.error})")

    # Find winner
    best_orientation = max(votes.keys(), key=lambda o: votes[o])
    total_confidence = sum(votes.values())

    if total_confidence > 0:
        overall_confidence = votes[best_orientation] / total_confidence
    else:
        overall_confidence = 0.0
        best_orientation = Orientation.CORRECT  # Default to no rotation

    voting_breakdown = {str(o.value): float(votes[o]) for o in votes}

    logger.info(f"Orientation analysis complete: {best_orientation.value}° (confidence: {overall_confidence:.2f})")
    logger.info(f"Voting breakdown: {voting_breakdown}")

    # Ensure all result details are JSON-serializable
    for result in results:
        result.details = _convert_numpy(result.details)
        result.confidence = float(result.confidence)

    return OrientationAnalysis(
        recommended_orientation=best_orientation,
        overall_confidence=float(overall_confidence),
        results=results,
        voting_breakdown=voting_breakdown
    )


def apply_orientation_correction(image: np.ndarray, orientation: Orientation) -> np.ndarray:
    """Apply the detected orientation correction to an image."""
    if orientation == Orientation.CORRECT:
        return image
    elif orientation == Orientation.ROTATE_90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == Orientation.ROTATE_180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == Orientation.ROTATE_270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image
