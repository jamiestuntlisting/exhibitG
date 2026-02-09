"""
Claude Vision-guided image correction loop for Exhibit G forms.

Instead of fragile OpenCV heuristics, this module sends the image to Claude
Vision to analyze orientation, keystone distortion, and skew — then applies
the prescribed corrections and loops until Claude approves the result.
"""

import os
import json
import math
import base64
import tempfile
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

ANALYSIS_PROMPT = """You are an image correction assistant. You are looking at a photo of a SAG-AFTRA Exhibit G form (Actors Production Time Report).

I'm also showing you the blank template of this form so you know what it should look like when properly corrected.

Your job is to decide if the photo needs geometric corrections to be readable by OCR software. DO NOT over-correct. A slightly imperfect but readable image is MUCH BETTER than one that has been over-corrected.

This is correction iteration {iteration} of {max_iterations}. {iteration_context}

The image dimensions are {width}x{height} pixels.

ANALYZE THE IMAGE AND RETURN JSON:

{{
  "approved": <bool - true if the image is good enough for OCR. Set true if: the text is readable, the form is right-side up with "SAG-AFTRA" visible at top, and the table lines are within ~3 degrees of horizontal. Minor imperfections, small borders, slight tilt are ALL acceptable. When in doubt, APPROVE>,

  "rotate_90": <int - 0, 90, 180, or 270. ONLY use non-zero if the form is clearly sideways or upside down. 0 if roughly upright>,

  "rotation_degrees": <float - fine skew. Set to 0 unless the table lines are VISIBLY tilted by more than ~2 degrees. Only prescribe REMAINING correction needed>,

  "document_corners": <object or null - ONLY provide if the paper edges are CLEARLY forming a visible trapezoid (one side significantly shorter than the opposite). If the paper looks roughly rectangular, set to null. When provided, corners MUST trace outer edges of the white paper and MUST cover at least 30% of image area. Format: {{"top_left": [x,y], "top_right": [x,y], "bottom_right": [x,y], "bottom_left": [x,y]}}>,

  "needs_crop": <bool - true ONLY if more than ~15% of image area is clearly non-document (desk, other papers, dark background). A thin border is fine>,

  "crop_box": <[x1, y1, x2, y2] or null - null if no crop needed>,

  "issues": <list of strings - empty list [] if approved>
}}

IMPORTANT RULES:
- The form is US Letter LANDSCAPE (wider than tall, 11" x 8.5")
- DO NOT prescribe corrections for minor imperfections — only for CLEAR, SIGNIFICANT problems
- If the photo was taken relatively flat and straight, APPROVE IT immediately
- Perspective correction is ONLY for obvious trapezoid distortion — NOT for slightly imperfect rectangles
- rotation_degrees less than 1.5 is usually NOT worth correcting — set to 0
- If you cannot clearly see that lines are crooked, they are straight enough — approve
- Return ONLY valid JSON, no other text"""


# ──────────────────────────────────────────────
# Claude API Interaction
# ──────────────────────────────────────────────

def _encode_image_file(path: str, max_bytes: int = 4_500_000) -> Tuple[str, str, float]:
    """Encode an image file as base64 JPEG, resizing if needed to stay under max_bytes.

    Claude Vision has a 5MB per-image limit. We target 4.5MB to leave margin.
    Always encodes as JPEG for smaller file sizes (PNG can be huge).

    Returns:
        (base64_data, media_type, scale_factor)
        scale_factor is the ratio used_size / original_size for coordinate mapping.
    """
    # Read the image with OpenCV
    img = cv2.imread(path)
    if img is None:
        # Fallback: read raw file
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        return data, "image/png", 1.0

    orig_h, orig_w = img.shape[:2]

    # Start with quality 85 JPEG encoding
    quality = 85
    scale = 1.0

    for attempt in range(5):
        if scale < 1.0:
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = img

        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        raw_bytes = buf.tobytes()

        if len(raw_bytes) <= max_bytes:
            data = base64.standard_b64encode(raw_bytes).decode("utf-8")
            logger.info(f"Encoded image: {resized.shape[1]}x{resized.shape[0]}, "
                       f"quality={quality}, size={len(raw_bytes)/1024:.0f}KB")
            return data, "image/jpeg", scale

        # Too large — reduce scale and/or quality
        logger.info(f"Image too large ({len(raw_bytes)/1024:.0f}KB), reducing...")
        if quality > 60:
            quality -= 10
        else:
            scale *= 0.75

    # Last resort: aggressive resize
    new_w, new_h = min(orig_w, 1600), min(orig_h, 1200)
    final_scale = new_w / orig_w
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 50])
    data = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
    return data, "image/jpeg", final_scale


def _call_claude_vision(image_path: str, template_path: str,
                        orig_width: int, orig_height: int,
                        api_key: str,
                        iteration: int = 1,
                        max_iterations: int = 3) -> Tuple[Optional[Dict], float]:
    """
    Send an image + template to Claude Vision and parse the JSON response.

    Args:
        image_path: Path to the current image being corrected
        template_path: Path to the blank Exhibit G template
        orig_width: Original image width (before any API scaling)
        orig_height: Original image height (before any API scaling)
        api_key: Anthropic API key
        iteration: Current iteration number (1-based)
        max_iterations: Total iterations allowed

    Returns:
        (Parsed JSON dict or None, image_scale_factor)
    """
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        img_data, img_media, img_scale = _encode_image_file(image_path)
        tmpl_data, tmpl_media, _ = _encode_image_file(template_path)

        # Build iteration context
        if iteration == 1:
            iteration_context = "This is the original uncorrected photo."
        else:
            iteration_context = (
                f"This image has ALREADY been partially corrected in {iteration - 1} previous "
                f"iteration(s). Only prescribe REMAINING corrections needed. "
                f"If the image looks reasonably good and readable, set approved=true."
            )

        # Tell Claude the dimensions of the image it's actually seeing
        seen_width = int(orig_width * img_scale)
        seen_height = int(orig_height * img_scale)
        prompt = ANALYSIS_PROMPT.format(
            width=seen_width, height=seen_height,
            iteration=iteration, max_iterations=max_iterations,
            iteration_context=iteration_context
        )

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the blank Exhibit G template for reference:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": tmpl_media,
                            "data": tmpl_data,
                        }
                    },
                    {
                        "type": "text",
                        "text": "Here is the phone photo to analyze and correct:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img_media,
                            "data": img_data,
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        response_text = message.content[0].text.strip()

        # Strip markdown code blocks if present
        if "```" in response_text:
            lines = response_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            response_text = "\n".join(lines).strip()

        # Extract JSON from response — Claude may include explanatory text
        # Find the first { and last } to extract the JSON object
        first_brace = response_text.find("{")
        last_brace = response_text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            response_text = response_text[first_brace:last_brace + 1]

        plan = json.loads(response_text)
        logger.info(f"Claude correction plan: {json.dumps(plan, indent=2)}")
        return plan, img_scale

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        logger.error(f"Response was: {response_text[:500]}")
        return None, 1.0
    except Exception as e:
        logger.error(f"Claude Vision API call failed: {e}")
        return None, 1.0


# ──────────────────────────────────────────────
# Objective Validation (Hough Lines)
# ──────────────────────────────────────────────

def _detect_hough_lines(image: np.ndarray) -> Tuple[list, list, Optional[np.ndarray]]:
    """Detect horizontal and vertical lines using Hough transform.

    Returns:
        (horiz_lines, vert_lines, raw_lines)
        Each line is [x1, y1, x2, y2].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Use a shorter min line length to catch more document edges
    min_line_length = min(image.shape[1], image.shape[0]) // 6
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min_line_length, maxLineGap=10)

    if lines is None:
        return [], [], None

    horiz, vert = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1 and y2 == y1:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < 30:
            horiz.append(l[0])
        elif abs(angle) > 60:
            vert.append(l[0])

    return horiz, vert, lines


def _line_intersection(line1, line2) -> Optional[Tuple[float, float]]:
    """Compute intersection point of two line segments (extended to infinite lines).

    Each line is [x1, y1, x2, y2].
    Returns (x, y) or None if lines are parallel.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None  # Parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


def _detect_table_grid_for_perspective(image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Detect table grid trapezoid corners for perspective correction.

    Uses morphological line detection to find actual line segments, then
    computes the intersection of extreme lines to get the real trapezoid
    shape (not just axis-aligned bounds).

    This captures perspective distortion in BOTH directions:
    - Horizontal skew (table lines not level)
    - Vertical keystone (left/right edges not parallel)

    Returns:
        (corners, skew_angle) — corners is 4x2 array [TL, TR, BR, BL] or None,
        skew_angle is the median angle of horizontal table lines in degrees.
    """
    h, w = image.shape[:2]

    # Convert to grayscale and binarize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal line segments using morphological ops
    horiz_kernel_len = max(w // 30, 40)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
    horiz_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

    # Detect vertical line segments
    vert_kernel_len = max(h // 30, 40)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
    vert_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, iterations=2)

    # Extract horizontal line segments with their actual geometry
    horiz_cnts, _ = cv2.findContours(horiz_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_segments = []
    for cnt in horiz_cnts:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw > w * 0.15:  # Only significant lines
            # Fit a line to get the actual angle
            if len(cnt) >= 5:
                [vx, vy, cx, cy] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = float(math.degrees(math.atan2(float(vy), float(vx))))
            else:
                angle = 0.0
            h_segments.append({
                'y': y + ch // 2,
                'angle': angle,
                'x1': x,
                'x2': x + cw,
                'cy': float(cy) if 'cy' in dir() else y + ch // 2,
                'vx': float(vx) if 'vx' in dir() else 1.0,
                'vy': float(vy) if 'vy' in dir() else 0.0,
            })

    # Extract vertical line segments
    vert_cnts, _ = cv2.findContours(vert_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_segments = []
    for cnt in vert_cnts:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch > h * 0.08:  # Only significant lines
            if len(cnt) >= 5:
                [vx, vy, cx, cy] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = float(math.degrees(math.atan2(float(vy), float(vx))))
            else:
                angle = 90.0
            v_segments.append({
                'x': x + cw // 2,
                'angle': angle,
                'y1': y,
                'y2': y + ch,
                'cx': float(cx) if 'cx' in dir() else x + cw // 2,
                'vx': float(vx) if 'vx' in dir() else 0.0,
                'vy': float(vy) if 'vy' in dir() else 1.0,
            })

    if len(h_segments) < 2 or len(v_segments) < 2:
        logger.info(f"Table grid detection: not enough line segments "
                   f"(H={len(h_segments)}, V={len(v_segments)})")
        return None, 0.0

    # Sort by position
    h_segments.sort(key=lambda s: s['y'])
    v_segments.sort(key=lambda s: s['x'])

    # Get extreme lines (topmost H, bottommost H, leftmost V, rightmost V)
    top_h = h_segments[0]
    bot_h = h_segments[-1]
    left_v = v_segments[0]
    right_v = v_segments[-1]

    # Calculate median skew from all horizontal lines
    h_angles = [s['angle'] for s in h_segments]
    skew = float(np.median(h_angles))

    # Calculate vertical deviation (perspective)
    v_angles = [s['angle'] for s in v_segments]
    v_deviation = float(np.median([abs(a) - 90 if a > 0 else abs(a) + 90 for a in v_angles]))

    logger.info(f"Table grid detection: H_segments={len(h_segments)}, V_segments={len(v_segments)}, "
               f"skew={skew:.2f}°, V_deviation={v_deviation:.2f}°")

    # Compute the 4 corner intersections
    # For each corner, extend the H and V lines and find intersection

    def line_intersection_from_segments(h_seg, v_seg):
        """Compute intersection of a horizontal and vertical segment."""
        # Use the center point and direction vector from fitLine
        # H line: point (h_seg['x1'], h_seg['y']) with direction (1, tan(angle))
        # V line: point (v_seg['x'], v_seg['y1']) with direction (tan(angle-90), 1)

        h_y = h_seg['y']
        h_angle_rad = math.radians(h_seg['angle'])

        v_x = v_seg['x']
        v_angle_rad = math.radians(v_seg['angle'])

        # For near-horizontal line: y = h_y + (x - h_x1) * tan(h_angle)
        # For near-vertical line: x = v_x + (y - v_y1) * tan(v_angle - 90)

        # Simplified: assume H line passes through (h_seg['x1'], h_seg['y'])
        # and V line passes through (v_seg['x'], v_seg['y1'])

        # H line equation: y - h_y = tan(h_angle) * (x - h_x1)
        # V line equation: x - v_x = tan(v_angle - 90) * (y - v_y1)

        # For small angles, approximate:
        # At x = v_x, the H line gives y = h_y + tan(h_angle) * (v_x - h_seg['x1'])
        # At y = h_y, the V line gives x = v_x + tan(v_angle - 90) * (h_y - v_seg['y1'])

        # Better approach: solve the system
        # y = h_y + (x - h_x1) * tan(h_angle)
        # x = v_x + (y - v_y1) / tan(v_angle)  [since tan(90+a) = -cot(a) = -1/tan(a)]

        tan_h = math.tan(h_angle_rad) if abs(h_angle_rad) > 0.001 else 0.0
        tan_v = math.tan(v_angle_rad) if abs(v_angle_rad - math.pi/2) > 0.001 else 1000.0

        # Iterative refinement (2 steps is enough for small angles)
        x = v_x
        y = h_y + tan_h * (x - h_seg['x1'])

        if abs(tan_v) > 0.001:
            x = v_x + (y - v_seg['y1']) / tan_v
            y = h_y + tan_h * (x - h_seg['x1'])

        return (x, y)

    # Compute 4 corners
    tl = line_intersection_from_segments(top_h, left_v)
    tr = line_intersection_from_segments(top_h, right_v)
    br = line_intersection_from_segments(bot_h, right_v)
    bl = line_intersection_from_segments(bot_h, left_v)

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # Clamp corners to image bounds (with margin)
    margin = max(w, h) * 0.05
    corners[:, 0] = np.clip(corners[:, 0], -margin, w + margin)
    corners[:, 1] = np.clip(corners[:, 1], -margin, h + margin)

    # Validate: check coverage and that it forms a reasonable quadrilateral
    quad_area = cv2.contourArea(corners.reshape(4, 1, 2))
    coverage = quad_area / (h * w)

    if coverage < 0.15:
        logger.info(f"Table grid detection: quad too small ({coverage:.1%})")
        return None, 0.0

    # Check if corners form a convex quadrilateral
    if not cv2.isContourConvex(corners.reshape(4, 1, 2).astype(np.int32)):
        logger.info("Table grid detection: corners don't form convex quad")
        return None, 0.0

    logger.info(f"Table grid detection: corners TL={tl}, TR={tr}, BR={br}, BL={bl}, "
               f"coverage={coverage:.1%}, skew={skew:.2f}°")

    return corners, skew


def _measure_actual_skew(image: np.ndarray) -> float:
    """Measure actual skew angle using Hough line detection as an objective check.

    Checks BOTH horizontal and vertical lines for a more robust measurement.

    Returns the median angle (in degrees) of detected lines
    relative to their expected orientation. Positive = counter-clockwise tilt.

    Returns 0.0 if no clear lines detected (fail-safe: assume straight).
    """
    horiz, vert, _ = _detect_hough_lines(image)

    # Collect skew angles from horizontal lines
    h_angles = []
    for x1, y1, x2, y2 in horiz:
        if x2 == x1:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        h_angles.append(angle)

    # Collect skew angles from vertical lines (deviation from 90°)
    v_angles = []
    for x1, y1, x2, y2 in vert:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Normalize to deviation from vertical (±90°)
        if angle > 0:
            v_angles.append(angle - 90)
        else:
            v_angles.append(angle + 90)

    all_angles = h_angles + v_angles

    if not all_angles:
        logger.info(f"Hough skew check: no H/V lines found "
                   f"(H={len(horiz)}, V={len(vert)}), assuming straight")
        return 0.0

    median_angle = float(np.median(all_angles))
    logger.info(f"Hough skew check: {len(h_angles)}H + {len(v_angles)}V lines, "
               f"median angle = {median_angle:.2f}°")
    return median_angle


# ──────────────────────────────────────────────
# Correction Application
# ──────────────────────────────────────────────

def _apply_corrections(image: np.ndarray, plan: Dict) -> np.ndarray:
    """
    Apply the corrections prescribed by Claude's analysis.

    Uses the existing CV execution functions from image_preprocessor.

    Order of operations:
    1. Major rotation (90° increments)
    2. Perspective/keystone correction (if corners provided)
    3. Fine rotation (sub-degree skew)
    4. Crop to document area
    5. Ensure landscape
    """
    from services.image_preprocessor import (
        rotate_image_90, correct_perspective, correct_keystone,
        correct_skew, ensure_landscape, order_corners
    )

    result = image.copy()
    perspective_applied = False

    # 1. Major rotation (0, 90, 180, 270)
    rotate_90 = plan.get("rotate_90", 0)
    if rotate_90 in (90, 180, 270):
        result = rotate_image_90(result, rotate_90)
        logger.info(f"Applied {rotate_90}° major rotation")

    # 2. Perspective correction
    # Only use Claude-provided corners here. Fallback to table grid detection
    # happens in the post-loop (_apply_table_grid_correction).
    corners = plan.get("document_corners")
    corner_array = None

    if corners is not None:
        try:
            corner_array = np.array([
                corners["top_left"],
                corners["top_right"],
                corners["bottom_right"],
                corners["bottom_left"],
            ], dtype=np.float32)
            logger.info("Using Claude-provided document corners")
        except (KeyError, ValueError) as e:
            logger.warning(f"Invalid document_corners format: {e}")
            corner_array = None

    if corner_array is not None:
        h, w = result.shape[:2]

        # Validate corners are within image bounds
        valid = True
        for pt in corner_array:
            if pt[0] < 0 or pt[0] > w or pt[1] < 0 or pt[1] > h:
                logger.warning(f"Corner {pt} out of bounds ({w}x{h}), skipping perspective")
                valid = False
                break

        if valid:
            ordered = order_corners(corner_array)
            w_top = np.linalg.norm(ordered[1] - ordered[0])
            w_bot = np.linalg.norm(ordered[2] - ordered[3])
            h_left = np.linalg.norm(ordered[3] - ordered[0])
            h_right = np.linalg.norm(ordered[2] - ordered[1])
            quad_area = cv2.contourArea(ordered.reshape(4, 1, 2))
            img_area = h * w
            coverage = quad_area / img_area

            # Check if the quad is nearly rectangular (already flat)
            width_ratio = min(w_top, w_bot) / max(w_top, w_bot) if max(w_top, w_bot) > 0 else 1
            height_ratio = min(h_left, h_right) / max(h_left, h_right) if max(h_left, h_right) > 0 else 1

            # Check if edges are already parallel
            top_angle = math.degrees(math.atan2(
                ordered[1][1] - ordered[0][1], ordered[1][0] - ordered[0][0]))
            bottom_angle = math.degrees(math.atan2(
                ordered[2][1] - ordered[3][1], ordered[2][0] - ordered[3][0]))
            left_angle = math.degrees(math.atan2(
                ordered[3][1] - ordered[0][1], ordered[3][0] - ordered[0][0]))
            right_angle = math.degrees(math.atan2(
                ordered[2][1] - ordered[1][1], ordered[2][0] - ordered[1][0]))
            horiz_parallel = abs(top_angle - bottom_angle)
            vert_parallel = abs(left_angle - right_angle)

            logger.info(f"Perspective check: coverage={coverage:.1%}, "
                       f"w_ratio={width_ratio:.3f}, h_ratio={height_ratio:.3f}, "
                       f"H_parallel={horiz_parallel:.1f}°, V_parallel={vert_parallel:.1f}°")

            if width_ratio > 0.97 and height_ratio > 0.97:
                logger.info("Document is already rectangular — skipping perspective")
            elif horiz_parallel < 2.0 and vert_parallel < 2.0:
                logger.info("Document edges are already parallel — skipping perspective")
            elif coverage < 0.20:
                logger.warning(f"Corners cover only {coverage:.1%} — too small, skipping")
            elif coverage > 0.5:
                result = correct_perspective(result, ordered)
                perspective_applied = True
                logger.info(f"Applied perspective correction (coverage={coverage:.1%})")
            else:
                result = correct_keystone(result, corner_array)
                perspective_applied = True
                logger.info(f"Applied keystone correction (coverage={coverage:.1%})")

    # 3. Fine rotation — validate with objective Hough line measurement
    rotation_deg = plan.get("rotation_degrees", 0)
    if abs(rotation_deg) >= 1.5:
        # Claude thinks rotation is needed — verify with Hough lines
        measured_skew = _measure_actual_skew(result)
        if abs(measured_skew) < 1.0:
            # Hough lines say the image is already straight — veto Claude
            logger.info(f"VETO: Claude prescribed {rotation_deg:.1f}° but Hough lines "
                       f"measure only {measured_skew:.2f}° — image is already straight")
        else:
            # Both agree rotation is needed — apply, but cap to 2x measured
            capped = np.clip(rotation_deg, -abs(measured_skew) * 2, abs(measured_skew) * 2)
            result = correct_skew(result, float(capped))
            logger.info(f"Applied rotation: {capped:.2f}° "
                       f"(Claude: {rotation_deg:.1f}°, measured: {measured_skew:.2f}°)")
    elif abs(rotation_deg) > 0:
        logger.info(f"Skipping tiny rotation ({rotation_deg:.2f}° < 1.5° threshold)")

    # 4. Crop to document area
    # Skip crop if perspective correction was applied — the crop coordinates
    # were for the pre-correction image and are no longer valid. The
    # perspective correction already reframes the document.
    if plan.get("needs_crop") and plan.get("crop_box") and not perspective_applied:
        try:
            x1, y1, x2, y2 = plan["crop_box"]
            h, w = result.shape[:2]
            # Clamp to image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            # Only crop if the result preserves at least 50% of the image
            crop_area = (x2 - x1) * (y2 - y1)
            if crop_area > (h * w * 0.5) and x2 > x1 + 200 and y2 > y1 + 200:
                result = result[y1:y2, x1:x2]
                logger.info(f"Cropped to [{x1},{y1},{x2},{y2}]")
            else:
                logger.warning(f"Crop would remove too much ({crop_area/(h*w):.0%} remaining), skipping")
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid crop_box: {e}")
    elif plan.get("needs_crop") and perspective_applied:
        logger.info("Skipping crop — perspective correction already reframed the document")

    # 5. Ensure landscape
    result = ensure_landscape(result)

    return result


def _rescale_plan_coordinates(plan: Dict, scale: float) -> Dict:
    """Rescale pixel coordinates in a correction plan by a scale factor.

    When the image is resized before sending to Claude, the pixel coordinates
    in Claude's response are for the resized image. This function scales them
    back to the original image dimensions.
    """
    plan = plan.copy()

    # Rescale document corners
    corners = plan.get("document_corners")
    if corners is not None:
        for key in ("top_left", "top_right", "bottom_right", "bottom_left"):
            if key in corners:
                corners[key] = [corners[key][0] * scale, corners[key][1] * scale]
        plan["document_corners"] = corners

    # Rescale crop box
    crop_box = plan.get("crop_box")
    if crop_box is not None:
        plan["crop_box"] = [v * scale for v in crop_box]

    return plan


# ──────────────────────────────────────────────
# Main Correction Loop
# ──────────────────────────────────────────────

def _apply_table_grid_correction(image: np.ndarray) -> np.ndarray:
    """Apply perspective + rotation correction using table grid detection.

    This runs independently of Claude's analysis. It's the safety net that
    catches perspective distortion when Claude returns document_corners=null
    or approves an image that actually has trapezoid distortion.

    Uses the table grid (which produces strong, reliable lines) as the
    reference for correction. Detects and corrects:
    - Horizontal skew (table lines not level)
    - Vertical keystone (left/right edges not parallel)

    Returns the corrected image, or the original if no correction needed.
    """
    from services.image_preprocessor import correct_keystone, correct_skew, order_corners

    # Detect table grid trapezoid and measure skew
    corners, skew = _detect_table_grid_for_perspective(image)

    if corners is None:
        logger.info("Table grid correction: no grid found — skipping")
        return image

    h, w = image.shape[:2]

    # Order corners properly
    ordered = order_corners(corners)

    # Validate table bounds cover reasonable area
    quad_area = cv2.contourArea(ordered.reshape(4, 1, 2))
    coverage = quad_area / (h * w)

    if coverage < 0.15:
        logger.info(f"Table grid correction: grid too small ({coverage:.1%}) — skipping")
        return image

    # Measure how rectangular the quad is
    # Width at top vs bottom
    w_top = np.linalg.norm(ordered[1] - ordered[0])
    w_bot = np.linalg.norm(ordered[2] - ordered[3])
    width_ratio = min(w_top, w_bot) / max(w_top, w_bot) if max(w_top, w_bot) > 0 else 1

    # Height at left vs right
    h_left = np.linalg.norm(ordered[3] - ordered[0])
    h_right = np.linalg.norm(ordered[2] - ordered[1])
    height_ratio = min(h_left, h_right) / max(h_left, h_right) if max(h_left, h_right) > 0 else 1

    # Check edge angles for parallelism
    top_angle = math.degrees(math.atan2(
        ordered[1][1] - ordered[0][1], ordered[1][0] - ordered[0][0]))
    bottom_angle = math.degrees(math.atan2(
        ordered[2][1] - ordered[3][1], ordered[2][0] - ordered[3][0]))
    left_angle = math.degrees(math.atan2(
        ordered[3][1] - ordered[0][1], ordered[3][0] - ordered[0][0]))
    right_angle = math.degrees(math.atan2(
        ordered[2][1] - ordered[1][1], ordered[2][0] - ordered[1][0]))

    horiz_parallel = abs(top_angle - bottom_angle)
    vert_parallel = abs(left_angle - right_angle)

    logger.info(f"Table grid correction: coverage={coverage:.1%}, "
               f"w_ratio={width_ratio:.3f}, h_ratio={height_ratio:.3f}, "
               f"H_parallel={horiz_parallel:.1f}°, V_parallel={vert_parallel:.1f}°, "
               f"skew={skew:.1f}°")

    # Check if correction is needed:
    # - Horizontal skew > 1.0° (lowered from 1.5° to catch more cases)
    # - OR width/height ratio deviation > 2% (perspective) - lowered from 3%
    # - OR edges not parallel > 1.0° (lowered from 2°)
    needs_correction = (
        abs(skew) >= 1.0 or
        width_ratio < 0.98 or
        height_ratio < 0.98 or
        horiz_parallel > 1.0 or
        vert_parallel > 1.0
    )

    if not needs_correction:
        logger.info("Table grid correction: quad is already rectangular — skipping")
        return image

    result = image

    # Apply keystone correction using actual trapezoid corners
    logger.info(f"Table grid correction: applying keystone "
               f"(w_ratio={width_ratio:.3f}, h_ratio={height_ratio:.3f})")
    result = correct_keystone(result, ordered)

    # Sanity check keystone result
    rh, rw = result.shape[:2]
    if rw < 200 or rh < 200:
        logger.warning(f"Keystone produced too-small image ({rw}x{rh}). Reverting.")
        return image

    # After keystone, the table grid should be rectangular.
    # Now apply rotation if there was horizontal skew.
    if abs(skew) >= 1.0:
        logger.info(f"Table grid correction: applying rotation ({skew:.1f}°)")
        result = correct_skew(result, skew)

    return result


def claude_correction_loop(image: np.ndarray, template_path: str,
                           max_iterations: int = 3,
                           api_key: Optional[str] = None) -> np.ndarray:
    """
    Claude Vision-guided image correction loop.

    Sends the image to Claude for analysis, applies prescribed corrections,
    then sends the result back for approval. Loops until approved or
    max_iterations reached.

    After the Claude loop, runs an independent Hough-based perspective check
    to catch distortion that Claude may have missed.

    Args:
        image: BGR numpy array (loaded, EXIF-corrected, resized)
        template_path: Path to the blank Exhibit G template image
        max_iterations: Maximum correction attempts (default 3)
        api_key: Anthropic API key

    Returns:
        Corrected BGR numpy array (ready for lighting normalization)
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        logger.error("No API key for Claude correction loop")
        return image

    if not os.path.exists(template_path):
        logger.error(f"Template not found: {template_path}")
        return image

    current_image = image.copy()
    perspective_already_applied = False

    for iteration in range(max_iterations):
        logger.info(f"=== Claude correction loop: iteration {iteration + 1}/{max_iterations} ===")

        # Save current image to temp JPEG for Claude (much smaller than PNG)
        h, w = current_image.shape[:2]
        temp_path = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(temp_path, current_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

        try:
            # Ask Claude to analyze — returns (plan, scale_factor)
            plan, img_scale = _call_claude_vision(
                temp_path, template_path, w, h, api_key,
                iteration=iteration + 1, max_iterations=max_iterations
            )

            if plan is None:
                logger.warning(f"Claude returned no plan on iteration {iteration + 1}. Stopping.")
                break

            # Check if approved
            if plan.get("approved", False):
                logger.info(f"Claude approved the image on iteration {iteration + 1}!")
                break

            # If the image was scaled down for Claude, rescale pixel coordinates back up
            if img_scale < 1.0:
                inv_scale = 1.0 / img_scale
                logger.info(f"Rescaling coordinates by {inv_scale:.2f}x (image was scaled to {img_scale:.2f})")
                plan = _rescale_plan_coordinates(plan, inv_scale)

            # Log the issues Claude found
            issues = plan.get("issues", [])
            logger.info(f"Issues found: {issues}")

            # Apply corrections
            corrected = _apply_corrections(current_image, plan)

            # Track if perspective was applied by Claude
            if plan.get("document_corners") is not None:
                perspective_already_applied = True

            # Sanity check: make sure we didn't make the image tiny or empty
            ch, cw = corrected.shape[:2]
            if cw < 200 or ch < 200:
                logger.warning(f"Correction produced too-small image ({cw}x{ch}). Reverting.")
                break

            current_image = corrected

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ── Post-loop: Table grid-based perspective + rotation check ──
    # This runs AFTER the Claude loop, regardless of whether Claude approved.
    # It catches perspective distortion that Claude missed (document_corners=null).
    # Skip if Claude already applied perspective correction in the loop.
    if not perspective_already_applied:
        logger.info("=== Post-loop table grid correction ===")
        current_image = _apply_table_grid_correction(current_image)
    else:
        logger.info("Skipping post-loop grid check — Claude already applied perspective correction")

    return current_image


# ──────────────────────────────────────────────
# Iterative Grid Correction Algorithm (V2)
# ──────────────────────────────────────────────

def _find_document_region(image: np.ndarray) -> Tuple[int, int, int, int]:
    """Find the grid table region in the Exhibit G form.

    Uses a multi-stage approach:
    1. Try to find the grid by detecting where horizontal and vertical lines cluster
    2. Fall back to template matching if line detection fails
    3. Final fallback to brightness-based detection

    Returns (x, y, w, h) bounding box of the grid table area.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    img_h, img_w = gray.shape[:2]

    # Try line-density based detection first
    grid_region = _find_grid_by_line_density(gray)
    if grid_region is not None:
        logger.info(f"Found grid region by line density: {grid_region}")
        return grid_region

    # Fall back to brightness-based detection
    logger.info("Line density detection failed, falling back to brightness detection")
    return _find_region_by_brightness(gray)


def _find_grid_by_line_density(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Find the grid region by detecting where lines cluster most densely.

    The Exhibit G grid has many closely-spaced horizontal and vertical lines.
    This function finds the rectangular region where these lines concentrate.
    """
    img_h, img_w = gray.shape[:2]

    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines with relaxed thresholds
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=80,
                            minLineLength=int(min(img_w, img_h) * 0.05),
                            maxLineGap=20)

    if lines is None or len(lines) < 10:
        return None

    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # Horizontal: angle near 0° or 180°
        if abs(angle) < 20 or abs(abs(angle) - 180) < 20:
            h_lines.append((y1 + y2) // 2)  # Y position
        # Vertical: angle near 90° or -90°
        elif abs(abs(angle) - 90) < 20:
            v_lines.append((x1 + x2) // 2)  # X position

    if len(h_lines) < 5 or len(v_lines) < 5:
        return None

    # Find the region with highest concentration of lines
    # Create histograms of line positions
    h_hist = np.zeros(img_h, dtype=int)
    for y in h_lines:
        if 0 <= y < img_h:
            h_hist[max(0, y-10):min(img_h, y+10)] += 1

    v_hist = np.zeros(img_w, dtype=int)
    for x in v_lines:
        if 0 <= x < img_w:
            v_hist[max(0, x-10):min(img_w, x+10)] += 1

    # Find contiguous regions with high line density
    h_threshold = max(h_hist) * 0.3
    v_threshold = max(v_hist) * 0.3

    # Find Y bounds (horizontal lines)
    h_mask = h_hist > h_threshold
    h_indices = np.where(h_mask)[0]
    if len(h_indices) < 2:
        return None
    y_min, y_max = h_indices[0], h_indices[-1]

    # Find X bounds (vertical lines)
    v_mask = v_hist > v_threshold
    v_indices = np.where(v_mask)[0]
    if len(v_indices) < 2:
        return None
    x_min, x_max = v_indices[0], v_indices[-1]

    # Validate the region
    region_w = x_max - x_min
    region_h = y_max - y_min

    # Grid should be reasonably sized (at least 20% of image in each dimension)
    if region_w < img_w * 0.2 or region_h < img_h * 0.2:
        return None

    # Grid should be wider than tall (typical Exhibit G aspect ratio)
    aspect_ratio = region_w / region_h if region_h > 0 else 0
    if aspect_ratio < 1.0 or aspect_ratio > 5.0:
        return None

    # Add small margin
    margin = 20
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    region_w = min(img_w - x_min, region_w + 2 * margin)
    region_h = min(img_h - y_min, region_h + 2 * margin)

    return (x_min, y_min, region_w, region_h)


def _find_region_by_brightness(gray: np.ndarray) -> Tuple[int, int, int, int]:
    """Fallback: Find the document region using brightness detection.

    This is the original method - finds bright (white paper) regions.
    """
    h, w = gray.shape[:2]

    # Find bright regions (white paper)
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    # Find the largest bright region (the document)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: use center 70% of image
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        return (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)

    # Get the largest contour by area
    largest = max(contours, key=cv2.contourArea)
    x, y, rw, rh = cv2.boundingRect(largest)

    # Add inward margin to exclude paper edges
    margin_x = int(rw * 0.10)  # Increased from 5% to 10%
    margin_y = int(rh * 0.10)

    x = x + margin_x
    y = y + margin_y
    rw = rw - 2 * margin_x
    rh = rh - 2 * margin_y

    # Ensure minimum size
    rw = max(rw, 200)
    rh = max(rh, 200)

    return (x, y, rw, rh)


def _detect_all_grid_lines(image: np.ndarray, orientation: str = 'horizontal',
                           doc_region: Optional[Tuple[int, int, int, int]] = None) -> List[Dict]:
    """Detect table grid lines using Hough Line Transform.

    Uses HoughLinesP which is better at detecting thin printed lines.
    Exhibit G has ~18 horizontal lines and ~26 vertical lines max.

    Args:
        image: BGR or grayscale image
        orientation: 'horizontal' or 'vertical'
        doc_region: Optional (x, y, w, h) to restrict detection to document area

    Returns:
        List of dicts with line properties and fit parameters.
    """
    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # If no region specified, find the document
    if doc_region is None:
        doc_region = _find_document_region(image)

    doc_x, doc_y, doc_w, doc_h = doc_region

    # Extract just the document region for processing
    doc_gray = gray[doc_y:doc_y+doc_h, doc_x:doc_x+doc_w]

    # Use Canny edge detection - good for finding thin printed lines
    edges = cv2.Canny(doc_gray, 50, 150, apertureSize=3)

    # Use Hough Line Transform
    if orientation == 'horizontal':
        # For horizontal lines: angle near 0° (or 180°)
        min_length = int(doc_w * 0.30)  # At least 30% of document width
        max_gap = int(doc_w * 0.05)  # Allow small gaps
        max_lines = 20
        angle_tolerance = 10  # Lines within 10° of horizontal
    else:
        # For vertical lines: angle near 90°
        min_length = int(doc_h * 0.15)  # At least 15% of document height
        max_gap = int(doc_h * 0.03)
        max_lines = 30
        angle_tolerance = 10  # Lines within 10° of vertical

    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=100,
                            minLineLength=min_length,
                            maxLineGap=max_gap)

    if lines is None:
        logger.warning(f"No {orientation} lines detected by Hough")
        return []

    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate angle
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # Filter by orientation
        if orientation == 'horizontal':
            # Accept lines near 0° or near 180° or near -180° (all are horizontal)
            if abs(angle) > angle_tolerance and abs(abs(angle) - 180) > angle_tolerance:
                continue
            # Normalize angle to -90 to +90 range (near 0° for horizontal)
            # This prevents averaging 5° and -175° to get 90° (wrong!)
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            length = abs(x2 - x1)
            pos = doc_y + (y1 + y2) // 2  # Y position in full image coords
        else:
            # Accept lines near 90° or near -90° (both are vertical)
            if abs(abs(angle) - 90) > angle_tolerance:
                continue
            # Normalize vertical angles to all be positive (90° side)
            # This prevents averaging 89° and -90° to get near 0°
            if angle < 0:
                angle += 180  # -90° becomes 90°, -85° becomes 95°
            length = abs(y2 - y1)
            pos = doc_x + (x1 + x2) // 2  # X position in full image coords

        # Calculate direction vector (normalized)
        dx, dy = x2 - x1, y2 - y1
        norm = math.sqrt(dx*dx + dy*dy)
        if norm > 0:
            vx, vy = dx / norm, dy / norm
        else:
            vx, vy = (1.0, 0.0) if orientation == 'horizontal' else (0.0, 1.0)

        # Convert coordinates back to full image
        cx = doc_x + (x1 + x2) / 2
        cy = doc_y + (y1 + y2) / 2

        segments.append({
            'angle': angle,
            'pos': pos,
            'length': length,
            'x1': doc_x + x1,
            'y1': doc_y + y1,
            'x2': doc_x + x2,
            'y2': doc_y + y2,
            # Line fit parameters for intersection calculation
            'vx': vx,
            'vy': vy,
            'cx': cx,
            'cy': cy,
        })

    # Cluster nearby lines (same line detected multiple times)
    segments = _cluster_nearby_lines(segments, orientation, merge_threshold=20)

    # Filter out lines that cross other lines (invalid for a parallel grid)
    segments = _filter_crossing_lines(segments, orientation, img_w, img_h)

    # Sort by position
    segments.sort(key=lambda s: s['pos'])

    # Limit to max expected lines (take the longest ones if too many)
    if len(segments) > max_lines:
        segments.sort(key=lambda s: s['length'], reverse=True)
        segments = segments[:max_lines]
        segments.sort(key=lambda s: s['pos'])

    return segments


def _cluster_nearby_lines(segments: List[Dict], orientation: str,
                          merge_threshold: int = 20) -> List[Dict]:
    """Merge lines that are very close together (likely the same line detected multiple times).

    Args:
        segments: List of line segments
        orientation: 'horizontal' or 'vertical'
        merge_threshold: Pixels - lines closer than this are merged

    Returns:
        Merged list of segments
    """
    if not segments:
        return []

    # Sort by position
    segments = sorted(segments, key=lambda s: s['pos'])

    merged = []
    current_group = [segments[0]]

    for seg in segments[1:]:
        if abs(seg['pos'] - current_group[-1]['pos']) < merge_threshold:
            # Same line, add to group
            current_group.append(seg)
        else:
            # New line - merge current group and start new
            merged.append(_merge_line_group(current_group))
            current_group = [seg]

    # Don't forget the last group
    merged.append(_merge_line_group(current_group))

    return merged


def _merge_line_group(group: List[Dict]) -> Dict:
    """Merge a group of similar lines into one, taking the longest/best one.

    Only merges lines with similar angles to avoid averaging a vertical line
    with a diagonal artifact.
    """
    if len(group) == 1:
        return group[0]

    # Take the longest line as the base
    best = max(group, key=lambda s: s['length'])

    # Only include lines with angles similar to the best line (within 15°)
    base_angle = best['angle']
    similar_lines = [s for s in group if abs(s['angle'] - base_angle) < 15 or
                     abs(s['angle'] - base_angle + 180) < 15 or
                     abs(s['angle'] - base_angle - 180) < 15]

    if not similar_lines:
        similar_lines = [best]

    # Use weighted average of angles (weight by length) for similar lines only
    total_length = sum(s['length'] for s in similar_lines)
    avg_angle = sum(s['angle'] * s['length'] for s in similar_lines) / total_length
    avg_pos = sum(s['pos'] * s['length'] for s in similar_lines) / total_length

    # Recalculate direction vector from average angle
    rad = math.radians(avg_angle)
    vx = math.cos(rad)
    vy = math.sin(rad)

    return {
        'angle': avg_angle,
        'pos': avg_pos,
        'length': best['length'],
        'x1': best['x1'],
        'y1': best['y1'],
        'x2': best['x2'],
        'y2': best['y2'],
        'vx': vx,
        'vy': vy,
        'cx': best['cx'],
        'cy': best['cy'],
    }


def _line_quality(seg: Dict, orientation: str) -> float:
    """Score a line's quality. Higher = more likely a real grid line.

    Factors:
    - Length: longer lines are more likely real grid lines
    - Angle deviation: lines closer to expected angle (0° for H, 90° for V) are better
    """
    length_score = seg['length']

    if orientation == 'horizontal':
        # For horizontal, angle should be near 0°
        angle_score = 1000 - abs(seg['angle']) * 100
    else:
        # For vertical, angle should be near 90°
        angle_score = 1000 - abs(seg['angle'] - 90) * 100

    return length_score + angle_score


def _filter_crossing_lines(segments: List[Dict], orientation: str,
                           img_w: int, img_h: int) -> List[Dict]:
    """Remove lines that intersect other lines of the same orientation.

    For a valid grid, all horizontal lines should be parallel (never cross),
    and all vertical lines should be parallel (never cross).

    When two lines cross within the image bounds, we keep the higher-quality
    line (longer, better angle) and discard the other.

    Args:
        segments: List of line segment dictionaries
        orientation: 'horizontal' or 'vertical'
        img_w: Image width
        img_h: Image height

    Returns:
        Filtered list with crossing lines removed
    """
    if len(segments) < 2:
        return segments

    valid = []
    removed_count = 0

    for seg in segments:
        is_valid = True
        seg_line = [seg['x1'], seg['y1'], seg['x2'], seg['y2']]

        to_remove = []
        for other in valid:
            other_line = [other['x1'], other['y1'], other['x2'], other['y2']]
            intersection = _line_intersection(seg_line, other_line)

            if intersection is not None:
                ix, iy = intersection
                # Check if intersection is within image bounds (with small margin)
                margin = 50  # Allow small margin outside image
                if -margin <= ix <= img_w + margin and -margin <= iy <= img_h + margin:
                    # These "parallel" lines cross - keep the better one
                    seg_quality = _line_quality(seg, orientation)
                    other_quality = _line_quality(other, orientation)

                    if seg_quality < other_quality:
                        is_valid = False
                        removed_count += 1
                        break
                    else:
                        to_remove.append(other)
                        removed_count += 1

        # Remove lower-quality crossing lines
        for item in to_remove:
            valid.remove(item)

        if is_valid:
            valid.append(seg)

    if removed_count > 0:
        logger.info(f"Filtered {removed_count} crossing {orientation} lines")

    return valid


def _get_line_edge_intersections(seg: Dict, img_width: int, img_height: int,
                                  orientation: str) -> Tuple[float, float]:
    """Calculate where a fitted line intersects the image edges.

    For horizontal lines: returns (left_y, right_y) - where line crosses x=0 and x=width
    For vertical lines: returns (top_x, bottom_x) - where line crosses y=0 and y=height

    Uses the parametric line equation from cv2.fitLine:
        point = (cx, cy) + t * (vx, vy)
    """
    vx, vy, cx, cy = seg['vx'], seg['vy'], seg['cx'], seg['cy']

    if orientation == 'horizontal':
        # Find y at x=0 (left edge)
        if abs(vx) > 1e-6:
            t_left = (0 - cx) / vx
            left_y = cy + t_left * vy
            # Find y at x=width (right edge)
            t_right = (img_width - cx) / vx
            right_y = cy + t_right * vy
        else:
            # Vertical line (shouldn't happen for horizontal detection)
            left_y = right_y = cy
        return (left_y, right_y)
    else:
        # Find x at y=0 (top edge)
        if abs(vy) > 1e-6:
            t_top = (0 - cy) / vy
            top_x = cx + t_top * vx
            # Find x at y=height (bottom edge)
            t_bottom = (img_height - cy) / vy
            bottom_x = cx + t_bottom * vx
        else:
            # Horizontal line (shouldn't happen for vertical detection)
            top_x = bottom_x = cx
        return (top_x, bottom_x)


def _compute_keystone_corners_from_angles(
    image: np.ndarray,
    top_angle: float,
    bottom_angle: float,
    left_angle: float,
    right_angle: float
) -> np.ndarray:
    """Compute trapezoid corners based on measured edge angles.

    Given the angles that the top/bottom/left/right edges SHOULD be at
    (after correction), compute where the corners would need to be.

    This creates a synthetic trapezoid that represents the distortion.
    """
    h, w = image.shape[:2]

    # Start with a rectangle
    margin = min(w, h) * 0.05

    # Top edge: if tilted, one side is higher than the other
    top_rise = (w - 2*margin) * math.tan(math.radians(top_angle))
    bottom_rise = (w - 2*margin) * math.tan(math.radians(bottom_angle))

    # Left edge: if tilted, top is left/right of bottom
    left_run = (h - 2*margin) * math.tan(math.radians(left_angle - 90))
    right_run = (h - 2*margin) * math.tan(math.radians(right_angle - 90))

    # Construct corners
    tl = [margin - left_run/2, margin - top_rise/2]
    tr = [w - margin - right_run/2, margin + top_rise/2]
    br = [w - margin + right_run/2, h - margin + bottom_rise/2]
    bl = [margin + left_run/2, h - margin - bottom_rise/2]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _detect_orientation(image: np.ndarray) -> int:
    """Detect if the image needs a 90° multiple rotation.

    Uses line detection to determine if the document is rotated by approximately
    90°, 180°, or 270° from the expected orientation.

    For an Exhibit G form, we expect:
    - More horizontal lines than vertical (it's a wide table)
    - Horizontal lines should be roughly horizontal (within ±45° of 0°)

    Returns:
        Rotation needed in degrees: 0, 90, 180, or 270
    """
    img_h, img_w = image.shape[:2]

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=min(img_w, img_h) // 8,
                            maxLineGap=20)

    if lines is None or len(lines) < 5:
        logger.info("Orientation detection: Not enough lines found, assuming correct orientation")
        return 0

    # Calculate angles for all lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Get angle in degrees (-90 to 90)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Categorize lines by orientation
    # Horizontal-ish: -45 to 45 degrees
    # Vertical-ish: 45 to 90 or -90 to -45 degrees
    horizontal_count = sum(1 for a in angles if -45 <= a <= 45)
    vertical_count = sum(1 for a in angles if a > 45 or a < -45)

    logger.info(f"Orientation detection: {horizontal_count} horizontal-ish, {vertical_count} vertical-ish lines")

    # For Exhibit G forms (wide tables), we expect more horizontal lines
    # If we see more vertical lines, the image is likely rotated 90°

    # Calculate the median angle of "horizontal" lines
    h_angles = [a for a in angles if -45 <= a <= 45]
    v_angles = [a for a in angles if a > 45 or a < -45]

    if len(h_angles) >= len(v_angles):
        # More horizontal lines - check if they're actually horizontal
        median_h = np.median(h_angles) if h_angles else 0

        # If median is close to 0, orientation is correct
        if abs(median_h) < 45:
            logger.info(f"Orientation: Correct (median H angle: {median_h:.1f}°)")
            return 0
    else:
        # More vertical lines than horizontal - image is likely rotated 90°
        # Check median angle of vertical lines
        # Normalize vertical angles to be relative to vertical (90°)
        v_normalized = []
        for a in v_angles:
            if a > 45:
                v_normalized.append(a - 90)  # e.g., 85° -> -5°
            else:
                v_normalized.append(a + 90)  # e.g., -85° -> 5°

        median_v = np.median(v_normalized) if v_normalized else 0

        # Determine rotation direction based on which way lines lean
        # If what should be horizontal lines are near +90°, rotate 90° CCW
        # If what should be horizontal lines are near -90°, rotate 90° CW

        # Check the dominant "vertical" angle
        raw_v_median = np.median(v_angles) if v_angles else 90

        if raw_v_median > 0:
            # Lines are near +90° (pointing up-right), need 90° CW rotation
            logger.info(f"Orientation: Need 270° (90° CW) (vertical lines at ~{raw_v_median:.1f}°)")
            return 270
        else:
            # Lines are near -90° (pointing down-right), need 90° CCW rotation
            logger.info(f"Orientation: Need 90° CCW (vertical lines at ~{raw_v_median:.1f}°)")
            return 90

    # Check for 180° rotation - this is harder to detect without text analysis
    # For now, we'll skip 180° detection as it requires OCR or other methods
    # The form should still work even if upside down (lines will still align)

    return 0


def iterative_grid_correction(
    image: np.ndarray,
    save_intermediates: bool = True,
    output_dir: Optional[str] = None
) -> Tuple[np.ndarray, List[Dict]]:
    """Apply iterative grid-based correction algorithm.

    NEW Algorithm - Alternating H/V correction with re-detection:

    Loop until converged (or max iterations):
    1. Detect document/grid region
    2. Detect horizontal lines, apply ONE correction pass
    3. Re-detect document/grid region
    4. Detect vertical lines, apply ONE correction pass
    5. Check if changes are small enough to stop

    This approach re-detects the grid region after each correction,
    ensuring we're always working with accurate bounds.

    Args:
        image: BGR numpy array
        save_intermediates: If True, save images at each step
        output_dir: Directory to save intermediate images

    Returns:
        (corrected_image, steps) where steps is a list of dicts describing each step
    """
    from services.image_preprocessor import correct_skew

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="grid_correction_")

    os.makedirs(output_dir, exist_ok=True)

    steps = []
    current = image.copy()
    step_num = 0

    def _convert_numpy(obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_numpy(v) for v in obj]
        return obj

    def save_step(img, name, description, data=None):
        nonlocal step_num
        step_num += 1
        filename = f"step_{step_num:02d}_{name}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        step_info = {
            'step': step_num,
            'name': name,
            'description': description,
            'filename': filename,
            'filepath': filepath,
            'data': _convert_numpy(data) if data else {}
        }
        steps.append(step_info)
        logger.info(f"Step {step_num}: {description}")
        return step_info

    # ─── Step 0: Original image ───
    save_step(current, 'original', 'Original input image')

    # ═══════════════════════════════════════════════════════════════════
    # ORIENTATION DETECTION: Multi-faceted analysis
    # ═══════════════════════════════════════════════════════════════════

    from services.orientation_detector import analyze_orientation, apply_orientation_correction, Orientation

    img_h, img_w = current.shape[:2]

    # Run comprehensive orientation analysis
    logger.info("Running multi-faceted orientation detection...")
    orientation_analysis = analyze_orientation(current)

    # Convert analysis to serializable format for the step data
    orientation_data = {
        "recommended_rotation": orientation_analysis.recommended_orientation.value,
        "overall_confidence": orientation_analysis.overall_confidence,
        "voting_breakdown": orientation_analysis.voting_breakdown,
        "methods": []
    }

    for result in orientation_analysis.results:
        method_data = {
            "method": result.method,
            "orientation": result.orientation.value if result.orientation else None,
            "confidence": result.confidence,
            "error": result.error,
            "details": _convert_numpy(result.details)
        }
        orientation_data["methods"].append(method_data)

    # Save orientation analysis step
    save_step(current, 'orientation_analysis',
              f'Orientation analysis: {len(orientation_analysis.results)} methods, '
              f'recommended {orientation_analysis.recommended_orientation.value}° '
              f'(confidence: {orientation_analysis.overall_confidence:.2f})',
              orientation_data)

    orientation_rotation = orientation_analysis.recommended_orientation.value

    # Track all transformation matrices for single-pass high-quality output
    # Each matrix is 3x3 homogeneous. We'll compose them at the end.
    accumulated_transforms: List[np.ndarray] = []

    if orientation_rotation != 0:
        logger.info(f"Detected orientation requires {orientation_rotation}° rotation")

        # Apply 90° rotation(s) - these are lossless operations
        if orientation_rotation == 90:
            current = cv2.rotate(current, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orientation_rotation == 180:
            current = cv2.rotate(current, cv2.ROTATE_180)
        elif orientation_rotation == 270:
            current = cv2.rotate(current, cv2.ROTATE_90_CLOCKWISE)

        # Update dimensions after rotation
        img_h, img_w = current.shape[:2]

        # Create the rotation matrix for accumulation
        # For 90° multiples, we use special rotation matrices
        center_x, center_y = img_w / 2, img_h / 2
        if orientation_rotation == 90:
            # 90° CCW: (x,y) -> (y, w-x) but dimensions swap
            # We need to account for the dimension change
            orig_h, orig_w = image.shape[:2]
            M_orient = np.array([
                [0, -1, orig_h],
                [1, 0, 0],
                [0, 0, 1]
            ], dtype=np.float64)
        elif orientation_rotation == 180:
            orig_h, orig_w = image.shape[:2]
            M_orient = np.array([
                [-1, 0, orig_w],
                [0, -1, orig_h],
                [0, 0, 1]
            ], dtype=np.float64)
        elif orientation_rotation == 270:
            orig_h, orig_w = image.shape[:2]
            M_orient = np.array([
                [0, 1, 0],
                [-1, 0, orig_w],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            M_orient = np.eye(3, dtype=np.float64)

        accumulated_transforms.append(M_orient)

        save_step(current, 'orientation',
                  f'Orientation correction: rotated {orientation_rotation}°',
                  {'rotation_degrees': orientation_rotation})

    # ═══════════════════════════════════════════════════════════════════
    # MAIN LOOP: Rotation then keystone corrections
    # ═══════════════════════════════════════════════════════════════════

    MAX_ITERATIONS = 5
    CONVERGENCE_THRESHOLD = 0.5  # Degrees
    MIN_CORRECTION_PX = 5.0

    for iteration in range(MAX_ITERATIONS):
        logger.info(f"=== Iteration {iteration+1}/{MAX_ITERATIONS} ===")

        # ─────────────────────────────────────────────────────────────
        # STEP A: Detect document region and lines
        # ─────────────────────────────────────────────────────────────
        doc_region = _find_document_region(current)
        doc_x, doc_y, doc_w, doc_h = doc_region
        img_h, img_w = current.shape[:2]

        vis_doc = current.copy()
        cv2.rectangle(vis_doc, (doc_x, doc_y), (doc_x + doc_w, doc_y + doc_h), (0, 255, 0), 3)
        save_step(vis_doc, f'iter{iteration+1}_doc',
                  f'Iter {iteration+1}: Doc region: ({doc_x}, {doc_y}) {doc_w}x{doc_h}',
                  {'iteration': iteration+1, 'x': doc_x, 'y': doc_y, 'w': doc_w, 'h': doc_h})

        # Detect both H and V lines
        h_lines = _detect_all_grid_lines(current, 'horizontal', doc_region)
        v_lines = _detect_all_grid_lines(current, 'vertical', doc_region)

        h_angles = [seg['angle'] for seg in h_lines] if h_lines else [0]
        v_deviations = [seg['angle'] - 90 for seg in v_lines] if v_lines else [0]

        h_median = float(np.median(h_angles))
        v_median = float(np.median(v_deviations))
        h_max = float(max(abs(a) for a in h_angles)) if h_angles else 0
        v_max = float(max(abs(d) for d in v_deviations)) if v_deviations else 0

        # Visualize detected lines
        vis = current.copy()
        for seg in h_lines:
            left_y, right_y = _get_line_edge_intersections(seg, img_w, img_h, 'horizontal')
            cv2.line(vis, (0, int(left_y)), (img_w, int(right_y)), (0, 255, 0), 2)
        for seg in v_lines:
            top_x, bot_x = _get_line_edge_intersections(seg, img_w, img_h, 'vertical')
            cv2.line(vis, (int(top_x), 0), (int(bot_x), img_h), (0, 128, 255), 2)

        save_step(vis, f'iter{iteration+1}_detect',
                  f'Iter {iteration+1}: H={len(h_lines)} lines ({h_median:.2f}°), V={len(v_lines)} lines ({v_median:.2f}° from vert)',
                  {'h_lines': len(h_lines), 'v_lines': len(v_lines), 'h_median': h_median, 'v_median': v_median})

        total_correction_px = 0.0

        # ─────────────────────────────────────────────────────────────
        # STEP B: ROTATION - Use H lines to determine skew
        # ─────────────────────────────────────────────────────────────
        # Horizontal lines give the best measure of rotation (tilt)
        # because they span the full width of the document

        if len(h_lines) >= 2 and abs(h_median) >= CONVERGENCE_THRESHOLD:
            # Rotate to level horizontal lines
            # If line tilts down-right (+angle), rotate image CCW (+angle) to level it
            # If line tilts up-right (-angle), rotate image CW (-angle) to level it
            # So: rotation_angle = +h_median (same sign)
            damping = 0.8
            rotation_angle = h_median * damping
            center = (img_w // 2, img_h // 2)
            M_rot = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            current = cv2.warpAffine(current, M_rot, (img_w, img_h),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))

            # Convert 2x3 affine to 3x3 homogeneous and accumulate
            M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])
            accumulated_transforms.append(M_rot_3x3)

            rotation_px = abs(math.tan(math.radians(rotation_angle)) * img_w / 2)
            total_correction_px += rotation_px

            save_step(current, f'iter{iteration+1}_rotate',
                      f'Iter {iteration+1}: H lines at {h_median:.2f}°, rotated by {rotation_angle:.2f}°',
                      {'h_median': h_median, 'rotation': rotation_angle, 'rotation_px': rotation_px})

            # Re-detect for keystone step
            doc_region = _find_document_region(current)
            h_lines = _detect_all_grid_lines(current, 'horizontal', doc_region)
            v_lines = _detect_all_grid_lines(current, 'vertical', doc_region)

        # ─────────────────────────────────────────────────────────────
        # STEP C: HORIZONTAL KEYSTONE
        # ─────────────────────────────────────────────────────────────
        if len(h_lines) >= 2:
            h_correction_px = 0.0
            current, h_correction_px, h_matrix = _apply_horizontal_correction(current, h_lines, save_step, iteration, doc_region)
            total_correction_px += h_correction_px
            if h_matrix is not None:
                accumulated_transforms.append(h_matrix)

        # ─────────────────────────────────────────────────────────────
        # STEP D: VERTICAL KEYSTONE
        # ─────────────────────────────────────────────────────────────
        # Re-detect after H keystone
        doc_region = _find_document_region(current)
        v_lines = _detect_all_grid_lines(current, 'vertical', doc_region)

        if len(v_lines) >= 2:
            v_correction_px = 0.0
            current, v_correction_px, v_matrix = _apply_vertical_correction(current, v_lines, save_step, iteration, doc_region)
            total_correction_px += v_correction_px
            if v_matrix is not None:
                accumulated_transforms.append(v_matrix)

        # ─────────────────────────────────────────────────────────────
        # STEP E: Check convergence
        # ─────────────────────────────────────────────────────────────
        logger.info(f"Iter {iteration+1}: H_max={h_max:.2f}°, V_max={v_max:.2f}°, Total_px={total_correction_px:.1f}")

        if h_max < CONVERGENCE_THRESHOLD and v_max < CONVERGENCE_THRESHOLD:
            logger.info(f"Converged after {iteration+1} iterations (both H and V below {CONVERGENCE_THRESHOLD}°)")
            break

        if total_correction_px < MIN_CORRECTION_PX:
            logger.info(f"Converged after {iteration+1} iterations (corrections < {MIN_CORRECTION_PX}px)")
            break

    # ─── Final step: Iterative result (may have quality loss) ───
    save_step(current, 'final_iterative', 'Final corrected image (iterative)')

    # ═══════════════════════════════════════════════════════════════════
    # HIGH-QUALITY SINGLE-PASS CORRECTION
    # ═══════════════════════════════════════════════════════════════════
    # Compose all accumulated transforms into a single matrix and apply
    # once to the original image. This avoids cumulative interpolation artifacts.

    if accumulated_transforms:
        logger.info(f"Composing {len(accumulated_transforms)} transforms for single-pass correction")

        # Check if orientation rotation was applied (dimensions changed)
        orig_h, orig_w = image.shape[:2]
        final_h, final_w = current.shape[:2]
        orientation_changed = (orig_w != final_w) or (orig_h != final_h)

        if orientation_changed and orientation_rotation in (90, 270):
            # For 90° rotations, we can't easily compose with perspective transforms
            # because the coordinate systems are different. Instead:
            # 1. Apply orientation rotation to original (lossless)
            # 2. Then apply remaining transforms

            # First, rotate the original image
            if orientation_rotation == 90:
                rotated_original = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif orientation_rotation == 270:
                rotated_original = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            else:
                rotated_original = cv2.rotate(image, cv2.ROTATE_180)

            # Compose only the non-orientation transforms (skip the first one)
            if len(accumulated_transforms) > 1:
                M_combined = np.eye(3, dtype=np.float64)
                for M in accumulated_transforms[1:]:  # Skip orientation matrix
                    M_combined = M @ M_combined

                high_quality = cv2.warpPerspective(
                    rotated_original,
                    M_combined,
                    (final_w, final_h),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)
                )
            else:
                # Only orientation was applied
                high_quality = rotated_original
        else:
            # No orientation change, or 180° rotation (same dimensions)
            # Compose all transforms
            M_combined = np.eye(3, dtype=np.float64)
            for M in accumulated_transforms:
                M_combined = M @ M_combined

            high_quality = cv2.warpPerspective(
                image,
                M_combined,
                (final_w, final_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )

        save_step(high_quality, 'final', 'Final corrected image (single-pass, high quality)',
                  {'num_transforms': len(accumulated_transforms)})

        logger.info(f"Iterative grid correction complete: {len(steps)} steps saved to {output_dir}")
        return high_quality, steps
    else:
        # No transforms were applied, just rename the iterative final
        save_step(current, 'final', 'Final corrected image (no corrections needed)')
        logger.info(f"Iterative grid correction complete: {len(steps)} steps saved to {output_dir}")
        return current, steps


def _apply_horizontal_correction(current: np.ndarray, h_lines: List[Dict],
                                  save_step, iteration: int,
                                  doc_region: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """Apply horizontal keystone correction (perspective).

    Returns: (corrected_image, correction_magnitude_px, transform_matrix_3x3)
    """
    img_h, img_w = current.shape[:2]

    # Use 2nd line from top and 2nd from bottom (more reliable than edge lines)
    if len(h_lines) >= 4:
        top_line = h_lines[1]     # 2nd from top
        bot_line = h_lines[-2]    # 2nd from bottom
        logger.info(f"Iter {iteration+1} H: Using 2nd-from-edge lines for keystone")
    else:
        top_line = h_lines[0]
        bot_line = h_lines[-1]
        logger.info(f"Iter {iteration+1} H: Using edge lines for keystone (only {len(h_lines)} lines)")

    # Calculate median angle for top half and bottom half
    mid_y = (h_lines[0]['pos'] + h_lines[-1]['pos']) / 2
    top_half = [s for s in h_lines if s['pos'] < mid_y]
    bot_half = [s for s in h_lines if s['pos'] >= mid_y]

    top_angles = [s['angle'] for s in top_half] if top_half else [0]
    bot_angles = [s['angle'] for s in bot_half] if bot_half else [0]
    top_median_angle = float(np.median(top_angles))
    bot_median_angle = float(np.median(bot_angles))

    logger.info(f"Iter {iteration+1} H: Top half median: {top_median_angle:.2f}°, Bot half median: {bot_median_angle:.2f}°")

    # Calculate where reference lines intersect the image edges
    top_left_y, top_right_y = _get_line_edge_intersections(top_line, img_w, img_h, 'horizontal')
    bot_left_y, bot_right_y = _get_line_edge_intersections(bot_line, img_w, img_h, 'horizontal')

    # Adjust intersections based on median angle if reference line differs
    top_line_angle = top_line['angle']
    bot_line_angle = bot_line['angle']

    if abs(top_median_angle) > abs(top_line_angle):
        adjustment = math.tan(math.radians(top_median_angle - top_line_angle)) * img_w
        top_left_y -= adjustment / 2
        top_right_y += adjustment / 2
    if abs(bot_median_angle) > abs(bot_line_angle):
        adjustment = math.tan(math.radians(bot_median_angle - bot_line_angle)) * img_w
        bot_left_y -= adjustment / 2
        bot_right_y += adjustment / 2

    # Clip to image bounds
    tl_y = float(np.clip(top_left_y, 0, img_h))
    tr_y = float(np.clip(top_right_y, 0, img_h))
    bl_y = float(np.clip(bot_left_y, 0, img_h))
    br_y = float(np.clip(bot_right_y, 0, img_h))

    # Check if keystone correction is needed
    top_diff = abs(tl_y - tr_y)
    bot_diff = abs(bl_y - br_y)

    if top_diff < 5.0 and bot_diff < 5.0:
        logger.info(f"Iter {iteration+1} H: Keystone negligible (top_diff={top_diff:.1f}px, bot_diff={bot_diff:.1f}px), skipping")
        return current, 0.0, None

    # Visualize the trapezoid
    vis2 = current.copy()
    pts = np.array([[0, tl_y], [img_w, tr_y], [img_w, br_y], [0, bl_y]], dtype=np.int32)
    cv2.polylines(vis2, [pts], True, (0, 0, 255), 3)
    for px, py in pts:
        cv2.circle(vis2, (px, py), 8, (255, 0, 0), -1)

    save_step(vis2, f'iter{iteration+1}_h_trapezoid',
              f'Iter {iteration+1} H trapezoid: TL_y={tl_y:.0f}, TR_y={tr_y:.0f}, BL_y={bl_y:.0f}, BR_y={br_y:.0f}',
              {'tl_y': tl_y, 'tr_y': tr_y, 'bl_y': bl_y, 'br_y': br_y})

    # Apply perspective transform with damping
    damping = 0.7
    full_dst_top_y = (tl_y + tr_y) / 2
    full_dst_bot_y = (bl_y + br_y) / 2

    dst_tl_y = tl_y + (full_dst_top_y - tl_y) * damping
    dst_tr_y = tr_y + (full_dst_top_y - tr_y) * damping
    dst_bl_y = bl_y + (full_dst_bot_y - bl_y) * damping
    dst_br_y = br_y + (full_dst_bot_y - br_y) * damping

    src = np.array([
        [0, tl_y], [img_w, tr_y], [img_w, br_y], [0, bl_y]
    ], dtype=np.float32)

    dst = np.array([
        [0, dst_tl_y], [img_w, dst_tr_y], [img_w, dst_br_y], [0, dst_bl_y]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    corrected = cv2.warpPerspective(current, M, (img_w, img_h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))

    keystone_px = max(top_diff, bot_diff)

    save_step(corrected, f'iter{iteration+1}_h_keystone',
              f'Iter {iteration+1} H: Keystone correction (top_diff={top_diff:.1f}px, bot_diff={bot_diff:.1f}px)',
              {'iteration': iteration+1, 'top_diff_px': top_diff, 'bot_diff_px': bot_diff})

    return corrected, keystone_px, M.astype(np.float64)


def _apply_vertical_correction(current: np.ndarray, v_lines: List[Dict],
                                save_step, iteration: int,
                                doc_region: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """Apply vertical keystone correction (perspective).

    Returns: (corrected_image, correction_magnitude_px, transform_matrix_3x3)
    """
    img_h, img_w = current.shape[:2]

    # Use 2nd line from left and 2nd from right (more reliable than edge lines)
    if len(v_lines) >= 4:
        left_line = v_lines[1]     # 2nd from left
        right_line = v_lines[-2]   # 2nd from right
        logger.info(f"Iter {iteration+1} V: Using 2nd-from-edge lines for keystone")
    else:
        left_line = v_lines[0]
        right_line = v_lines[-1]
        logger.info(f"Iter {iteration+1} V: Using edge lines for keystone (only {len(v_lines)} lines)")

    # Calculate median deviation for left half and right half
    mid_x = (v_lines[0]['pos'] + v_lines[-1]['pos']) / 2
    left_half = [s for s in v_lines if s['pos'] < mid_x]
    right_half = [s for s in v_lines if s['pos'] >= mid_x]

    left_devs = [s['angle'] - 90 for s in left_half] if left_half else [0]
    right_devs = [s['angle'] - 90 for s in right_half] if right_half else [0]
    left_median_dev = float(np.median(left_devs))
    right_median_dev = float(np.median(right_devs))

    logger.info(f"Iter {iteration+1} V: Left half median: {left_median_dev:.2f}°, Right half median: {right_median_dev:.2f}°")

    # Calculate where reference lines intersect the image edges
    left_top_x, left_bot_x = _get_line_edge_intersections(left_line, img_w, img_h, 'vertical')
    right_top_x, right_bot_x = _get_line_edge_intersections(right_line, img_w, img_h, 'vertical')

    # Adjust intersections based on median deviation if reference line differs
    left_line_dev = left_line['angle'] - 90
    right_line_dev = right_line['angle'] - 90

    if abs(left_median_dev) > abs(left_line_dev):
        adjustment = math.tan(math.radians(left_median_dev - left_line_dev)) * img_h
        left_top_x -= adjustment / 2
        left_bot_x += adjustment / 2
    if abs(right_median_dev) > abs(right_line_dev):
        adjustment = math.tan(math.radians(right_median_dev - right_line_dev)) * img_h
        right_top_x -= adjustment / 2
        right_bot_x += adjustment / 2

    # Clip to image bounds
    lt_x = float(np.clip(left_top_x, 0, img_w))
    lb_x = float(np.clip(left_bot_x, 0, img_w))
    rt_x = float(np.clip(right_top_x, 0, img_w))
    rb_x = float(np.clip(right_bot_x, 0, img_w))

    # Check if keystone correction is needed
    left_diff = abs(lt_x - lb_x)
    right_diff = abs(rt_x - rb_x)

    if left_diff < 5.0 and right_diff < 5.0:
        logger.info(f"Iter {iteration+1} V: Keystone negligible (left_diff={left_diff:.1f}px, right_diff={right_diff:.1f}px), skipping")
        return current, 0.0, None

    # Visualize the trapezoid
    vis2 = current.copy()
    pts = np.array([[lt_x, 0], [rt_x, 0], [rb_x, img_h], [lb_x, img_h]], dtype=np.int32)
    cv2.polylines(vis2, [pts], True, (0, 0, 255), 3)
    for px, py in pts:
        cv2.circle(vis2, (px, py), 8, (255, 0, 0), -1)

    save_step(vis2, f'iter{iteration+1}_v_trapezoid',
              f'Iter {iteration+1} V trapezoid: LT_x={lt_x:.0f}, LB_x={lb_x:.0f}, RT_x={rt_x:.0f}, RB_x={rb_x:.0f}',
              {'lt_x': lt_x, 'lb_x': lb_x, 'rt_x': rt_x, 'rb_x': rb_x})

    # Apply perspective transform with damping
    damping = 0.7
    full_dst_left_x = (lt_x + lb_x) / 2
    full_dst_right_x = (rt_x + rb_x) / 2

    dst_tl_x = lt_x + (full_dst_left_x - lt_x) * damping
    dst_bl_x = lb_x + (full_dst_left_x - lb_x) * damping
    dst_tr_x = rt_x + (full_dst_right_x - rt_x) * damping
    dst_br_x = rb_x + (full_dst_right_x - rb_x) * damping

    src = np.array([
        [lt_x, 0], [rt_x, 0], [rb_x, img_h], [lb_x, img_h]
    ], dtype=np.float32)

    dst = np.array([
        [dst_tl_x, 0], [dst_tr_x, 0], [dst_br_x, img_h], [dst_bl_x, img_h]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    corrected = cv2.warpPerspective(current, M, (img_w, img_h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))

    keystone_px = max(left_diff, right_diff)

    save_step(corrected, f'iter{iteration+1}_v_keystone',
              f'Iter {iteration+1} V: Keystone correction (left_diff={left_diff:.1f}px, right_diff={right_diff:.1f}px)',
              {'iteration': iteration+1, 'left_diff_px': left_diff, 'right_diff_px': right_diff})

    return corrected, keystone_px, M.astype(np.float64)
