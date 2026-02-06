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
from typing import Dict, Optional, Tuple

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
