"""
Image preprocessing service for Exhibit G forms.
Transforms phone photos into flatbed-scan quality images.

AGGRESSIVE pipeline for arbitrary phone photo orientations:
1. Load image (HEIC/JPG/PNG)
2. Apply EXIF orientation tag (auto-rotate from camera metadata)
3. Resize if needed
4. Detect major orientation (0°/90°/180°/270°) via pytesseract OSD
5. Multi-rotation trial: try all 4 orientations, pick best document detection
6. Detect document boundary (find 4 corners of the paper)
7. Perspective correction (warp trapezoid to rectangle)
8. Fine rotation correction (Hough lines, any angle)
9. Lighting normalization (division-based + CLAHE)
10. Binarize (adaptive threshold)
11. Denoise (median blur)
"""

import os
import uuid
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

# Maximum dimension for initial resize (before perspective correction)
MAX_DIMENSION = 3000

# Standard Exhibit G dimensions (US Letter landscape @ 200 DPI)
EXHIBIT_G_WIDTH = 2200   # 11 inches
EXHIBIT_G_HEIGHT = 1700  # 8.5 inches

# Minimum document area as fraction of image area
MIN_DOC_AREA_FRACTION = 0.08  # 8% - very relaxed for distant/angled photos


# ──────────────────────────────────────────────
# Step 1: Image Loading + EXIF Orientation
# ──────────────────────────────────────────────

def convert_heic_to_pil(file_path: str) -> Image.Image:
    """Convert HEIC/HEIF file to PIL Image."""
    from pillow_heif import register_heif_opener
    register_heif_opener()
    return Image.open(file_path)


def apply_exif_orientation(pil_img: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation tag to properly orient the image.
    Phone cameras store the physical orientation in EXIF tag 274.
    Without this, a photo taken sideways appears sideways.
    """
    try:
        exif = pil_img.getexif()
        orientation_tag = 274  # EXIF Orientation tag number

        if orientation_tag not in exif:
            return pil_img

        orientation = exif[orientation_tag]
        logger.info(f"EXIF orientation tag: {orientation}")

        # Standard EXIF orientation transformations
        if orientation == 2:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            pil_img = pil_img.rotate(180, expand=True)
        elif orientation == 4:
            pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
        elif orientation == 6:
            pil_img = pil_img.rotate(270, expand=True)
        elif orientation == 7:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        elif orientation == 8:
            pil_img = pil_img.rotate(90, expand=True)

    except Exception as e:
        logger.warning(f"Could not read EXIF orientation: {e}")

    return pil_img


def load_image(file_path: str) -> np.ndarray:
    """
    Load an image from any supported format (JPG, PNG, HEIC) as BGR numpy array.
    Applies EXIF orientation to handle phone camera rotation metadata.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ('.heic', '.heif'):
        pil_img = convert_heic_to_pil(file_path)
    else:
        pil_img = Image.open(file_path)

    # Always apply EXIF orientation
    pil_img = apply_exif_orientation(pil_img)

    # Convert to BGR numpy array for OpenCV
    rgb_array = np.array(pil_img.convert('RGB'))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    return bgr_array


def resize_if_needed(image: np.ndarray) -> np.ndarray:
    """Resize image if either dimension exceeds MAX_DIMENSION."""
    h, w = image.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


# ──────────────────────────────────────────────
# Step 2: Major Orientation Detection (OSD)
# ──────────────────────────────────────────────

def detect_orientation_osd(image: np.ndarray) -> int:
    """
    Use pytesseract's Orientation and Script Detection (OSD)
    to detect if the image is rotated 0°, 90°, 180°, or 270°.

    Returns the rotation angle needed to correct the image.
    Returns 0 if detection fails or image is already upright.
    """
    try:
        import pytesseract

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Convert to PIL for pytesseract
        pil_img = Image.fromarray(gray)

        osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
        rotation = osd.get('rotate', 0)
        confidence = osd.get('orientation_conf', 0)

        logger.info(f"OSD detected rotation={rotation}°, confidence={confidence}")

        if confidence >= 1.0 and rotation in (0, 90, 180, 270):
            return rotation

    except Exception as e:
        logger.warning(f"OSD orientation detection failed: {e}")

    return 0


def rotate_image_90(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate image by exactly 0, 90, 180, or 270 degrees.
    Uses cv2.rotate for lossless 90-degree rotations.
    """
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image


# ──────────────────────────────────────────────
# Step 3: Multi-Rotation Document Detection
# ──────────────────────────────────────────────

def score_document_detection(image: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """
    Score how well we can detect a document in this orientation.

    Returns:
        (score, corners) where score is higher for better detections.
        score = 0 means no document found.
    """
    h, w = image.shape[:2]
    image_area = h * w
    min_area = image_area * MIN_DOC_AREA_FRACTION

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Try all strategies
    for method in ["canny", "adaptive", "canny_relaxed", "canny_very_relaxed"]:
        corners = _find_quad_contour(gray, min_area, method=method)
        if corners is not None:
            # Score based on:
            # 1. Area of the detected quadrilateral (bigger = better)
            # 2. How rectangular it is (closer to rectangle = better)
            # 3. Whether it looks landscape (Exhibit G is landscape)
            quad_area = cv2.contourArea(corners.reshape(4, 1, 2))
            area_ratio = quad_area / image_area

            # Check rectangularity: ratio of quad area to bounding rect area
            rect = cv2.minAreaRect(corners.reshape(4, 1, 2))
            rect_area = rect[1][0] * rect[1][1]
            rectangularity = quad_area / max(rect_area, 1)

            # Check if landscape orientation after warping
            ordered = order_corners(corners)
            width_top = np.linalg.norm(ordered[1] - ordered[0])
            width_bottom = np.linalg.norm(ordered[2] - ordered[3])
            height_left = np.linalg.norm(ordered[3] - ordered[0])
            height_right = np.linalg.norm(ordered[2] - ordered[1])
            avg_w = (width_top + width_bottom) / 2
            avg_h = (height_left + height_right) / 2
            aspect_ratio = max(avg_w, avg_h) / max(min(avg_w, avg_h), 1)

            # Exhibit G is roughly 11:8.5 ≈ 1.29
            aspect_bonus = 1.0
            if 1.15 < aspect_ratio < 1.45:
                aspect_bonus = 1.5  # Looks like letter-landscape

            # Landscape bonus: wider than tall is expected
            landscape_bonus = 1.2 if avg_w > avg_h else 1.0

            score = area_ratio * rectangularity * aspect_bonus * landscape_bonus
            logger.info(
                f"Document score ({method}): {score:.4f} "
                f"(area={area_ratio:.2f}, rect={rectangularity:.2f}, "
                f"aspect={aspect_ratio:.2f})"
            )
            return score, corners

    return 0.0, None


def find_best_rotation(image: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
    """
    Try all four 90° rotations and return the one that gives the best
    document detection score.

    This handles photos taken sideways, upside down, or at any 90° increment.

    Returns:
        (best_angle, best_corners) - angle is 0, 90, 180, or 270
    """
    best_score = -1
    best_angle = 0
    best_corners = None

    for angle in [0, 90, 180, 270]:
        rotated = rotate_image_90(image, angle)
        score, corners = score_document_detection(rotated)

        logger.info(f"Rotation {angle}°: score={score:.4f}")

        # Only replace best if strictly better (prefer 0° on ties)
        if score > best_score + 0.001:
            best_score = score
            best_angle = angle
            best_corners = corners

    # If no rotation found a document, try OSD as a tiebreaker
    if best_score <= 0:
        osd_angle = detect_orientation_osd(image)
        if osd_angle != 0:
            logger.info(f"No document detected; using OSD angle: {osd_angle}°")
            return osd_angle, None

    logger.info(f"Best rotation: {best_angle}° (score={best_score:.4f})")
    return best_angle, best_corners


# ──────────────────────────────────────────────
# Step 4: Document Boundary Detection
# ──────────────────────────────────────────────

def detect_document_boundary(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the 4 corners of the paper document in a phone photo.

    Tries multiple strategies with increasing relaxation:
    1. Canny edge detection → largest quadrilateral contour
    2. Adaptive threshold → contour search
    3. Relaxed Canny (stronger blur, lower thresholds)
    4. Very relaxed Canny (heavy blur, very low thresholds)
    5. Morphological close → contour search

    Args:
        image: BGR color image

    Returns:
        numpy array of shape (4, 2) with corner coordinates, or None if not found
    """
    h, w = image.shape[:2]
    image_area = h * w
    min_doc_area = image_area * MIN_DOC_AREA_FRACTION

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    strategies = ["canny", "adaptive", "canny_relaxed", "canny_very_relaxed", "morph_close"]

    for strategy in strategies:
        corners = _find_quad_contour(gray, min_doc_area, method=strategy)
        if corners is not None:
            logger.info(f"Document boundary found via {strategy}.")
            return corners

    logger.warning("Could not detect document boundary. Skipping perspective correction.")
    return None


def _find_quad_contour(
    gray: np.ndarray, min_area: float, method: str
) -> Optional[np.ndarray]:
    """
    Find a 4-sided contour in the image using the specified method.
    Uses both RETR_EXTERNAL and RETR_TREE for more robust detection.
    """
    edges = _get_edges(gray, method)
    if edges is None:
        return None

    # Try both external-only and full hierarchy contour modes
    for retr_mode in [cv2.RETR_EXTERNAL, cv2.RETR_TREE]:
        contours, _ = cv2.findContours(edges, retr_mode, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Sort by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Check the top contours for a valid quadrilateral
        for contour in contours[:15]:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)

            # Try different epsilon values for polygon approximation
            for epsilon_factor in [0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
                approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    return approx.reshape(4, 2)

            # Also try: if we have 4-6 vertices, use the convex hull
            # and try to simplify to exactly 4 points
            hull = cv2.convexHull(contour)
            hull_perimeter = cv2.arcLength(hull, True)
            for epsilon_factor in [0.02, 0.04, 0.06, 0.08, 0.10]:
                approx = cv2.approxPolyDP(hull, epsilon_factor * hull_perimeter, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    hull_area = cv2.contourArea(approx)
                    if hull_area >= min_area:
                        return approx.reshape(4, 2)

    return None


def _get_edges(gray: np.ndarray, method: str) -> Optional[np.ndarray]:
    """Generate edge map using the specified method."""
    if method == "canny":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 200)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges

    elif method == "canny_relaxed":
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        return edges

    elif method == "canny_very_relaxed":
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blurred, 10, 50)
        kernel = np.ones((7, 7), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=3)
        # Close gaps
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        return edges

    elif method == "adaptive":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        return edges

    elif method == "morph_close":
        # Heavy morphological approach for very noisy images
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        kernel = np.ones((9, 9), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        return closed

    return None


# ──────────────────────────────────────────────
# Step 5: Perspective Correction
# ──────────────────────────────────────────────

def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: [top-left, top-right, bottom-right, bottom-left].

    Uses the sum/difference method:
    - TL has the smallest x+y
    - BR has the largest x+y
    - TR has the smallest y-x
    - BL has the largest y-x
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    ordered[0] = pts[np.argmin(s)]   # Top-left
    ordered[2] = pts[np.argmax(s)]   # Bottom-right
    ordered[1] = pts[np.argmin(d)]   # Top-right
    ordered[3] = pts[np.argmax(d)]   # Bottom-left

    return ordered


def is_plausible_page_boundary(corners: np.ndarray, image_shape: tuple) -> bool:
    """
    Check whether detected corners look like a full page boundary
    rather than just an inner table or sub-region.

    An Exhibit G page is US Letter landscape (11" x 8.5"), aspect ≈ 1.29.
    If the detected quad is way off from that, it's probably the inner
    table grid, not the page edges.
    """
    ordered = order_corners(corners)
    w_top = np.linalg.norm(ordered[1] - ordered[0])
    w_bot = np.linalg.norm(ordered[2] - ordered[3])
    h_left = np.linalg.norm(ordered[3] - ordered[0])
    h_right = np.linalg.norm(ordered[2] - ordered[1])

    avg_w = (w_top + w_bot) / 2
    avg_h = (h_left + h_right) / 2

    ratio = max(avg_w, avg_h) / max(min(avg_w, avg_h), 1)

    # US Letter is 1.29:1. Accept anything between 1.05 and 1.6.
    # Anything above ~1.7 is likely the table grid (which is very wide).
    is_plausible = 1.05 < ratio < 1.6

    logger.info(
        f"Page boundary check: {avg_w:.0f}x{avg_h:.0f}, "
        f"ratio={ratio:.2f}, plausible={is_plausible}"
    )
    return is_plausible


def correct_perspective(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Warp the image so the document fills a standard rectangle.

    Uses the Exhibit G standard page dimensions (letter landscape)
    since we know what the document should look like.

    Args:
        image: Full input image (BGR or grayscale)
        corners: Ordered corners [TL, TR, BR, BL] as float32

    Returns:
        Warped image at standard Exhibit G dimensions
    """
    src = corners.astype(np.float32)

    # Calculate detected dimensions to determine orientation
    width_top = np.linalg.norm(src[1] - src[0])
    width_bottom = np.linalg.norm(src[2] - src[3])
    height_left = np.linalg.norm(src[3] - src[0])
    height_right = np.linalg.norm(src[2] - src[1])

    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2

    # Determine if landscape (standard) or portrait (rotated photo)
    if avg_width >= avg_height:
        out_w, out_h = EXHIBIT_G_WIDTH, EXHIBIT_G_HEIGHT
    else:
        out_w, out_h = EXHIBIT_G_HEIGHT, EXHIBIT_G_WIDTH

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)

    border_val = (255, 255, 255) if len(image.shape) == 3 else 255
    warped = cv2.warpPerspective(
        image, M, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val
    )

    return warped


def correct_keystone(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Apply keystone (perspective) correction to the ENTIRE image using a
    detected sub-region as reference. Unlike correct_perspective() which
    crops to just the quad, this transforms the full image so that the
    detected quad becomes rectangular, and everything else comes along.

    This is used when we detect the inner table grid (not the page boundary).
    We straighten the whole photo so the table lines become horizontal/vertical,
    without losing any surrounding content (header, margins, etc.).

    Args:
        image: Full input image (BGR)
        corners: 4 corners of the detected sub-region (e.g., table grid)

    Returns:
        Perspective-corrected full image
    """
    h, w = image.shape[:2]
    src = order_corners(corners).astype(np.float32)

    # Compute what the quad's rectangle should look like (same position, but
    # straightened to an axis-aligned rectangle)
    width_top = np.linalg.norm(src[1] - src[0])
    width_bottom = np.linalg.norm(src[2] - src[3])
    height_left = np.linalg.norm(src[3] - src[0])
    height_right = np.linalg.norm(src[2] - src[1])

    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2

    # Place the straightened rectangle at the centroid of the original quad
    cx = np.mean(src[:, 0])
    cy = np.mean(src[:, 1])

    dst = np.array([
        [cx - avg_width / 2, cy - avg_height / 2],
        [cx + avg_width / 2, cy - avg_height / 2],
        [cx + avg_width / 2, cy + avg_height / 2],
        [cx - avg_width / 2, cy + avg_height / 2],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)

    # Transform all 4 image corners through M to find the output bounds
    img_corners = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype=np.float32).reshape(-1, 1, 2)

    transformed_corners = cv2.perspectiveTransform(img_corners, M).reshape(-1, 2)

    # Compute the bounding box of the transformed image
    x_min = transformed_corners[:, 0].min()
    y_min = transformed_corners[:, 1].min()
    x_max = transformed_corners[:, 0].max()
    y_max = transformed_corners[:, 1].max()

    # Shift everything so it fits in positive coordinates
    shift = np.array([[1, 0, -x_min],
                      [0, 1, -y_min],
                      [0, 0, 1]], dtype=np.float64)
    M_shifted = shift @ M

    out_w = int(np.ceil(x_max - x_min))
    out_h = int(np.ceil(y_max - y_min))

    # Cap output size to prevent memory explosion
    max_out = MAX_DIMENSION + 500
    if out_w > max_out or out_h > max_out:
        scale = max_out / max(out_w, out_h)
        out_w = int(out_w * scale)
        out_h = int(out_h * scale)
        scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
        M_shifted = scale_mat @ M_shifted

    border_val = (255, 255, 255) if len(image.shape) == 3 else 255
    warped = cv2.warpPerspective(
        image, M_shifted, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val
    )

    logger.info(
        f"Keystone corrected: {w}x{h} -> {out_w}x{out_h} "
        f"(using {avg_width:.0f}x{avg_height:.0f} reference quad)"
    )

    return warped


# ──────────────────────────────────────────────
# Step 6: Fine Rotation Correction (any angle)
# ──────────────────────────────────────────────

def detect_skew_angle(gray: np.ndarray, threshold: int = 80, post_perspective: bool = False) -> float:
    """
    Detect skew angle using Hough line transform.

    After perspective correction, residual skew is typically small (< 5°),
    so we use a tight filter. Without perspective correction, we allow
    larger deviations.

    Args:
        gray: Grayscale image
        threshold: Hough accumulator threshold
        post_perspective: If True, only look for small residual skew (±5°)

    Returns angle in degrees to rotate the image.
    """
    # After perspective correction, skew should be small
    max_dev = 5.0 if post_perspective else 30.0

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Strategy 1: Probabilistic Hough (length-weighted, more reliable)
    angle = _probabilistic_hough_angle(edges, gray.shape, max_deviation=max_dev)
    if angle is not None:
        return angle

    # Strategy 2: Standard Hough
    angle = _hough_angle(edges, threshold, max_deviation=max_dev)
    if angle is not None:
        return angle

    # Strategy 3: Lower threshold
    angle = _hough_angle(edges, max(threshold // 2, 30), max_deviation=max_dev)
    if angle is not None:
        return angle

    return 0.0


def _hough_angle(edges: np.ndarray, threshold: int, max_deviation: float = 10) -> Optional[float]:
    """
    Extract skew angle from Hough lines.
    max_deviation: maximum angle from horizontal to consider (degrees).
    """
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    if lines is None:
        return None

    angles = []
    for rho, theta in lines[:, 0]:
        angle_deg = (theta * 180 / np.pi) - 90
        # Accept lines within max_deviation of horizontal
        if -max_deviation < angle_deg < max_deviation:
            angles.append(angle_deg)

    if not angles:
        return None

    # Use median to be robust to outliers
    median_angle = float(np.median(angles))

    if abs(median_angle) < 0.1:
        return 0.0

    return median_angle


def _probabilistic_hough_angle(edges: np.ndarray, shape: tuple, max_deviation: float = 10) -> Optional[float]:
    """
    Use probabilistic Hough transform for line detection.
    Weight angles by line length so long table lines dominate over short noise.
    """
    h, w = shape[:2]
    min_length = min(h, w) // 6  # Require fairly long lines

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=50,
        minLineLength=min_length,
        maxLineGap=20
    )

    if lines is None:
        return None

    angles = []
    weights = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only near-horizontal lines within max_deviation
        if -max_deviation < angle < max_deviation:
            angles.append(angle)
            weights.append(length)

    if not angles:
        return None

    # Length-weighted average (long lines more reliable than short ones)
    angles = np.array(angles)
    weights = np.array(weights)
    weighted_angle = float(np.average(angles, weights=weights))

    logger.info(
        f"Probabilistic Hough: {len(angles)} lines, "
        f"weighted angle={weighted_angle:.2f}°"
    )

    if abs(weighted_angle) < 0.1:
        return 0.0

    return weighted_angle


def correct_skew(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the image to correct skew. Handles any angle."""
    if abs(angle) < 0.1:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    border_val = (255, 255, 255) if len(image.shape) == 3 else 255
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val
    )
    return rotated


# ──────────────────────────────────────────────
# Step 7: Ensure Landscape Orientation
# ──────────────────────────────────────────────

def ensure_landscape(image: np.ndarray) -> np.ndarray:
    """
    Exhibit G is a landscape form. If the image is portrait after all
    corrections, rotate it 90° to landscape.
    """
    h, w = image.shape[:2]
    if h > w:
        logger.info("Image is portrait; rotating to landscape for Exhibit G.")
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _ocr_keyword_score(gray_strip: np.ndarray) -> int:
    """
    Run pytesseract on a grayscale image strip and count how many
    known Exhibit G header keywords appear in the recognized text.

    Returns the number of keyword matches found.
    """
    # Known words that appear in the top header area of an Exhibit G form
    EXHIBIT_G_KEYWORDS = [
        "exhibit", "sag", "aftra", "actors", "production", "time", "report",
        "day", "date", "player", "role", "status", "work", "meal",
        "category", "serial", "episode", "prod", "no", "call",
        "makeup", "wardrobe", "travel", "stunt", "adjustment",
        "ndb", "ndd",  # ND Breakfast / ND Dinner abbreviations on form
    ]

    try:
        import pytesseract
        pil_img = Image.fromarray(gray_strip)
        text = pytesseract.image_to_string(pil_img, config='--psm 6').lower()
        score = sum(1 for kw in EXHIBIT_G_KEYWORDS if kw in text)
        logger.info(f"OCR keyword score: {score} (text snippet: {text[:120]!r})")
        return score
    except Exception as e:
        logger.warning(f"OCR keyword scoring failed: {e}")
        return 0


def check_upside_down(image: np.ndarray, doc_region: Optional[np.ndarray] = None) -> bool:
    """
    Determine if the document is upside down using OCR keyword matching.

    If doc_region (4 corners of detected document area) is provided,
    we crop to just the document area first. Otherwise we use the
    central band of the image to avoid background areas.

    Strategy:
    1. OCR the top portion and look for known Exhibit G keywords.
       Then OCR the top of the 180°-rotated version.
    2. Whichever has more keywords is correct.
    3. On tie: assume upright (don't flip on inconclusive data).

    Returns True if the image should be flipped 180°.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape[:2]

    # If we know where the document is and it's a meaningful sub-region
    # (covers < 85% of the image), crop to it for more reliable OCR
    if doc_region is not None:
        ordered = order_corners(doc_region)
        y_min = max(0, int(min(ordered[:, 1])))
        y_max = min(h, int(max(ordered[:, 1])))
        x_min = max(0, int(min(ordered[:, 0])))
        x_max = min(w, int(max(ordered[:, 0])))
        region_area = (y_max - y_min) * (x_max - x_min)
        image_area = h * w

        if region_area < image_area * 0.85:
            doc_gray = gray[y_min:y_max, x_min:x_max]
            dh, dw = doc_gray.shape[:2]
            logger.info(f"Upside-down check: cropped to doc region ({dw}x{dh})")
        else:
            # Region covers almost the whole image, use center band instead
            margin_y = int(h * 0.10)
            margin_x = int(w * 0.05)
            doc_gray = gray[margin_y:h - margin_y, margin_x:w - margin_x]
            dh, dw = doc_gray.shape[:2]
            logger.info(f"Upside-down check: using center band ({dw}x{dh})")
    else:
        doc_gray = gray
        dh, dw = h, w

    # --- OCR keyword matching ---
    top_strip = doc_gray[0:int(dh * 0.25), :]
    flipped = cv2.rotate(doc_gray, cv2.ROTATE_180)
    flipped_top_strip = flipped[0:int(dh * 0.25), :]

    upright_score = _ocr_keyword_score(top_strip)
    flipped_score = _ocr_keyword_score(flipped_top_strip)

    logger.info(f"Upside-down check: upright_keywords={upright_score}, flipped_keywords={flipped_score}")

    # Need a clear winner — require at least 2-point margin to flip
    if flipped_score >= upright_score + 2:
        logger.info("OCR keywords indicate image is upside down. Rotating 180°.")
        return True
    elif upright_score >= flipped_score:
        logger.info("OCR keywords confirm image is upright (or inconclusive — keeping as-is).")
        return False

    # Slight flipped advantage (1 point) — not enough confidence to flip
    logger.info("OCR keyword difference too small to flip. Keeping current orientation.")
    return False


# ──────────────────────────────────────────────
# Step 8: Lighting Normalization
# ──────────────────────────────────────────────

def normalize_lighting(gray: np.ndarray) -> np.ndarray:
    """
    Normalize uneven lighting by dividing by a large-blur approximation
    of the background illumination.

    Removes shadows, flash hotspots, and ambient light gradients
    to produce the even illumination of a flatbed scanner.
    """
    blur_size = max(gray.shape[1] // 5, 51)
    if blur_size % 2 == 0:
        blur_size += 1

    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    background = np.maximum(background, 1).astype(np.float32)

    normalized = (gray.astype(np.float32) / background) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return normalized


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE for local contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def normalize_lighting_color(image: np.ndarray) -> np.ndarray:
    """
    Normalize lighting on a color image using LAB color space.
    Only the L (lightness) channel is normalized, preserving color.
    """
    if len(image.shape) != 3:
        return normalize_lighting(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_normalized = normalize_lighting(l_channel)
    l_enhanced = apply_clahe(l_normalized)

    lab_normalized = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)

    return result


# ──────────────────────────────────────────────
# Step 9 & 10: Binarize and Denoise
# ──────────────────────────────────────────────

def binarize(gray: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding for binarization."""
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=10
    )
    return binary


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove noise using median blur."""
    denoised = cv2.medianBlur(image, 3)
    return denoised


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def _legacy_heuristic_correction(color_image: np.ndarray) -> np.ndarray:
    """
    Legacy heuristic-based correction pipeline using OpenCV.
    Used as fallback when Claude API is unavailable.

    Steps: multi-rotation trial → perspective/keystone → landscape → upside-down check → fine skew
    """
    # Find best rotation by trying all four 90° orientations
    best_angle, best_corners = find_best_rotation(color_image)

    if best_angle != 0:
        color_image = rotate_image_90(color_image, best_angle)
        logger.info(f"[Legacy] Applied {best_angle}° rotation")

    # Detect document boundary
    if best_corners is not None and best_angle == 0:
        corners = best_corners
    else:
        corners = detect_document_boundary(color_image)

    # Perspective / keystone correction
    doc_region_for_ocr = corners
    perspective_applied = False
    if corners is not None:
        if is_plausible_page_boundary(corners, color_image.shape):
            ordered = order_corners(corners)
            color_image = correct_perspective(color_image, ordered)
            perspective_applied = True
            doc_region_for_ocr = None
        else:
            color_image = correct_keystone(color_image, corners)
            perspective_applied = True
            doc_region_for_ocr = detect_document_boundary(color_image)

    color_image = ensure_landscape(color_image)

    if check_upside_down(color_image, doc_region=doc_region_for_ocr):
        color_image = cv2.rotate(color_image, cv2.ROTATE_180)
        logger.info("[Legacy] Flipped upside-down image 180°")

    # Fine rotation
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    hough_threshold = 80 if perspective_applied else 60
    skew_angle = detect_skew_angle(
        gray, threshold=hough_threshold, post_perspective=perspective_applied
    )
    if abs(skew_angle) > 0.1:
        color_image = correct_skew(color_image, skew_angle)
        logger.info(f"[Legacy] Fine rotation corrected: {skew_angle:.2f}°")

    return color_image


def preprocess_image(file_path: str) -> dict:
    """
    Preprocessing pipeline for phone photos of Exhibit G forms.

    Uses Claude Vision to guide image correction when API key is available,
    with fallback to legacy OpenCV heuristics.

    Pipeline:
    1. Load image with EXIF orientation correction
    2. Resize if too large
    3. Lighting normalization (division + CLAHE) — so Claude sees clean, even lighting
    4. Claude Vision correction loop (or legacy heuristic fallback)
    5. Binarize + denoise

    Returns:
        dict with keys: corrected_image_path, color_corrected_path,
        binary_image_path, skew_angle, gray_image, binary_image, color_image
    """
    # Step 1-2: Load with EXIF orientation and resize
    color_image = load_image(file_path)
    color_image = resize_if_needed(color_image)
    logger.info(f"Loaded image: {color_image.shape[1]}x{color_image.shape[0]}")

    # Step 3: Lighting normalization FIRST — gives Claude a cleaner image to analyze
    color_image = normalize_lighting_color(color_image)
    logger.info("Lighting normalized (before geometric correction)")

    # Step 4: Geometric correction (Claude Vision loop or legacy fallback)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    template_path = os.path.join(BASE_DIR, "static", "exhibit_g_template.png")
    skew_angle = 0.0

    if api_key and os.path.exists(template_path):
        logger.info("Using Claude Vision correction loop")
        try:
            from services.claude_corrector import claude_correction_loop
            color_image = claude_correction_loop(
                color_image, template_path,
                max_iterations=3, api_key=api_key
            )
        except Exception as e:
            logger.error(f"Claude correction loop failed: {e}. Falling back to legacy.")
            color_image = _legacy_heuristic_correction(color_image)
    else:
        if not api_key:
            logger.info("No ANTHROPIC_API_KEY set — using legacy heuristic correction")
        elif not os.path.exists(template_path):
            logger.info(f"Template not found at {template_path} — using legacy heuristic correction")
        color_image = _legacy_heuristic_correction(color_image)

    # Step 5: Final grayscale + CLAHE for OCR engines, binarize + denoise
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray_normalized = normalize_lighting(gray)
    gray_normalized = apply_clahe(gray_normalized)
    color_normalized = color_image  # Already lighting-normalized from step 3
    logger.info("Grayscale processing complete")

    binary = binarize(gray_normalized)
    binary_clean = denoise(binary)

    # Save all versions
    file_id = str(uuid.uuid4())[:8]
    corrected_filename = f"{file_id}_corrected.png"
    color_corrected_filename = f"{file_id}_color_corrected.png"
    binary_filename = f"{file_id}_binary.png"

    corrected_path = os.path.join(PROCESSED_DIR, corrected_filename)
    color_corrected_path = os.path.join(PROCESSED_DIR, color_corrected_filename)
    binary_path = os.path.join(PROCESSED_DIR, binary_filename)

    cv2.imwrite(color_corrected_path, color_normalized)
    cv2.imwrite(corrected_path, gray_normalized)
    cv2.imwrite(binary_path, binary_clean)

    logger.info(f"Saved: {color_corrected_filename} (skew={skew_angle:.2f}°)")

    return {
        "corrected_image_path": corrected_path,
        "color_corrected_path": color_corrected_path,
        "binary_image_path": binary_path,
        "skew_angle": skew_angle,
        "gray_image": gray_normalized,
        "binary_image": binary_clean,
        "color_image": color_normalized,
    }
