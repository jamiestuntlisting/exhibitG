"""
Claude Vision API integration for Exhibit G form transcription.
Uses the full corrected image and asks Claude to extract structured data.
"""

import os
import base64
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


EXHIBIT_G_PROMPT = """You are reading a SAG-AFTRA Exhibit G form (Actors Production Time Report).

First, extract the HEADER fields from the top of the form:
- production_co: Production company name
- series_title: Series/show title
- episode_name: Episode name/number
- prod_number: Production number (Prod#)
- date: Date shown on the form
- day_x_of_y: "Day X of Y" value
- contact: Contact name
- phone: Phone number

Then, read the TABLE containing cast/crew time data. The table columns are (in order from left to right):
1. cast_number - Row number (1, 2, 3, etc.)
2. cast_name - Actor/performer name (CAST column)
3. character - Character name
4. status_code - Status code: W (Worked), H (Hold), S (Start), WF (Work/Finish), F (Finish), R (Rehearsal), FT (Fitting), TR (Travel), T (Test), N/P (Not Photographed)
5. makeup_time - Make-up/Hair/Wardrobe report time
6. report_on_set - Report on Set time
7. dismiss_on_set - Dismiss on Set time
8. dismiss_mu_hair - Dismiss Make-up/Hair/Wardrobe time
9. nd_breakfast_out - Non-Deductible Breakfast Out time
10. nd_breakfast_in - Non-Deductible Breakfast In time
11. first_meal_in - 1st Meal In (start) time
12. first_meal_out - 1st Meal Out (end) time
13. second_meal_in - 2nd Meal In (start) time
14. second_meal_out - 2nd Meal Out (end) time
15. leave_for_loc - Leave for Location time
16. arrive_on_loc - Arrive on Location time
17. leave_loc - Leave Location time
18. arrive_hotel - Arrive at Hotel/Studio time
19. stunt_adjust - Stunt Adjustment
20. mileage - Mileage
21. mpv_fc - MPV or Forced Calls

IMPORTANT INSTRUCTIONS:
- Read ALL rows that contain handwritten data. Skip completely empty rows.
- For time values, preserve the exact format as written (e.g., "10:30A", "12P", "5:45P", "11A", "1138A")
- Times often use shorthand: "A" = AM, "P" = PM
- If a cell is empty, use an empty string ""
- Read names carefully - they are handwritten
- Status codes are usually single letters or two-letter abbreviations
- Some cells may have superscript or subscript notation for minutes

Return your response as a JSON object with this exact structure:
{
  "header": {
    "production_co": "",
    "series_title": "",
    "episode_name": "",
    "prod_number": "",
    "date": "",
    "day_x_of_y": "",
    "contact": "",
    "phone": ""
  },
  "rows": [
    {
      "cast_number": "1",
      "cast_name": "",
      "character": "",
      "status_code": "",
      "makeup_time": "",
      "report_on_set": "",
      "dismiss_on_set": "",
      "dismiss_mu_hair": "",
      "nd_breakfast_out": "",
      "nd_breakfast_in": "",
      "first_meal_in": "",
      "first_meal_out": "",
      "second_meal_in": "",
      "second_meal_out": "",
      "leave_for_loc": "",
      "arrive_on_loc": "",
      "leave_loc": "",
      "arrive_hotel": "",
      "stunt_adjust": "",
      "mileage": "",
      "mpv_fc": ""
    }
  ]
}

Return ONLY the JSON object, no other text."""


def encode_image_to_base64(image_path: str, max_bytes: int = 4_500_000) -> str:
    """Read an image file and encode it as base64 JPEG, resizing if needed.

    Claude Vision has a 5MB per-image limit. We compress to JPEG and resize
    if necessary to stay under the limit.
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        # Fallback: read raw file
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    quality = 85
    scale = 1.0
    orig_h, orig_w = img.shape[:2]

    for attempt in range(5):
        if scale < 1.0:
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = img

        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        raw_bytes = buf.tobytes()

        if len(raw_bytes) <= max_bytes:
            logger.info(f"OCR image encoded: {resized.shape[1]}x{resized.shape[0]}, "
                       f"quality={quality}, size={len(raw_bytes)/1024:.0f}KB")
            return base64.standard_b64encode(raw_bytes).decode("utf-8")

        if quality > 60:
            quality -= 10
        else:
            scale *= 0.75

    # Last resort
    resized = cv2.resize(img, (min(orig_w, 1600), min(orig_h, 1200)),
                         interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def get_media_type(image_path: str) -> str:
    """Always return JPEG since we re-encode images as JPEG for size."""
    return "image/jpeg"


def read_with_claude(image_path: str, api_key: Optional[str] = None) -> Dict:
    """
    Send the full Exhibit G image to Claude Vision API for transcription.

    Args:
        image_path: Path to the corrected/preprocessed image
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var)

    Returns:
        Dict with "header" and "rows" keys
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        logger.error("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")
        return {"header": {}, "rows": []}

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        image_data = encode_image_to_base64(image_path)
        media_type = get_media_type(image_path)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": EXHIBIT_G_PROMPT,
                        },
                    ],
                }
            ],
        )

        # Parse the response
        response_text = message.content[0].text.strip()

        # Try to extract JSON from the response
        # Sometimes Claude wraps it in ```json ... ```
        if response_text.startswith("```"):
            # Remove markdown code block
            lines = response_text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or not line.strip().startswith("```"):
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        result = json.loads(response_text)

        # Validate structure
        if "header" not in result:
            result["header"] = {}
        if "rows" not in result:
            result["rows"] = []

        logger.info(f"Claude extracted {len(result['rows'])} rows from image.")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        logger.error(f"Response was: {response_text[:500]}")
        return {"header": {}, "rows": []}
    except Exception as e:
        logger.error(f"Claude OCR error: {e}")
        return {"header": {}, "rows": []}


def claude_results_to_cell_readings(claude_result: Dict) -> List[Dict]:
    """
    Convert Claude's structured response to individual cell readings
    matching the format from ocr_engine.read_all_cells().

    Returns:
        List of dicts with keys: row_index, col_index, column_name, text, confidence, engine
    """
    readings = []
    column_names = [
        "cast_number", "cast_name", "character", "status_code",
        "makeup_time", "report_on_set", "dismiss_on_set", "dismiss_mu_hair",
        "nd_breakfast_out", "nd_breakfast_in",
        "first_meal_in", "first_meal_out",
        "second_meal_in", "second_meal_out",
        "leave_for_loc", "arrive_on_loc", "leave_loc", "arrive_hotel",
        "stunt_adjust", "mileage", "mpv_fc",
    ]

    for row_idx, row_data in enumerate(claude_result.get("rows", [])):
        for col_idx, col_name in enumerate(column_names):
            value = row_data.get(col_name, "")
            readings.append({
                "row_index": row_idx,
                "col_index": col_idx,
                "column_name": col_name,
                "text": str(value) if value else "",
                "confidence": 0.95,  # Claude doesn't provide confidence scores
                "engine": "claude",
            })

    return readings
