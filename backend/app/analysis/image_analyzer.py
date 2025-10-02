# backend/app/analysis/image_analyzer.py (Corrected)

import exifread
import easyocr
from typing import Dict, Any
from . import text_analyzer

try:
    # Initialize the OCR reader once for efficiency
    reader = easyocr.Reader(['en'])
except Exception as e:
    print(f"Failed to initialize EasyOCR: {e}")
    reader = None

def get_exif_data(image_path: str) -> Dict[str, str]:
    """Extracts EXIF metadata from an image file."""
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            # Filter out the large thumbnail data
            return {str(key): str(val) for key, val in tags.items() if key != 'JPEGThumbnail'}
    except Exception as e:
        print(f"Could not read EXIF data: {e}")
        return {"error": str(e)}

def analyze_image(image_path: str) -> Dict[str, Any]:
    """Analyzes an image for metadata and OCR text, then scores the text for PII."""
    if not reader:
        return {"error": "OCR reader not available."}

    exif_data = get_exif_data(image_path)
    try:
        ocr_results = reader.readtext(image_path, detail=0, paragraph=True)
        ocr_text = " ".join(ocr_results)
    except Exception as e:
        print(f"OCR failed: {e}")
        ocr_text = ""

    # --- THIS IS THE CORRECTED BLOCK ---
    # If no text is found, return a valid response that matches the Pydantic models.
    if not ocr_text:
        return {
            "metadata": exif_data,
            "ocr_text": "No text found.",
            "sentiment": None,
            "evidence": [],
            "score": {
                "rawscore": 0.0,
                "exposurescore": 0,
                "details": []
            }
        }
    
    # If text is found, run it through the advanced text analyzer
    text_analysis_results = text_analyzer.analyze_text(ocr_text)
    
    # Return the combined results
    return {
        "metadata": exif_data,
        "ocr_text": ocr_text,
        "score": text_analysis_results.get('score'),
        "evidence": text_analysis_results.get('evidence'),
        "sentiment": text_analysis_results.get('sentiment')
    }
