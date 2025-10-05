# backend/app/analysis/image_analyzer.py (Phase 1: YOLOv8 + Document Detection)

import exifread
import easyocr
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
from ultralytics import YOLO
from . import text_analyzer
from ..fusion.multimodal_fusion import multimodal_analyzer

try:
    # Initialize the OCR reader once for efficiency
    reader = easyocr.Reader(['en'])
except Exception as e:
    print(f"Failed to initialize EasyOCR: {e}")
    reader = None

try:
    # Initialize YOLOv8 model for document detection
    # Using COCO pretrained model which can detect various objects including documents
    yolo_model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Failed to initialize YOLOv8 model: {e}")
    yolo_model = None

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

def detect_documents(image_path: str) -> List[Dict[str, Any]]:
    """
    Uses YOLOv8 to detect document-like objects in the image.
    Returns list of detected documents with bounding boxes, confidence, and OCR text.
    """
    if not yolo_model or not reader:
        return []
    
    detected_documents = []
    
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        # Run YOLOv8 inference
        results = yolo_model(image)
        
        # COCO class IDs that might represent documents/cards
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus, 6: train, 7: truck
        # 8: boat, 9: traffic light, 10: fire hydrant, 11: stop sign, 12: parking meter
        # 13: bench, 14: bird, 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow
        # 20: elephant, 21: bear, 22: zebra, 23: giraffe, 24: backpack, 25: umbrella
        # 26: handbag, 27: tie, 28: suitcase, 29: frisbee, 30: skis, 31: snowboard
        # 32: sports ball, 33: kite, 34: baseball bat, 35: baseball glove, 36: skateboard
        # 37: surfboard, 38: tennis racket, 39: bottle, 40: wine glass, 41: cup
        # 42: fork, 43: knife, 44: spoon, 45: bowl, 46: banana, 47: apple
        # 48: sandwich, 49: orange, 50: broccoli, 51: carrot, 52: hot dog, 53: pizza
        # 54: donut, 55: cake, 56: chair, 57: couch, 58: potted plant, 59: bed
        # 60: dining table, 61: toilet, 62: tv, 63: laptop, 64: mouse, 65: remote
        # 66: keyboard, 67: cell phone, 68: microwave, 69: oven, 70: toaster
        # 71: sink, 72: refrigerator, 73: book, 74: clock, 75: vase, 76: scissors
        # 77: teddy bear, 78: hair drier, 79: toothbrush
        
        # Focus on objects that could be documents or cards
        document_related_classes = [73]  # book - most likely to be documents
        confidence_threshold = 0.3
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if it's a document-related class and meets confidence threshold
                    if class_id in document_related_classes and confidence >= confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Crop the detected region
                        cropped_image = image[y1:y2, x1:x2]
                        
                        # Run OCR on the cropped region
                        try:
                            ocr_results = reader.readtext(cropped_image, detail=0, paragraph=True)
                            ocr_text = " ".join(ocr_results) if ocr_results else ""
                        except Exception as e:
                            print(f"OCR failed on cropped region: {e}")
                            ocr_text = ""
                        
                        # Store detection information
                        detected_doc = {
                            "class_id": class_id,
                            "class_name": yolo_model.names[class_id],
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2],
                            "ocr_text": ocr_text
                        }
                        
                        detected_documents.append(detected_doc)
        
        # Also check for any rectangular regions that might be documents
        # This is a fallback for when YOLO doesn't detect documents but there might be cards/IDs
        if len(detected_documents) == 0:
            detected_documents = detect_rectangular_regions(image, image_path)
            
    except Exception as e:
        print(f"Document detection failed: {e}")
        return []
    
    return detected_documents

def detect_rectangular_regions(image: np.ndarray, image_path: str) -> List[Dict[str, Any]]:
    """
    Fallback method to detect rectangular regions that might be documents/cards.
    Uses contour detection to find rectangular shapes.
    """
    detected_documents = []
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for rectangular contours that might be documents
        for contour in contours:
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners) and has reasonable size
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                image_area = image.shape[0] * image.shape[1]
                
                # Filter for reasonably sized rectangular regions
                if area > image_area * 0.01 and area < image_area * 0.8:  # 1% to 80% of image
                    aspect_ratio = w / h
                    # Look for card-like aspect ratios (roughly 1.4:1 to 1.8:1)
                    if 0.5 <= aspect_ratio <= 2.0:
                        # Crop and run OCR
                        cropped_image = image[y:y+h, x:x+w]
                        
                        try:
                            ocr_results = reader.readtext(cropped_image, detail=0, paragraph=True)
                            ocr_text = " ".join(ocr_results) if ocr_results else ""
                            
                            # Only include if OCR found some text
                            if ocr_text.strip():
                                detected_doc = {
                                    "class_id": 999,  # Custom ID for rectangular regions
                                    "class_name": "rectangular_region",
                                    "confidence": 0.5,  # Medium confidence for fallback method
                                    "bbox": [x, y, x+w, y+h],
                                    "ocr_text": ocr_text
                                }
                                detected_documents.append(detected_doc)
                        except Exception as e:
                            print(f"OCR failed on rectangular region: {e}")
                            continue
                            
    except Exception as e:
        print(f"Rectangular region detection failed: {e}")
    
    return detected_documents

def analyze_image(image_path: str) -> Dict[str, Any]:
    """Analyzes an image for metadata, OCR text, document detection, then scores the text for PII."""
    if not reader:
        return {"error": "OCR reader not available."}

    exif_data = get_exif_data(image_path)
    
    # Run full-image OCR
    try:
        ocr_results = reader.readtext(image_path, detail=0, paragraph=True)
        ocr_text = " ".join(ocr_results)
    except Exception as e:
        print(f"Full image OCR failed: {e}")
        ocr_text = ""
    
    # Detect documents in the image
    detected_documents = detect_documents(image_path)
    
    # Extract visual regions for multimodal analysis
    visual_regions = []
    visual_meta = []
    if detected_documents:
        try:
            image = cv2.imread(image_path)
            if image is not None:
                for doc in detected_documents:
                    x1, y1, x2, y2 = doc["bbox"]
                    cropped_region = image[y1:y2, x1:x2]
                    visual_regions.append(cropped_region)
                    visual_meta.append({
                        "bbox": doc.get("bbox"),
                        "class_name": doc.get("class_name"),
                        "confidence": doc.get("confidence"),
                    })
        except Exception as e:
            print(f"Failed to extract visual regions: {e}")
    
    # Prepare text content for multimodal analysis
    text_content = []
    text_meta = []
    if ocr_text.strip():
        text_content.append(ocr_text)
        text_meta.append({
            "source": "full_image_ocr",
            "text_span": [0, min(len(ocr_text), 200)]  # coarse span for preview
        })
    if detected_documents:
        for doc in detected_documents:
            if doc["ocr_text"].strip():
                text_content.append(doc["ocr_text"])
                text_meta.append({
                    "source": "document_region_ocr",
                    "text_span": [0, min(len(doc["ocr_text"]), 200)]
                })
    
    # Run multimodal fusion analysis
    fusion_results = multimodal_analyzer.analyze_multimodal_content(
        visual_regions=visual_regions,
        text_content=text_content,
        audio_segments=None,  # No audio for images
        visual_meta=visual_meta,
        text_meta=text_meta,
        audio_meta=None,
    )
    
    # If no text is found in full image and no documents detected, return minimal response
    if not ocr_text and not detected_documents:
        return {
            "metadata": exif_data,
            "ocr_text": "No text found.",
            "detected_documents": [],
            "fusion_analysis": fusion_results,
            "sentiment": None,
            "evidence": [],
            "score": {
                "rawscore": 0.0,
                "exposurescore": 0,
                "details": []
            }
        }
    
    # Combine OCR text from full image and detected documents
    all_ocr_text = ocr_text
    if detected_documents:
        document_texts = [doc["ocr_text"] for doc in detected_documents if doc["ocr_text"]]
        if document_texts:
            all_ocr_text += " " + " ".join(document_texts)
    
    # If no combined text is found, return results without text analysis
    if not all_ocr_text.strip():
        return {
            "metadata": exif_data,
            "ocr_text": ocr_text,
            "detected_documents": detected_documents,
            "fusion_analysis": fusion_results,
            "sentiment": None,
            "evidence": [],
            "score": {
                "rawscore": 0.0,
                "exposurescore": 0,
                "details": []
            }
        }
    
    # Run text analysis on combined OCR text
    text_analysis_results = text_analyzer.analyze_text(all_ocr_text)
    
    # Return the combined results
    return {
        "metadata": exif_data,
        "ocr_text": ocr_text,
        "detected_documents": detected_documents,
        "fusion_analysis": fusion_results,
        "score": text_analysis_results.get('score'),
        "evidence": text_analysis_results.get('evidence'),
        "sentiment": text_analysis_results.get('sentiment')
    }