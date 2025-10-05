import os
import whisper
import moviepy.editor as mp
import cv2
import numpy as np
from typing import Dict, Any, List
from ultralytics import YOLO
import easyocr
from . import text_analyzer
from ..fusion.multimodal_fusion import multimodal_analyzer

try:
    # Load the Whisper model once when the application starts
    whisper_model = whisper.load_model("base")
except Exception as e:
    print(f"FATAL: Failed to load Whisper model: {e}")
    whisper_model = None

try:
    # Initialize YOLOv8 model for document detection
    yolo_model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Failed to initialize YOLOv8 model: {e}")
    yolo_model = None

try:
    # Initialize OCR reader
    reader = easyocr.Reader(['en'])
except Exception as e:
    print(f"Failed to initialize EasyOCR: {e}")
    reader = None

def detect_documents_in_frame(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detects documents in a single video frame using YOLOv8.
    Returns list of detected documents with bounding boxes and OCR text.
    """
    if not yolo_model or not reader:
        return []
    
    detected_documents = []
    
    try:
        # Run YOLOv8 inference on frame
        results = yolo_model(frame)
        
        # Focus on document-related classes - expanded for better detection
        document_related_classes = [
            73,  # book - documents, IDs, cards
            67,  # cell phone - may show sensitive info on screen
            63,  # laptop - may show sensitive info on screen
            0,   # person - faces are PII
        ]
        confidence_threshold = 0.25  # Lower threshold to catch more potential documents
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in document_related_classes and confidence >= confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Crop the detected region
                        cropped_frame = frame[y1:y2, x1:x2]
                        
                        # Run OCR on the cropped region
                        try:
                            ocr_results = reader.readtext(cropped_frame, detail=0, paragraph=True)
                            ocr_text = " ".join(ocr_results) if ocr_results else ""
                        except Exception as e:
                            print(f"OCR failed on cropped frame region: {e}")
                            ocr_text = ""
                        
                        # Store detection information
                        detected_doc = {
                            "class_id": class_id,
                            "class_name": yolo_model.names[class_id],
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2],
                            "ocr_text": ocr_text,
                            "frame_timestamp": None  # Will be set by caller
                        }
                        
                        detected_documents.append(detected_doc)
        
    except Exception as e:
        print(f"Document detection failed on frame: {e}")
    
    return detected_documents

def extract_key_frames(video_path: str, max_frames: int = 10) -> List[tuple]:
    """
    Extracts key frames from video for document detection.
    Returns list of (frame, timestamp) tuples.
    """
    key_frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return key_frames
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame indices to sample
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        # Extract frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                timestamp = frame_idx / fps if fps > 0 else 0
                key_frames.append((frame, timestamp))
        
        cap.release()
        
    except Exception as e:
        print(f"Key frame extraction failed: {e}")
    
    return key_frames

def analyze_video(video_path: str) -> Dict[str, Any]:
    """
    Analyzes a video file by:
    1. Extracting audio and transcribing it
    2. Extracting key frames and detecting documents
    3. Running multimodal fusion analysis with explainability metadata
    4. Running text analysis on combined content
    """
    if not whisper_model:
        return {
            "transcribed_text": "",
            "detected_documents": [],
            "fusion_analysis": {"fusion_score": 0.0, "modality_contributions": {"visual": 0.0, "text": 0.0, "audio": 0.0}},
            "analysis": {
                "inputtext": "",
                "sentiment": {"label": "Error", "score": 0.0},
                "evidence": [],
                "score": {"rawscore": 0.0, "exposurescore": 0, "details": []},
            }
        }

    # Define a temporary path for the audio file
    audio_path = "temp_extracted_audio.wav"
    all_detected_documents: List[Dict[str, Any]] = []
    visual_regions: List[np.ndarray] = []
    visual_meta: List[Dict[str, Any]] = []
    audio_segments: List[np.ndarray] = []
    audio_meta: List[Dict[str, Any]] = []

    try:
        # Use moviepy to extract the audio from the video file
        print(f"Analyzing video: {video_path}")
        with mp.VideoFileClip(video_path) as video:
            if video.audio is None:
                return {
                    "transcribed_text": "",
                    "detected_documents": [],
                    "fusion_analysis": {"fusion_score": 0.0, "modality_contributions": {"visual": 0.0, "text": 0.0, "audio": 0.0}},
                    "analysis": {
                        "inputtext": "",
                        "sentiment": {"label": "Error", "score": 0.0},
                        "evidence": [],
                        "score": {"rawscore": 0.0, "exposurescore": 0, "details": []},
                    }
                }
            video.audio.write_audiofile(audio_path, codec='pcm_s16le')

        # Transcribe the extracted audio using Whisper
        print("Transcribing audio...")
        transcription_result = whisper_model.transcribe(audio_path)
        transcribed_text = transcription_result.get("text", "")
        print("Transcription complete.")

        # Extract key frames and detect documents
        print("Extracting key frames and detecting documents...")
        key_frames = extract_key_frames(video_path, max_frames=15)
        for frame, timestamp in key_frames:
            detected_docs = detect_documents_in_frame(frame)
            for doc in detected_docs:
                doc["frame_timestamp"] = timestamp
                all_detected_documents.append(doc)

                # Extract visual regions for multimodal analysis
                x1, y1, x2, y2 = doc["bbox"]
                cropped_region = frame[y1:y2, x1:x2]
                visual_regions.append(cropped_region)
                visual_meta.append({
                    "bbox": doc.get("bbox"),
                    "class_name": doc.get("class_name"),
                    "confidence": doc.get("confidence"),
                    "frame_timestamp": doc.get("frame_timestamp"),
                })

        print(f"Document detection complete. Found {len(all_detected_documents)} document regions.")

        # Extract audio segments for multimodal analysis (5s chunks)
        try:
            import librosa
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            segment_length = sample_rate * 5  # 5 seconds
            for i in range(0, len(audio_data), segment_length):
                segment = audio_data[i:i + segment_length]
                if len(segment) > 0:
                    audio_segments.append(segment)
                    start_t = i / sample_rate
                    end_t = min((i + segment_length) / sample_rate, len(audio_data) / sample_rate)
                    audio_meta.append({
                        "time_range": [float(start_t), float(end_t)]
                    })
        except ImportError:
            print("librosa not available, skipping audio segmentation")
        except Exception as e:
            print(f"Audio segmentation failed: {e}")

    except Exception as e:
        print(f"ERROR in video processing: {e}")
        return {
            "transcribed_text": "",
            "detected_documents": [],
            "fusion_analysis": {"fusion_score": 0.0, "modality_contributions": {"visual": 0.0, "text": 0.0, "audio": 0.0}},
            "analysis": {
                "inputtext": "",
                "sentiment": {"label": "Error", "score": 0.0},
                "evidence": [],
                "score": {"rawscore": 0.0, "exposurescore": 0, "details": []},
            }
        }
    
    finally:
        # Ensure the temporary audio file is always deleted
        if os.path.exists(audio_path):
            os.remove(audio_path)

    # Prepare text content and metadata for multimodal analysis
    text_content: List[str] = []
    text_meta: List[Dict[str, Any]] = []
    if transcribed_text.strip():
        text_content.append(transcribed_text)
        text_meta.append({
            "source": "transcription",
            "text_span": [0, min(len(transcribed_text), 200)]
        })
    if all_detected_documents:
        for doc in all_detected_documents:
            if doc["ocr_text"].strip():
                text_content.append(doc["ocr_text"])
                text_meta.append({
                    "source": "document_region_ocr",
                    "text_span": [0, min(len(doc["ocr_text"]), 200)]
                })

    # Run multimodal fusion analysis with metadata
    fusion_results = multimodal_analyzer.analyze_multimodal_content(
        visual_regions=visual_regions,
        text_content=text_content,
        audio_segments=audio_segments,
        visual_meta=visual_meta,
        text_meta=text_meta,
        audio_meta=audio_meta,
    )

    # Combine transcribed text with OCR text from detected documents
    all_text = transcribed_text
    if all_detected_documents:
        document_texts = [doc["ocr_text"] for doc in all_detected_documents if doc["ocr_text"]]
        if document_texts:
            all_text += " " + " ".join(document_texts)

    if not transcribed_text and not all_detected_documents:
        return {
            "transcribed_text": "No speech could be detected in the video and no documents found.",
            "detected_documents": [],
            "fusion_analysis": fusion_results,
            "analysis": {
                "inputtext": "",
                "sentiment": {"label": "Neutral", "score": 0.5},
                "evidence": [],
                "score": {"rawscore": 0.0, "exposurescore": 0, "details": []},
            }
        }

    # If we have text content, run analysis
    if all_text.strip():
        print("Running text analysis on combined content...")
        text_analysis_results = text_analyzer.analyze_text(all_text)
        
        return {
            "transcribed_text": transcribed_text,
            "detected_documents": all_detected_documents,
            "fusion_analysis": fusion_results,
            "analysis": text_analysis_results
        }
    else:
        # Return results without text analysis if no text found
        return {
            "transcribed_text": transcribed_text,
            "detected_documents": all_detected_documents,
            "fusion_analysis": fusion_results,
            "analysis": {
                "inputtext": "",
                "sentiment": {"label": "Neutral", "score": 0.5},
                "evidence": [],
                "score": {"rawscore": 0.0, "exposurescore": 0, "details": []},
            }
        }