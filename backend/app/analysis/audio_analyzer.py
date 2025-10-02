# backend/app/analysis/audio_analyzer.py (New File)

import os
import whisper
from typing import Dict, Any
from . import text_analyzer

try:
    # This loads the Whisper model once when the server starts
    whisper_model = whisper.load_model("base")
except Exception as e:
    print(f"Failed to load Whisper model for audio analyzer: {e}")
    whisper_model = None

def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """Transcribes an audio file and analyzes the resulting text for PII."""
    if not whisper_model:
        return {"error": "Whisper model not available."}

    try:
        # Transcribe the audio file directly
        transcription_result = whisper_model.transcribe(audio_path)
        transcribed_text = transcription_result.get('text', '').strip()

    except Exception as e:
        print(f"ERROR in audio transcription: {e}")
        return {"error": f"Failed to process audio file. Details: {e}"}

    # If no text was transcribed, return a clean response
    if not transcribed_text:
        return {
            "transcribed_text": "No speech found in audio.",
            "analysis": None
        }

    # If text was found, run it through the text analyzer
    text_analysis = text_analyzer.analyze_text(transcribed_text)
    
    return {
        "transcribed_text": transcribed_text,
        "analysis": text_analysis
    }
