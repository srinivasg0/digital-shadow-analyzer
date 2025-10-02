import os
import whisper
import moviepy.editor as mp
from typing import Dict, Any
from . import text_analyzer

try:
    # Load the Whisper model once when the application starts
    whisper_model = whisper.load_model("base")
except Exception as e:
    print(f"FATAL: Failed to load Whisper model: {e}")
    whisper_model = None

def analyze_video(video_path: str) -> Dict[str, Any]:
    """
    Analyzes a video file by extracting audio, transcribing it,
    and then running the full text analysis pipeline on the transcription.
    """
    if not whisper_model:
        # Provide a friendly message in the required response shape
        return {
            "transcribed_text": "",
            "analysis": {
                "inputtext": "",
                "sentiment": {"label": "Error", "score": 0.0},
                "evidence": [],
                "score": {"rawscore": 0.0, "exposurescore": 0, "details": []},
            }
        }

    # Define a temporary path for the audio file
    audio_path = "temp_extracted_audio.wav"

    try:
        # Use moviepy to extract the audio from the video file
        print(f"Analyzing video: {video_path}")
        with mp.VideoFileClip(video_path) as video:
            if video.audio is None:
                # Return a user-friendly message when no audio track is present
                return {
                    "transcribed_text": "",
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

    except Exception as e:
        # If any part of the video/audio processing fails, return a specific error.
        print(f"ERROR in video processing: {e}")
        # Return a friendly message in the required response shape
        return {
            "transcribed_text": "",
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

    if not transcribed_text:
        return {
            "transcribed_text": "No speech could be detected in the video.",
            "analysis": {
                "inputtext": "",
                "sentiment": {"label": "Neutral", "score": 0.5},
                "evidence": [],
                "score": {"rawscore": 0.0, "exposurescore": 0, "details": []},
            }
        }

    # If transcription is successful, run it through our perfected text analyzer
    print("Running text analysis on transcription...")
    text_analysis_results = text_analyzer.analyze_text(transcribed_text)

    # Return a combined result with the transcription and the full text analysis
    return {
        "transcribed_text": transcribed_text,
        "analysis": text_analysis_results
    }
