# backend/app/main.py (Final Version with Correct CORS)

import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from starlette.concurrency import run_in_threadpool

# --- Import all four analyzers ---
from app.analysis import text_analyzer, image_analyzer, video_analyzer, audio_analyzer

# --- Pydantic Models for API Response Structure ---
class SentimentResponse(BaseModel):
    label: str
    score: float

class ScoreDetailItem(BaseModel):
    evidence: Dict[str, Any]
    weight: Optional[float] = None
    factor: Optional[float] = None
    contribution: float

class ScoreResponse(BaseModel):
    rawscore: float
    exposurescore: int
    details: List[ScoreDetailItem]

class TextAnalysisResponse(BaseModel):
    inputtext: str
    sentiment: SentimentResponse
    evidence: List[Dict[str, Any]]
    score: ScoreResponse

class ImageAnalysisResponse(BaseModel):
    metadata: Dict[str, Any]
    ocr_text: str
    sentiment: Optional[SentimentResponse] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    score: Optional[ScoreResponse] = None

class VideoAnalysisResponse(BaseModel):
    transcribed_text: str
    analysis: Optional[TextAnalysisResponse] = None

# --- FastAPI App Initialization ---
app = FastAPI(title="Digital Shadow Analyzer API")

# --- THIS IS THE FIX ---
# We are specifically allowing requests from our Next.js frontend's address.
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Digital Shadow Analyzer API is running."}

class TextRequest(BaseModel):
    text: str

@app.post("/analyze-text/", response_model=TextAnalysisResponse)
async def analyze_text_endpoint(request: TextRequest):
    try:
        result = await run_in_threadpool(text_analyzer.analyze_text, request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during text analysis: {e}")

@app.post("/analyze-image/", response_model=ImageAnalysisResponse)
async def analyze_image_endpoint(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    try:
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = await run_in_threadpool(image_analyzer.analyze_image, temp_path)
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        shutil.rmtree(temp_dir)

@app.post("/analyze-video/", response_model=VideoAnalysisResponse)
async def analyze_video_endpoint(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    try:
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = await run_in_threadpool(video_analyzer.analyze_video, temp_path)
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        shutil.rmtree(temp_dir)

@app.post("/analyze-audio/", response_model=VideoAnalysisResponse)
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    try:
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = await run_in_threadpool(audio_analyzer.analyze_audio, temp_path)
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        shutil.rmtree(temp_dir)
