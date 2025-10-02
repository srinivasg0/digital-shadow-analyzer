from pydantic import BaseModel
from typing import List, Optional, Any, Tuple

class TextAnalysisRequest(BaseModel):
    text: str

class Sentiment(BaseModel):
    label: str
    score: float
    
class EvidenceItem(BaseModel):
    source: str
    label: str
    text: str
    span: Tuple[int, int]
    score: float

class ScoreDetail(BaseModel):
    evidence: EvidenceItem
    contribution: float

class Score(BaseModel):
    exposure_score: float
    details: List[ScoreDetail]

class TextAnalysisResponse(BaseModel):
    input_text: str
    sentiment: Sentiment
    evidence: List[EvidenceItem]
    score: Score
    
class ImageAnalysisResponse(BaseModel):
    metadata: Optional[dict] = None
    ocr_text: Optional[str] = None
    score: dict
