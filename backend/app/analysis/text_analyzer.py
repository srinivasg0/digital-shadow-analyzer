# backend/app/analysis/text_analyzer.py (Definitive Final Version)

import re
import spacy
from transformers import pipeline
from typing import Dict, List, Any
import numpy as np  # Make sure numpy is imported

# --- Model Loading ---
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    hf_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return nlp, hf_ner, sentiment_pipe

nlp, hf_ner, sentiment_pipe = load_models()

# --- PII Detection and Scoring ---
# Using raw strings (r"...") to prevent SyntaxWarning
PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9._%-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"\b\d{6,15}\b"),
    "url": re.compile(r"https?://[^\s]+"),
    "creditcardlike": re.compile(r"\b\d{13,16}\b"),
}
WEIGHTS = {
    "email": 25, "phone": 25, "creditcardlike": 40, "url": 10,
    "addresslike": 20, "PERSON": 15, "GPE": 15, "LOC": 12, "ORG": 12,
    "DATE": 8, "TIME": 6, "MONEY": 20, "defaultentity": 5,
}

# --- Evidence Extraction Functions ---
def detect_pii_regex(text: str) -> List[Dict[str, Any]]:
    hits = []
    for label, pattern in PII_PATTERNS.items():
        for m in pattern.finditer(text):
            hits.append({"source": "regex", "type": label, "text": m.group(0), "span": m.span()})
    return hits

def run_spacy_ner(text: str) -> List[Dict[str, Any]]:
    doc = nlp(text)
    return [{"source": "spacy", "label": ent.label_, "text": ent.text, "span": (ent.start_char, ent.end_char)} for ent in doc.ents]

def run_hf_ner(text: str) -> List[Dict[str, Any]]:
    out = hf_ner(text)
    entities = []
    for e in out:
        label = e["entity_group"]
        word = e["word"]
        score = e.get("score", 1.0)
        entities.append({"source": "hf", "label": label, "text": word, "score": score})
    return entities

def extract_evidence(text: str) -> List[Dict[str, Any]]:
    evidence = []
    evidence.extend(detect_pii_regex(text))
    evidence.extend(run_spacy_ner(text))
    evidence.extend(run_hf_ner(text))
    return evidence

def compute_exposure_score(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    details = []
    for e in evidence:
        t = e.get("type") or e.get("label") or ""
        w = WEIGHTS.get(t, WEIGHTS.get(t.upper(), WEIGHTS["defaultentity"]))
        score_factor = float(e.get("score", 1.0))
        text_len_factor = min(1.0, len(e.get("text", ""))/30.0 + 0.2)
        contrib = w * score_factor * text_len_factor
        details.append({"evidence": e, "weight": w, "factor": round(score_factor * text_len_factor, 2), "contribution": round(contrib, 2)})
        total += contrib
    
    return {"rawscore": round(total, 2), "exposurescore": min(100, round(total)), "details": details}

# --- THIS IS THE CRUCIAL HELPER FUNCTION ---
def convert_numpy_types(obj):
    """Recursively converts numpy numeric types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        # Convert tuples (e.g., spans) to lists for JSON compatibility
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

# --- Main Analysis Function (with the fix applied) ---
def analyze_text(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {
            "inputtext": text or "",
            "sentiment": {"label": "N/A", "score": 0},
            "evidence": [],
            "score": {"rawscore": 0, "exposurescore": 0, "details": []}
        }
    
    evidence = extract_evidence(text)
    score = compute_exposure_score(evidence)
    sentiment = sentiment_pipe(text[:512])[0]
    
    # --- THIS IS THE FIX ---
    # Convert both the sentiment and score dictionaries before returning.
    final_sentiment = convert_numpy_types(sentiment)
    final_score = convert_numpy_types(score)
    final_evidence = convert_numpy_types(evidence)
    
    return {
        "inputtext": text,
        "sentiment": final_sentiment,
        "evidence": final_evidence,
        "score": final_score,
    }
