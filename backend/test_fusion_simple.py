#!/usr/bin/env python3
"""
Simple test to demonstrate fusion score calculation logic
"""

import numpy as np
import re

def calculate_fusion_score(visual_regions, text_content, audio_segments, 
                          visual_meta=None, text_meta=None, audio_meta=None):
    """
    Simplified fusion score calculation for demonstration
    """
    # Calculate modality scores
    visual_score = 0.0
    text_score = 0.0
    audio_score = 0.0
    
    # Visual scoring
    if len(visual_regions) > 0:
        visual_score = min(1.0, len(visual_regions) * 0.25)
        if visual_meta:
            confidences = [m.get('confidence', 0.5) for m in visual_meta if m.get('confidence')]
            if confidences:
                avg_confidence = np.mean(confidences)
                visual_score *= avg_confidence
    
    # Text scoring
    if len(text_content) > 0:
        combined_text = " ".join(text_content)
        text_length = len(combined_text)
        
        text_score = min(1.0, text_length / 500.0)
        
        # Check for PII patterns
        pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',  # email
            r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',  # phone
            r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',  # aadhaar/card
            r'\b[A-Z]{5}\d{4}[A-Z]\b',  # PAN
        ]
        
        pii_count = 0
        for pattern in pii_patterns:
            pii_count += len(re.findall(pattern, combined_text))
        
        if pii_count > 0:
            text_score = min(1.0, text_score + (pii_count * 0.15))
    
    # Audio scoring
    if audio_segments and len(audio_segments) > 0:
        audio_score = min(1.0, len(audio_segments) * 0.2)
    
    # Calculate weighted fusion score
    total_weight = 0
    weighted_sum = 0
    
    if visual_score > 0:
        weighted_sum += visual_score * 0.4
        total_weight += 0.4
    if text_score > 0:
        weighted_sum += text_score * 0.5
        total_weight += 0.5
    if audio_score > 0:
        weighted_sum += audio_score * 0.3
        total_weight += 0.3
    
    fusion_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Boost if high-risk PII detected
    if text_score > 0.7:
        fusion_score = min(1.0, fusion_score * 1.2)
    
    # Calculate modality contributions
    contributions_raw = {
        "visual": visual_score * 0.4,
        "text": text_score * 0.5,
        "audio": audio_score * 0.3
    }
    total_contrib = sum(contributions_raw.values())
    modality_contributions = {
        k: v / total_contrib if total_contrib > 0 else 0.0
        for k, v in contributions_raw.items()
    }
    
    primary_modality = max(modality_contributions, key=modality_contributions.get)
    
    return {
        "fusion_score": fusion_score,
        "modality_contributions": modality_contributions,
        "primary_modality": primary_modality,
        "component_scores": {
            "visual": visual_score,
            "text": text_score,
            "audio": audio_score
        }
    }

def print_result(name, result):
    """Pretty print results"""
    print(f"\n{name}")
    print("-" * 60)
    score = result['fusion_score']
    percentage = score * 100
    
    # Determine risk level
    if percentage < 20:
        level = "LOW"
        color = "🟢"
    elif percentage < 40:
        level = "MODERATE"
        color = "🟡"
    elif percentage < 60:
        level = "ELEVATED"
        color = "🟠"
    elif percentage < 80:
        level = "HIGH"
        color = "🔴"
    else:
        level = "CRITICAL"
        color = "🔴🔴"
    
    print(f"Fusion Score: {score:.3f} ({percentage:.1f}%) {color} {level}")
    print(f"Primary Modality: {result['primary_modality']}")
    
    print("\nComponent Scores:")
    for modality, score in result['component_scores'].items():
        print(f"  {modality:8s}: {score:.3f}")
    
    print("\nModality Contributions:")
    for modality, contrib in result['modality_contributions'].items():
        bar = "█" * int(contrib * 40)
        print(f"  {modality:8s}: {contrib:5.1%} {bar}")

# Test Cases
print("="*60)
print("FUSION SCORE DEMONSTRATION")
print("="*60)

# Test 1: Text only with PII
result1 = calculate_fusion_score(
    visual_regions=[],
    text_content=["My email is john@example.com and phone is 555-123-4567"],
    audio_segments=None,
    text_meta=[{"source": "input"}]
)
print_result("TEST 1: Text with Email + Phone", result1)

# Test 2: Visual + Text
result2 = calculate_fusion_score(
    visual_regions=[1, 2],  # 2 documents
    text_content=["ID Card", "Name: John Doe"],
    audio_segments=None,
    visual_meta=[
        {"confidence": 0.85},
        {"confidence": 0.90}
    ],
    text_meta=[{"source": "ocr"}]
)
print_result("TEST 2: 2 Documents + OCR Text", result2)

# Test 3: High-risk multimodal
result3 = calculate_fusion_score(
    visual_regions=[1, 2, 3],  # 3 documents
    text_content=[
        "Email: sensitive@company.com, Phone: +1-555-987-6543",
        "Aadhaar: 9876 5432 1098",
        "PAN: ABCDE1234F"
    ],
    audio_segments=[1, 2, 3, 4],  # 4 segments
    visual_meta=[
        {"confidence": 0.80},
        {"confidence": 0.75},
        {"confidence": 0.85}
    ],
    text_meta=[{"source": "ocr"}],
    audio_meta=[{"time_range": [0, 5]}]
)
print_result("TEST 3: High-Risk Multimodal (3 docs + PII + audio)", result3)

# Test 4: Low risk
result4 = calculate_fusion_score(
    visual_regions=[],
    text_content=["Hello world, this is a test"],
    audio_segments=None
)
print_result("TEST 4: Low Risk (no PII)", result4)

# Test 5: Visual only
result5 = calculate_fusion_score(
    visual_regions=[1],
    text_content=[],
    audio_segments=None,
    visual_meta=[{"confidence": 0.60}]
)
print_result("TEST 5: Visual Only (1 document, 60% confidence)", result5)

# Test 6: Critical risk
result6 = calculate_fusion_score(
    visual_regions=[1, 2, 3, 4],  # 4 documents
    text_content=[
        "SSN: 123-45-6789, Email: test@example.com",
        "Phone: 555-1234, Aadhaar: 1234 5678 9012",
        "PAN: ABCDE1234F, Credit Card: 1234-5678-9012-3456"
    ],
    audio_segments=[1, 2, 3, 4, 5],
    visual_meta=[
        {"confidence": 0.95},
        {"confidence": 0.90},
        {"confidence": 0.92},
        {"confidence": 0.88}
    ]
)
print_result("TEST 6: CRITICAL Risk (4 docs + multiple PIIs + audio)", result6)

print("\n" + "="*60)
print("SUMMARY: Risk Level Distribution")
print("="*60)
print("  0-20%:  🟢 LOW       - Minimal privacy exposure")
print(" 21-40%:  🟡 MODERATE  - Some sensitive content")
print(" 41-60%:  🟠 ELEVATED  - Multiple PII elements")
print(" 61-80%:  🔴 HIGH      - Significant privacy risk")
print("81-100%:  🔴🔴 CRITICAL - Severe exposure")
print("="*60 + "\n")
