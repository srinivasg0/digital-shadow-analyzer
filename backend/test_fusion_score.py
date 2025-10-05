#!/usr/bin/env python3
"""
Test script to demonstrate fusion score calculation
"""

import sys
import numpy as np
from app.fusion.multimodal_fusion import multimodal_analyzer

def test_text_only():
    """Test with text content only"""
    print("\n" + "="*60)
    print("TEST 1: Text-Only Analysis")
    print("="*60)
    
    text_content = [
        "My email is john.doe@example.com",
        "Phone: 555-123-4567",
        "Aadhaar: 1234 5678 9012"
    ]
    
    result = multimodal_analyzer.analyze_multimodal_content(
        visual_regions=[],
        text_content=text_content,
        audio_segments=None,
        visual_meta=[],
        text_meta=[
            {"source": "input", "text_span": [0, 100]},
            {"source": "input", "text_span": [0, 100]},
            {"source": "input", "text_span": [0, 100]}
        ],
        audio_meta=[]
    )
    
    print(f"Fusion Score: {result['fusion_score']:.2f} ({result['fusion_score']*100:.1f}%)")
    print(f"Primary Modality: {result['primary_modality']}")
    print(f"Modality Contributions:")
    for modality, contrib in result['modality_contributions'].items():
        print(f"  - {modality}: {contrib:.2%}")
    print(f"\nExplanation: {result.get('explanation', 'N/A')}")

def test_visual_and_text():
    """Test with visual regions and text"""
    print("\n" + "="*60)
    print("TEST 2: Visual + Text Analysis")
    print("="*60)
    
    # Simulate 2 detected document regions
    visual_regions = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    ]
    
    text_content = ["ID Card detected", "Name: John Doe"]
    
    result = multimodal_analyzer.analyze_multimodal_content(
        visual_regions=visual_regions,
        text_content=text_content,
        audio_segments=None,
        visual_meta=[
            {"bbox": [10, 20, 110, 120], "class_name": "book", "confidence": 0.85},
            {"bbox": [200, 100, 300, 200], "class_name": "book", "confidence": 0.90}
        ],
        text_meta=[
            {"source": "ocr", "text_span": [0, 50]},
            {"source": "ocr", "text_span": [0, 50]}
        ],
        audio_meta=[]
    )
    
    print(f"Fusion Score: {result['fusion_score']:.2f} ({result['fusion_score']*100:.1f}%)")
    print(f"Primary Modality: {result['primary_modality']}")
    print(f"Modality Contributions:")
    for modality, contrib in result['modality_contributions'].items():
        print(f"  - {modality}: {contrib:.2%}")
    print(f"\nExplanation: {result.get('explanation', 'N/A')}")
    
    print(f"\nContributing Factors ({len(result.get('contributing_factors', []))}):")
    for factor in result.get('contributing_factors', [])[:3]:
        print(f"  - {factor['modality']}: importance={factor['importance']:.3f}")

def test_multimodal():
    """Test with all modalities"""
    print("\n" + "="*60)
    print("TEST 3: Full Multimodal Analysis (Visual + Text + Audio)")
    print("="*60)
    
    # Simulate 3 visual regions
    visual_regions = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    ]
    
    # Rich text with multiple PIIs
    text_content = [
        "Email: sensitive@company.com, Phone: +1-555-987-6543",
        "Aadhaar Number: 9876 5432 1098",
        "PAN: ABCDE1234F"
    ]
    
    # Simulate 4 audio segments (20 seconds)
    audio_segments = [
        np.random.randn(8000),  # 1 second at 8kHz
        np.random.randn(8000),
        np.random.randn(8000),
        np.random.randn(8000)
    ]
    
    result = multimodal_analyzer.analyze_multimodal_content(
        visual_regions=visual_regions,
        text_content=text_content,
        audio_segments=audio_segments,
        visual_meta=[
            {"bbox": [10, 20, 110, 120], "class_name": "book", "confidence": 0.80},
            {"bbox": [200, 100, 300, 200], "class_name": "laptop", "confidence": 0.75},
            {"bbox": [400, 300, 500, 400], "class_name": "cell phone", "confidence": 0.85}
        ],
        text_meta=[
            {"source": "ocr", "text_span": [0, 100]},
            {"source": "transcription", "text_span": [0, 100]},
            {"source": "ocr", "text_span": [0, 100]}
        ],
        audio_meta=[
            {"time_range": [0.0, 5.0]},
            {"time_range": [5.0, 10.0]},
            {"time_range": [10.0, 15.0]},
            {"time_range": [15.0, 20.0]}
        ]
    )
    
    print(f"Fusion Score: {result['fusion_score']:.2f} ({result['fusion_score']*100:.1f}%)")
    print(f"Primary Modality: {result['primary_modality']}")
    print(f"\nModality Contributions:")
    for modality, contrib in result['modality_contributions'].items():
        print(f"  - {modality}: {contrib:.2%}")
    
    print(f"\nEmbedding Counts:")
    print(f"  - Visual: {result['visual_embeddings_count']}")
    print(f"  - Text: {result['text_embeddings_count']}")
    print(f"  - Audio: {result['audio_embeddings_count']}")
    
    print(f"\nExplanation: {result.get('explanation', 'N/A')}")
    
    print(f"\nTop Contributing Factors:")
    for i, factor in enumerate(result.get('contributing_factors', [])[:5], 1):
        print(f"  {i}. {factor['modality'].upper()}: importance={factor['importance']:.3f}")

def test_risk_levels():
    """Test different risk levels"""
    print("\n" + "="*60)
    print("TEST 4: Risk Level Calibration")
    print("="*60)
    
    test_cases = [
        ("Low Risk", [], ["Hello world"], None),
        ("Moderate Risk", [], ["My email is test@example.com"], None),
        ("High Risk", [], ["Email: test@example.com, Phone: 555-1234, SSN: 123-45-6789"], None),
    ]
    
    for name, visual, text, audio in test_cases:
        result = multimodal_analyzer.analyze_multimodal_content(
            visual_regions=visual,
            text_content=text,
            audio_segments=audio,
            visual_meta=[],
            text_meta=[{"source": "test", "text_span": [0, 100]}] if text else [],
            audio_meta=[]
        )
        
        score = result['fusion_score']
        percentage = score * 100
        
        if percentage < 20:
            level = "LOW"
        elif percentage < 40:
            level = "MODERATE"
        elif percentage < 60:
            level = "ELEVATED"
        elif percentage < 80:
            level = "HIGH"
        else:
            level = "CRITICAL"
        
        print(f"{name:20s}: {score:.2f} ({percentage:5.1f}%) - {level}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FUSION SCORE TEST SUITE")
    print("="*60)
    
    try:
        test_text_only()
        test_visual_and_text()
        test_multimodal()
        test_risk_levels()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
