# backend/app/fusion/multimodal_fusion.py (Phase 2: Multimodal Fusion)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import re
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalFusionModel(nn.Module):
    """
    Transformer-based multimodal fusion model for privacy exposure scoring.
    Combines visual, textual, and audio embeddings into a unified exposure score.
    """
    
    def __init__(self, 
                 visual_dim: int = 512,
                 text_dim: int = 768, 
                 audio_dim: int = 1024,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        
        # Projection layers to map different modalities to common space
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Positional encoding for sequence modeling
        self.pos_encoding = nn.Parameter(torch.randn(100, hidden_dim))
        
        # Transformer encoder for cross-modal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head for exposure scoring
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Modality importance prediction
        self.modality_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # visual, text, audio
            nn.Softmax(dim=-1)
        )
        
    def forward(self, visual_embeddings: torch.Tensor, 
                text_embeddings: torch.Tensor,
                audio_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the multimodal fusion model.
        
        Args:
            visual_embeddings: [batch_size, num_regions, visual_dim]
            text_embeddings: [batch_size, num_texts, text_dim] 
            audio_embeddings: [batch_size, num_audio, audio_dim]
            
        Returns:
            exposure_scores: [batch_size, 1] - unified exposure scores
            modality_weights: [batch_size, 3] - importance weights for each modality
            attention_weights: [batch_size, total_tokens, total_tokens] (placeholder)
            token_importance: [batch_size, total_tokens] - per-token importance proxy
        """
        batch_size = visual_embeddings.size(0)
        
        # Project all modalities to common hidden dimension
        visual_proj = self.visual_proj(visual_embeddings)  # [B, num_regions, hidden_dim]
        text_proj = self.text_proj(text_embeddings)        # [B, num_texts, hidden_dim]
        audio_proj = self.audio_proj(audio_embeddings)     # [B, num_audio, hidden_dim]
        
        # Concatenate all modalities
        all_embeddings = torch.cat([visual_proj, text_proj, audio_proj], dim=1)  # [B, total_tokens, hidden_dim]
        
        # Add positional encoding
        seq_len = all_embeddings.size(1)
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        all_embeddings = all_embeddings + pos_enc
        
        # Pass through transformer
        transformer_output = self.transformer(all_embeddings)  # [B, total_tokens, hidden_dim]
        
        # Global average pooling
        pooled_output = torch.mean(transformer_output, dim=1)  # [B, hidden_dim]
        
        # Predict exposure score
        exposure_scores = self.classifier(pooled_output)  # [B, 1]
        
        # Predict modality importance
        modality_weights = self.modality_classifier(pooled_output)  # [B, 3]
        
        # Get attention weights (simplified - using last layer attention)
        attention_weights = torch.ones(batch_size, seq_len, seq_len)  # Placeholder

        # Token importance proxy: normalized L2 norm of token representations
        tok_norms = torch.norm(transformer_output, p=2, dim=-1)  # [B, total_tokens]
        token_importance = F.softmax(tok_norms, dim=-1)  # [B, total_tokens]

        return exposure_scores, modality_weights, attention_weights, token_importance

class EmbeddingExtractor:
    """
    Extracts embeddings from different modalities using pre-trained models.
    """
    
    def __init__(self):
        self.text_model = None
        self.text_tokenizer = None
        self.sentence_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize pre-trained models for embedding extraction."""
        try:
            # Initialize BERT for text embeddings
            self.text_model = AutoModel.from_pretrained('bert-base-uncased')
            self.text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {e}")
        
        try:
            # Initialize sentence transformer for general text embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
    
    def extract_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract embeddings from text using BERT or sentence transformer.
        
        Args:
            texts: List of text strings
            
        Returns:
            embeddings: [num_texts, embedding_dim] numpy array
        """
        if not texts:
            return np.zeros((0, 768))
        
        if self.sentence_model is not None:
            # Use sentence transformer for simplicity
            embeddings = self.sentence_model.encode(texts)
            return embeddings
        elif self.text_model is not None and self.text_tokenizer is not None:
            # Use BERT
            embeddings = []
            for text in texts:
                inputs = self.text_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = self.text_model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    embeddings.append(embedding)
            return np.array(embeddings)
        else:
            # Fallback: return zero embeddings
            logger.warning("No text model available, returning zero embeddings")
            return np.zeros((len(texts), 768))
    
    def extract_visual_embeddings(self, image_regions: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from image regions using YOLOv8 backbone features.
        
        Args:
            image_regions: List of cropped image regions as numpy arrays
            
        Returns:
            embeddings: [num_regions, embedding_dim] numpy array
        """
        if not image_regions:
            return np.zeros((0, 512))
        
        # For now, use a simple CNN-based feature extractor
        # In a production system, you'd extract features from YOLOv8's backbone
        embeddings = []
        for region in image_regions:
            # Simple feature extraction: resize and flatten
            # This is a placeholder - in practice, use YOLOv8's feature extractor
            resized = cv2.resize(region, (224, 224))
            features = resized.flatten()[:512]  # Take first 512 features
            if len(features) < 512:
                features = np.pad(features, (0, 512 - len(features)))
            embeddings.append(features)
        
        return np.array(embeddings)
    
    def extract_audio_embeddings(self, audio_segments: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from audio segments.
        
        Args:
            audio_segments: List of audio arrays
            
        Returns:
            embeddings: [num_segments, embedding_dim] numpy array
        """
        if not audio_segments:
            return np.zeros((0, 1024))
        
        # Placeholder audio embedding extraction
        # In practice, use Whisper's encoder or wav2vec2
        embeddings = []
        for audio in audio_segments:
            # Simple feature extraction: MFCC-like features
            # This is a placeholder - use proper audio models in production
            features = np.random.randn(1024)  # Placeholder
            embeddings.append(features)
        
        return np.array(embeddings)

class MultimodalFusionAnalyzer:
    """
    Main class for multimodal fusion analysis.
    Combines visual, textual, and audio features to produce unified exposure scores.
    """
    
    def __init__(self):
        self.embedding_extractor = EmbeddingExtractor()
        self.fusion_model = None
        self._initialize_fusion_model()
    
    def _initialize_fusion_model(self):
        """Initialize the fusion model."""
        try:
            self.fusion_model = MultimodalFusionModel()
            # Load pre-trained weights if available, otherwise use random initialization
            logger.info("Multimodal fusion model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize fusion model: {e}")
            self.fusion_model = None
    
    def analyze_multimodal_content(self, 
                                 visual_regions: List[np.ndarray],
                                 text_content: List[str],
                                 audio_segments: List[np.ndarray] = None,
                                 visual_meta: Optional[List[Dict[str, Any]]] = None,
                                 text_meta: Optional[List[Dict[str, Any]]] = None,
                                 audio_meta: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze multimodal content and return fusion results.
        Uses heuristic-based scoring when fusion model is not trained.
        
        Args:
            visual_regions: List of cropped image regions
            text_content: List of text strings (OCR results, transcriptions)
            audio_segments: List of audio segments (optional)
            
        Returns:
            Dictionary containing fusion analysis results
        """
        # Use heuristic scoring instead of untrained model
        use_heuristic = True  # Set to True until model is properly trained
        
        try:
            # Extract embeddings
            visual_embeddings = self.embedding_extractor.extract_visual_embeddings(visual_regions)
            text_embeddings = self.embedding_extractor.extract_text_embeddings(text_content)
            audio_embeddings = self.embedding_extractor.extract_audio_embeddings(audio_segments or [])
            
            # Ensure we have at least one modality
            if len(visual_embeddings) == 0 and len(text_embeddings) == 0 and len(audio_embeddings) == 0:
                return {
                    "fusion_score": 0.0,
                    "modality_contributions": {"visual": 0.0, "text": 0.0, "audio": 0.0},
                    "error": "No content to analyze"
                }
            
            # Use heuristic-based scoring for better accuracy
            if use_heuristic:
                return self._heuristic_fusion_analysis(
                    visual_regions, text_content, audio_segments or [],
                    visual_meta, text_meta, audio_meta,
                    visual_embeddings, text_embeddings, audio_embeddings
                )
            
            # Default metadata containers
            visual_meta = visual_meta or [{} for _ in range(len(visual_embeddings))]
            text_meta = text_meta or [{} for _ in range(len(text_embeddings))]
            audio_meta = audio_meta or [{} for _ in range(len(audio_embeddings))]

            # Pad sequences to same length for batch processing
            max_length = max(len(visual_embeddings), len(text_embeddings), len(audio_embeddings), 1)
            
            # Pad visual embeddings
            if len(visual_embeddings) == 0:
                visual_embeddings = np.zeros((1, 512))
            visual_padded = np.zeros((max_length, visual_embeddings.shape[1]))
            visual_padded[:len(visual_embeddings)] = visual_embeddings
            visual_tensor = torch.FloatTensor(visual_padded).unsqueeze(0)
            
            # Pad text embeddings
            if len(text_embeddings) == 0:
                text_embeddings = np.zeros((1, 768))
            text_padded = np.zeros((max_length, text_embeddings.shape[1]))
            text_padded[:len(text_embeddings)] = text_embeddings
            text_tensor = torch.FloatTensor(text_padded).unsqueeze(0)
            
            # Pad audio embeddings
            if len(audio_embeddings) == 0:
                audio_embeddings = np.zeros((1, 1024))
            audio_padded = np.zeros((max_length, audio_embeddings.shape[1]))
            audio_padded[:len(audio_embeddings)] = audio_embeddings
            audio_tensor = torch.FloatTensor(audio_padded).unsqueeze(0)
            
            # Run fusion model
            with torch.no_grad():
                exposure_scores, modality_weights, attention_weights, token_importance = self.fusion_model(
                    visual_tensor, text_tensor, audio_tensor
                )
            
            # Convert to numpy and format results
            fusion_score = float(exposure_scores.squeeze().item())
            modality_contributions = {
                "visual": float(modality_weights[0, 0].item()),
                "text": float(modality_weights[0, 1].item()),
                "audio": float(modality_weights[0, 2].item())
            }
            
            # Determine primary modality
            primary_modality = max(modality_contributions, key=modality_contributions.get)

            # Build per-token importance breakdown mapped back to modalities
            n_v = len(visual_embeddings) if len(visual_embeddings) > 0 else 1
            n_t = len(text_embeddings) if len(text_embeddings) > 0 else 1
            n_a = len(audio_embeddings) if len(audio_embeddings) > 0 else 1
            total_tokens = n_v + n_t + n_a

            tok_imp = token_importance.squeeze(0).cpu().numpy().tolist()  # length = total_tokens
            # Indices for slicing
            v_start, v_end = 0, n_v
            t_start, t_end = v_end, v_end + n_t
            a_start, a_end = t_end, t_end + n_a

            visual_token_importance = tok_imp[v_start:v_end]
            text_token_importance = tok_imp[t_start:t_end]
            audio_token_importance = tok_imp[a_start:a_end]

            # Top contributing evidence per modality
            def top_indices(vals: List[float], k: int = 3):
                if not vals:
                    return []
                idxs = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)[:k]
                return [(i, float(vals[i])) for i in idxs]

            top_visual = top_indices(visual_token_importance)
            top_text = top_indices(text_token_importance)
            top_audio = top_indices(audio_token_importance)

            contributing_factors: List[Dict[str, Any]] = []
            for idx, imp in top_visual:
                meta = visual_meta[idx] if idx < len(visual_meta) else {}
                contributing_factors.append({
                    "modality": "visual",
                    "index": idx,
                    "importance": imp,
                    "region": {
                        "bbox": meta.get("bbox"),
                        "class_name": meta.get("class_name"),
                        "confidence": meta.get("confidence"),
                        "frame_timestamp": meta.get("frame_timestamp"),
                    }
                })
            for idx, imp in top_text:
                meta = text_meta[idx] if idx < len(text_meta) else {}
                contributing_factors.append({
                    "modality": "text",
                    "index": idx,
                    "importance": imp,
                    "text_span": meta.get("text_span"),
                    "source": meta.get("source"),
                })
            for idx, imp in top_audio:
                meta = audio_meta[idx] if idx < len(audio_meta) else {}
                contributing_factors.append({
                    "modality": "audio",
                    "index": idx,
                    "importance": imp,
                    "time_range": meta.get("time_range"),
                })

            # Simple templated explanation
            def compose_explanation() -> str:
                mod = primary_modality
                parts = []
                if mod == "visual" and top_visual:
                    parts.append("visual regions with detected document-like areas")
                if mod == "text" and top_text:
                    parts.append("OCR/transcript text spans containing potential identifiers")
                if mod == "audio" and top_audio:
                    parts.append("audio segments with sensitive spoken content")
                reason = ", and ".join(parts) if parts else mod
                return f"Risk is primarily driven by {reason}. The fusion score weighs cross-modal evidence and highlights the most influential regions/spans/segments."

            explanation = compose_explanation()

            return {
                "fusion_score": fusion_score,
                "modality_contributions": modality_contributions,
                "primary_modality": primary_modality,
                "visual_embeddings_count": len(visual_embeddings),
                "text_embeddings_count": len(text_embeddings),
                "audio_embeddings_count": len(audio_embeddings),
                "token_importance": {
                    "visual": visual_token_importance,
                    "text": text_token_importance,
                    "audio": audio_token_importance,
                },
                "contributing_factors": contributing_factors,
                "explanation": explanation,
            }
            
        except Exception as e:
            logger.error(f"Error in multimodal analysis: {e}")
            return {
                "fusion_score": 0.0,
                "modality_contributions": {"visual": 0.33, "text": 0.33, "audio": 0.34},
                "error": str(e)
            }
    
    def _heuristic_fusion_analysis(self,
                                   visual_regions: List[np.ndarray],
                                   text_content: List[str],
                                   audio_segments: List[np.ndarray],
                                   visual_meta: Optional[List[Dict[str, Any]]],
                                   text_meta: Optional[List[Dict[str, Any]]],
                                   audio_meta: Optional[List[Dict[str, Any]]],
                                   visual_embeddings: np.ndarray,
                                   text_embeddings: np.ndarray,
                                   audio_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Heuristic-based fusion scoring that provides more accurate results
        than an untrained neural model.
        """
        # Calculate modality scores based on content presence and quality
        visual_score = 0.0
        text_score = 0.0
        audio_score = 0.0
        
        # Visual scoring: based on number and confidence of detected regions
        if len(visual_regions) > 0:
            visual_score = min(1.0, len(visual_regions) * 0.25)  # More regions = higher risk
            if visual_meta:
                avg_confidence = np.mean([m.get('confidence', 0.5) for m in visual_meta if m.get('confidence')])
                visual_score *= avg_confidence
        
        # Text scoring: based on content length and PII indicators
        if len(text_content) > 0:
            combined_text = " ".join(text_content)
            text_length = len(combined_text)
            
            # Base score on text length (more text = potentially more exposure)
            text_score = min(1.0, text_length / 500.0)  # Normalize to 500 chars
            
            # Check for PII patterns
            pii_indicators = [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',  # email
                r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',  # phone
                r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',  # aadhaar/card
                r'\b[A-Z]{5}\d{4}[A-Z]\b',  # PAN
            ]
            
            pii_count = 0
            for pattern in pii_indicators:
                pii_count += len(re.findall(pattern, combined_text))
            
            # Boost score based on PII findings
            if pii_count > 0:
                text_score = min(1.0, text_score + (pii_count * 0.15))
        
        # Audio scoring: based on segment count and duration
        if len(audio_segments) > 0:
            audio_score = min(1.0, len(audio_segments) * 0.2)
        
        # Calculate weighted fusion score
        total_weight = 0
        weighted_sum = 0
        
        if visual_score > 0:
            weighted_sum += visual_score * 0.4  # Visual evidence is strong
            total_weight += 0.4
        if text_score > 0:
            weighted_sum += text_score * 0.5  # Text is most reliable
            total_weight += 0.5
        if audio_score > 0:
            weighted_sum += audio_score * 0.3
            total_weight += 0.3
        
        # Calculate fusion score (0-1 range for consistency with frontend display)
        fusion_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate modality contributions (normalized to sum to 1.0)
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
        
        # Determine primary modality
        primary_modality = max(modality_contributions, key=modality_contributions.get)
        
        # Boost fusion score if high-risk PII detected in text
        if text_score > 0.7:  # High text risk
            fusion_score = min(1.0, fusion_score * 1.2)  # Boost by 20%
        
        # Build contributing factors
        contributing_factors = []
        
        # Add visual factors
        if visual_meta:
            for idx, meta in enumerate(visual_meta[:3]):  # Top 3
                contributing_factors.append({
                    "modality": "visual",
                    "index": idx,
                    "importance": float(visual_score / max(len(visual_meta), 1)),
                    "region": {
                        "bbox": meta.get("bbox"),
                        "class_name": meta.get("class_name"),
                        "confidence": meta.get("confidence"),
                        "frame_timestamp": meta.get("frame_timestamp"),
                    }
                })
        
        # Add text factors
        if text_meta:
            for idx, meta in enumerate(text_meta[:3]):  # Top 3
                contributing_factors.append({
                    "modality": "text",
                    "index": idx,
                    "importance": float(text_score / max(len(text_meta), 1)),
                    "text_span": meta.get("text_span"),
                    "source": meta.get("source"),
                })
        
        # Add audio factors
        if audio_meta:
            for idx, meta in enumerate(audio_meta[:3]):  # Top 3
                contributing_factors.append({
                    "modality": "audio",
                    "index": idx,
                    "importance": float(audio_score / max(len(audio_meta), 1)),
                    "time_range": meta.get("time_range"),
                })
        
        # Generate explanation
        explanation_parts = []
        if visual_score > 0.3:
            explanation_parts.append(f"{len(visual_regions)} document/sensitive region(s) detected")
        if text_score > 0.3:
            explanation_parts.append("text content contains potential PII")
        if audio_score > 0.3:
            explanation_parts.append("audio contains spoken content")
        
        explanation = f"Privacy risk detected: {', '.join(explanation_parts) if explanation_parts else 'minimal exposure'}. "
        explanation += f"The {primary_modality} modality contributes most significantly to the risk assessment."
        
        return {
            "fusion_score": float(fusion_score),
            "modality_contributions": modality_contributions,
            "primary_modality": primary_modality,
            "visual_embeddings_count": len(visual_embeddings),
            "text_embeddings_count": len(text_embeddings),
            "audio_embeddings_count": len(audio_embeddings),
            "contributing_factors": contributing_factors,
            "explanation": explanation,
        }

# Global instance for use across the application
multimodal_analyzer = MultimodalFusionAnalyzer()
