# Digital Shadow Analyzer

> **An Explainable Multimodal AI System for Privacy Exposure Detection**

A cutting-edge privacy analysis tool that uses computer vision, NLP, speech recognition, and transformer-based multimodal fusion to detect and quantify personally identifiable information (PII) across text, images, videos, and audio. The system provides **human-interpretable explanations** showing exactly what, where, and why privacy risks were detected.

---

## 🎯 What It Does

- **Text Analysis**: Detects PII (names, emails, phone numbers, Aadhaar, PAN) using NER and regex patterns
- **Image Analysis**: Uses YOLOv8 to detect documents/ID cards, extracts text via OCR, analyzes EXIF metadata
- **Video Analysis**: Transcribes speech with Whisper, detects documents in frames, links evidence to timestamps
- **Audio Analysis**: Transcribes audio and analyzes for spoken PII
- **Multimodal Fusion**: Combines visual, textual, and audio evidence using a transformer-based fusion model
- **Explainability**: Provides token-level importance, contributing factors, and natural language explanations

---

## 🏗️ Architecture

### Phase 1: Single-Modality Analyzers
- **Text**: spaCy NER + regex patterns → exposure scoring
- **Image**: YOLOv8 object detection + EasyOCR → document extraction
- **Video**: Whisper ASR + frame sampling + YOLOv8 → multimodal content
- **Audio**: Whisper transcription → text analysis pipeline

### Phase 2: Multimodal Fusion
- **Transformer-based fusion model** (4 layers, 8 attention heads)
- Projects visual (512-dim), text (768-dim), audio (1024-dim) embeddings to common space
- Outputs unified exposure score and modality contribution weights

### Phase 3: Explainability (Current)
- **Token-level importance** mapped to evidence
- **Contributing factors** with bounding boxes, timestamps, text spans
- **Natural language explanations** for transparency
- **Interactive frontend** displaying evidence and reasoning

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Node.js 16+** and npm/yarn
- **Git**

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/srinivasg0/digital-shadow-analyzer.git
   cd digital-shadow-analyzer
   ```

2. **Set up Python virtual environment**
   ```bash
   cd backend
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Download YOLOv8 weights** (automatic on first run, or manual)
   ```bash
   # The yolov8n.pt model will download automatically when you first run the app
   # Or download manually from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   # Place it in the backend/ directory
   ```

6. **Start the backend server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

   The UI will be available at `http://localhost:3000`

---

## 📖 Usage

1. **Open your browser** and navigate to `http://localhost:3000`

2. **Choose an analysis mode** from the tabs:
   - **Text**: Paste text or upload a .txt file
   - **Image**: Upload images (JPG, PNG, etc.)
   - **Video**: Upload video files (MP4, AVI, MOV, etc.)
   - **Audio**: Upload audio files (MP3, WAV, etc.)

3. **View results**:
   - **Exposure Score**: 0-100 privacy risk score
   - **Fusion Analysis**: Multimodal score with modality contributions
   - **Explanation**: Human-readable reasoning
   - **Contributing Evidence**: Top factors with bounding boxes, timestamps, spans
   - **Detected Documents**: List of found documents with OCR text
   - **PII Evidence**: Specific entities detected (names, IDs, etc.)

---

## 🔧 API Endpoints

### Text Analysis
```bash
POST /analyze-text/
Content-Type: application/json

{
  "text": "My name is John Doe and my Aadhaar is 1234-5678-9012"
}
```

### Image Analysis
```bash
POST /analyze-image/
Content-Type: multipart/form-data

file: <image_file>
```

### Video Analysis
```bash
POST /analyze-video/
Content-Type: multipart/form-data

file: <video_file>
```

### Audio Analysis
```bash
POST /analyze-audio/
Content-Type: multipart/form-data

file: <audio_file>
```

---

## 📁 Project Structure

```
digital-shadow-analyzer/
├── backend/
│   ├── app/
│   │   ├── analysis/
│   │   │   ├── text_analyzer.py      # NER + regex PII detection
│   │   │   ├── image_analyzer.py     # YOLOv8 + OCR
│   │   │   ├── video_analyzer.py     # Video → frames + audio
│   │   │   └── audio_analyzer.py     # Whisper transcription
│   │   ├── fusion/
│   │   │   └── multimodal_fusion.py  # Transformer fusion + explainability
│   │   └── main.py                   # FastAPI endpoints
│   ├── requirements.txt
│   └── yolov8n.pt                    # YOLOv8 weights (auto-downloaded)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AnalysisTabs.tsx      # Main UI
│   │   │   └── FileUploader.tsx      # Upload component
│   │   └── pages/
│   │       └── index.tsx             # Home page
│   ├── package.json
│   └── next.config.js
└── README.md
```

---

## 🧪 Example Use Cases

### 1. Social Media Privacy Check
Upload a photo before posting to detect visible ID cards, documents, or sensitive text in the background.

### 2. Corporate Compliance
Scan documents and videos for PII before sharing externally to ensure GDPR/compliance.

### 3. Digital Footprint Analysis
Analyze your online content to understand your privacy exposure score.

### 4. Forensic Investigation
Examine leaked content to assess what sensitive information was exposed.

---

## 🎓 Technical Highlights

### Multimodal Fusion Model
- **Architecture**: 4-layer transformer encoder with 8 attention heads
- **Input**: Visual (512-dim), Text (768-dim), Audio (1024-dim) embeddings
- **Output**: Unified exposure score (0-1), modality weights, token importance

### Explainability Features
- **Token-level importance**: Softmax over L2 norms of transformer representations
- **Evidence linking**: Maps importance back to bounding boxes, timestamps, text spans
- **Contributing factors**: Top-k evidence per modality with metadata
- **Natural language explanations**: Templated summaries of risk drivers

### Models Used
- **YOLOv8n**: Document/object detection
- **EasyOCR**: Text extraction from images
- **Whisper (base)**: Speech-to-text transcription
- **spaCy (en_core_web_sm)**: Named entity recognition
- **Sentence-BERT (all-MiniLM-L6-v2)**: Text embeddings

---

## ⚠️ Current Limitations

### 1. **Placeholder Embeddings**
- **Visual**: Uses resized image patches instead of YOLOv8 backbone features
- **Audio**: Random embeddings instead of Whisper encoder or wav2vec2
- **Impact**: Fusion model doesn't leverage full feature richness

### 2. **Attention Weights**
- Currently a placeholder (all-ones tensor)
- Real cross-modal attention not extracted from transformer layers
- Limits fine-grained interpretability

### 3. **No Visual Saliency Maps**
- Token importance exists but no heatmap overlays on images/video
- Users can't see *exactly* which pixels drove the score

### 4. **Templated Explanations**
- Simple rule-based summaries
- Not using GPT or advanced NLG for richer narratives

### 5. **No Model Training**
- Fusion model uses random initialization
- Not fine-tuned on privacy exposure datasets
- Scores may not be calibrated to real-world risk

### 6. **Limited Document Classes**
- YOLOv8 COCO model only detects "book" class as proxy for documents
- Misses ID cards, passports, credit cards (would need custom training)

### 7. **Performance**
- Video processing is slow (frame extraction + OCR + transcription)
- No GPU acceleration configured by default
- Large videos may timeout

### 8. **No Persistent Storage**
- Results are not saved
- No user accounts or history tracking

---

## 🚧 What Could Have Been Done (Future Improvements)

### **High Priority**

1. **Real Visual Embeddings**
   - Extract features from YOLOv8 backbone (C3, C4, C5 layers)
   - Use ResNet/EfficientNet for region embeddings
   - **Impact**: Better visual understanding in fusion

2. **Grad-CAM Saliency Maps**
   - Compute gradients w.r.t. input images
   - Generate heatmaps showing influential regions
   - Overlay on frontend with bounding boxes
   - **Impact**: Visual proof of "why this region is risky"

3. **Real Attention Extraction**
   - Hook transformer encoder layers
   - Extract cross-modal attention weights
   - Visualize which text tokens attend to which image regions
   - **Impact**: Deeper interpretability

4. **Custom Document Detector**
   - Fine-tune YOLOv8 on ID cards, passports, credit cards, Aadhaar, PAN
   - Create labeled dataset (or use synthetic data)
   - **Impact**: 10x better document detection accuracy

5. **GPT-Powered Explanations**
   - Integrate OpenAI API or local LLaMA
   - Generate context-aware narratives: *"Your Aadhaar card is clearly visible in the bottom-left corner at timestamp 0:23, with the number partially legible. Risk: High."*
   - **Impact**: Human-friendly, actionable insights

### **Medium Priority**

6. **Fine-Tune Fusion Model**
   - Create/curate privacy exposure dataset
   - Train with contrastive loss (high-risk vs. low-risk pairs)
   - Calibrate scores to real-world risk levels
   - **Impact**: Accurate, trustworthy scores

7. **Frontend Evidence Overlays**
   - `EvidenceViewer.tsx`: Draw bounding boxes on images
   - `SaliencyOverlay.tsx`: Show heatmaps
   - Video timeline with clickable timestamps
   - Text highlighting for spans
   - **Impact**: Interactive, visual evidence exploration

8. **Audio Embeddings**
   - Use Whisper encoder or wav2vec2 features
   - Segment by speaker or topic
   - **Impact**: Better audio understanding in fusion

9. **Real-Time Analysis**
   - WebRTC for live camera/microphone input
   - Streaming analysis with incremental results
   - **Impact**: Privacy check before recording/posting

10. **Multi-Language Support**
    - Extend to Hindi, Spanish, French, etc.
    - Use multilingual models (mBERT, XLM-R)
    - **Impact**: Global usability

### **Low Priority (Nice-to-Have)**

11. **User Accounts & History**
    - Save analysis results
    - Track exposure trends over time
    - Privacy dashboard
    - **Impact**: Longitudinal privacy insights

12. **Redaction Tool**
    - Auto-blur detected PII in images/videos
    - Generate "safe" versions for sharing
    - **Impact**: Actionable privacy protection

13. **Browser Extension**
    - Analyze images before uploading to social media
    - Real-time warnings
    - **Impact**: Proactive privacy protection

14. **Mobile App**
    - iOS/Android with on-device inference
    - Camera integration
    - **Impact**: Accessibility and convenience

15. **Adversarial Robustness**
    - Test against obfuscation attacks (blur, noise, rotation)
    - Improve OCR resilience
    - **Impact**: Reliability in real-world scenarios

16. **Differential Privacy**
    - Add noise to embeddings for privacy-preserving analysis
    - Federated learning for model updates
    - **Impact**: Meta-privacy (analyzing privacy without compromising it)

---

## 🛡️ Privacy & Security

- **Local Processing**: All analysis runs locally; no data sent to external servers (except optional GPT API)
- **No Data Storage**: Uploaded files are processed in-memory and deleted immediately
- **Open Source**: Full transparency; audit the code yourself

---

## 📊 Performance Tips

### Speed Up Video Processing
- Reduce `max_frames` in `video_analyzer.py` (default: 15)
- Use shorter videos for testing
- Enable GPU acceleration (install `torch` with CUDA)

### Improve Accuracy
- Use higher-resolution images
- Ensure good lighting and focus
- Use clear audio recordings

### GPU Acceleration
```bash
# Install PyTorch with CUDA (if you have an NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Custom document detection dataset
- Better embedding extractors
- Saliency map visualization
- GPT explanation generator
- Frontend overlays
- Performance optimization

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **Whisper** by OpenAI
- **EasyOCR** by JaidedAI
- **spaCy** by Explosion AI
- **Sentence-BERT** by UKPLab
- **Material-UI** by MUI

---

## 📧 Contact

For questions, suggestions, or collaboration:
- **GitHub**: [srinivasg0/digital-shadow-analyzer](https://github.com/srinivasg0/digital-shadow-analyzer)

---

## 🎯 Why This Project Matters

In an era of oversharing and data breaches, **privacy awareness is critical**. This tool empowers users to:
- **Understand** their digital footprint
- **Detect** hidden PII before it's too late
- **Trust** AI explanations instead of black-box scores

**Digital Shadow Analyzer transforms privacy analysis from opaque to transparent, from reactive to proactive.**

---

**Built with ❤️ for privacy-conscious users and AI enthusiasts.**
