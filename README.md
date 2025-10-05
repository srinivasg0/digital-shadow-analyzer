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

## 🚀 Future Enhancements

The system is fully functional and production-ready. Here are some advanced features that could further enhance it:

### **Visual Enhancements**
- **Grad-CAM Saliency Maps**: Overlay heatmaps on images/videos showing exactly which pixels influenced the risk score
- **Interactive Evidence Viewer**: Clickable bounding boxes and video timeline navigation
- **Custom Document Detector**: Fine-tune YOLOv8 on specific ID types (passports, credit cards, etc.)

### **Explainability Upgrades**
- **GPT-Powered Narratives**: Generate richer, context-aware explanations using LLMs
- **Cross-Modal Attention Visualization**: Show which text tokens attend to which image regions
- **Real-Time Feedback**: Live analysis during video recording or camera preview

### **Production Optimizations**
- **GPU Acceleration**: Configure CUDA for faster processing
- **Model Fine-Tuning**: Train fusion model on privacy exposure datasets for calibrated scores
- **Multi-Language Support**: Extend to Hindi, Spanish, French, etc.
- **User Accounts**: Save analysis history and track privacy trends over time

### **Advanced Features**
- **Auto-Redaction Tool**: Automatically blur detected PII for safe sharing
- **Browser Extension**: Analyze images before uploading to social media
- **Mobile App**: On-device inference for iOS/Android

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
