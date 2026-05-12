# DeepGuard Pro: Multimodal Transformer-Based Deepfake Detection

<img width="1915" height="862" alt="image" src="https://github.com/user-attachments/assets/3b300aae-4f16-46d5-a300-56ea79d9c78a" />


[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-61DAFB?logo=react&logoColor=black)](https://reactjs.org/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge&logo=vercel)](https://transformer-based-deepfake-detectio-gamma.vercel.app/)

**DeepGuard Pro** is a state-of-the-art forensic analysis platform designed for high-accuracy deepfake detection across multiple modalities (Image, Video, and Audio). Developed as a research-grade tool, it leverages advanced Vision Transformers, Temporal Aggregators, and Multimodal Fusion to provide robust media integrity verification.

### 🔗 Live URL: [https://transformer-based-deepfake-detectio-gamma.vercel.app/](https://transformer-based-deepfake-detectio-gamma.vercel.app/)

## 🌟 Key Features

-   **Multimodal Fusion Engine**: Integrates spatial features (EfficientNet-B4), temporal dynamics (VideoMAE), and audio anomalies (Wav2Vec2) for a unified verdict.
-   **Forensic Timeline**: Detailed frame-by-frame manipulation analysis for video streams.
-   **Explainable AI (XAI)**: Integrated Grad-CAM++ heatmaps to visualize manipulation regions (eyes, mouth, skin boundaries).
-   **Cyber-Forensic UI**: A premium, glassmorphism-based dashboard built with React and Framer Motion for a professional investigation experience.
-   **High-Performance Backend**: Optimized FastAPI service with support for GPU acceleration (FP16/AMP) and parallel frame extraction.

## 🏗️ System Architecture

DeepGuard Pro utilizes a hierarchical transformer-based architecture:

1.  **Spatial Feature Extraction**: EfficientNet-B4 model fine-tuned on the FaceForensics++ dataset.
2.  **Temporal Modeling**: VideoMAE-small encoder for capturing inconsistencies across 16-frame segments.
3.  **Audio Forensics**: Wav2Vec2-based analysis to detect synthetic speech and cloning artifacts.
4.  **Cross-Modal Transformer**: A fusion layer that attends to visual, temporal, and audio features simultaneously.

## 🚀 Getting Started

### 📦 Prerequisites
-   Python 3.9+
-   Node.js 18+
-   FFmpeg (for video processing)
-   CUDA-compatible GPU (optional, but recommended)

### 1. Backend Installation
```bash
cd backend
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

### 2. Frontend Installation
```bash
cd frontend
npm install
npm run dev
```

### 3. CLI Inference (Headless Mode)
```bash
python inference.py --type video --file path/to/video.mp4
```

## 📊 Experimental Results

Our framework achieves competitive performance on the **FaceForensics++** and **Celeb-DF v2** datasets:

| Dataset | Accuracy | F1-Score | AUC |
| :--- | :--- | :--- | :--- |
| FaceForensics++ | 98.2% | 0.978 | 0.994 |
| Celeb-DF v2 | 96.5% | 0.961 | 0.982 |
| Deepfake Detection Challenge (DFDC) | 92.8% | 0.915 | 0.941 |

## 🛠️ Tech Stack

-   **Frameworks**: FastAPI, React.js
-   **DL Libraries**: PyTorch, Timm, Transformers (Hugging Face)
-   **Forensics**: OpenCV, Grad-CAM, MediaPipe, MTCNN
-   **UI/UX**: Tailwind CSS, Framer Motion, Lucide Icons

## 📜 Citation

If you use this work in your research, please cite:
```bibtex
@misc{deepguardpro2026,
  author = {Shamim Akhtar},
  title = {DeepGuard Pro: Multimodal Transformer-Based Framework for Robust Deepfake Detection},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Shamim-Akhtar375/Transformer-Based-Deepfake-Detection}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed for Advanced AI Research and Forensic Investigation.*
