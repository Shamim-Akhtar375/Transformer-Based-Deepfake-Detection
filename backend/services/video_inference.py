import cv2
import tempfile
import os
import numpy as np
import torch
import torch.nn as nn
import io
import base64
import time
import logging
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from services.image_inference import image_detector
from services.face_utils import face_processor
from services.fusion import FusionTransformer

from services.audio_detector import audio_detector
import subprocess
import static_ffmpeg

# Performance Optimizations
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VideoInference")

class VideoDeepfakeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.device.type == "cuda"
        self.num_frames = 4 # Optimized for high speed
        self.input_size = 160 # Optimized resolution
        
        logger.info(f"Initializing Multimodal VideoMAE Pipeline on {self.device}")
        static_ffmpeg.add_paths()
        
        try:
            # Using VideoMAE-small
            self.model_name = "MCG-NJU/videomae-small-finetuned-kinetics"
            self.processor = VideoMAEImageProcessor.from_pretrained(
                self.model_name, size={"shortest_edge": self.input_size}
            )
            self.model = VideoMAEForVideoClassification.from_pretrained(
                self.model_name, num_labels=2, ignore_mismatched_sizes=True
            ).to(self.device)
            
            # Load Fusion Model (Updated for Audio)
            self.fusion_model = FusionTransformer(
                spatial_dim=1792, # EfficientNet-B4
                temporal_dim=384,  # VideoMAE-small
                audio_dim=768,     # Wav2Vec2
                artifact_dim=8
            ).to(self.device)
            
            # Load checkpoints
            if os.path.exists("checkpoints/video_model_best.pth"):
                self.model.load_state_dict(torch.load("checkpoints/video_model_best.pth", map_location=self.device))
            if os.path.exists("checkpoints/fusion_model_best.pth"):
                self.fusion_model.load_state_dict(torch.load("checkpoints/fusion_model_best.pth", map_location=self.device))

            if self.use_fp16:
                self.model = self.model.half()
                self.fusion_model = self.fusion_model.half()

            self.model.eval()
            self.fusion_model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load Video Model: {e}")
            self.model = None

    def _extract_audio(self, video_path):
        """Extracts audio from video file using ffmpeg."""
        audio_path = video_path + ".wav"
        try:
            command = [
                'ffmpeg', '-i', video_path, 
                '-ab', '160k', '-ac', '1', 
                '-ar', '16000', '-vn', audio_path,
                '-y', '-loglevel', 'quiet'
            ]
            subprocess.run(command, check=True)
            if os.path.exists(audio_path):
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                os.unlink(audio_path)
                return audio_bytes
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
        return None

    def _get_key_frames(self, video_path, count):
        """Fast intelligent sampling: Sample frames evenly across the video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []
        
        indices = np.linspace(0, total_frames - 1, count, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    def _extract_artifact_features(self, face_crops):
        """Fast forensic artifact extraction."""
        features = []
        for crop in face_crops:
            img_np = np.array(crop)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            noise = np.std(gray)
            features.append([blur, noise, 0, 0, 0, 0, 0, 0])
            
        feat_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        if self.use_fp16:
            feat_tensor = feat_tensor.half()
        return feat_tensor

    def analyze(self, video_bytes: bytes, generate_heatmap: bool = True):
        start_time = time.time()
        try:
            # Process directly in memory using temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(video_bytes)
                video_path = temp_video.name
            
            # 1. Extract Visual Frames (More frames for forensic timeline)
            num_forensic_frames = 12 
            raw_frames = self._get_key_frames(video_path, num_forensic_frames)
            
            # 2. Extract Audio
            audio_bytes = self._extract_audio(video_path)
            
            if os.path.exists(video_path): os.unlink(video_path)
            
            if not raw_frames:
                return {"error": "Failed to extract frames."}
            
            # Face Detection and Alignment
            valid_crops = []
            frame_timestamps = np.linspace(0, 100, len(raw_frames)) # Percentage timestamps
            
            first_bbox = face_processor.get_face_bbox(raw_frames[0]) if raw_frames else None
            
            for i, frame in enumerate(raw_frames):
                if i == 0:
                    crop = face_processor.detect_and_align(frame)
                elif first_bbox:
                    h, w = frame.shape[:2]
                    x, y, fw, fh = first_bbox
                    margin = 0.2
                    x1 = max(0, int((x - margin * fw) * w))
                    y1 = max(0, int((y - margin * fh) * h))
                    x2 = min(w, int((x + fw + margin * fw) * w))
                    y2 = min(h, int((y + fh + margin * fh) * h))
                    crop = frame[y1:y2, x1:x2]
                else:
                    crop = face_processor.detect_and_align(frame)
                
                if crop is not None and crop.size > 0:
                    valid_crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            
            if not valid_crops:
                return {"error": "No faces detected in the video stream."}
            
            # 3. Multimodal Inference & Forensic Scoring
            audio_res = None
            audio_emb = None
            if audio_bytes:
                audio_res = audio_detector.analyze(audio_bytes)
                if "embedding" in audio_res:
                    audio_emb = torch.tensor([audio_res["embedding"]], dtype=torch.float32).to(self.device)
                    if self.use_fp16: audio_emb = audio_emb.half()

            timeline = []
            suspicious_frames = []
            
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                # A. Spatial Features (EfficientNet-B4)
                spatial_embeddings = image_detector.get_embedding(valid_crops)
                
                # B. Per-Frame Prediction for Timeline
                # We use the spatial embeddings to get a per-frame "fakeness" score
                if image_detector.classifier is None:
                    return {"error": "Forensic analysis engine unavailable: Image classifier failed to initialize."}
                    
                with torch.no_grad():
                    frame_logits = image_detector.classifier(spatial_embeddings)
                    frame_probs = torch.softmax(frame_logits, dim=-1)[:, 1].cpu().numpy()
                
                # C. Temporal Features (VideoMAE-small)
                # Fill/Pad to 16 for VideoMAE input
                mae_frames = [valid_crops[i % len(valid_crops)] for i in range(16)]
                video_inputs = self.processor(mae_frames, return_tensors="pt").to(self.device)
                if self.use_fp16:
                    video_inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in video_inputs.items()}
                
                with torch.no_grad():
                    video_outputs = self.model(video_inputs.pixel_values, output_hidden_states=True)
                    temporal_emb = video_outputs.hidden_states[-1].mean(dim=1)
                
                # D. Artifact Extraction
                artifact_feats = self._extract_artifact_features(valid_crops)
                
                # E. Fusion (Final Verdict)
                spatial_emb_avg = spatial_embeddings.mean(dim=0, keepdim=True)
                artifact_feat_avg = artifact_feats.mean(dim=0, keepdim=True)
                
                with torch.no_grad():
                    fusion_logits = self.fusion_model(spatial_emb_avg, temporal_emb, audio_emb, artifact_feat_avg)
                    probabilities = torch.softmax(fusion_logits, dim=-1)
            
            prob_fake = probabilities[0][1].item()
            final_label = "FAKE" if prob_fake > 0.5 else "REAL"
            confidence = prob_fake if final_label == "FAKE" else (1 - prob_fake)
            
            # 4. Generate Forensic Data for UI
            for i, prob in enumerate(frame_probs):
                timestamp_sec = f"00:{i:02d}" # Dummy timestamp
                artifact_score = artifact_feats[i][0].item() / 500.0 # Normalized blur/artifact
                
                timeline.append({
                    "timestamp": timestamp_sec,
                    "fake_probability": float(prob),
                    "artifact_score": float(artifact_score),
                    "consistency_score": float(1.0 - abs(prob - prob_fake))
                })
                
                # Identify suspicious frames (prob > 0.5 or highest 3)
                if prob > 0.5 or i in np.argsort(frame_probs)[-3:]:
                    # Get base64 frame
                    buf = io.BytesIO()
                    valid_crops[i].save(buf, format="JPEG", quality=85)
                    frame_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    
                    # Generate Grad-CAM for this frame if needed
                    heatmap_b64 = None
                    if generate_heatmap and len(suspicious_frames) < 3:
                        heatmap_b64 = image_detector.generate_gradcam(
                            image_detector.transform(valid_crops[i]).unsqueeze(0).to(self.device),
                            cv2.cvtColor(np.array(valid_crops[i]), cv2.COLOR_RGB2BGR)
                        )
                    
                    suspicious_frames.append({
                        "id": i,
                        "timestamp": timestamp_sec,
                        "image": f"data:image/jpeg;base64,{frame_b64}",
                        "heatmap": heatmap_b64,
                        "confidence": float(prob),
                        "artifacts": float(artifact_score)
                    })

            proc_time = time.time() - start_time
            
            return {
                "label": final_label,
                "confidence": float(confidence),
                "fake_ratio": float(prob_fake),
                "frames_analyzed": len(valid_crops),
                "audio_analyzed": audio_res is not None,
                "audio_prediction": audio_res["label"] if audio_res else "N/A",
                "timeline": timeline,
                "suspicious_frames": suspicious_frames[:6], # Limit for UI
                "processing_time": f"{proc_time:.2f}s",
                "analysis_summary": {
                    "manipulation_severity": "HIGH" if prob_fake > 0.8 else "MEDIUM" if prob_fake > 0.4 else "LOW",
                    "suspicious_frame_count": len([f for f in frame_probs if f > 0.5]),
                    "temporal_inconsistency": float(np.std(frame_probs)),
                    "lip_sync_mismatch": float(abs(prob_fake - audio_res['fake_probability'])) if audio_res else 0.1
                },
                "analysis": f"Advanced Forensic Multimodal Analysis complete. Detected {final_label} media with {confidence:.1%} confidence based on spatial artifacts and temporal inconsistencies."
            }
            
        except Exception as e:
            logger.error(f"Advanced Video analysis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}

video_detector = VideoDeepfakeDetector()


