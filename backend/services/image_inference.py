import torch
import torch.nn as nn
import timm
from PIL import Image
import io
import base64
import cv2
import numpy as np
import time
import logging
import os
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from services.face_utils import face_processor

logger = logging.getLogger("ImageInference")

class ImageDeepfakeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.device.type == "cuda"
        self.model = None
        self.classifier = None
        
        logger.info(f"Initializing Optimized Image AI Model (EfficientNet-B4) on {self.device}")
        
        try:
            self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2).to(self.device)
            self.classifier = self.model.classifier
            
            checkpoint_path = "checkpoints/image_model_best.pth"
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                logger.info("Loaded fine-tuned EfficientNet-B4 weights.")
            
            if self.use_fp16:
                self.model = self.model.half()
            
            self.model.eval()
            
            # Note: torch.compile is disabled for Windows compatibility (avoids cl.exe dependency)
                
            self.input_size = 160 # Optimized from 224
            self.transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Grad-CAM++ Initialization (Only used if requested)
            target_layers = [self.model.conv_head] if hasattr(self.model, 'conv_head') else [self.model.blocks[-1]]
            self.cam = GradCAMPlusPlus(model=self.model, target_layers=target_layers)
            
        except Exception as e:
            logger.error(f"Failed to load Image Model: {e}")
            self.model = None

    def analyze(self, image_bytes: bytes, generate_heatmap: bool = False):
        start_time = time.time()
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            face_crop = face_processor.detect_and_align(img_cv)
            if face_crop is None:
                face_crop = img_cv
            
            image_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            if self.use_fp16:
                input_tensor = input_tensor.half()
            
            if self.model is None:
                return {"error": "Image model not initialized."}

            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=-1)
            
            prob_fake = probabilities[0][1].item()
            prob_real = probabilities[0][0].item()
            
            final_label = "FAKE" if prob_fake > 0.5 else "REAL"
            confidence = prob_fake if final_label == "FAKE" else prob_real
            
            heatmap_base64 = None
            if generate_heatmap:
                heatmap_base64 = self.generate_gradcam(input_tensor, face_crop)
            
            proc_time = time.time() - start_time
            
            return {
                "label": final_label,
                "confidence": float(confidence),
                "heatmap": heatmap_base64,
                "processing_time": f"{proc_time:.2f}s",
                "analysis": f"High-speed forensic scan complete ({proc_time:.2f}s). Verdict: {final_label}."
            }
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return {"error": str(e)}

    def get_embedding(self, pil_images):
        """Returns spatial embeddings for a batch of PIL images."""
        if self.model is None: return None
        
        tensors = torch.stack([self.transform(img) for img in pil_images]).to(self.device)
        if self.use_fp16:
            tensors = tensors.half()
            
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                if hasattr(self.model, 'forward_features'):
                    features = self.model.forward_features(tensors)
                    embedding = self.model.global_pool(features)
                else:
                    embedding = self.model(tensors)
        return embedding

    def generate_gradcam(self, input_tensor, face_cv):
        try:
            targets = [ClassifierOutputTarget(1)]
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            img_normalized = cv2.resize(face_cv, (self.input_size, self.input_size)).astype(np.float32) / 255
            visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
            
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', visualization_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Grad-CAM error: {e}")
            return None

image_detector = ImageDeepfakeDetector()
