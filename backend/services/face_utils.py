import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

class FaceProcessor:
    def __init__(self):
        # Initialize MTCNN (High accuracy and speed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            image_size=160, margin=20, min_face_size=40,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )
        
    def detect_and_align(self, img_cv, margin=0.15):
        """MTCNN-based face detection and cropping."""
        if img_cv is None: return None
        
        # Convert to RGB for MTCNN
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Detect bboxes
        boxes, _ = self.detector.detect(pil_img)
        
        if boxes is None or len(boxes) == 0:
            return None

        # Pick the largest detection
        box = boxes[0] # MTCNN sorts by confidence/size
        
        h, w = img_cv.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        
        # Add margin
        fw = x2 - x1
        fh = y2 - y1
        x1 = max(0, int(x1 - margin * fw))
        y1 = max(0, int(y1 - margin * fh))
        x2 = min(w, int(x2 + margin * fw))
        y2 = min(h, int(y2 + margin * fh))
        
        face_crop = img_cv[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
            
        return face_crop

    def get_face_bbox(self, img_cv):
        """Returns relative bbox [xmin, ymin, width, height] for the largest face."""
        if img_cv is None: return None
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        boxes, _ = self.detector.detect(pil_img)
        if boxes is None or len(boxes) == 0:
            return None
            
        box = boxes[0]
        h, w = img_cv.shape[:2]
        
        # Convert to relative
        return [box[0]/w, box[1]/h, (box[2]-box[0])/w, (box[3]-box[1])/h]

face_processor = FaceProcessor()


