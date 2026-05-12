import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import timm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import os
import cv2
import glob
from PIL import Image
import numpy as np
from torchvision import transforms
from services.face_utils import face_processor
from services.fusion import FusionTransformer

class ProductionDeepfakeDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, is_training=True):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.is_training = is_training
        self.samples = []
        
        # Load samples (FF++ structure)
        # 0: REAL, 1: FAKE
        for label, folder in [(0, "original_sequences"), (1, "manipulated_sequences")]:
            path = os.path.join(root_dir, folder)
            if os.path.exists(path):
                for v in glob.glob(f"{path}/**/*.mp4", recursive=True):
                    self.samples.append((v, label))
        
        self.spatial_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.RandomRotation(10),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if is_training else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

        # Load frames (Sample 8 then repeat to 16 for VideoMAE architecture)
        indices = np.linspace(0, total - 1, 8, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                face = face_processor.detect_and_align(frame)
                if face is not None:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(face))
                else:
                    h, w = frame.shape[:2]
                    s = min(h, w)
                    crop = frame[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
                    frames.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            if len(frames) == 8: break
        cap.release()
        
        # Repeat to 16
        if len(frames) >= 8:
            frames = [f for f in frames for _ in (0, 1)][:16]
        while len(frames) < 16:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
        return frames

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._load_video(path)
        
        # Return tensors for all frames
        tensors = torch.stack([self.spatial_transform(f) for f in frames])
        return tensors, torch.tensor(label, dtype=torch.long)

def train_production_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "data/ff-c23" # Path to FF++ C23
    
    # 1. Initialize Models
    spatial_model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2).to(device)
    temporal_model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-small-finetuned-kinetics", num_labels=2, ignore_mismatched_sizes=True
    ).to(device)
    fusion_model = FusionTransformer(spatial_dim=1792, temporal_dim=384, artifact_dim=8).to(device)
    
    # 2. Dataset & Loader
    dataset = ProductionDeepfakeDataset(dataset_path, num_frames=16)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    
    # 3. Optimizer & Criterion
    params = list(spatial_model.parameters()) + list(temporal_model.parameters()) + list(fusion_model.parameters())
    optimizer = AdamW(params, lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    print("Starting Production Pipeline Training...")
    
    for epoch in range(10):
        spatial_model.train()
        temporal_model.train()
        fusion_model.train()
        
        for batch_idx, (tensors, labels) in enumerate(train_loader):
            # tensors: [B, T, C, H, W]
            tensors, labels = tensors.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                # Get Spatial Embeddings (mean over frames)
                B, T, C, H, W = tensors.shape
                flat_tensors = tensors.view(-1, C, H, W)
                spatial_feats = spatial_model.forward_features(flat_tensors)
                spatial_emb = spatial_model.global_pool(spatial_feats) # [B*T, 1792]
                spatial_emb = spatial_emb.view(B, T, -1).mean(dim=1) # [B, 1792]
                
                # Get Temporal Embeddings
                # VideoMAE expects [B, T, C, H, W] but different normalization usually
                # For simplicity here we use the same tensors
                temp_outputs = temporal_model(tensors, output_hidden_states=True)
                temporal_emb = temp_outputs.hidden_states[-1].mean(dim=1) # [B, 384]
                
                # Artifact features (dummy for training logic)
                artifact_emb = torch.zeros((B, 8), device=device)
                
                # Fusion
                logits = fusion_model(spatial_emb, temporal_emb, artifact_emb)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")
        
        # Save checkoints
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(spatial_model.state_dict(), "checkpoints/image_model_best.pth")
        torch.save(temporal_model.state_dict(), "checkpoints/video_model_best.pth")
        torch.save(fusion_model.state_dict(), "checkpoints/fusion_model_best.pth")

if __name__ == "__main__":
    train_production_pipeline()
