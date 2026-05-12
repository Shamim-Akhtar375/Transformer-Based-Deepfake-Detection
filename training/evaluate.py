import torch
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from sklearn.metrics import accuracy_score, precision_recall_f1_score_support, roc_auc_score, confusion_matrix
import numpy as np
import os
import glob
from train import VideoDeepfakeDataset
from torch.utils.data import DataLoader
import kagglehub

def evaluate_temporal_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Temporal Model for Evaluation on {device}")
    
    model_path = "../checkpoints/best_video_model"
    if not os.path.exists(model_path):
        print("Model checkpoint not found. Please run train.py first.")
        return
        
    model = TimesformerForVideoClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    print("Locating Dataset...")
    dataset_path = kagglehub.dataset_download("xdxd003/ff-c23")
    
    # Load validation data
    dataset = VideoDeepfakeDataset(dataset_path, num_frames=32, is_training=False)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    y_true = []
    y_pred_probs = []
    
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            outputs = model(videos)
            probs = torch.softmax(outputs.logits, dim=-1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred_probs.extend(probs[:, 1].cpu().numpy()) # Probability of FAKE
            
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    # Reduced threshold for false negative optimization
    y_pred = (y_pred_probs > 0.4).astype(int) 
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_f1_score_support(y_true, y_pred, average="binary")
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except:
        roc_auc = 0.0
        
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print("\n=====================================")
    print("      TEMPORAL MODEL EVALUATION      ")
    print("=====================================")
    print(f"Accuracy:  {accuracy:.4f} (Target: >0.92)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f} (Crucial for False Negatives)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix [TN, FP] [FN, TP]:")
    print(conf_matrix)
    print("=====================================")

if __name__ == "__main__":
    evaluate_temporal_model()
