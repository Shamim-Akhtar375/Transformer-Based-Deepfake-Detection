import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    def __init__(self, spatial_dim=1792, temporal_dim=384, audio_dim=768, artifact_dim=8, num_heads=4):
        """
        Multimodal Fusion Transformer.
        spatial_dim: EfficientNet-B4 embedding size
        temporal_dim: VideoMAE-small embedding size
        audio_dim: Wav2Vec2 embedding size
        artifact_dim: Number of forensic artifact features
        """
        super().__init__()
        self.fusion_dim = 256
        
        # Projections to common embedding space
        self.spatial_proj = nn.Linear(spatial_dim, self.fusion_dim)
        self.temporal_proj = nn.Linear(temporal_dim, self.fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, self.fusion_dim)
        self.artifact_proj = nn.Linear(artifact_dim, self.fusion_dim)
        
        # Transformer Attention Fusion Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim, 
            nhead=num_heads, 
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, spatial_emb, temporal_emb, audio_emb=None, artifact_emb=None):
        # spatial_emb: [B, S_dim]
        # temporal_emb: [B, T_dim]
        # audio_emb: [B, Au_dim] (Optional)
        # artifact_emb: [B, Ar_dim] (Optional)
        
        tokens = []
        
        s = self.spatial_proj(spatial_emb).unsqueeze(1) # [B, 1, 256]
        tokens.append(s)
        
        t = self.temporal_proj(temporal_emb).unsqueeze(1) # [B, 1, 256]
        tokens.append(t)
        
        if audio_emb is not None:
            au = self.audio_proj(audio_emb).unsqueeze(1) # [B, 1, 256]
            tokens.append(au)
            
        if artifact_emb is not None:
            ar = self.artifact_proj(artifact_emb).unsqueeze(1) # [B, 1, 256]
            tokens.append(ar)
        
        # Concatenate tokens
        fusion_input = torch.cat(tokens, dim=1) # [B, N, 256]
        
        # Self-attention over modalities
        fusion_output = self.transformer_encoder(fusion_input)
        
        # Global average pooling
        combined = fusion_output.mean(dim=1) # [B, 256]
        
        return self.classifier(combined)

fusion_model = FusionTransformer()

