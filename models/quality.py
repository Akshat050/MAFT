import torch
import torch.nn as nn


class QualityEstimator(nn.Module):
    """
    Estimates per-token quality scores in [0, 1] from input features.
    
    Architecture: LayerNorm → Linear(H→H/2) → GELU → Linear(H/2→1) → Sigmoid
    Masks padding tokens to 0.
    
    Args:
        hidden_dim (int): Hidden dimension size (H)
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, features, mask=None):
        """
        Compute per-token quality scores.
        
        Args:
            features (torch.Tensor): Input features of shape [B, L, H]
            mask (torch.Tensor, optional): Boolean mask of shape [B, L] where True indicates PADDING
        
        Returns:
            torch.Tensor: Quality scores of shape [B, L] in range [0, 1]
        """
        # features: [B, L, H]
        x = self.layer_norm(features)  # [B, L, H]
        x = self.fc1(x)  # [B, L, H/2]
        x = self.gelu(x)  # [B, L, H/2]
        x = self.fc2(x)  # [B, L, 1]
        x = self.sigmoid(x)  # [B, L, 1]
        x = x.squeeze(-1)  # [B, L]
        
        # Mask padding tokens to 0 (mask is True for padding)
        if mask is not None:
            x = x * (~mask).float()
        
        return x
