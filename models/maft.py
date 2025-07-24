import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .encoders import TextEncoder, AudioEncoder, VisualEncoder
from .fusion import FusionTransformer, ModalityDropout


class MultiTaskHead(nn.Module):
    """Dual heads for classification and regression tasks."""
    
    def __init__(self, hidden_dim: int = 768, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch_size, hidden_dim] - CLS token features
        Returns:
            classification_logits: [batch_size, num_classes]
            regression_output: [batch_size, 1]
        """
        cls_logits = self.classification_head(features)
        reg_output = self.regression_head(features)
        return cls_logits, reg_output


class MAFT(nn.Module):
    """Multimodal Attention Fusion Transformer."""
    
    def __init__(self, text_model_name: str = "bert-base-uncased",
                 hidden_dim: int = 768, num_heads: int = 12, num_layers: int = 1,
                 audio_input_dim: int = 74, visual_input_dim: int = 35,
                 num_classes: int = 2, dropout: float = 0.1,
                 modality_dropout_rate: float = 0.1, freeze_bert: bool = False,
                 return_attention: bool = False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.return_attention = return_attention
        
        # Modality encoders
        self.text_encoder = TextEncoder(text_model_name, hidden_dim, dropout=dropout, freeze_bert=freeze_bert)
        self.audio_encoder = AudioEncoder(audio_input_dim, hidden_dim, dropout=dropout)
        self.visual_encoder = VisualEncoder(visual_input_dim, hidden_dim, dropout=dropout)
        
        # Fusion transformer
        self.fusion_transformer = FusionTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            return_attention=return_attention
        )
        
        # Multi-task head
        self.multi_task_head = MultiTaskHead(hidden_dim, num_classes, dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Modality dropout
        self.modality_dropout = ModalityDropout(modality_dropout_rate)
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                audio_features: torch.Tensor,
                audio_mask: torch.Tensor,
                visual_features: torch.Tensor,
                visual_mask: torch.Tensor,
                audio_lengths: Optional[torch.Tensor] = None,
                visual_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, text_len]
            attention_mask: [batch_size, text_len]
            audio_features: [batch_size, audio_len, audio_dim]
            audio_mask: [batch_size, audio_len]
            visual_features: [batch_size, visual_len, visual_dim]
            visual_mask: [batch_size, visual_len]
            audio_lengths: [batch_size] - optional sequence lengths for audio
            visual_lengths: [batch_size] - optional sequence lengths for visual
        Returns:
            Dictionary containing:
                - classification_logits: [batch_size, num_classes]
                - regression_output: [batch_size, 1]
                - attention_weights: [batch_size, num_heads, total_len, total_len] (optional)
        """
        batch_size = input_ids.size(0)
        
        # Encode each modality
        text_features = self.text_encoder(input_ids, attention_mask)
        audio_features = self.audio_encoder(audio_features, audio_lengths)
        visual_features = self.visual_encoder(visual_features, visual_lengths)
        
        # Apply modality dropout during training
        text_features, audio_features, visual_features, attention_mask, audio_mask, visual_mask = \
            self.modality_dropout(text_features, audio_features, visual_features, 
                                attention_mask, audio_mask, visual_mask)
        
        # Add CLS token to the beginning of text sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        text_features = torch.cat([cls_tokens, text_features], dim=1)
        attention_mask = torch.cat([torch.ones(batch_size, 1, device=input_ids.device), attention_mask], dim=1)
        
        # Fuse modalities through transformer
        if self.return_attention:
            fused_features, attention_weights = self.fusion_transformer(
                text_features, audio_features, visual_features,
                attention_mask, audio_mask, visual_mask
            )
        else:
            fused_features = self.fusion_transformer(
                text_features, audio_features, visual_features,
                attention_mask, audio_mask, visual_mask
            )
            attention_weights = None
        
        # Extract CLS token features (first token)
        cls_features = fused_features[:, 0, :]  # [batch_size, hidden_dim]
        
        # Multi-task prediction
        classification_logits, regression_output = self.multi_task_head(cls_features)
        
        output_dict = {
            'classification_logits': classification_logits,
            'regression_output': regression_output,
            'cls_features': cls_features,
            'fused_features': fused_features
        }
        
        if attention_weights is not None:
            output_dict['attention_weights'] = attention_weights
        
        return output_dict
    
    def get_attention_weights(self, 
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            audio_features: torch.Tensor,
                            audio_mask: torch.Tensor,
                            visual_features: torch.Tensor,
                            visual_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights from the fusion transformer for visualization.
        """
        # Temporarily enable attention extraction
        original_return_attention = self.return_attention
        self.return_attention = True
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids, attention_mask, audio_features, audio_mask,
                visual_features, visual_mask
            )
            attention_weights = outputs.get('attention_weights')
        
        # Restore original setting
        self.return_attention = original_return_attention
        
        return attention_weights


class MAFTLoss(nn.Module):
    """Combined loss for multi-task learning."""
    
    def __init__(self, classification_weight: float = 0.5, regression_weight: float = 0.5):
        super().__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, 
                classification_logits: torch.Tensor,
                regression_output: torch.Tensor,
                classification_targets: torch.Tensor,
                regression_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            classification_logits: [batch_size, num_classes]
            regression_output: [batch_size, 1]
            classification_targets: [batch_size]
            regression_targets: [batch_size]
        Returns:
            Dictionary containing total loss and individual losses
        """
        # Classification loss
        cls_loss = self.ce_loss(classification_logits, classification_targets)
        
        # Regression loss
        reg_loss = self.mse_loss(regression_output.squeeze(-1), regression_targets)
        
        # Combined loss
        total_loss = (self.classification_weight * cls_loss + 
                     self.regression_weight * reg_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'regression_loss': reg_loss
        } 