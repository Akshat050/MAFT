import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from transformers import BertModel, BertTokenizer

from .encoders import TextEncoder, AudioEncoder, VisualEncoder


class TextOnlyBERT(nn.Module):
    """Text-only BERT baseline - no multimodal fusion."""
    
    def __init__(self, model_name: str = "bert-base-uncased", hidden_dim: int = 768,
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-task heads
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        
        # Multi-task prediction
        classification_logits = self.classification_head(pooled_output)
        regression_output = self.regression_head(pooled_output)
        
        return {
            'classification_logits': classification_logits,
            'regression_output': regression_output,
            'cls_features': pooled_output
        }


class LateFusion(nn.Module):
    """Simple late fusion baseline - concatenate features then linear head."""
    
    def __init__(self, text_model_name: str = "bert-base-uncased", 
                 audio_input_dim: int = 74, visual_input_dim: int = 35,
                 hidden_dim: int = 768, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Modality encoders
        self.text_encoder = TextEncoder(text_model_name, hidden_dim, dropout=dropout)
        self.audio_encoder = AudioEncoder(audio_input_dim, hidden_dim, dropout=dropout)
        self.visual_encoder = VisualEncoder(visual_input_dim, hidden_dim, dropout=dropout)
        
        # Global pooling for each modality
        self.text_pooler = nn.AdaptiveAvgPool1d(1)
        self.audio_pooler = nn.AdaptiveAvgPool1d(1)
        self.visual_pooler = nn.AdaptiveAvgPool1d(1)
        
        # Late fusion (concatenation + linear)
        fusion_dim = hidden_dim * 3  # text + audio + visual
        self.fusion_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task heads
        self.classification_head = nn.Linear(hidden_dim, num_classes)
        self.regression_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                audio_features: torch.Tensor, audio_mask: torch.Tensor,
                visual_features: torch.Tensor, visual_mask: torch.Tensor,
                audio_lengths: Optional[torch.Tensor] = None,
                visual_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Encode each modality
        text_features = self.text_encoder(input_ids, attention_mask)  # [B, L, H]
        audio_features = self.audio_encoder(audio_features, audio_lengths)  # [B, L, H]
        visual_features = self.visual_encoder(visual_features, visual_lengths)  # [B, L, H]
        
        # Global pooling (average over sequence length)
        text_pooled = self.text_pooler(text_features.transpose(1, 2)).squeeze(-1)  # [B, H]
        audio_pooled = self.audio_pooler(audio_features.transpose(1, 2)).squeeze(-1)  # [B, H]
        visual_pooled = self.visual_pooler(visual_features.transpose(1, 2)).squeeze(-1)  # [B, H]
        
        # Late fusion: concatenate
        fused_features = torch.cat([text_pooled, audio_pooled, visual_pooled], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # Multi-task prediction
        classification_logits = self.classification_head(fused_features)
        regression_output = self.regression_head(fused_features)
        
        return {
            'classification_logits': classification_logits,
            'regression_output': regression_output,
            'cls_features': fused_features
        }


class MAGBERT(nn.Module):
    """MAG-BERT baseline with gating mechanism."""
    
    def __init__(self, text_model_name: str = "bert-base-uncased",
                 audio_input_dim: int = 74, visual_input_dim: int = 35,
                 hidden_dim: int = 768, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Modality encoders
        self.audio_encoder = AudioEncoder(audio_input_dim, hidden_dim, dropout=dropout)
        self.visual_encoder = VisualEncoder(visual_input_dim, hidden_dim, dropout=dropout)
        
        # Global pooling
        self.audio_pooler = nn.AdaptiveAvgPool1d(1)
        self.visual_pooler = nn.AdaptiveAvgPool1d(1)
        
        # Gating mechanism
        self.audio_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.visual_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # BERT with gated multimodal input
        self.bert = BertModel.from_pretrained(text_model_name)
        
        # Multi-task heads
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                audio_features: torch.Tensor, audio_mask: torch.Tensor,
                visual_features: torch.Tensor, visual_mask: torch.Tensor,
                audio_lengths: Optional[torch.Tensor] = None,
                visual_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Encode audio and visual
        audio_features = self.audio_encoder(audio_features, audio_lengths)
        visual_features = self.visual_encoder(visual_features, visual_lengths)
        
        # Global pooling
        audio_pooled = self.audio_pooler(audio_features.transpose(1, 2)).squeeze(-1)
        visual_pooled = self.visual_pooler(visual_features.transpose(1, 2)).squeeze(-1)
        
        # Gating mechanism
        audio_gate = self.audio_gate(audio_pooled)
        visual_gate = self.visual_gate(visual_pooled)
        
        # Apply gates
        gated_audio = audio_pooled * audio_gate
        gated_visual = visual_pooled * visual_gate
        
        # Combine gated features (this is a simplified version)
        # In practice, MAG-BERT has more complex integration
        multimodal_features = gated_audio + gated_visual
        
        # BERT encoding (text only, multimodal features used for initialization)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Add multimodal influence (simplified)
        pooled_output = pooled_output + 0.1 * multimodal_features
        
        # Multi-task prediction
        classification_logits = self.classification_head(pooled_output)
        regression_output = self.regression_head(pooled_output)
        
        return {
            'classification_logits': classification_logits,
            'regression_output': regression_output,
            'cls_features': pooled_output
        }


class MulT(nn.Module):
    """MulT baseline with cross-modal transformers."""
    
    def __init__(self, text_model_name: str = "bert-base-uncased",
                 audio_input_dim: int = 74, visual_input_dim: int = 35,
                 hidden_dim: int = 768, num_heads: int = 12, num_layers: int = 4,
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Modality encoders
        self.text_encoder = TextEncoder(text_model_name, hidden_dim, dropout=dropout)
        self.audio_encoder = AudioEncoder(audio_input_dim, hidden_dim, dropout=dropout)
        self.visual_encoder = VisualEncoder(visual_input_dim, hidden_dim, dropout=dropout)
        
        # Cross-modal transformers (MulT uses separate transformers for each pair)
        self.text_audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.text_visual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim),  # text + audio + visual
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task heads
        self.classification_head = nn.Linear(hidden_dim, num_classes)
        self.regression_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                audio_features: torch.Tensor, audio_mask: torch.Tensor,
                visual_features: torch.Tensor, visual_mask: torch.Tensor,
                audio_lengths: Optional[torch.Tensor] = None,
                visual_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Encode each modality
        text_features = self.text_encoder(input_ids, attention_mask)
        audio_features = self.audio_encoder(audio_features, audio_lengths)
        visual_features = self.visual_encoder(visual_features, visual_lengths)
        
        # Cross-modal attention (MulT style)
        # Text-Audio cross-modal attention
        text_audio_combined = torch.cat([text_features, audio_features], dim=1)
        text_audio_mask = torch.cat([attention_mask, audio_mask], dim=1)
        text_audio_fused = self.text_audio_transformer(
            text_audio_combined, 
            src_key_padding_mask=~text_audio_mask
        )
        
        # Text-Visual cross-modal attention
        text_visual_combined = torch.cat([text_features, visual_features], dim=1)
        text_visual_mask = torch.cat([attention_mask, visual_mask], dim=1)
        text_visual_fused = self.text_visual_transformer(
            text_visual_combined,
            src_key_padding_mask=~text_visual_mask
        )
        
        # Extract text portions from fused features
        text_len = text_features.size(1)
        text_audio_text = text_audio_fused[:, :text_len, :]
        text_visual_text = text_visual_fused[:, :text_len, :]
        
        # Global pooling
        text_pooled = text_features.mean(dim=1)  # [B, H]
        text_audio_pooled = text_audio_text.mean(dim=1)  # [B, H]
        text_visual_pooled = text_visual_text.mean(dim=1)  # [B, H]
        
        # Fusion
        fused_features = torch.cat([text_pooled, text_audio_pooled, text_visual_pooled], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # Multi-task prediction
        classification_logits = self.classification_head(fused_features)
        regression_output = self.regression_head(fused_features)
        
        return {
            'classification_logits': classification_logits,
            'regression_output': regression_output,
            'cls_features': fused_features
        }


class EarlyFusionMAFT(nn.Module):
    """MAFT with early fusion (fuse at first layer)."""
    
    def __init__(self, text_model_name: str = "bert-base-uncased",
                 hidden_dim: int = 768, num_heads: int = 12, num_layers: int = 1,
                 audio_input_dim: int = 74, visual_input_dim: int = 35,
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Modality encoders
        self.text_encoder = TextEncoder(text_model_name, hidden_dim, dropout=dropout)
        self.audio_encoder = AudioEncoder(audio_input_dim, hidden_dim, dropout=dropout)
        self.visual_encoder = VisualEncoder(visual_input_dim, hidden_dim, dropout=dropout)
        
        # Early fusion transformer (fuse immediately)
        from .fusion import FusionTransformer
        self.fusion_transformer = FusionTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Multi-task heads
        from .maft import MultiTaskHead
        self.multi_task_head = MultiTaskHead(hidden_dim, num_classes, dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                audio_features: torch.Tensor, audio_mask: torch.Tensor,
                visual_features: torch.Tensor, visual_mask: torch.Tensor,
                audio_lengths: Optional[torch.Tensor] = None,
                visual_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size = input_ids.size(0)
        
        # Encode each modality
        text_features = self.text_encoder(input_ids, attention_mask)
        audio_features = self.audio_encoder(audio_features, audio_lengths)
        visual_features = self.visual_encoder(visual_features, visual_lengths)
        
        # Add CLS token to text
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        text_features = torch.cat([cls_tokens, text_features], dim=1)
        attention_mask = torch.cat([torch.ones(batch_size, 1, device=input_ids.device), attention_mask], dim=1)
        
        # Early fusion
        fused_features = self.fusion_transformer(
            text_features, audio_features, visual_features,
            attention_mask, audio_mask, visual_mask
        )
        
        # Extract CLS token
        cls_features = fused_features[:, 0, :]
        
        # Multi-task prediction
        classification_logits, regression_output = self.multi_task_head(cls_features)
        
        return {
            'classification_logits': classification_logits,
            'regression_output': regression_output,
            'cls_features': cls_features,
            'fused_features': fused_features
        }


class LateFusionMAFT(nn.Module):
    """MAFT with late fusion (fuse at last layer)."""
    
    def __init__(self, text_model_name: str = "bert-base-uncased",
                 hidden_dim: int = 768, num_heads: int = 12, num_layers: int = 1,
                 audio_input_dim: int = 74, visual_input_dim: int = 35,
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Modality encoders
        self.text_encoder = TextEncoder(text_model_name, hidden_dim, dropout=dropout)
        self.audio_encoder = AudioEncoder(audio_input_dim, hidden_dim, dropout=dropout)
        self.visual_encoder = VisualEncoder(visual_input_dim, hidden_dim, dropout=dropout)
        
        # Separate transformers for each modality
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.visual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Late fusion transformer
        from .fusion import FusionTransformer
        self.late_fusion = FusionTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=1,  # Single layer for late fusion
            dropout=dropout
        )
        
        # Multi-task heads
        from .maft import MultiTaskHead
        self.multi_task_head = MultiTaskHead(hidden_dim, num_classes, dropout)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                audio_features: torch.Tensor, audio_mask: torch.Tensor,
                visual_features: torch.Tensor, visual_mask: torch.Tensor,
                audio_lengths: Optional[torch.Tensor] = None,
                visual_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size = input_ids.size(0)
        
        # Encode each modality
        text_features = self.text_encoder(input_ids, attention_mask)
        audio_features = self.audio_encoder(audio_features, audio_lengths)
        visual_features = self.visual_encoder(visual_features, visual_lengths)
        
        # Add CLS token to text
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        text_features = torch.cat([cls_tokens, text_features], dim=1)
        attention_mask = torch.cat([torch.ones(batch_size, 1, device=input_ids.device), attention_mask], dim=1)
        
        # Separate modality processing
        text_features = self.text_transformer(text_features, src_key_padding_mask=~attention_mask)
        audio_features = self.audio_transformer(audio_features, src_key_padding_mask=~audio_mask)
        visual_features = self.visual_transformer(visual_features, src_key_padding_mask=~visual_mask)
        
        # Late fusion
        fused_features = self.late_fusion(
            text_features, audio_features, visual_features,
            attention_mask, audio_mask, visual_mask
        )
        
        # Extract CLS token
        cls_features = fused_features[:, 0, :]
        
        # Multi-task prediction
        classification_logits, regression_output = self.multi_task_head(cls_features)
        
        return {
            'classification_logits': classification_logits,
            'regression_output': regression_output,
            'cls_features': cls_features,
            'fused_features': fused_features
        } 