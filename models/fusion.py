import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ModalityEmbedding(nn.Module):
    """Learnable modality embeddings to distinguish between text, audio, and visual tokens."""
    
    def __init__(self, num_modalities: int = 3, hidden_dim: int = 768):
        super().__init__()
        self.modality_embeddings = nn.Embedding(num_modalities, hidden_dim)
        self.hidden_dim = hidden_dim
        
    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modality_ids: [batch_size, seq_len] - 0 for text, 1 for audio, 2 for visual
        Returns:
            modality_embeddings: [batch_size, seq_len, hidden_dim]
        """
        return self.modality_embeddings(modality_ids)


class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformer."""
    
    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, hidden_dim]
        Returns:
            x + positional_encoding: [seq_len, batch_size, hidden_dim]
        """
        return x + self.pe[:x.size(0), :]


class FusionTransformer(nn.Module):
    """Unified fusion transformer with cross-modal attention."""
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12, num_layers: int = 1,
                 dropout: float = 0.1, return_attention: bool = False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.return_attention = return_attention
        
        # Modality embeddings
        self.modality_embedding = nn.Embedding(3, hidden_dim)  # 0: text, 1: audio, 2: visual
        
        # Positional encoding
        self.pos_encoding = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=1
        )
        
        # Main transformer
        self.transformer = nn.TransformerEncoder(
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
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_features: torch.Tensor, audio_features: torch.Tensor, 
                visual_features: torch.Tensor, text_mask: torch.Tensor,
                audio_mask: torch.Tensor, visual_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: [batch_size, text_len, hidden_dim]
            audio_features: [batch_size, audio_len, hidden_dim]
            visual_features: [batch_size, visual_len, hidden_dim]
            text_mask: [batch_size, text_len]
            audio_mask: [batch_size, audio_len]
            visual_mask: [batch_size, visual_len]
        Returns:
            fused_features: [batch_size, total_len, hidden_dim]
            attention_weights: [batch_size, num_heads, total_len, total_len] (optional)
        """
        batch_size = text_features.size(0)
        text_len = text_features.size(1)
        audio_len = audio_features.size(1)
        visual_len = visual_features.size(1)
        total_len = text_len + audio_len + visual_len
        
        # Create modality IDs: 0 for text, 1 for audio, 2 for visual
        text_ids = torch.zeros(batch_size, text_len, dtype=torch.long, device=text_features.device)
        audio_ids = torch.ones(batch_size, audio_len, dtype=torch.long, device=audio_features.device)
        visual_ids = torch.full((batch_size, visual_len), 2, dtype=torch.long, device=visual_features.device)
        
        # Concatenate features and modality IDs
        fused_features = torch.cat([text_features, audio_features, visual_features], dim=1)
        modality_ids = torch.cat([text_ids, audio_ids, visual_ids], dim=1)
        
        # Add modality embeddings
        modality_emb = self.modality_embedding(modality_ids)
        fused_features = fused_features + modality_emb
        
        # Add positional encoding
        fused_features = fused_features.transpose(0, 1)  # [total_len, batch_size, hidden_dim]
        fused_features = self.pos_encoding(fused_features)
        fused_features = fused_features.transpose(0, 1)
        
        # Create attention mask
        attention_mask = torch.cat([text_mask, audio_mask, visual_mask], dim=1)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, total_len]
        attention_mask = (1.0 - attention_mask) * -10000.0  # Convert to large negative values
        
        # Apply transformer with attention extraction
        fused_features = self.dropout(fused_features)
        
        if self.return_attention:
            # Extract attention weights from transformer layers
            attention_weights = []
            x = fused_features
            
            for layer in self.transformer.layers:
                # Get self-attention weights
                attn_output, attn_weights = layer.self_attn(
                    x, x, x, 
                    attn_mask=None,
                    key_padding_mask=~attention_mask.squeeze(1).squeeze(1),
                    need_weights=True
                )
                attention_weights.append(attn_weights)
                
                # Apply layer norm and feedforward
                x = layer.norm1(x + layer.dropout1(attn_output))
                ff_output = layer.linear2(layer.dropout2(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout3(ff_output))
            
            fused_features = x
            # Average attention weights across layers
            attention_weights = torch.stack(attention_weights, dim=1).mean(dim=1)
        else:
            fused_features = self.transformer(fused_features, src_key_padding_mask=~attention_mask.squeeze(1).squeeze(1))
            attention_weights = None
        
        fused_features = self.layer_norm(fused_features)
        
        if self.return_attention and attention_weights is not None:
            return fused_features, attention_weights
        else:
            return fused_features


class ModalityDropout(nn.Module):
    """Randomly drops modalities during training for robustness."""
    
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        
    def forward(self, text_features: torch.Tensor, audio_features: torch.Tensor,
                visual_features: torch.Tensor, text_mask: torch.Tensor,
                audio_mask: torch.Tensor, visual_mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            text_features, audio_features, visual_features: modality features
            text_mask, audio_mask, visual_mask: modality masks
        Returns:
            Tuple of (features, masks) with some modalities potentially zeroed out
        """
        if self.training and self.dropout_rate > 0:
            # Randomly drop modalities
            if torch.rand(1) < self.dropout_rate:
                modality_to_drop = torch.randint(0, 3, (1,)).item()
                
                if modality_to_drop == 0:  # Drop text
                    text_features = torch.zeros_like(text_features)
                    text_mask = torch.zeros_like(text_mask)
                elif modality_to_drop == 1:  # Drop audio
                    audio_features = torch.zeros_like(audio_features)
                    audio_mask = torch.zeros_like(audio_mask)
                else:  # Drop visual
                    visual_features = torch.zeros_like(visual_features)
                    visual_mask = torch.zeros_like(visual_mask)
        
        return text_features, audio_features, visual_features, text_mask, audio_mask, visual_mask 