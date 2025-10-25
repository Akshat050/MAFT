import torch
import torch.nn as nn
from typing import Tuple


class TextEncoder(nn.Module):
    """Minimal text encoder without HuggingFace dependency"""
    
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        hidden_dim: int = 768,
        freeze_bert: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(30522, hidden_dim, padding_idx=0)  # vocab size
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [B, L] token ids
            attention_mask: [B, L] 1 for valid tokens, 0 for padding
        
        Returns:
            features: [B, L, H] encoded features
            pad_mask: [B, L] True for padding tokens
        """
        x = self.embedding(input_ids)  # [B, L, H]
        x = self.dropout(x)
        x = self.proj(x)  # [B, L, H]
        
        pad_mask = attention_mask == 0  # True for padding
        return x, pad_mask


class AudioEncoder(nn.Module):
    """Minimal audio encoder"""
    
    def __init__(
        self,
        input_dim: int = 74,
        hidden_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, audio: torch.Tensor, audio_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio: [B, L, D] audio features
            audio_mask: [B, L] 1 for valid frames, 0 for padding
        
        Returns:
            features: [B, L, H] encoded features
            pad_mask: [B, L] True for padding frames
        """
        x = self.proj(audio)  # [B, L, H]
        x = self.dropout(x)
        
        # Run LSTM without packing (simpler, more reliable)
        x, _ = self.lstm(x)  # [B, L, H]
        
        pad_mask = audio_mask == 0  # True for padding
        return x, pad_mask


class VisualEncoder(nn.Module):
    """Minimal visual encoder"""
    
    def __init__(
        self,
        input_dim: int = 35,
        hidden_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, visual: torch.Tensor, visual_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual: [B, L, D] visual features
            visual_mask: [B, L] 1 for valid frames, 0 for padding
        
        Returns:
            features: [B, L, H] encoded features
            pad_mask: [B, L] True for padding frames
        """
        x = self.proj(visual)  # [B, L, H]
        x = self.dropout(x)
        
        # Run LSTM without packing (simpler, more reliable)
        x, _ = self.lstm(x)  # [B, L, H]
        
        pad_mask = visual_mask == 0  # True for padding
        return x, pad_mask
