import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from typing import Optional, Tuple


class TextEncoder(nn.Module):
    """BERT-based text encoder with fine-tuning capability."""
    
    def __init__(self, model_name: str = "bert-base-uncased", hidden_dim: int = 768, 
                 freeze_bert: bool = False, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(768, hidden_dim) if hidden_dim != 768 else nn.Identity()
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            text_features: [batch_size, seq_len, hidden_dim]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state
        text_features = self.dropout(text_features)
        text_features = self.projection(text_features)
        return text_features


class AudioEncoder(nn.Module):
    """BiLSTM-based audio encoder for COVAREP/MFCC features."""
    
    def __init__(self, input_dim: int = 74, hidden_dim: int = 768, num_layers: int = 2, 
                 dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, audio_features: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            audio_features: [batch_size, seq_len, input_dim]
            lengths: [batch_size] - sequence lengths for packing
        Returns:
            audio_features: [batch_size, seq_len, hidden_dim]
        """
        if lengths is not None:
            # Pack sequences for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                audio_features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            audio_features, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            audio_features, _ = self.lstm(audio_features)
        
        audio_features = self.dropout(audio_features)
        audio_features = self.projection(audio_features)
        return audio_features


class VisualEncoder(nn.Module):
    """BiLSTM-based visual encoder for facial action units and head pose."""
    
    def __init__(self, input_dim: int = 35, hidden_dim: int = 768, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, visual_features: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            visual_features: [batch_size, seq_len, input_dim]
            lengths: [batch_size] - sequence lengths for packing
        Returns:
            visual_features: [batch_size, seq_len, hidden_dim]
        """
        if lengths is not None:
            # Pack sequences for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                visual_features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            visual_features, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            visual_features, _ = self.lstm(visual_features)
        
        visual_features = self.dropout(visual_features)
        visual_features = self.projection(visual_features)
        return visual_features 