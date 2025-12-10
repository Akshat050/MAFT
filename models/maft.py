import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .encoders import TextEncoder, AudioEncoder, VisualEncoder
from .fusion import FusionTransformer, ModalityDropout


class MultiTaskHead(nn.Module):
    """Multi-task prediction head for classification and regression."""
    
    def __init__(self, hidden_dim: int = 768, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.reg = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.log_temp = nn.Parameter(torch.zeros(1))  # for calibration

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.cls(z) / torch.exp(self.log_temp)
        reg = self.reg(z)
        return {"logits": logits, "reg": reg}


def _mean_pool(feats: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Compute mean pooling over valid (non-padded) tokens.
    
    Args:
        feats: [B, L, H] Feature tensor
        pad: [B, L] Boolean mask where True indicates padding
    
    Returns:
        [B, H] Mean-pooled features
    """
    # feats [B, L, H], pad [B, L] True for pad
    mask = (~pad).unsqueeze(-1).float()
    summed = (feats * mask).sum(1)
    denom = mask.sum(1).clamp_min(1.0)
    return summed / denom


class MAFT(nn.Module):
    """
    Multimodal Attention Fusion Transformer (MAFT).

    A unified transformer architecture for robust multimodal sentiment and behavior
    analysis. MAFT processes text, audio, and visual inputs through modality-specific
    encoders, then fuses them using a shared transformer with modality-aware embeddings.

    Key features:
        - Unified cross-modal self-attention (simpler than pairwise attention)
        - Scheduled modality dropout for robustness training
        - Bottleneck token aggregation for efficiency
        - Symmetric KL consistency loss between modality-specific predictions

    Args:
        text_model_name: Name of pretrained text model (default: "bert-base-uncased")
        hidden_dim: Hidden dimension size (default: 768)
        num_heads: Number of attention heads (default: 12)
        num_layers: Number of transformer layers (default: 2)
        audio_input_dim: Input dimension of audio features (default: 74)
        visual_input_dim: Input dimension of visual features (default: 35)
        num_classes: Number of classification classes (default: 2)
        dropout: Dropout probability (default: 0.1)
        modality_dropout_rate: Probability of dropping entire modalities (default: 0.1)
        freeze_bert: Whether to freeze BERT parameters (default: False)

    Architecture:
        Input → Unimodal Encoders → Modality Dropout → Fusion Transformer 
        → Bottleneck Aggregation → Task Heads

    Example:
        >>> model = MAFT(hidden_dim=768, num_layers=2, num_classes=2)
        >>> outputs = model(batch)
        >>> loss, parts = compute_loss(outputs, batch, lambdas)
    """
    
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 2,
        audio_input_dim: int = 74,
        visual_input_dim: int = 35,
        num_classes: int = 2,
        dropout: float = 0.1,
        modality_dropout_rate: float = 0.1,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.text_enc = TextEncoder(text_model_name, hidden_dim, freeze_bert, dropout)
        self.audio_enc = AudioEncoder(audio_input_dim, hidden_dim, 2, dropout, True)
        self.visual_enc = VisualEncoder(visual_input_dim, hidden_dim, 2, dropout, True)

        # Projection layer for pre-computed text embeddings (e.g., GloVe 300d -> hidden_dim)
        self.text_proj = nn.Linear(300, hidden_dim)  # For CMU-MOSEI GloVe embeddings

        self.fusion = FusionTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_bottlenecks=8,
        )

        self.moddrop = ModalityDropout(modality_dropout_rate)
        self.head = MultiTaskHead(hidden_dim, num_classes, dropout)

        # modality-specific heads for consistency
        self.cls_t = nn.Linear(hidden_dim, num_classes)
        self.cls_a = nn.Linear(hidden_dim, num_classes)
        self.cls_v = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MAFT model.

        Args:
            batch: Dictionary containing:
                - input_ids: [B, L] Token IDs for text (when using BERT encoder)
                - attention_mask: [B, L] Attention mask (1=valid, 0=padding)
                - text: [B, L, H] Pre-computed text embeddings (optional, for CMU-MOSEI)
                - audio: [B, La, D_a] Audio features
                - audio_mask: [B, La] Audio mask (True=padding)
                - visual: [B, Lv, D_v] Visual features
                - visual_mask: [B, Lv] Visual mask (True=padding)

        Returns:
            Dictionary containing:
                - logits: [B, C] Main classification logits
                - reg: [B, 1] Regression predictions
                - logits_text: [B, C] Text-only auxiliary logits
                - logits_audio: [B, C] Audio-only auxiliary logits
                - logits_visual: [B, C] Visual-only auxiliary logits

        Note:
            During training, modality dropout randomly zeros entire modalities
            with probability modality_dropout_rate (scheduled from 0.1 to 0.35).
            
            The model supports two input modes:
            1. BERT tokenizer mode: Uses input_ids via text_enc
            2. Pre-computed embeddings: Uses 'text' directly (e.g., GloVe from CMU-MOSEI)
        """
        # Handle text input - support both BERT tokens and pre-computed embeddings
        if "text" in batch and batch["text"].dim() == 3:
            # Pre-computed embeddings (e.g., GloVe from CMU-MOSEI)
            T = self.text_proj(batch["text"])  # [B, Lt, 300] -> [B, Lt, H]
            T_pad = batch["attention_mask"] == 0  # Convert to padding mask (True=padding)
        else:
            # BERT tokenizer path (standard)
            T, T_pad = self.text_enc(batch["input_ids"], batch["attention_mask"])  # [B, Lt, H], [B, Lt] True=PAD
        
        A, A_pad = self.audio_enc(batch["audio"], batch["audio_mask"])  # [B, La, H], [B, La]
        V, V_pad = self.visual_enc(batch["visual"], batch["visual_mask"])  # [B, Lv, H], [B, Lv]

        # modality dropout
        T, A, V, T_pad, A_pad, V_pad, drops = self.moddrop(T, A, V, T_pad, A_pad, V_pad)

        # fusion
        _, Z = self.fusion(T, A, V, T_pad, A_pad, V_pad)

        # main heads
        out = self.head(Z)  # dict with logits and reg

        # modality-specific quick logits for consistency
        out["logits_text"] = self.cls_t(_mean_pool(T, T_pad))
        out["logits_audio"] = self.cls_a(_mean_pool(A, A_pad))
        out["logits_visual"] = self.cls_v(_mean_pool(V, V_pad))

        return out