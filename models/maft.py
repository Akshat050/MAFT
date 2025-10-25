import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .encoders import TextEncoder, AudioEncoder, VisualEncoder
from .fusion import FusionTransformer, ModalityDropout
from .quality import QualityEstimator


class MultiTaskHead(nn.Module):
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
    # feats [B, L, H], pad [B, L] True for pad
    mask = (~pad).unsqueeze(-1).float()
    summed = (feats * mask).sum(1)
    denom = mask.sum(1).clamp_min(1.0)
    return summed / denom


class MAFT(nn.Module):
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
        self.text_enc = TextEncoder(text_model_name, hidden_dim, freeze_bert, dropout)
        self.audio_enc = AudioEncoder(audio_input_dim, hidden_dim, 2, dropout, True)
        self.visual_enc = VisualEncoder(visual_input_dim, hidden_dim, 2, dropout, True)

        self.q_text = QualityEstimator(hidden_dim)
        self.q_audio = QualityEstimator(hidden_dim)
        self.q_visual = QualityEstimator(hidden_dim)

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
        T, T_pad = self.text_enc(batch["input_ids"], batch["attention_mask"])  # [B, Lt, H], [B, Lt] True=PAD
        A, A_pad = self.audio_enc(batch["audio"], batch["audio_mask"])  # [B, La, H], [B, La]
        V, V_pad = self.visual_enc(batch["visual"], batch["visual_mask"])  # [B, Lv, H], [B, Lv]

        # quality
        Q_t = self.q_text(T, T_pad)
        Q_a = self.q_audio(A, A_pad)
        Q_v = self.q_visual(V, V_pad)

        # save pre-drop summaries for reconstruction targets
        A_sum = _mean_pool(A, A_pad).detach()
        V_sum = _mean_pool(V, V_pad).detach()

        # modality dropout
        T, A, V, T_pad, A_pad, V_pad, drops = self.moddrop(T, A, V, T_pad, A_pad, V_pad)

        # fusion
        _, Z = self.fusion(T, A, V, T_pad, A_pad, V_pad, q_text=Q_t, q_audio=Q_a, q_visual=Q_v)

        # main heads
        out = self.head(Z)  # dict with logits and reg

        # modality-specific quick logits for consistency
        out["logits_text"] = self.cls_t(_mean_pool(T, T_pad))
        out["logits_audio"] = self.cls_a(_mean_pool(A, A_pad))
        out["logits_visual"] = self.cls_v(_mean_pool(V, V_pad))

        # reconstruction heads: predict A_sum and V_sum from Z only if dropped
        out["rec_audio"] = Z  # project outside in loss for simplicity
        out["rec_visual"] = Z
        out["targets_audio_sum"] = A_sum
        out["targets_visual_sum"] = V_sum
        out["drops"] = drops

        return out
