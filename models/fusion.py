import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformers."""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ModalityEmbedding(nn.Module):
    """Learned embeddings to distinguish text/audio/visual tokens."""
    
    def __init__(self, num_modalities: int = 3, hidden_dim: int = 768):
        super().__init__()
        self.emb = nn.Embedding(num_modalities, hidden_dim)

    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(modality_ids)


class ModalityDropout(nn.Module):
    """Randomly drops entire modalities during training for robustness."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, t, a, v, tm, am, vm):
        if not self.training or self.p <= 0.0:
            return t, a, v, tm, am, vm, dict(drop_t=None, drop_a=None, drop_v=None)

        B = t.size(0)
        dt = torch.rand(B, device=t.device) < self.p
        da = torch.rand(B, device=t.device) < self.p
        dv = torch.rand(B, device=t.device) < self.p

        # Clone to avoid in-place operations on tensors that need gradients
        t = t.clone()
        a = a.clone()
        v = v.clone()
        tm = tm.clone()
        am = am.clone()
        vm = vm.clone()

        t[dt] = 0.0
        tm[dt] = True
        a[da] = 0.0
        am[da] = True
        v[dv] = 0.0
        vm[dv] = True

        return t, a, v, tm, am, vm, dict(drop_t=dt, drop_a=da, drop_v=dv)


class FusionTransformer(nn.Module):
    """Unified transformer that fuses text, audio, and visual via self-attention with bottleneck aggregation."""

    def __init__(self, hidden_dim=768, num_heads=12, num_layers=2, dropout=0.1, num_bottlenecks=8):
        super().__init__()
        self.mod_emb = ModalityEmbedding(3, hidden_dim)
        self.pos = SinusoidalPositionalEncoding(hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.bottlenecks = nn.Parameter(torch.randn(1, num_bottlenecks, hidden_dim))

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_bottlenecks),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        text_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        visual_feats: torch.Tensor,
        text_pad: torch.Tensor,
        audio_pad: torch.Tensor,
        visual_pad: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Lt, H = text_feats.shape
        La = audio_feats.size(1)
        Lv = visual_feats.size(1)

        X = torch.cat([text_feats, audio_feats, visual_feats], dim=1)  # [B, Ltot, H]

        mids = torch.cat(
            [
                torch.zeros(B, Lt, dtype=torch.long, device=X.device),
                torch.ones(B, La, dtype=torch.long, device=X.device),
                torch.full((B, Lv), 2, dtype=torch.long, device=X.device),
            ],
            dim=1,
        )
        pad = torch.cat([text_pad, audio_pad, visual_pad], dim=1)  # True for pad

        X = X + self.mod_emb(mids)
        X = self.pos(X)

        bn = self.bottlenecks.expand(B, -1, -1)  # [B, K, H]
        X = torch.cat([bn, X], dim=1)  # [B, K+L, H]
        pad = torch.cat([torch.zeros(B, bn.size(1), dtype=torch.bool, device=pad.device), pad], dim=1)

        Y = self.encoder(X, src_key_padding_mask=pad)  # [B, K+L, H]
        Y_bn, Y_tok = Y[:, : bn.size(1)], Y[:, bn.size(1) :]

        gate_w = self.gate(Y_bn.mean(1))  # [B, K]
        Z = (Y_bn * gate_w.unsqueeze(-1)).sum(1)  # [B, H]

        return Y_tok, Z
