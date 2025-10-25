import torch


def make_batch(B=2, Lt=8, La=12, Lv=10, Da=74, Dv=35, C=3, device="cpu"):
    return {
        "input_ids": torch.randint(1000, (B, Lt), device=device),
        "attention_mask": torch.ones(B, Lt, dtype=torch.long, device=device),
        "audio": torch.randn(B, La, Da, device=device),
        "audio_mask": torch.ones(B, La, dtype=torch.long, device=device),
        "visual": torch.randn(B, Lv, Dv, device=device),
        "visual_mask": torch.ones(B, Lv, dtype=torch.long, device=device),
        "classification_targets": torch.randint(C, (B,), device=device),
    }
