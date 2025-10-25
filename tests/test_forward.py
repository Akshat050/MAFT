import torch
from models.maft import MAFT
from tests.synthetic_loader import make_batch


def test_forward():
    model = MAFT(
        hidden_dim=256,
        num_heads=4,
        num_layers=1,
        audio_input_dim=74,
        visual_input_dim=35,
        num_classes=3,
        dropout=0.1,
        modality_dropout_rate=0.1,
    )
    batch = make_batch(C=3)
    out = model(batch)
    assert out["logits"].shape == (2, 3)
    assert out["reg"].shape == (2, 1)
