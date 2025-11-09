import torch

from models.maft import MAFT
from train import compute_loss


def test_clean_architecture():
    """Test that cleaned MAFT architecture works correctly."""
    print("Testing cleaned MAFT architecture...")
    
    # 1. Create model
    model = MAFT(
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        audio_input_dim=74,
        visual_input_dim=35,
        num_classes=3,
        dropout=0.1,
        modality_dropout_rate=0.1,
    )
    print("[OK] Model created successfully")
    
    # 2. Create synthetic batch
    batch_size = 4
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, 50)),
        'attention_mask': torch.ones(batch_size, 50),
        'audio': torch.randn(batch_size, 100, 74),
        'audio_mask': torch.ones(batch_size, 100),
        'visual': torch.randn(batch_size, 100, 35),
        'visual_mask': torch.ones(batch_size, 100),
        'classification_targets': torch.randint(0, 3, (batch_size,)),
        'regression_targets': torch.randn(batch_size),
    }
    print("[OK] Batch created")
    
    # 3. Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    print("[OK] Forward pass successful")
    
    # 4. Check output keys (should NOT have reconstruction or drops)
    expected_keys = {'logits', 'reg', 'logits_text', 'logits_audio', 'logits_visual'}
    assert set(outputs.keys()) == expected_keys, f"Wrong output keys: {outputs.keys()}"
    print("[OK] Output keys correct (no reconstruction outputs)")
    
    # 5. Check shapes
    assert outputs['logits'].shape == (batch_size, 3), "Wrong logits shape"
    assert outputs['reg'].shape == (batch_size, 1), "Wrong reg shape"
    print("[OK] Output shapes correct")
    
    # 6. Test backward pass
    model.train()
    outputs = model(batch)
    
    # 7. Compute loss (should NOT have reconstruction loss)
    lambdas = {'reg': 0.5, 'cons': 0.1}
    loss, parts = compute_loss(outputs, batch, lambdas)
    
    # Check loss parts (should NOT have reconstruction_loss)
    expected_parts = {'classification_loss', 'regression_loss', 'consistency_loss'}
    assert set(parts.keys()) == expected_parts, f"Wrong loss parts: {parts.keys()}"
    print("[OK] Loss computation correct (no reconstruction loss)")
    
    # 8. Backward pass
    loss.backward()
    print("[OK] Backward pass successful")
    
    # 9. Check gradients
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad_params.append(name)
    
    assert len(no_grad_params) == 0, f"Parameters without gradients: {no_grad_params}"
    print("[OK] All parameters have gradients")
    
    print("\n" + "="*50)
    print("[SUCCESS] ALL TESTS PASSED - CLEAN ARCHITECTURE VERIFIED")
    print("="*50)


if __name__ == "__main__":
    test_clean_architecture()

