#!/usr/bin/env python3
"""
MPS Compatibility Fix for models/fusion.py

This script automatically adds enable_nested_tensor=False to TransformerEncoder
to fix the MPS error on Apple Silicon.

Usage:
    python fix_mps_compatibility.py
"""

import sys
from pathlib import Path


def fix_fusion_file():
    """Apply MPS compatibility fix to models/fusion.py"""
    
    fusion_path = Path("models/fusion.py")
    
    if not fusion_path.exists():
        print("❌ Error: models/fusion.py not found")
        print("   Make sure you're running this from the MAFT directory")
        return False
    
    print("="*70)
    print("MPS COMPATIBILITY FIX")
    print("="*70)
    
    # Read file
    with open(fusion_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if "enable_nested_tensor=False" in content:
        print("✅ Fix already applied!")
        return True
    
    # Create backup
    backup_path = fusion_path.parent / f"{fusion_path.stem}.backup{fusion_path.suffix}"
    print(f"\n📋 Creating backup: {backup_path}")
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # Apply fix
    print("🔧 Applying fix...")
    
    # Find and replace the TransformerEncoder line
    old_line = "self.encoder = nn.TransformerEncoder(layer, num_layers)"
    new_lines = """self.encoder = nn.TransformerEncoder(
            layer,
            num_layers,
            enable_nested_tensor=False  # Disable for MPS compatibility
        )"""
    
    if old_line in content:
        fixed_content = content.replace(old_line, new_lines)
        
        # Write fixed file
        with open(fusion_path, 'w') as f:
            f.write(fixed_content)
        
        print("✅ Fix applied successfully!")
        print("\nChanged:")
        print("  FROM: self.encoder = nn.TransformerEncoder(layer, num_layers)")
        print("  TO:   self.encoder = nn.TransformerEncoder(")
        print("            layer,")
        print("            num_layers,")
        print("            enable_nested_tensor=False  # Disable for MPS compatibility")
        print("        )")
        
        return True
    else:
        print("⚠️  Could not find expected code pattern")
        print("   You may need to apply the fix manually")
        return False


def verify_fix():
    """Verify the fix was applied correctly"""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    try:
        # Try importing the module
        import sys
        sys.path.insert(0, '.')
        from models.fusion import MAFTFusion
        
        print("✅ Module imports successfully")
        
        # Check if TransformerEncoder was created correctly
        import torch
        import torch.nn as nn
        
        test_fusion = MAFTFusion(hidden_dim=128, num_heads=4, num_layers=2)
        print("✅ MAFTFusion instantiates correctly")
        
        # Check enable_nested_tensor setting
        if hasattr(test_fusion.encoder, 'enable_nested_tensor'):
            if not test_fusion.encoder.enable_nested_tensor:
                print("✅ enable_nested_tensor=False is set correctly")
            else:
                print("⚠️  enable_nested_tensor is still True")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    print("\n")
    
    # Apply fix
    if not fix_fusion_file():
        print("\n❌ Fix failed!")
        sys.exit(1)
    
    # Verify
    if verify_fix():
        print("\n" + "="*70)
        print("✅ SUCCESS! MPS compatibility fix applied and verified")
        print("="*70)
        print("\nYou can now run training with MPS:")
        print("  python train_improved.py --config configs/cpu_test_config_improved.yaml --device mps")
        print("\nOr use fallback mode:")
        print("  export PYTORCH_ENABLE_MPS_FALLBACK=1")
        print("  python train_improved.py --config configs/m4_pro_config_improved.yaml --device mps")
        print("="*70 + "\n")
    else:
        print("\n⚠️  Fix applied but verification failed")
        print("   Try running training to see if it works")


if __name__ == '__main__':
    main()