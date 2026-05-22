#!/usr/bin/env python3
"""
Download CMU-MOSEI using official SDK
Updated for new repo location
"""

try:
    from mmsdk import mmdatasdk
except ImportError:
    print("❌ SDK not installed!")
    print("\nInstall with:")
    print("  pip3 install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git")
    exit(1)

import os

print("="*70)
print("CMU-MOSEI DATASET DOWNLOAD")
print("="*70)
print("\nThis will download ~3GB of data")
print("Time: 30-60 minutes")
print("="*70 + "\n")

os.makedirs('data/mosei_raw', exist_ok=True)

try:
    # Download MOSEI
    print("📥 Downloading CMU-MOSEI...\n")
    
    dataset = mmdatasdk.mmdataset(
        mmdatasdk.cmu_mosei.highlevel, 
        'data/mosei_raw/'
    )
    
    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*70)
    print("\nData location: data/mosei_raw/")
    print("\nNext: python3 prepare_mosei_data.py")
    print("="*70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nIf download fails, you can:")
    print("1. Try again (downloads can resume)")
    print("2. Use mock data: python3 download_mosei_simple.py")
    print("3. Manual download from: http://immortal.multicomp.cs.cmu.edu/")

