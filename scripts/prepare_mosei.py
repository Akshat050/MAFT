#!/usr/bin/env python3
"""
CMU-MOSEI Dataset Preparation Script

This script downloads and preprocesses the CMU-MOSEI dataset for MAFT training.
Uses the CMU Multimodal SDK to get the actual data.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse
import subprocess
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import normalize_features


def install_cmu_sdk():
    """Install CMU Multimodal SDK if not already installed."""
    try:
        import cmumosei
        print("✅ CMU Multimodal SDK already installed")
        return True
    except ImportError:
        print("📦 Installing CMU Multimodal SDK...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/A2Zadeh/CMU-MultimodalSDK.git"
            ])
            print("✅ CMU Multimodal SDK installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install CMU Multimodal SDK")
            print("📝 Please install manually: pip install git+https://github.com/A2Zadeh/CMU-MultimodalSDK.git")
            return False


def download_mosei_data(output_dir: str, use_mock: bool = False):
    """
    Download CMU-MOSEI dataset.
    
    Args:
        output_dir: Directory to save processed data
        use_mock: If True, create mock data instead of downloading real data
    """
    print("📥 Downloading CMU-MOSEI dataset...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if use_mock:
        print("🎭 Using mock data for demonstration...")
        create_mock_mosei_data(output_path)
        return
    
    # Try to install and use CMU SDK
    if not install_cmu_sdk():
        print("⚠️  Falling back to mock data...")
        create_mock_mosei_data(output_path)
        return
    
    try:
        # Try importing the original CMU SDK (legacy)
        try:
            from cmumosei import CMUMOSEI
            sdk_type = 'cmumosei'
        except ImportError:
            # Try importing the official mmsdk
            from mmsdk import mmdatasdk
            sdk_type = 'mmsdk'

        print("🔍 Loading CMU-MOSEI dataset...")

        # Initialize dataset
        if sdk_type == 'cmumosei':
            mosei = CMUMOSEI()
            # Process each split
            splits = ['train', 'val', 'test']
            for split in splits:
                print(f"\n📊 Processing {split} split...")
                process_mosei_split(mosei, split, output_path)
        else:
            # mmsdk usage: download and align data
            # This is a simplified example; adapt as needed for your use case
            dataset_path = str(output_path)
            mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel, dataset_path)
            # You may want to add alignment and further processing here
            print("✅ Downloaded CMU-MOSEI using mmsdk. Please check and align features as needed.")
            # Optionally, you can call your own process_mosei_split here if you want to convert to your format

        # Create dataset info
        create_dataset_info(output_path)

        print(f"\n✅ MOSEI data processed and saved to {output_path}")

    except Exception as e:
        print(f"❌ Error loading CMU-MOSEI: {e}")
        if not use_mock:
            raise RuntimeError("Failed to download or process real CMU-MOSEI data. Please check your SDK installation and internet connection.") from e
        print("⚠️  Falling back to mock data...")
        create_mock_mosei_data(output_path)


def process_mosei_split(mosei, split: str, output_path: Path):
    """Process a single split of the MOSEI dataset."""
    
    split_dir = output_path / split
    split_dir.mkdir(exist_ok=True)
    
    # Get data for this split
    if split == 'train':
        data = mosei.train()
    elif split == 'val':
        data = mosei.val()
    else:  # test
        data = mosei.test()
    
    print(f"📈 Processing {len(data)} samples...")
    
    processed_samples = []
    
    for i, sample in enumerate(tqdm(data, desc=f"Processing {split}")):
        try:
            processed_sample = process_mosei_sample(sample, i, split)
            processed_samples.append(processed_sample)
            
            # Save individual sample
            sample_path = split_dir / f"sample_{i:06d}.json"
            with open(sample_path, 'w') as f:
                json.dump(processed_sample, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Skipping sample {i} due to error: {e}")
            continue
    
    # Save all samples as pickle for faster loading
    with open(split_dir / 'samples.pkl', 'wb') as f:
        pickle.dump(processed_samples, f)
    
    # Create metadata
    metadata = {
        'split': split,
        'num_samples': len(processed_samples),
        'features': {
            'text_max_length': 512,
            'audio_dim': 74,  # COVAREP features
            'visual_dim': 35,  # FACET features
            'audio_max_length': 1000,
            'visual_max_length': 1000
        }
    }
    
    with open(split_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ {split} split: {len(processed_samples)} samples processed")


def process_mosei_sample(sample, sample_id: int, split: str) -> dict:
    """Process a single MOSEI sample."""
    
    # Extract text (transcript)
    text = sample.get('text', '')
    if not text:
        text = "No transcript available"
    
    # Extract audio features (COVAREP)
    if 'covarep' not in sample:
        raise KeyError(f"Sample {sample_id} in split '{split}' is missing 'covarep' audio features. Please check your data.")
    audio_features = sample.get('covarep', None)
    if audio_features is None:
        # Create mock audio features if not available
        audio_length = np.random.randint(50, 200)
        audio_features = np.random.randn(audio_length, 74)
    else:
        audio_features = np.array(audio_features)
    
    # Extract visual features (FACET)
    if 'facet' not in sample:
        raise KeyError(f"Sample {sample_id} in split '{split}' is missing 'facet' visual features. Please check your data.")
    visual_features = sample.get('facet', None)
    if visual_features is None:
        # Create mock visual features if not available
        visual_length = np.random.randint(50, 200)
        visual_features = np.random.randn(visual_length, 35)
    else:
        visual_features = np.array(visual_features)
    
    # Normalize features
    audio_features, _ = normalize_features(audio_features)
    visual_features, _ = normalize_features(visual_features)
    
    # Extract labels
    sentiment_label = sample.get('sentiment', 0)  # Binary sentiment
    sentiment_score = sample.get('sentiment_score', 0.0)  # Continuous sentiment
    
    # Ensure proper types
    if isinstance(sentiment_label, (list, np.ndarray)):
        sentiment_label = sentiment_label[0] if len(sentiment_label) > 0 else 0
    if isinstance(sentiment_score, (list, np.ndarray)):
        sentiment_score = sentiment_score[0] if len(sentiment_score) > 0 else 0.0
    
    # Convert to binary if needed
    if sentiment_score > 0:
        sentiment_label = 1
    else:
        sentiment_label = 0
    
    processed_sample = {
        'id': f'mosei_{split}_{sample_id:06d}',
        'text': text,
        'text_length': len(text.split()),
        'audio_features': audio_features.tolist(),
        'audio_length': audio_features.shape[0],
        'visual_features': visual_features.tolist(),
        'visual_length': visual_features.shape[0],
        'sentiment_label': int(sentiment_label),
        'sentiment_score': float(sentiment_score)
    }
    
    return processed_sample


def create_dataset_info(output_path: Path):
    """Create dataset information file."""
    
    info = {
        'dataset_name': 'CMU-MOSEI',
        'description': 'CMU Multimodal Opinion Sentiment, Emotions and Intensity Dataset',
        'paper': 'https://arxiv.org/abs/1806.00562',
        'features': {
            'text': 'BERT tokenized transcripts',
            'audio': 'COVAREP features (74-dimensional)',
            'visual': 'FACET features (35-dimensional)'
        },
        'labels': {
            'sentiment_label': 'Binary sentiment (0: negative, 1: positive)',
            'sentiment_score': 'Continuous sentiment score (-3 to 3)'
        },
        'splits': ['train', 'val', 'test'],
        'processing_date': pd.Timestamp.now().isoformat()
    }
    
    with open(output_path / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)


def create_mock_mosei_data(output_path: Path):
    """Create mock MOSEI data for demonstration purposes."""
    
    # Create splits
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        # Number of samples per split
        if split == 'train':
            num_samples = 1000
        elif split == 'val':
            num_samples = 200
        else:  # test
            num_samples = 300
        
        print(f"Creating {split} split with {num_samples} samples...")
        
        processed_samples = []
        
        for i in tqdm(range(num_samples), desc=f"Creating {split} data"):
            # Generate mock sample
            sample = generate_mock_mosei_sample(i, split)
            processed_samples.append(sample)
            
            # Save sample
            sample_path = split_dir / f"sample_{i:06d}.json"
            with open(sample_path, 'w') as f:
                json.dump(sample, f, indent=2)
        
        # Save all samples as pickle for faster loading
        with open(split_dir / 'samples.pkl', 'wb') as f:
            pickle.dump(processed_samples, f)
        
        # Create metadata file
        metadata = {
            'split': split,
            'num_samples': num_samples,
            'features': {
                'text_max_length': 512,
                'audio_dim': 74,
                'visual_dim': 35,
                'audio_max_length': 1000,
                'visual_max_length': 1000
            }
        }
        
        with open(split_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


def generate_mock_mosei_sample(sample_id: int, split: str) -> dict:
    """Generate a mock MOSEI sample."""
    
    # Mock text (in practice, this would be actual transcripts)
    mock_texts = [
        "I really enjoyed this movie, it was fantastic!",
        "This product is terrible, I hate it.",
        "The service was okay, nothing special.",
        "I love this restaurant, the food is amazing!",
        "This is the worst experience I've ever had.",
        "The performance was incredible, I'm impressed.",
        "I'm disappointed with the quality.",
        "This is exactly what I was looking for!",
        "The customer service was helpful and friendly.",
        "I'm not sure how I feel about this."
    ]
    
    text = np.random.choice(mock_texts)
    
    # Generate sequence lengths
    text_length = np.random.randint(10, 50)
    audio_length = np.random.randint(50, 200)
    visual_length = np.random.randint(50, 200)
    
    # Generate features
    audio_features = np.random.randn(audio_length, 74)  # Mock COVAREP features
    visual_features = np.random.randn(visual_length, 35)  # Mock FACET features
    
    # Normalize features
    audio_features, _ = normalize_features(audio_features)
    visual_features, _ = normalize_features(visual_features)
    
    # Generate labels
    # Binary sentiment (0: negative, 1: positive)
    sentiment_label = np.random.randint(0, 2)
    
    # Continuous sentiment score (-3 to 3)
    if sentiment_label == 0:
        sentiment_score = np.random.uniform(-3, 0)
    else:
        sentiment_score = np.random.uniform(0, 3)
    
    sample = {
        'id': f'mosei_{split}_{sample_id:06d}',
        'text': text,
        'text_length': text_length,
        'audio_features': audio_features.tolist(),
        'audio_length': audio_length,
        'visual_features': visual_features.tolist(),
        'visual_length': visual_length,
        'sentiment_label': int(sentiment_label),
        'sentiment_score': float(sentiment_score)
    }
    
    return sample


def main():
    parser = argparse.ArgumentParser(description='Prepare CMU-MOSEI dataset')
    parser.add_argument('--output_dir', type=str, default='data/mosei',
                       help='Output directory for processed data')
    parser.add_argument('--use_mock', action='store_true',
                       help='Use mock data instead of downloading real data')
    args = parser.parse_args()
    
    # Download and prepare data
    download_mosei_data(args.output_dir, args.use_mock)
    
    print("\n📋 Dataset preparation completed!")
    print(f"📁 Data saved to: {args.output_dir}")
    print("\n📝 Next steps:")
    print("1. Run training: python train.py --config configs/mosei_config.yaml")
    print("2. Run evaluation: python evaluate.py --config configs/mosei_config.yaml")
    print("3. Run attention analysis: python scripts/analyze_attention.py --config configs/mosei_config.yaml")


if __name__ == '__main__':
    main() 