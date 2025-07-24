#!/usr/bin/env python3
"""
Interview Dataset Preparation Script

This script prepares interview data for MAFT training.
In practice, you would use actual job interview data with audio/video/transcripts.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import normalize_features


def prepare_interview_data(output_dir: str):
    """
    Prepare interview dataset.
    
    Note: This is a placeholder function. In practice, you would:
    1. Collect real job interview data with audio/video recordings
    2. Extract transcripts from audio
    3. Extract audio features (MFCC, prosody, etc.)
    4. Extract visual features (facial expressions, head pose, etc.)
    5. Create hire/no-hire labels and performance scores
    """
    print("📥 Preparing Interview dataset...")
    print("⚠️  Note: This is a placeholder. Please use actual interview data.")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # For demonstration, we'll create mock data
    # In practice, replace this with actual interview data processing
    create_mock_interview_data(output_path)
    
    print(f"✅ Mock interview data created in {output_path}")


def create_mock_interview_data(output_path: Path):
    """Create mock interview data for demonstration purposes."""
    
    # Create splits
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        # Number of samples per split
        if split == 'train':
            num_samples = 500
        elif split == 'val':
            num_samples = 100
        else:  # test
            num_samples = 150
        
        print(f"Creating {split} split with {num_samples} samples...")
        
        for i in tqdm(range(num_samples), desc=f"Creating {split} data"):
            # Generate mock sample
            sample = generate_mock_interview_sample(i, split)
            
            # Save sample
            sample_path = split_dir / f"sample_{i:06d}.json"
            with open(sample_path, 'w') as f:
                json.dump(sample, f, indent=2)
        
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


def generate_mock_interview_sample(sample_id: int, split: str) -> dict:
    """Generate a mock interview sample."""
    
    # Mock interview responses
    mock_responses = [
        "I have five years of experience in software development, primarily working with Python and machine learning.",
        "In my previous role, I led a team of three developers and successfully delivered a project ahead of schedule.",
        "I'm passionate about solving complex problems and I enjoy collaborating with cross-functional teams.",
        "I believe in continuous learning and I'm always looking to improve my technical skills.",
        "I have experience with both frontend and backend development, and I'm comfortable with full-stack projects.",
        "I'm detail-oriented and I always ensure code quality through proper testing and documentation.",
        "I enjoy mentoring junior developers and sharing knowledge with the team.",
        "I'm comfortable working in agile environments and I adapt quickly to changing requirements.",
        "I have strong communication skills and I can explain technical concepts to non-technical stakeholders.",
        "I'm excited about this opportunity and I believe I can contribute significantly to your team."
    ]
    
    response = np.random.choice(mock_responses)
    
    # Generate sequence lengths
    text_length = np.random.randint(15, 60)
    audio_length = np.random.randint(80, 250)
    visual_length = np.random.randint(80, 250)
    
    # Generate features
    audio_features = np.random.randn(audio_length, 74)  # Mock audio features
    visual_features = np.random.randn(visual_length, 35)  # Mock visual features
    
    # Normalize features
    audio_features, _ = normalize_features(audio_features)
    visual_features, _ = normalize_features(visual_features)
    
    # Generate labels
    # Binary hire decision (0: no hire, 1: hire)
    hire_label = np.random.randint(0, 2)
    
    # Performance score (0-10)
    if hire_label == 0:
        performance_score = np.random.uniform(0, 5)  # Lower scores for no-hire
    else:
        performance_score = np.random.uniform(5, 10)  # Higher scores for hire
    
    sample = {
        'id': f'interview_{split}_{sample_id:06d}',
        'text': response,
        'text_length': text_length,
        'audio_features': audio_features.tolist(),
        'audio_length': audio_length,
        'visual_features': visual_features.tolist(),
        'visual_length': visual_length,
        'hire_label': int(hire_label),
        'performance_score': float(performance_score)
    }
    
    return sample


def main():
    parser = argparse.ArgumentParser(description='Prepare Interview dataset')
    parser.add_argument('--output_dir', type=str, default='data/interview',
                       help='Output directory for processed data')
    args = parser.parse_args()
    
    # Prepare data
    prepare_interview_data(args.output_dir)
    
    print("\n📋 Dataset preparation completed!")
    print(f"📁 Data saved to: {args.output_dir}")
    print("\n📝 Next steps:")
    print("1. Replace mock data with actual interview data")
    print("2. Ensure proper alignment between audio/video/transcript")
    print("3. Run training: python train.py --config configs/interview_config.yaml")


if __name__ == '__main__':
    main() 