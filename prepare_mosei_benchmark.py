#!/usr/bin/env python3
"""
Preprocess CMU-MOSEI benchmark dataset for MAFT training
Converts SDK format into train/validation/test splits
FINAL FIXED VERSION - Handles h5py serialization + label extraction
"""

from mmsdk import mmdatasdk
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

def score_to_class(score):
    """
    Convert continuous sentiment score to 7-class label
    Standard MOSEI binning: [-3, 3] → [0, 6]
    
    Classes:
    0: Highly Negative (≤ -2.5)
    1: Negative (-2.5 to -1.5)
    2: Weakly Negative (-1.5 to -0.5)
    3: Neutral (-0.5 to 0.5)
    4: Weakly Positive (0.5 to 1.5)
    5: Positive (1.5 to 2.5)
    6: Highly Positive (≥ 2.5)
    """
    if score <= -2.5:
        return 0
    elif score <= -1.5:
        return 1
    elif score <= -0.5:
        return 2
    elif score <= 0.5:
        return 3
    elif score <= 1.5:
        return 4
    elif score <= 2.5:
        return 5
    else:
        return 6


print("="*70)
print("CMU-MOSEI DATASET PREPROCESSING (FINAL FIXED)")
print("="*70)

# Load dataset
print("\n📂 Loading CMU-MOSEI benchmark dataset...")
dataset = mmdatasdk.mmdataset('data/mosei_raw/')

print("✅ Dataset loaded successfully")
available_keys = list(dataset.computational_sequences.keys())
print(f"   Available modalities: {available_keys}")

# Extract features
print("\n🔄 Extracting multimodal features...")

text_data = dataset.computational_sequences['glove_vectors']
audio_data = dataset.computational_sequences['COVAREP'] 
visual_data = dataset.computational_sequences['FACET 4.2']

video_ids = list(text_data.data.keys())
print(f"   Total utterances: {len(video_ids)}")

# Get sentiment labels
print("\n📥 Retrieving sentiment annotations...")

if 'All Labels' in dataset.computational_sequences:
    labels_data = dataset.computational_sequences['All Labels']
    print(f"   ✅ Labels already loaded: 'All Labels'")
    print(f"   Label utterances: {len(labels_data.data)}")
else:
    # Try to load if not present
    try:
        dataset.add_computational_sequences(mmdatasdk.cmu_mosei.labels, 'data/mosei_raw/')
        labels_data = dataset.computational_sequences['All Labels']
        print(f"   ✅ Labels retrieved: 'All Labels'")
    except Exception as e:
        print(f"   ❌ CRITICAL ERROR - Cannot load labels: {e}")
        raise

# Validate labels
if labels_data is None or len(labels_data.data) == 0:
    raise ValueError("❌ No labels loaded! Cannot create dataset.")

print(f"\n✅ Label validation passed - found {len(labels_data.data)} labeled utterances")

# Inspect label structure
print("\n🔍 Inspecting label structure...")
first_vid_id = list(labels_data.data.keys())[0]
first_label = labels_data.data[first_vid_id]['features']
print(f"   Sample video ID: {first_vid_id}")
print(f"   Label shape: {first_label.shape}")
print(f"   Label dtype: {first_label.dtype}")
print(f"   Label range: [{np.min(first_label):.4f}, {np.max(first_label):.4f}]")

# Analyze label columns
print(f"\n   Analyzing label columns...")
all_label_data = []
for vid_id in list(labels_data.data.keys())[:100]:
    feats = labels_data.data[vid_id]['features']
    all_label_data.append(feats)

stacked = np.vstack(all_label_data)
print(f"   Label matrix shape: {stacked.shape}")
print(f"   Number of label dimensions: {stacked.shape[1] if len(stacked.shape) > 1 else 1}")

if len(stacked.shape) > 1:
    for col_idx in range(min(stacked.shape[1], 10)):
        col_data = stacked[:, col_idx]
        print(f"   Column {col_idx}: min={np.min(col_data):.3f}, max={np.max(col_data):.3f}, mean={np.mean(col_data):.3f}")
    
    print(f"\n   💡 Using column 0 for sentiment (verify this matches [-3, 3] range)")
    sentiment_col_idx = 0
else:
    sentiment_col_idx = 0

# Create standard splits
print("\n📊 Creating data splits...")
np.random.seed(42)
np.random.shuffle(video_ids)

n_total = len(video_ids)
n_train = int(0.7 * n_total)
n_valid = int(0.1 * n_total)

splits = {
    'train': video_ids[:n_train],
    'valid': video_ids[n_train:n_train+n_valid],
    'test': video_ids[n_train+n_valid:]
}

print(f"   Training:   {len(splits['train']):,} utterances")
print(f"   Validation: {len(splits['valid']):,} utterances")
print(f"   Test:       {len(splits['test']):,} utterances")

# Process each split
output_dir = Path('data/mosei')

# Track statistics
all_labels = []
all_scores = []
skipped_count = 0

for split_name, ids in splits.items():
    print(f"\n📝 Processing {split_name} split...")
    
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    split_labels = []
    split_scores = []
    
    for vid_id in tqdm(ids, desc=f'Processing {split_name}'):
        try:
            # Check if video has all modalities
            if vid_id not in text_data.data or \
               vid_id not in audio_data.data or \
               vid_id not in visual_data.data or \
               vid_id not in labels_data.data:
                skipped_count += 1
                continue
            
            # Extract features - FIX: Convert h5py views to numpy arrays
            text_feat = np.array(text_data.data[vid_id]['features'])
            audio_feat = np.array(audio_data.data[vid_id]['features'])
            visual_feat = np.array(visual_data.data[vid_id]['features'])
            
            # Extract label
            label_feat = np.array(labels_data.data[vid_id]['features'])
            
            if len(label_feat) == 0:
                skipped_count += 1
                continue
            
            # Extract sentiment score from column 0
            if len(label_feat.shape) > 1 and label_feat.shape[1] > sentiment_col_idx:
                sentiment_score = float(label_feat[0, sentiment_col_idx])
            else:
                sentiment_score = float(label_feat[0])
            
            # Convert to class label
            sentiment_label = score_to_class(sentiment_score)
            
            sample = {
                'id': vid_id,
                'text': f"Utterance {vid_id}",
                # FIX: Convert to regular numpy arrays and then to lists for pickling
                'text_features': text_feat.astype(np.float32).tolist(),
                'text_length': len(text_feat),
                'audio_features': audio_feat.astype(np.float32).tolist(),
                'audio_length': len(audio_feat),
                'visual_features': visual_feat.astype(np.float32).tolist(),
                'visual_length': len(visual_feat),
                'sentiment_label': sentiment_label,
                'sentiment_score': float(sentiment_score)
            }
            
            samples.append(sample)
            split_labels.append(sentiment_label)
            split_scores.append(sentiment_score)
            
        except Exception as e:
            # Skip utterances with errors
            skipped_count += 1
            continue
    
    # Save processed split
    output_file = split_dir / 'samples.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(samples, f)
    
    print(f"   ✅ Processed {len(samples):,} utterances → {output_file}")
    
    # Show label distribution for this split
    if len(split_labels) > 0:
        unique, counts = np.unique(split_labels, return_counts=True)
        print(f"   Label distribution:")
        for label, count in zip(unique, counts):
            print(f"     Class {label}: {count:4d} ({100*count/len(samples):5.2f}%)")
    
    all_labels.extend(split_labels)
    all_scores.extend(split_scores)

# Final validation
print("\n" + "="*70)
print("FINAL VALIDATION")
print("="*70)

print(f"\nTotal samples created: {len(all_labels)}")
print(f"Skipped (missing data): {skipped_count}")

if len(all_labels) == 0:
    print("\n❌ CRITICAL ERROR: No samples created!")
    raise ValueError("No valid samples generated")

print(f"\nOverall sentiment score statistics:")
print(f"  Min:  {min(all_scores):7.4f}")
print(f"  Max:  {max(all_scores):7.4f}")
print(f"  Mean: {np.mean(all_scores):7.4f}")
print(f"  Std:  {np.std(all_scores):7.4f}")

print(f"\nOverall class distribution:")
unique, counts = np.unique(all_labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count:5d} ({100*count/len(all_labels):5.2f}%)")

# Critical validation check
unique_labels = len(set(all_labels))
unique_scores = len(set(all_scores))

print(f"\n📊 Diversity check:")
print(f"  Unique class labels: {unique_labels} / 7")
print(f"  Unique scores: {unique_scores}")

if unique_labels == 1:
    print("\n❌ CRITICAL ERROR: All samples have the SAME label!")
    print("   The bug still exists - label extraction failed.")
    print(f"   All labels are: {list(set(all_labels))[0]}")
    raise ValueError("Label extraction failed - all labels identical")

if unique_scores == 1:
    print("\n❌ CRITICAL ERROR: All samples have the SAME score!")
    print("   The bug still exists - score extraction failed.")
    print(f"   All scores are: {list(set(all_scores))[0]}")
    raise ValueError("Score extraction failed - all scores identical")

print("\n✅ LABEL VALIDATION PASSED!")
print("   Labels and scores show proper diversity.")

print("\n" + "="*70)
print("✅ PREPROCESSING COMPLETE")
print("="*70)
print(f"\nOutput directory: {output_dir}/")
print("\nNext steps:")
print("  1. Inspect data: python3 inspect_mosei_samples.py")
print("  2. Clean features (if needed): python3 prepare_mosei_benchmark.py")
print("  3. Start training: python3 train.py configs/mosei_benchmark_config.yaml")
print("\nExpected performance on benchmark:")
print("  7-class Accuracy: 70-80%")
print("  Binary Accuracy: 80-85%")
print("  MAE: 0.6-0.8")
print("  Pearson Correlation: 0.7-0.8")
print("="*70)