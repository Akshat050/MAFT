#!/usr/bin/env python3
"""
Comprehensive MOSEI Data Preprocessing & Quality Analysis
For research-grade data preparation
"""

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler


class MOSEIDataCleaner:
    """Analyzes and cleans MOSEI data with detailed statistics"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.stats = defaultdict(dict)
        
    def analyze_split(self, split='train'):
        """Comprehensive analysis of data quality"""
        print(f"\n{'='*70}")
        print(f"ANALYZING {split.upper()} SPLIT")
        print(f"{'='*70}")
        
        split_dir = self.data_dir / split
        with open(split_dir / 'samples.pkl', 'rb') as f:
            samples = pickle.load(f)
        
        print(f"Total samples: {len(samples)}")
        
        # Analyze each modality
        self._analyze_modality(samples, 'text_features', 'Text (GloVe)')
        self._analyze_modality(samples, 'audio_features', 'Audio (COVAREP)')
        self._analyze_modality(samples, 'visual_features', 'Visual (FACET)')
        self._analyze_targets(samples)
        
        return samples
    
    def _analyze_modality(self, samples, key, name):
        """Detailed analysis of a single modality"""
        print(f"\n{name} Features:")
        print("-" * 50)
        
        all_features = []
        nan_count = 0
        inf_count = 0
        zero_sequences = 0
        sequence_lengths = []
        
        for sample in samples:
            feat = np.array(sample[key])
            all_features.append(feat)
            sequence_lengths.append(len(feat))
            
            # Check for problematic values
            if np.any(np.isnan(feat)):
                nan_count += 1
            if np.any(np.isinf(feat)):
                inf_count += 1
            if np.all(feat == 0):
                zero_sequences += 1
        
        # Concatenate for statistics
        all_feat_concat = np.vstack(all_features)
        
        print(f"  Feature dimension: {all_feat_concat.shape[1]}")
        print(f"  Sequence lengths: min={min(sequence_lengths)}, "
              f"max={max(sequence_lengths)}, mean={np.mean(sequence_lengths):.1f}")
        print(f"\n  Data Quality Issues:")
        print(f"    • Sequences with NaN: {nan_count} ({100*nan_count/len(samples):.2f}%)")
        print(f"    • Sequences with Inf: {inf_count} ({100*inf_count/len(samples):.2f}%)")
        print(f"    • All-zero sequences: {zero_sequences} ({100*zero_sequences/len(samples):.2f}%)")
        
        # Count individual NaN/Inf values
        total_values = all_feat_concat.size
        nan_values = np.sum(np.isnan(all_feat_concat))
        inf_values = np.sum(np.isinf(all_feat_concat))
        
        print(f"\n  Individual Value Issues:")
        print(f"    • NaN values: {nan_values:,} / {total_values:,} ({100*nan_values/total_values:.4f}%)")
        print(f"    • Inf values: {inf_values:,} / {total_values:,} ({100*inf_values/total_values:.4f}%)")
        
        # Statistical analysis (on clean data)
        clean_data = all_feat_concat[np.isfinite(all_feat_concat)]
        if len(clean_data) > 0:
            print(f"\n  Statistics (finite values only):")
            print(f"    • Min: {np.min(clean_data):.4f}")
            print(f"    • Max: {np.max(clean_data):.4f}")
            print(f"    • Mean: {np.mean(clean_data):.4f}")
            print(f"    • Std: {np.std(clean_data):.4f}")
            print(f"    • Median: {np.median(clean_data):.4f}")
            
            # Check for outliers
            q1 = np.percentile(clean_data, 25)
            q3 = np.percentile(clean_data, 75)
            iqr = q3 - q1
            outlier_count = np.sum((clean_data < q1 - 3*iqr) | (clean_data > q3 + 3*iqr))
            print(f"    • Outliers (3×IQR): {outlier_count:,} ({100*outlier_count/len(clean_data):.4f}%)")
        
        # Store stats for later use
        self.stats[key] = {
            'nan_count': nan_count,
            'inf_count': inf_count,
            'zero_sequences': zero_sequences,
            'dim': all_feat_concat.shape[1],
            'total_samples': len(samples)
        }
    
    def _analyze_targets(self, samples):
        """Analyze target distribution"""
        print(f"\nTarget Labels:")
        print("-" * 50)
        
        labels = [s['sentiment_label'] for s in samples]
        scores = [s['sentiment_score'] for s in samples]
        
        # Classification label distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Classification (7-class):")
        for label, count in zip(unique, counts):
            print(f"    Class {label}: {count} ({100*count/len(samples):.2f}%)")
        
        # Check for class imbalance
        max_count = max(counts)
        min_count = min(counts)
        print(f"  Imbalance ratio: {max_count/min_count:.2f}:1")
        
        # Regression score distribution
        print(f"\n  Regression scores:")
        print(f"    Min: {min(scores):.4f}")
        print(f"    Max: {max(scores):.4f}")
        print(f"    Mean: {np.mean(scores):.4f}")
        print(f"    Std: {np.std(scores):.4f}")
    
    def clean_and_save(self, samples, split, strategy='robust'):
        """
        Clean data and save preprocessed version
        
        Strategies:
        - 'zero': Replace NaN/Inf with 0 (simple but loses information)
        - 'median': Replace with per-feature median (better for sparse outliers)
        - 'interpolate': Linear interpolation within sequence (temporal coherence)
        - 'robust': Clip extreme values + median imputation (recommended)
        """
        print(f"\n{'='*70}")
        print(f"CLEANING {split.upper()} SPLIT - Strategy: {strategy}")
        print(f"{'='*70}")
        
        cleaned_samples = []
        
        for sample in samples:
            cleaned_sample = sample.copy()
            
            # Clean each modality
            for key in ['text_features', 'audio_features', 'visual_features']:
                feat = np.array(sample[key])
                
                if strategy == 'zero':
                    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                
                elif strategy == 'median':
                    # Replace with per-feature median
                    finite_mask = np.isfinite(feat)
                    for dim in range(feat.shape[1]):
                        col = feat[:, dim]
                        if np.any(finite_mask[:, dim]):
                            median_val = np.median(col[finite_mask[:, dim]])
                            col[~finite_mask[:, dim]] = median_val
                        else:
                            col[:] = 0.0
                        feat[:, dim] = col
                
                elif strategy == 'robust':
                    # Clip extreme values first (3x IQR)
                    finite_mask = np.isfinite(feat)
                    if np.any(finite_mask):
                        q1 = np.percentile(feat[finite_mask], 25)
                        q3 = np.percentile(feat[finite_mask], 75)
                        iqr = q3 - q1
                        lower_bound = q1 - 3 * iqr
                        upper_bound = q3 + 3 * iqr
                        
                        # Clip
                        feat = np.clip(feat, lower_bound, upper_bound)
                    
                    # Then impute remaining NaN with median
                    for dim in range(feat.shape[1]):
                        col = feat[:, dim]
                        finite_in_col = np.isfinite(col)
                        if np.any(finite_in_col):
                            median_val = np.median(col[finite_in_col])
                            col[~finite_in_col] = median_val
                        else:
                            col[:] = 0.0
                        feat[:, dim] = col
                
                elif strategy == 'interpolate':
                    # Linear interpolation along time axis
                    for dim in range(feat.shape[1]):
                        col = feat[:, dim]
                        nans = np.isnan(col) | np.isinf(col)
                        
                        if np.any(nans) and not np.all(nans):
                            # Get valid indices
                            valid_idx = np.where(~nans)[0]
                            if len(valid_idx) > 0:
                                # Interpolate
                                col[nans] = np.interp(
                                    np.where(nans)[0],
                                    valid_idx,
                                    col[valid_idx]
                                )
                        elif np.all(nans):
                            col[:] = 0.0
                        
                        feat[:, dim] = col
                
                cleaned_sample[key] = feat.tolist()
            
            cleaned_samples.append(cleaned_sample)
        
        # Save cleaned data
        output_dir = self.data_dir / f"{split}_cleaned"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / 'samples.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(cleaned_samples, f)
        
        print(f"✅ Saved cleaned data to: {output_path}")
        print(f"   Total samples: {len(cleaned_samples)}")
        
        return cleaned_samples


def main():
    """Run comprehensive data analysis and cleaning"""
    
    data_dir = "data/mosei"
    
    cleaner = MOSEIDataCleaner(data_dir)
    
    # Analyze all splits
    print("\n" + "="*70)
    print("MOSEI DATA QUALITY ANALYSIS")
    print("="*70)
    
    train_samples = cleaner.analyze_split('train')
    valid_samples = cleaner.analyze_split('valid')
    test_samples = cleaner.analyze_split('test')
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    # Check if cleaning is needed
    needs_cleaning = False
    for key in ['text_features', 'audio_features', 'visual_features']:
        if cleaner.stats[key]['nan_count'] > 0 or cleaner.stats[key]['inf_count'] > 0:
            needs_cleaning = True
            break
    
    if needs_cleaning:
        print("\n⚠️  DATA QUALITY ISSUES DETECTED!")
        print("\nRecommended cleaning strategy: 'robust'")
        print("  • Clips extreme outliers (beyond 3×IQR)")
        print("  • Imputes NaN/Inf with per-feature median")
        print("  • Preserves data distribution while removing artifacts")
        
        response = input("\nProceed with cleaning? (yes/no): ").strip().lower()
        
        if response == 'yes':
            print("\nCleaning data...")
            cleaner.clean_and_save(train_samples, 'train', strategy='robust')
            cleaner.clean_and_save(valid_samples, 'valid', strategy='robust')
            cleaner.clean_and_save(test_samples, 'test', strategy='robust')
            
            print("\n✅ All splits cleaned and saved!")
            print("\nNext steps:")
            print("  1. Update mosei_dataloader.py to use cleaned data")
            print("  2. Remove np.nan_to_num() calls (data is now clean)")
            print("  3. Re-run training with clean data")
    else:
        print("\n✅ DATA IS CLEAN - No preprocessing needed!")
        print("   All features are finite and within reasonable ranges.")


if __name__ == '__main__':
    main()