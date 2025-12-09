"""
Data Augmentation for Multimodal Training

Implements various augmentation strategies to improve model robustness:
- Temporal masking (randomly mask portions of sequences)
- Feature noise (add Gaussian noise to audio/visual features)
- Modality dropout (randomly drop entire modality during training)
"""

import torch
import torch.nn.functional as F


class MultimodalAugmentation:
    """
    Data augmentation for multimodal sentiment analysis.
    
    Args:
        temporal_mask_prob: Probability of masking temporal regions (default: 0.1)
        feature_noise_std: Standard deviation of Gaussian noise (default: 0.05)
        modality_dropout_prob: Probability of dropping entire modality (default: 0.0)
    """
    
    def __init__(
        self,
        temporal_mask_prob=0.1,
        feature_noise_std=0.05,
        modality_dropout_prob=0.0
    ):
        self.temporal_mask_prob = temporal_mask_prob
        self.feature_noise_std = feature_noise_std
        self.modality_dropout_prob = modality_dropout_prob
    
    def __call__(self, batch, training=True):
        """
        Apply augmentation to a batch.
        
        Args:
            batch: Dictionary containing 'text', 'audio', 'visual' tensors
            training: If False, no augmentation is applied
        
        Returns:
            Augmented batch dictionary
        """
        if not training:
            return batch
        
        # Apply each augmentation with some probability
        batch = self._temporal_masking(batch)
        batch = self._feature_noise(batch)
        batch = self._modality_dropout(batch)
        
        return batch
    
    def _temporal_masking(self, batch):
        """
        Randomly mask temporal regions in sequences.
        Similar to SpecAugment but for all modalities.
        """
        for key in ['text', 'audio', 'visual']:
            if key not in batch or batch[key].dim() != 3:
                continue
            
            B, L, D = batch[key].shape
            mask_len = max(1, int(L * self.temporal_mask_prob))
            
            # Apply masking per sample in batch
            for i in range(B):
                if torch.rand(1).item() < 0.5:  # 50% chance of masking
                    start = torch.randint(0, L - mask_len + 1, (1,)).item()
                    batch[key][i, start:start+mask_len] = 0
        
        return batch
    
    def _feature_noise(self, batch):
        """
        Add Gaussian noise to audio and visual features.
        Text is not augmented as it uses discrete embeddings.
        """
        for key in ['audio', 'visual']:
            if key not in batch:
                continue
            
            noise = torch.randn_like(batch[key]) * self.feature_noise_std
            batch[key] = batch[key] + noise
        
        return batch
    
    def _modality_dropout(self, batch):
        """
        Randomly drop entire modality during training.
        Forces model to learn robust representations.
        """
        if self.modality_dropout_prob == 0.0:
            return batch
        
        for key in ['text', 'audio', 'visual']:
            if key not in batch:
                continue
            
            if torch.rand(1).item() < self.modality_dropout_prob:
                # Zero out entire modality
                batch[key] = torch.zeros_like(batch[key])
                
                # Also update padding mask if exists
                pad_key = f'{key}_padding_mask'
                if pad_key in batch:
                    batch[pad_key] = torch.ones_like(batch[pad_key], dtype=torch.bool)
        
        return batch


def get_augmentation_from_config(config):
    """
    Create augmentation instance from config dictionary.
    
    Args:
        config: Training config with 'augmentation' section
    
    Returns:
        MultimodalAugmentation instance or None
    """
    if not config.get('training', {}).get('use_augmentation', False):
        return None
    
    aug_config = config.get('training', {}).get('augmentation', {})
    
    return MultimodalAugmentation(
        temporal_mask_prob=aug_config.get('temporal_mask_prob', 0.1),
        feature_noise_std=aug_config.get('feature_noise_std', 0.05),
        modality_dropout_prob=aug_config.get('modality_dropout_prob', 0.0)
    )


# Example usage:
if __name__ == '__main__':
    print("="*70)
    print("MULTIMODAL AUGMENTATION - TEST")
    print("="*70)
    
    # Create dummy batch
    batch = {
        'text': torch.randn(4, 50, 768),      # [B, L, D]
        'audio': torch.randn(4, 500, 74),
        'visual': torch.randn(4, 500, 35),
    }
    
    # Create augmentation
    aug = MultimodalAugmentation(
        temporal_mask_prob=0.15,
        feature_noise_std=0.05,
        modality_dropout_prob=0.1
    )
    
    # Apply augmentation
    print("\nOriginal batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
    
    aug_batch = aug(batch, training=True)
    
    print("\nAugmented batch shapes:")
    for k, v in aug_batch.items():
        print(f"  {k}: {v.shape}")
    
    # Check if augmentation actually changed data
    print("\nAugmentation effects:")
    for k in batch.keys():
        diff = (aug_batch[k] != batch[k]).float().mean()
        print(f"  {k}: {diff*100:.1f}% of values changed")
    
    print("\n✅ Augmentation test complete!")