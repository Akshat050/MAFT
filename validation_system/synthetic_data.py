"""
Synthetic multimodal data generator for MAFT validation.

Generates realistic synthetic data with controllable cross-modal correlations
for testing the MAFT model before deployment.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import numpy as np


class SyntheticMultimodalDataset(Dataset):
    """
    Synthetic multimodal dataset with controllable cross-modal correlations.
    
    Generates text, audio, and visual features that correlate with a shared
    latent sentiment variable, simulating realistic multimodal data.
    
    Args:
        num_samples: Number of samples in the dataset
        seq_len_text: Maximum sequence length for text (default: 64)
        seq_len_audio: Maximum sequence length for audio (default: 200)
        seq_len_visual: Maximum sequence length for visual (default: 200)
        vocab_size: Vocabulary size for text tokens (default: 30000)
        audio_dim: Dimensionality of audio features (default: 74)
        visual_dim: Dimensionality of visual features (default: 35)
        num_classes: Number of classification classes (default: 2)
        correlation_strength: Strength of cross-modal correlation 0-1 (default: 0.7)
        seed: Random seed for reproducibility (default: 42)
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        seq_len_text: int = 64,
        seq_len_audio: int = 200,
        seq_len_visual: int = 200,
        vocab_size: int = 30000,
        audio_dim: int = 74,
        visual_dim: int = 35,
        num_classes: int = 2,
        correlation_strength: float = 0.7,
        seed: int = 42
    ):
        super().__init__()
        
        self.num_samples = num_samples
        self.seq_len_text = seq_len_text
        self.seq_len_audio = seq_len_audio
        self.seq_len_visual = seq_len_visual
        self.vocab_size = vocab_size
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.num_classes = num_classes
        self.correlation_strength = correlation_strength
        self.seed = seed
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate latent sentiment vectors for all samples
        # This is the shared "ground truth" that all modalities correlate with
        self.latent_sentiment = torch.randn(num_samples, 1)
        
        # Generate classification targets from latent sentiment
        # Positive sentiment -> class 1, negative -> class 0
        self.classification_targets = (self.latent_sentiment.squeeze() > 0).long()
        
        # Generate regression targets (continuous sentiment scores)
        # Normalize to range [-3, 3] like MOSEI
        self.regression_targets = torch.tanh(self.latent_sentiment) * 3.0
        
        # Create projection matrices for generating correlated features
        # These map from latent sentiment to feature spaces
        self.text_projection = nn.Linear(1, vocab_size, bias=False)
        self.audio_projection = nn.Linear(1, audio_dim, bias=False)
        self.visual_projection = nn.Linear(1, visual_dim, bias=False)
        
        # Initialize projections with small weights for stability
        nn.init.normal_(self.text_projection.weight, mean=0, std=0.1)
        nn.init.normal_(self.audio_projection.weight, mean=0, std=0.1)
        nn.init.normal_(self.visual_projection.weight, mean=0, std=0.1)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def _generate_text_features(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text tokens with variable length and correlation to sentiment.
        
        Returns:
            input_ids: [seq_len_text] integer tokens
            attention_mask: [seq_len_text] binary mask (1=valid, 0=padding)
        """
        # Variable sequence length (50-100% of max length)
        actual_len = np.random.randint(
            max(1, self.seq_len_text // 2),
            self.seq_len_text + 1
        )
        
        # Get sentiment for this sample
        sentiment = self.latent_sentiment[idx]
        
        # Generate token distribution correlated with sentiment
        with torch.no_grad():
            token_probs = torch.softmax(self.text_projection(sentiment), dim=-1)
        
        # Mix correlated and random tokens based on correlation_strength
        if self.correlation_strength > 0:
            # Sample from sentiment-correlated distribution
            correlated_tokens = torch.multinomial(
                token_probs.squeeze(),
                num_samples=actual_len,
                replacement=True
            )
        else:
            correlated_tokens = torch.zeros(actual_len, dtype=torch.long)
        
        # Add random noise tokens
        random_tokens = torch.randint(0, self.vocab_size, (actual_len,))
        
        # Blend based on correlation strength
        use_correlated = torch.rand(actual_len) < self.correlation_strength
        tokens = torch.where(use_correlated, correlated_tokens, random_tokens)
        
        # Create padded sequence
        input_ids = torch.zeros(self.seq_len_text, dtype=torch.long)
        input_ids[:actual_len] = tokens
        
        # Create attention mask
        attention_mask = torch.zeros(self.seq_len_text, dtype=torch.long)
        attention_mask[:actual_len] = 1
        
        return input_ids, attention_mask
    
    def _generate_audio_features(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate audio features with temporal structure and correlation to sentiment.
        
        Returns:
            audio: [seq_len_audio, audio_dim] float features
            audio_mask: [seq_len_audio] binary mask (1=valid, 0=padding)
        """
        # Variable sequence length
        actual_len = np.random.randint(
            max(1, self.seq_len_audio // 2),
            self.seq_len_audio + 1
        )
        
        # Get sentiment for this sample
        sentiment = self.latent_sentiment[idx]
        
        # Generate base features correlated with sentiment
        with torch.no_grad():
            base_features = self.audio_projection(sentiment)  # [1, audio_dim]
        
        # Create temporal sequence with smoothing (random walk)
        audio = torch.zeros(self.seq_len_audio, self.audio_dim)
        
        for t in range(actual_len):
            # Mix correlated base features with random noise
            correlated_part = base_features.squeeze(0) * self.correlation_strength
            random_part = torch.randn(self.audio_dim) * (1 - self.correlation_strength)
            
            # Add temporal smoothing (features change gradually)
            if t > 0:
                temporal_smooth = 0.7 * audio[t-1] + 0.3 * (correlated_part + random_part)
                audio[t] = temporal_smooth
            else:
                audio[t] = correlated_part + random_part
        
        # Create mask
        audio_mask = torch.zeros(self.seq_len_audio, dtype=torch.long)
        audio_mask[:actual_len] = 1
        
        return audio, audio_mask
    
    def _generate_visual_features(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate visual features with frame continuity and correlation to sentiment.
        
        Returns:
            visual: [seq_len_visual, visual_dim] float features
            visual_mask: [seq_len_visual] binary mask (1=valid, 0=padding)
        """
        # Variable sequence length
        actual_len = np.random.randint(
            max(1, self.seq_len_visual // 2),
            self.seq_len_visual + 1
        )
        
        # Get sentiment for this sample
        sentiment = self.latent_sentiment[idx]
        
        # Generate base features correlated with sentiment
        with torch.no_grad():
            base_features = self.visual_projection(sentiment)  # [1, visual_dim]
        
        # Create temporal sequence with frame-to-frame continuity
        visual = torch.zeros(self.seq_len_visual, self.visual_dim)
        
        for t in range(actual_len):
            # Mix correlated base features with random noise
            correlated_part = base_features.squeeze(0) * self.correlation_strength
            random_part = torch.randn(self.visual_dim) * (1 - self.correlation_strength)
            
            # Add frame-to-frame continuity (stronger than audio)
            if t > 0:
                frame_smooth = 0.8 * visual[t-1] + 0.2 * (correlated_part + random_part)
                visual[t] = frame_smooth
            else:
                visual[t] = correlated_part + random_part
        
        # Create mask
        visual_mask = torch.zeros(self.seq_len_visual, dtype=torch.long)
        visual_mask[:actual_len] = 1
        
        return visual, visual_mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys matching MAFT expected format:
            - input_ids: [seq_len_text] text tokens
            - attention_mask: [seq_len_text] text mask
            - audio: [seq_len_audio, audio_dim] audio features
            - audio_mask: [seq_len_audio] audio mask
            - visual: [seq_len_visual, visual_dim] visual features
            - visual_mask: [seq_len_visual] visual mask
            - classification_targets: scalar class label
            - regression_targets: scalar regression value
        """
        input_ids, attention_mask = self._generate_text_features(idx)
        audio, audio_mask = self._generate_audio_features(idx)
        visual, visual_mask = self._generate_visual_features(idx)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'audio': audio,
            'audio_mask': audio_mask,
            'visual': visual,
            'visual_mask': visual_mask,
            'classification_targets': self.classification_targets[idx],
            'regression_targets': self.regression_targets[idx].squeeze()
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """Get dataset statistics for validation."""
        # Calculate actual sequence lengths
        text_lengths = []
        audio_lengths = []
        visual_lengths = []
        
        for i in range(min(100, len(self))):  # Sample first 100
            sample = self[i]
            text_lengths.append(sample['attention_mask'].sum().item())
            audio_lengths.append(sample['audio_mask'].sum().item())
            visual_lengths.append(sample['visual_mask'].sum().item())
        
        return {
            'num_samples': self.num_samples,
            'text_length_mean': np.mean(text_lengths),
            'text_length_std': np.std(text_lengths),
            'audio_length_mean': np.mean(audio_lengths),
            'audio_length_std': np.std(audio_lengths),
            'visual_length_mean': np.mean(visual_lengths),
            'visual_length_std': np.std(visual_lengths),
            'num_classes': self.num_classes,
            'class_distribution': {
                i: (self.classification_targets == i).sum().item()
                for i in range(self.num_classes)
            },
            'regression_mean': self.regression_targets.mean().item(),
            'regression_std': self.regression_targets.std().item(),
            'correlation_strength': self.correlation_strength
        }


def get_synthetic_loaders(
    batch_size: int = 8,
    num_train_batches: int = 10,
    num_val_batches: int = 3,
    seq_len_text: int = 64,
    seq_len_audio: int = 200,
    seq_len_visual: int = 200,
    correlation_strength: float = 0.7,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders with synthetic data.
    
    Args:
        batch_size: Batch size for data loaders
        num_train_batches: Number of training batches (samples = batch_size * num_batches)
        num_val_batches: Number of validation batches
        seq_len_text: Maximum text sequence length
        seq_len_audio: Maximum audio sequence length
        seq_len_visual: Maximum visual sequence length
        correlation_strength: Cross-modal correlation strength (0-1)
        num_workers: Number of data loading workers
        seed: Random seed
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    # Calculate number of samples
    num_train_samples = batch_size * num_train_batches
    num_val_samples = batch_size * num_val_batches
    
    # Create datasets
    train_dataset = SyntheticMultimodalDataset(
        num_samples=num_train_samples,
        seq_len_text=seq_len_text,
        seq_len_audio=seq_len_audio,
        seq_len_visual=seq_len_visual,
        correlation_strength=correlation_strength,
        seed=seed
    )
    
    val_dataset = SyntheticMultimodalDataset(
        num_samples=num_val_samples,
        seq_len_text=seq_len_text,
        seq_len_audio=seq_len_audio,
        seq_len_visual=seq_len_visual,
        correlation_strength=correlation_strength,
        seed=seed + 1000  # Different seed for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader
