#!/usr/bin/env python3
"""MOSEI Data Loader for MAFT - FIXED VERSION"""

import torch
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class MOSEIDataset(Dataset):
    """MOSEI dataset for MAFT - properly handles GloVe embeddings"""
    
    def __init__(self, data_dir, split='train', max_text_len=50, 
                 max_audio_len=500, max_visual_len=500):
        self.data_dir = Path(data_dir) / split
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.max_visual_len = max_visual_len
        
        # Load samples
        with open(self.data_dir / 'samples.pkl', 'rb') as f:
            self.samples = pickle.load(f)
        
        print(f"Loaded {len(self.samples)} {split} samples")
        
        # Fit scalers on training data for normalization
        if split == 'train':
            self.audio_scaler = self._fit_scaler('audio_features')
            self.visual_scaler = self._fit_scaler('visual_features')
        else:
            # Load pre-fitted scalers (you'll need to save/load these)
            self.audio_scaler = None
            self.visual_scaler = None
    
    def _fit_scaler(self, feature_key):
        """Fit StandardScaler on all features"""
        all_features = []
        for sample in self.samples[:1000]:  # Use first 1000 for efficiency
            features = np.array(sample[feature_key])
            all_features.append(features)
        
        # Concatenate and fit
        all_features = np.vstack(all_features)
        scaler = StandardScaler()
        scaler.fit(all_features)
        return scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # === TEXT: GloVe embeddings (300-dim) ===
        text_feat = np.array(sample['text_features'][:self.max_text_len])
        text_len = len(text_feat)
        
        # Pad text features to max length
        if text_len < self.max_text_len:
            padding = np.zeros((self.max_text_len - text_len, 300))
            text_feat = np.vstack([text_feat, padding])
        
        text_feat = torch.FloatTensor(text_feat)  # [max_text_len, 300]
        
        # === AUDIO: COVAREP features (74-dim) ===
        audio_feat = np.array(sample['audio_features'][:self.max_audio_len])
        audio_len = len(audio_feat)
        
        # Normalize audio features
        if self.audio_scaler is not None:
            audio_feat = self.audio_scaler.transform(audio_feat)
        
        # Pad audio features
        if audio_len < self.max_audio_len:
            padding = np.zeros((self.max_audio_len - audio_len, 74))
            audio_feat = np.vstack([audio_feat, padding])
        
        audio_feat = torch.FloatTensor(audio_feat)  # [max_audio_len, 74]
        
        # === VISUAL: FACET features (35-dim) ===
        visual_feat = np.array(sample['visual_features'][:self.max_visual_len])
        visual_len = len(visual_feat)
        
        # Normalize visual features
        if self.visual_scaler is not None:
            visual_feat = self.visual_scaler.transform(visual_feat)
        
        # Pad visual features
        if visual_len < self.max_visual_len:
            padding = np.zeros((self.max_visual_len - visual_len, 35))
            visual_feat = np.vstack([visual_feat, padding])
        
        visual_feat = torch.FloatTensor(visual_feat)  # [max_visual_len, 35]
        
        # === MASKS: 1 = valid, 0 = padding ===
        text_mask = torch.zeros(self.max_text_len, dtype=torch.long)
        text_mask[:text_len] = 1
        
        audio_mask = torch.zeros(self.max_audio_len, dtype=torch.long)
        audio_mask[:audio_len] = 1
        
        visual_mask = torch.zeros(self.max_visual_len, dtype=torch.long)
        visual_mask[:visual_len] = 1
        
        # === TARGETS ===
        sentiment_label = int(sample['sentiment_label'])
        sentiment_score = float(sample['sentiment_score'])
        
        return {
            # Text as pre-computed embeddings (MAFT expects 'text' key, not 'input_ids')
            'text': text_feat,  # [max_text_len, 300]
            'attention_mask': text_mask,  # [max_text_len]
            
            # Audio features
            'audio': audio_feat,  # [max_audio_len, 74]
            'audio_mask': audio_mask,  # [max_audio_len]
            
            # Visual features  
            'visual': visual_feat,  # [max_visual_len, 35]
            'visual_mask': visual_mask,  # [max_visual_len]
            
            # Targets (match train.py expectations)
            'classification_targets': torch.LongTensor([sentiment_label]).squeeze(),
            'regression_targets': torch.FloatTensor([sentiment_score]).squeeze()
        }


def get_mosei_loaders(data_dir, batch_size=16, max_text_len=50, 
                      max_audio_len=500, max_visual_len=500, num_workers=0):
    """Create MOSEI data loaders with proper normalization"""
    
    train_dataset = MOSEIDataset(data_dir, 'train', max_text_len, 
                                  max_audio_len, max_visual_len)
    valid_dataset = MOSEIDataset(data_dir, 'valid', max_text_len, 
                                  max_audio_len, max_visual_len)
    test_dataset = MOSEIDataset(data_dir, 'test', max_text_len, 
                                 max_audio_len, max_visual_len)
    
    # Copy scalers from train to val/test
    valid_dataset.audio_scaler = train_dataset.audio_scaler
    valid_dataset.visual_scaler = train_dataset.visual_scaler
    test_dataset.audio_scaler = train_dataset.audio_scaler
    test_dataset.visual_scaler = train_dataset.visual_scaler
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader