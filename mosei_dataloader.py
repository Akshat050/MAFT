#!/usr/bin/env python3
"""MOSEI Data Loader for MAFT - Fixed key names"""

import torch
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class MOSEIDataset(Dataset):
    """MOSEI dataset for MAFT"""
    
    def __init__(self, data_dir, split='train', max_text_len=50, max_audio_len=500, max_visual_len=500):
        self.data_dir = Path(data_dir) / split
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.max_visual_len = max_visual_len
        
        with open(self.data_dir / 'samples.pkl', 'rb') as f:
            self.samples = pickle.load(f)
        
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get features
        text_feat = torch.FloatTensor(sample['text_features'][:self.max_text_len])
        audio_feat = torch.FloatTensor(sample['audio_features'][:self.max_audio_len])
        visual_feat = torch.FloatTensor(sample['visual_features'][:self.max_visual_len])
        
        # Get actual lengths
        text_len = len(text_feat)
        audio_len = len(audio_feat)
        visual_len = len(visual_feat)
        
        # Pad to max length
        if text_len < self.max_text_len:
            text_feat = torch.cat([text_feat, torch.zeros(self.max_text_len - text_len, text_feat.shape[1])])
        if audio_len < self.max_audio_len:
            audio_feat = torch.cat([audio_feat, torch.zeros(self.max_audio_len - audio_len, audio_feat.shape[1])])
        if visual_len < self.max_visual_len:
            visual_feat = torch.cat([visual_feat, torch.zeros(self.max_visual_len - visual_len, visual_feat.shape[1])])
        
        # Create padding masks (True = padding)
        text_mask = torch.ones(self.max_text_len, dtype=torch.bool)
        text_mask[:text_len] = False
        
        audio_mask = torch.ones(self.max_audio_len, dtype=torch.bool)
        audio_mask[:audio_len] = False
        
        visual_mask = torch.ones(self.max_visual_len, dtype=torch.bool)
        visual_mask[:visual_len] = False
        
        # Create dummy input_ids for text
        input_ids = torch.arange(text_len).long()
        if len(input_ids) < self.max_text_len:
            input_ids = torch.cat([input_ids, torch.zeros(self.max_text_len - len(input_ids)).long()])
        
        # Attention mask (inverse of padding mask)
        attention_mask = ~text_mask
        
        return {
            # Text modality (model expects these keys)
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            
            # Audio modality (model expects 'audio_mask' NOT 'audio_padding_mask')
            'audio': audio_feat,
            'audio_mask': audio_mask,  # Changed from audio_padding_mask
            
            # Visual modality (model expects 'visual_mask' NOT 'visual_padding_mask')
            'visual': visual_feat,
            'visual_mask': visual_mask,  # Changed from visual_padding_mask
            
            # Targets
            'classification_targets': torch.LongTensor([sample['sentiment_label']]).squeeze(),
            'regression_targets': torch.FloatTensor([sample['sentiment_score']]).squeeze()
        }


def get_mosei_loaders(data_dir, batch_size=32, max_text_len=50, max_audio_len=500, max_visual_len=500):
    """Create MOSEI data loaders"""
    
    train_dataset = MOSEIDataset(data_dir, 'train', max_text_len, max_audio_len, max_visual_len)
    valid_dataset = MOSEIDataset(data_dir, 'valid', max_text_len, max_audio_len, max_visual_len)
    test_dataset = MOSEIDataset(data_dir, 'test', max_text_len, max_audio_len, max_visual_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    print("Testing MOSEI data loader...")
    train_loader, valid_loader, test_loader = get_mosei_loaders('data/mosei', batch_size=16)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    print("\n✅ Data loader working!")
