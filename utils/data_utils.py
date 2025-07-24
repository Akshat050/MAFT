import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import Dict, List, Tuple, Optional
import json
import os
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import librosa


class MultimodalDataset(Dataset):
    """Base dataset class for multimodal data."""
    
    def __init__(self, data_path: str, tokenizer: BertTokenizer, max_length: int = 512,
                 audio_max_length: int = 1000, visual_max_length: int = 1000):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.audio_max_length = audio_max_length
        self.visual_max_length = visual_max_length
        
        # Load data
        self.data = self.load_data()
        
    def load_data(self) -> List[Dict]:
        """Load and preprocess data. Override in subclasses."""
        raise NotImplementedError
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample. Override in subclasses."""
        raise NotImplementedError


class MOSEIDataset(MultimodalDataset):
    """CMU-MOSEI dataset loader."""
    
    def __init__(self, data_path: str, tokenizer: BertTokenizer, split: str = 'train',
                 max_length: int = 512, audio_max_length: int = 1000, visual_max_length: int = 1000):
        self.split = split
        super().__init__(data_path, tokenizer, max_length, audio_max_length, visual_max_length)
        
    def load_data(self) -> List[Dict]:
        """Load CMU-MOSEI data."""
        split_path = Path(self.data_path) / self.split
        
        # Try to load from pickle first (faster)
        pickle_path = split_path / 'samples.pkl'
        if pickle_path.exists():
            print(f"📂 Loading {self.split} data from pickle...")
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Loaded {len(data)} samples from pickle")
            return data
        
        # Fallback to loading individual JSON files
        print(f"📂 Loading {self.split} data from JSON files...")
        data = []
        json_files = list(split_path.glob('sample_*.json'))
        
        for json_file in tqdm(json_files, desc=f"Loading {self.split} samples"):
            try:
                with open(json_file, 'r') as f:
                    sample = json.load(f)
                data.append(sample)
            except Exception as e:
                print(f"⚠️  Error loading {json_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(data)} samples from JSON files")
        return data
        
    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        
        # Tokenize text
        text_encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process audio features
        audio_features = torch.tensor(sample['audio_features'], dtype=torch.float32)
        audio_length = min(sample['audio_length'], self.audio_max_length)
        audio_features = audio_features[:audio_length]
        
        # Pad audio features
        if audio_features.size(0) < self.audio_max_length:
            padding = torch.zeros(self.audio_max_length - audio_features.size(0), audio_features.size(1))
            audio_features = torch.cat([audio_features, padding], dim=0)
        
        audio_mask = torch.ones(self.audio_max_length)
        audio_mask[audio_length:] = 0
        
        # Process visual features
        visual_features = torch.tensor(sample['visual_features'], dtype=torch.float32)
        visual_length = min(sample['visual_length'], self.visual_max_length)
        visual_features = visual_features[:visual_length]
        
        # Pad visual features
        if visual_features.size(0) < self.visual_max_length:
            padding = torch.zeros(self.visual_max_length - visual_features.size(0), visual_features.size(1))
            visual_features = torch.cat([visual_features, padding], dim=0)
        
        visual_mask = torch.ones(self.visual_max_length)
        visual_mask[visual_length:] = 0
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'audio_features': audio_features,
            'audio_mask': audio_mask,
            'audio_length': torch.tensor(audio_length),
            'visual_features': visual_features,
            'visual_mask': visual_mask,
            'visual_length': torch.tensor(visual_length),
            'sentiment_label': torch.tensor(sample['sentiment_label']),
            'sentiment_score': torch.tensor(sample['sentiment_score'], dtype=torch.float32)
        }


class InterviewDataset(MultimodalDataset):
    """MIT Interview dataset loader (or similar job interview data)."""
    
    def __init__(self, data_path: str, tokenizer: BertTokenizer, split: str = 'train',
                 max_length: int = 512, audio_max_length: int = 1000, visual_max_length: int = 1000):
        self.split = split
        super().__init__(data_path, tokenizer, max_length, audio_max_length, visual_max_length)
        
    def load_data(self) -> List[Dict]:
        """Load interview data."""
        # Mock interview dataset - replace with actual data loading
        data = []
        for i in range(500):  # Mock 500 interview samples
            sample = {
                'id': f'interview_{self.split}_{i}',
                'text': f"Interview response sample {i} about candidate's experience and skills",
                'audio_features': np.random.randn(150, 74),  # Mock audio features
                'visual_features': np.random.randn(150, 35),  # Mock visual features
                'hire_label': np.random.randint(0, 2),  # Binary hire decision
                'performance_score': np.random.uniform(0, 10),  # Interview performance score
                'text_length': np.random.randint(15, 60),
                'audio_length': np.random.randint(80, 250),
                'visual_length': np.random.randint(80, 250)
            }
            data.append(sample)
            
        return data
        
    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        
        # Tokenize text
        text_encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process audio features
        audio_features = torch.tensor(sample['audio_features'], dtype=torch.float32)
        audio_length = min(sample['audio_length'], self.audio_max_length)
        audio_features = audio_features[:audio_length]
        
        # Pad audio features
        if audio_features.size(0) < self.audio_max_length:
            padding = torch.zeros(self.audio_max_length - audio_features.size(0), audio_features.size(1))
            audio_features = torch.cat([audio_features, padding], dim=0)
        
        audio_mask = torch.ones(self.audio_max_length)
        audio_mask[audio_length:] = 0
        
        # Process visual features
        visual_features = torch.tensor(sample['visual_features'], dtype=torch.float32)
        visual_length = min(sample['visual_length'], self.visual_max_length)
        visual_features = visual_features[:visual_length]
        
        # Pad visual features
        if visual_features.size(0) < self.visual_max_length:
            padding = torch.zeros(self.visual_max_length - visual_features.size(0), visual_features.size(1))
            visual_features = torch.cat([visual_features, padding], dim=0)
        
        visual_mask = torch.ones(self.visual_max_length)
        visual_mask[visual_length:] = 0
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'audio_features': audio_features,
            'audio_mask': audio_mask,
            'audio_length': torch.tensor(audio_length),
            'visual_features': visual_features,
            'visual_mask': visual_mask,
            'visual_length': torch.tensor(visual_length),
            'hire_label': torch.tensor(sample['hire_label']),
            'performance_score': torch.tensor(sample['performance_score'], dtype=torch.float32)
        }


def create_dataloader(dataset: Dataset, batch_size: int = 8, shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """Create a DataLoader with proper collation."""
    
    def collate_fn(batch):
        """Custom collate function for multimodal data."""
        # Separate different types of data
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        audio_features = torch.stack([item['audio_features'] for item in batch])
        audio_mask = torch.stack([item['audio_mask'] for item in batch])
        audio_lengths = torch.stack([item['audio_length'] for item in batch])
        visual_features = torch.stack([item['visual_features'] for item in batch])
        visual_mask = torch.stack([item['visual_mask'] for item in batch])
        visual_lengths = torch.stack([item['visual_length'] for item in batch])
        
        # Handle different label types
        if 'sentiment_label' in batch[0]:
            classification_targets = torch.stack([item['sentiment_label'] for item in batch])
            regression_targets = torch.stack([item['sentiment_score'] for item in batch])
        elif 'hire_label' in batch[0]:
            classification_targets = torch.stack([item['hire_label'] for item in batch])
            regression_targets = torch.stack([item['performance_score'] for item in batch])
        else:
            raise ValueError("Unknown dataset type")
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'audio_features': audio_features,
            'audio_mask': audio_mask,
            'audio_lengths': audio_lengths,
            'visual_features': visual_features,
            'visual_mask': visual_mask,
            'visual_lengths': visual_lengths,
            'classification_targets': classification_targets,
            'regression_targets': regression_targets
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def normalize_features(features: np.ndarray, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """Normalize features using StandardScaler."""
    if scaler is None:
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
    else:
        features_normalized = scaler.transform(features)
    
    return features_normalized, scaler 