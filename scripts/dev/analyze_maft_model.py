#!/usr/bin/env python3
"""
MAFT Model Analysis - Final Working Version
Compatible with your actual MAFT implementation
"""

import torch
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import yaml
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix

print("="*70)
print("   MAFT MODEL ANALYSIS")
print("="*70)

# === CONFIGURATION ===
MODEL_PATH = "experiments/m4_pro_improved/best_model.pth"
CONFIG_PATH = "configs/mosei_benchmark_config.yaml"
DATA_DIR = "data/mosei"
DEVICE = "mps"

# === LOAD CONFIG ===
print(f"\n📄 Loading config...")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
print(f"✅ Config loaded")

# === LOAD MODEL ===
print(f"\n🤖 Loading model...")
import sys
sys.path.append('.')
from models.maft import MAFT

model_config = config['model']

# Create model with CORRECT parameter names from your MAFT class
model = MAFT(
    text_model_name="bert-base-uncased",
    hidden_dim=model_config['hidden_dim'],
    num_heads=model_config['num_heads'],
    num_layers=model_config['num_layers'],
    audio_input_dim=model_config['audio_input_dim'],
    visual_input_dim=model_config['visual_input_dim'],
    num_classes=model_config['num_classes'],
    dropout=model_config['dropout'],
    modality_dropout_rate=model_config.get('modality_dropout_rate', 0.1),
    freeze_bert=False
)

# Load weights
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"✅ Loaded checkpoint from epoch {epoch}")
else:
    model.load_state_dict(checkpoint)
    print(f"✅ Loaded model weights")

model.to(DEVICE)
model.eval()

# === MODEL ARCHITECTURE ===
print("\n" + "="*70)
print("   1. MODEL ARCHITECTURE")
print("="*70)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 Parameters:")
print(f"   Total: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

print(f"\n⚙️  Configuration:")
print(f"   Hidden dim: {model_config['hidden_dim']}")
print(f"   Num heads: {model_config['num_heads']}")
print(f"   Num layers: {model_config['num_layers']}")
print(f"   Dropout: {model_config['dropout']}")
print(f"   Num classes: {model_config['num_classes']}")
print(f"   Audio input: {model_config['audio_input_dim']}-dim")
print(f"   Visual input: {model_config['visual_input_dim']}-dim")

# === LOAD TEST DATA ===
print("\n" + "="*70)
print("   2. LOADING TEST DATA")
print("="*70)

test_dir = Path(DATA_DIR) / 'test'
with open(test_dir / 'samples.pkl', 'rb') as f:
    test_samples = pickle.load(f)

print(f"\n✅ Loaded {len(test_samples)} test samples")

# Get sequence lengths
max_text_len = config['dataset']['max_length']
max_audio_len = config['dataset']['audio_max_length']
max_visual_len = config['dataset']['visual_max_length']

print(f"\nSequence lengths:")
print(f"   Text: {max_text_len}")
print(f"   Audio: {max_audio_len}")
print(f"   Visual: {max_visual_len}")

# === PREDICTION ANALYSIS ===
print("\n" + "="*70)
print("   3. PREDICTION ANALYSIS")
print("="*70)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(f"\n🔮 Running inference on 200 test samples...")

predictions = []
ground_truth = []
confidences = []

with torch.no_grad():
    for i in tqdm(range(min(200, len(test_samples)))):
        sample = test_samples[i]
        
        # Prepare batch
        text = sample['text']
        encoded = tokenizer(text, max_length=max_text_len, padding='max_length',
                           truncation=True, return_tensors='pt')
        
        # Audio
        audio = torch.FloatTensor(sample['audio_features'][:max_audio_len]).unsqueeze(0)
        if audio.shape[1] < max_audio_len:
            padding = torch.zeros(1, max_audio_len - audio.shape[1], 74)
            audio = torch.cat([audio, padding], dim=1)
        
        # Visual
        visual = torch.FloatTensor(sample['visual_features'][:max_visual_len]).unsqueeze(0)
        if visual.shape[1] < max_visual_len:
            padding = torch.zeros(1, max_visual_len - visual.shape[1], 35)
            visual = torch.cat([visual, padding], dim=1)
        
        # Create batch dictionary (matching MAFT's expected format)
        batch = {
            'input_ids': encoded['input_ids'].to(DEVICE),
            'attention_mask': encoded['attention_mask'].to(DEVICE),
            'audio': audio.to(DEVICE),
            'audio_mask': torch.zeros(1, max_audio_len, dtype=torch.bool).to(DEVICE),  # False = valid
            'visual': visual.to(DEVICE),
            'visual_mask': torch.zeros(1, max_visual_len, dtype=torch.bool).to(DEVICE)  # False = valid
        }
        
        # Forward pass
        outputs = model(batch)
        
        # Get prediction (MAFT returns 'logits' key)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        conf = probs[0, pred].item()
        
        predictions.append(pred)
        ground_truth.append(sample['sentiment_label'])
        confidences.append(conf)

predictions = np.array(predictions)
ground_truth = np.array(ground_truth)
confidences = np.array(confidences)

# Calculate metrics
accuracy = (predictions == ground_truth).mean()

print(f"\n📈 Prediction Results:")
print(f"   Overall Accuracy: {accuracy*100:.2f}%")
print(f"   Average Confidence: {confidences.mean()*100:.2f}%")

# Per-class accuracy
print(f"\n📊 Per-Class Accuracy:")
unique_classes = np.unique(ground_truth)
for cls in unique_classes:
    mask = ground_truth == cls
    if mask.sum() > 0:
        cls_acc = (predictions[mask] == ground_truth[mask]).mean()
        count = mask.sum()
        print(f"   Class {cls}: {cls_acc*100:>6.2f}% ({count:>3d} samples)")

# Confusion matrix
cm = confusion_matrix(ground_truth, predictions)
print(f"\n📊 Confusion Matrix:")
print(f"    Predicted →")
for i in range(cm.shape[0]):
    if i == 0:
        print(f"Actual ↓  {cm[i]}")
    else:
        print(f"          {cm[i]}")

# === MODALITY ABLATION ===
print("\n" + "="*70)
print("   4. MODALITY IMPORTANCE")
print("="*70)

print(f"\n🔬 Testing modality ablation (50 samples)...")

ablation_results = {
    'all': [],
    'text_only': [],
    'audio_only': [],
    'visual_only': []
}

with torch.no_grad():
    for i in tqdm(range(min(50, len(test_samples)))):
        sample = test_samples[i]
        gt = sample['sentiment_label']
        
        # Prepare inputs
        text = sample['text']
        encoded = tokenizer(text, max_length=max_text_len, padding='max_length',
                           truncation=True, return_tensors='pt')
        
        audio = torch.FloatTensor(sample['audio_features'][:max_audio_len]).unsqueeze(0)
        if audio.shape[1] < max_audio_len:
            audio = torch.cat([audio, torch.zeros(1, max_audio_len - audio.shape[1], 74)], dim=1)
        
        visual = torch.FloatTensor(sample['visual_features'][:max_visual_len]).unsqueeze(0)
        if visual.shape[1] < max_visual_len:
            visual = torch.cat([visual, torch.zeros(1, max_visual_len - visual.shape[1], 35)], dim=1)
        
        # All modalities
        batch = {
            'input_ids': encoded['input_ids'].to(DEVICE),
            'attention_mask': encoded['attention_mask'].to(DEVICE),
            'audio': audio.to(DEVICE),
            'audio_mask': torch.zeros(1, max_audio_len, dtype=torch.bool).to(DEVICE),
            'visual': visual.to(DEVICE),
            'visual_mask': torch.zeros(1, max_visual_len, dtype=torch.bool).to(DEVICE)
        }
        outputs = model(batch)
        pred = torch.argmax(outputs['logits'], dim=-1).item()
        ablation_results['all'].append(pred == gt)
        
        # Text only (zero out audio and visual)
        batch['audio'] = torch.zeros_like(audio).to(DEVICE)
        batch['visual'] = torch.zeros_like(visual).to(DEVICE)
        outputs = model(batch)
        pred = torch.argmax(outputs['logits'], dim=-1).item()
        ablation_results['text_only'].append(pred == gt)
        
        # Audio only (zero out text and visual)
        batch['input_ids'] = torch.zeros_like(encoded['input_ids']).to(DEVICE)
        batch['attention_mask'] = torch.zeros_like(encoded['attention_mask']).to(DEVICE)
        batch['audio'] = audio.to(DEVICE)
        batch['visual'] = torch.zeros_like(visual).to(DEVICE)
        outputs = model(batch)
        pred = torch.argmax(outputs['logits'], dim=-1).item()
        ablation_results['audio_only'].append(pred == gt)
        
        # Visual only (zero out text and audio)
        batch['input_ids'] = torch.zeros_like(encoded['input_ids']).to(DEVICE)
        batch['attention_mask'] = torch.zeros_like(encoded['attention_mask']).to(DEVICE)
        batch['audio'] = torch.zeros_like(audio).to(DEVICE)
        batch['visual'] = visual.to(DEVICE)
        outputs = model(batch)
        pred = torch.argmax(outputs['logits'], dim=-1).item()
        ablation_results['visual_only'].append(pred == gt)

print(f"\n📊 Modality Ablation Results:")
all_acc = np.mean(ablation_results['all'])
print(f"   {'Configuration':<20s}  {'Accuracy':>10s}  {'vs All':>10s}")
print(f"   {'-'*20}  {'-'*10}  {'-'*10}")
print(f"   {'All Modalities':<20s}  {all_acc*100:>9.2f}%  {'baseline':>10s}")

for config in ['text_only', 'audio_only', 'visual_only']:
    acc = np.mean(ablation_results[config])
    diff = acc - all_acc
    diff_str = f"{diff*100:+.2f}%"
    name = config.replace('_', ' ').title()
    print(f"   {name:<20s}  {acc*100:>9.2f}%  {diff_str:>10s}")

# === SUMMARY ===
print("\n" + "="*70)
print("   SUMMARY & KEY FINDINGS")
print("="*70)

print(f"\n🎯 Model Performance:")
print(f"   • Overall Accuracy: {accuracy*100:.2f}%")
print(f"   • Total Parameters: {total_params:,}")
print(f"   • Trained for: {config['training']['num_epochs']} epochs max")
print(f"   • Batch size: {config['training']['batch_size']}")

print(f"\n🔍 Modality Analysis:")
best_modality = max([
    ('Text', np.mean(ablation_results['text_only'])),
    ('Audio', np.mean(ablation_results['audio_only'])),
    ('Visual', np.mean(ablation_results['visual_only']))
], key=lambda x: x[1])
print(f"   • Most important modality: {best_modality[0]} ({best_modality[1]*100:.2f}%)")
print(f"   • Multimodal fusion benefit: {(all_acc - best_modality[1])*100:+.2f}%")

print(f"\n💡 Insights:")
text_acc = np.mean(ablation_results['text_only'])
audio_acc = np.mean(ablation_results['audio_only'])
visual_acc = np.mean(ablation_results['visual_only'])

if text_acc > audio_acc and text_acc > visual_acc:
    print(f"   ✓ Text is dominant modality (as expected for sentiment)")
    print(f"   → Audio adds {(all_acc - text_acc)*100:.2f}% when combined")
    print(f"   → Visual adds {(all_acc - text_acc)*100:.2f}% when combined")
else:
    print(f"   ⚠ Unexpected modality dominance - check fusion mechanism")

print(f"\n📈 Next Steps to Improve:")
print(f"   1. Train longer: Current {config['training']['num_epochs']} → Try 60 epochs")
print(f"   2. Increase batch size: Current {config['training']['batch_size']} → Try 8-16")
print(f"   3. Tune regularization: dropout={model_config['dropout']}")
print(f"   4. Better fusion: Improve audio/visual contribution")

print("\n" + "="*70)
print("   ✅ ANALYSIS COMPLETE")
print("="*70)