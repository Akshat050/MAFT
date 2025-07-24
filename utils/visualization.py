import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import cv2
from pathlib import Path


def plot_attention_heatmap(attention_weights: np.ndarray, 
                          text_tokens: List[str] = None,
                          audio_tokens: List[str] = None,
                          visual_tokens: List[str] = None,
                          title: str = "Cross-Modal Attention",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8)):
    """
    Plot attention heatmap for cross-modal attention.
    
    Args:
        attention_weights: [num_heads, seq_len, seq_len] attention weights
        text_tokens: List of text tokens
        audio_tokens: List of audio token descriptions
        visual_tokens: List of visual token descriptions
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    num_heads, seq_len, _ = attention_weights.shape
    
    # Create subplots for each attention head
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    axes = axes.flatten()
    
    for head_idx in range(min(num_heads, 12)):  # Show first 12 heads
        ax = axes[head_idx]
        
        # Get attention weights for this head
        head_weights = attention_weights[head_idx]
        
        # Create heatmap
        im = ax.imshow(head_weights, cmap='Blues', aspect='auto')
        
        # Set labels
        if text_tokens and audio_tokens and visual_tokens:
            # Create combined token list
            all_tokens = ['[CLS]'] + text_tokens + audio_tokens + visual_tokens
            
            # Set tick positions
            text_end = len(text_tokens) + 1
            audio_end = text_end + len(audio_tokens)
            
            ax.set_xticks([0, text_end, audio_end, seq_len-1])
            ax.set_xticklabels(['CLS', 'Text', 'Audio', 'Visual'], rotation=45)
            ax.set_yticks([0, text_end, audio_end, seq_len-1])
            ax.set_yticklabels(['CLS', 'Text', 'Audio', 'Visual'])
        
        ax.set_title(f'Head {head_idx + 1}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label('Attention Weight')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_modality_importance(attention_weights: np.ndarray,
                           modality_names: List[str] = ['Text', 'Audio', 'Visual'],
                           title: str = "Modality Importance",
                           save_path: Optional[str] = None):
    """
    Plot modality importance based on attention weights.
    
    Args:
        attention_weights: [num_heads, seq_len, seq_len] attention weights
        modality_names: Names of modalities
        title: Plot title
        save_path: Path to save the plot
    """
    num_heads, seq_len, _ = attention_weights.shape
    
    # Calculate modality importance for each head
    modality_importance = np.zeros((num_heads, len(modality_names)))
    
    # Define modality boundaries (this should match your data structure)
    # Assuming: [CLS] + text + audio + visual
    text_start, text_end = 1, seq_len // 3
    audio_start, audio_end = text_end, 2 * seq_len // 3
    visual_start, visual_end = audio_end, seq_len
    
    modality_ranges = [
        (text_start, text_end),
        (audio_start, audio_end),
        (visual_start, visual_end)
    ]
    
    for head_idx in range(num_heads):
        for mod_idx, (start, end) in enumerate(modality_ranges):
            # Calculate average attention to this modality
            importance = np.mean(attention_weights[head_idx, :, start:end])
            modality_importance[head_idx, mod_idx] = importance
    
    # Plot modality importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap of modality importance across heads
    im = ax1.imshow(modality_importance.T, cmap='Blues', aspect='auto')
    ax1.set_xlabel('Attention Head')
    ax1.set_ylabel('Modality')
    ax1.set_yticks(range(len(modality_names)))
    ax1.set_yticklabels(modality_names)
    ax1.set_title('Modality Importance by Attention Head')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Importance Score')
    
    # Bar plot of average modality importance
    avg_importance = np.mean(modality_importance, axis=0)
    std_importance = np.std(modality_importance, axis=0)
    
    bars = ax2.bar(modality_names, avg_importance, yerr=std_importance, 
                   capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Average Importance Score')
    ax2.set_title('Average Modality Importance')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_importance):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_temporal_attention(attention_weights: np.ndarray,
                          timestamps: List[float] = None,
                          title: str = "Temporal Attention",
                          save_path: Optional[str] = None):
    """
    Plot temporal attention patterns.
    
    Args:
        attention_weights: [num_heads, seq_len, seq_len] attention weights
        timestamps: List of timestamps for each token
        title: Plot title
        save_path: Path to save the plot
    """
    num_heads, seq_len, _ = attention_weights.shape
    
    if timestamps is None:
        timestamps = list(range(seq_len))
    
    # Calculate temporal attention (attention to each time step)
    temporal_attention = np.mean(attention_weights, axis=1)  # Average across heads
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot temporal attention for different heads
    for head_idx in range(min(num_heads, 4)):
        ax = axes[head_idx]
        
        # Get attention pattern for this head
        attention_pattern = temporal_attention[head_idx]
        
        # Plot as line
        ax.plot(timestamps, attention_pattern, linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Head {head_idx + 1} Temporal Attention')
        ax.grid(True, alpha=0.3)
        
        # Highlight peaks
        peaks = np.where(attention_pattern > np.mean(attention_pattern) + np.std(attention_pattern))[0]
        if len(peaks) > 0:
            ax.scatter([timestamps[i] for i in peaks], 
                      [attention_pattern[i] for i in peaks], 
                      color='red', s=50, zorder=5, label='Peaks')
            ax.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(feature_importance: Dict[str, float],
                          title: str = "Feature Importance",
                          save_path: Optional[str] = None):
    """
    Plot feature importance for different modalities.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        title: Plot title
        save_path: Path to save the plot
    """
    features = list(feature_importance.keys())
    importance_scores = list(feature_importance.values())
    
    # Sort by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    features = [features[i] for i in sorted_indices]
    importance_scores = [importance_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(features, importance_scores, color='skyblue', alpha=0.7)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, importance_scores)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontweight='bold')
    
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='x')
    plt.gca().invert_yaxis()  # Show most important at top
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]],
                         metrics: List[str] = ['accuracy', 'f1_score', 'mse'],
                         title: str = "Model Comparison",
                         save_path: Optional[str] = None):
    """
    Plot comparison of different models/ablation results.
    
    Args:
        results: Dictionary mapping model names to metric dictionaries
        metrics: List of metrics to compare
        title: Plot title
        save_path: Path to save the plot
    """
    models = list(results.keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        values = [results[model].get(metric, 0) for model in models]
        
        bars = ax.bar(models, values, alpha=0.7, color='lightcoral')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels if needed
        if len(models) > 3:
            ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_attention_video(attention_weights: np.ndarray,
                          text_tokens: List[str],
                          output_path: str,
                          fps: int = 10):
    """
    Create a video showing attention patterns over time.
    
    Args:
        attention_weights: [num_frames, num_heads, seq_len, seq_len] attention weights
        text_tokens: List of text tokens
        output_path: Path to save the video
        fps: Frames per second
    """
    num_frames, num_heads, seq_len, _ = attention_weights.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (800, 600))
    
    for frame_idx in range(num_frames):
        # Create figure for this frame
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        axes = axes.flatten()
        
        # Plot attention for first 4 heads
        for head_idx in range(min(num_heads, 4)):
            ax = axes[head_idx]
            
            attention = attention_weights[frame_idx, head_idx]
            im = ax.imshow(attention, cmap='Blues', aspect='auto')
            ax.set_title(f'Head {head_idx + 1}, Frame {frame_idx}')
            
            # Set labels
            if frame_idx == 0:  # Only set labels for first frame
                ax.set_xticks(range(seq_len))
                ax.set_xticklabels(text_tokens, rotation=45, ha='right')
                ax.set_yticks(range(seq_len))
                ax.set_yticklabels(text_tokens)
        
        plt.tight_layout()
        
        # Convert matplotlib figure to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize to video dimensions
        img_bgr = cv2.resize(img_bgr, (800, 600))
        
        # Write frame
        out.write(img_bgr)
        
        plt.close(fig)
    
    out.release()
    print(f"🎬 Attention video saved to: {output_path}")


def plot_training_progress(log_file: str,
                          metrics: List[str] = ['loss', 'accuracy'],
                          save_path: Optional[str] = None):
    """
    Plot training progress from log file.
    
    Args:
        log_file: Path to training log file
        metrics: List of metrics to plot
        save_path: Path to save the plot
    """
    # This would parse your actual log file format
    # For now, we'll create a mock example
    
    epochs = list(range(1, 21))
    train_loss = [2.5 - 0.1*i + np.random.normal(0, 0.05) for i in range(20)]
    val_loss = [2.3 - 0.08*i + np.random.normal(0, 0.1) for i in range(20)]
    train_acc = [0.3 + 0.03*i + np.random.normal(0, 0.02) for i in range(20)]
    val_acc = [0.25 + 0.025*i + np.random.normal(0, 0.03) for i in range(20)]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_loss, label='Train Loss', marker='o')
    axes[0].plot(epochs, val_loss, label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_acc, label='Train Accuracy', marker='o')
    axes[1].plot(epochs, val_acc, label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 