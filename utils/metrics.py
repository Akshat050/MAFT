import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class MultimodalMetrics:
    """Comprehensive metrics for multimodal evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.classification_predictions = []
        self.classification_targets = []
        self.regression_predictions = []
        self.regression_targets = []
    
    def update(self, classification_logits: torch.Tensor, regression_output: torch.Tensor,
               classification_targets: torch.Tensor, regression_targets: torch.Tensor):
        """Update metrics with batch predictions."""
        # Classification
        classification_preds = torch.argmax(classification_logits, dim=1).cpu().numpy()
        self.classification_predictions.extend(classification_preds)
        self.classification_targets.extend(classification_targets.cpu().numpy())
        
        # Regression
        regression_preds = regression_output.squeeze(-1).cpu().numpy()
        self.regression_predictions.extend(regression_preds)
        self.regression_targets.extend(regression_targets.cpu().numpy())
    
    def compute_classification_metrics(self) -> Dict[str, float]:
        """Compute classification metrics."""
        if not self.classification_predictions:
            return {}
        
        y_pred = np.array(self.classification_predictions)
        y_true = np.array(self.classification_targets)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # ROC AUC (for binary classification)
        if len(np.unique(y_true)) == 2:
            try:
                auc = roc_auc_score(y_true, y_pred)
            except ValueError:
                auc = 0.0
        else:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm
        }
    
    def compute_regression_metrics(self) -> Dict[str, float]:
        """Compute regression metrics."""
        if not self.regression_predictions:
            return {}
        
        y_pred = np.array(self.regression_predictions)
        y_true = np.array(self.regression_targets)
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Pearson correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation
        }
    
    def compute_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute all metrics."""
        return {
            'classification': self.compute_classification_metrics(),
            'regression': self.compute_regression_metrics()
        }
    
    def print_metrics(self):
        """Print formatted metrics."""
        all_metrics = self.compute_all_metrics()
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        
        # Classification metrics
        cls_metrics = all_metrics['classification']
        if cls_metrics:
            print("\n📊 CLASSIFICATION METRICS:")
            print(f"  Accuracy:  {cls_metrics['accuracy']:.4f}")
            print(f"  Precision: {cls_metrics['precision']:.4f}")
            print(f"  Recall:    {cls_metrics['recall']:.4f}")
            print(f"  F1-Score:  {cls_metrics['f1_score']:.4f}")
            print(f"  AUC:       {cls_metrics['auc']:.4f}")
        
        # Regression metrics
        reg_metrics = all_metrics['regression']
        if reg_metrics:
            print("\n📈 REGRESSION METRICS:")
            print(f"  MSE:        {reg_metrics['mse']:.4f}")
            print(f"  MAE:        {reg_metrics['mae']:.4f}")
            print(f"  RMSE:       {reg_metrics['rmse']:.4f}")
            print(f"  Correlation: {reg_metrics['correlation']:.4f}")
        
        print("="*50)


def compute_modality_ablation_metrics(model, dataloader, device, 
                                    modalities_to_drop: List[str] = None) -> Dict[str, Dict[str, float]]:
    """Compute metrics with different modality combinations for ablation studies."""
    
    if modalities_to_drop is None:
        modalities_to_drop = ['text', 'audio', 'visual']
    
    results = {}
    
    for modality in modalities_to_drop:
        print(f"\n🔬 Testing without {modality} modality...")
        
        metrics = MultimodalMetrics()
        model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Drop specified modality
                if modality == 'text':
                    batch['input_ids'] = torch.zeros_like(batch['input_ids'])
                    batch['attention_mask'] = torch.zeros_like(batch['attention_mask'])
                elif modality == 'audio':
                    batch['audio_features'] = torch.zeros_like(batch['audio_features'])
                    batch['audio_mask'] = torch.zeros_like(batch['audio_mask'])
                elif modality == 'visual':
                    batch['visual_features'] = torch.zeros_like(batch['visual_features'])
                    batch['visual_mask'] = torch.zeros_like(batch['visual_mask'])
                
                # Forward pass
                outputs = model(**batch)
                
                # Update metrics
                metrics.update(
                    outputs['classification_logits'],
                    outputs['regression_output'],
                    batch['classification_targets'],
                    batch['regression_targets']
                )
        
        results[f'without_{modality}'] = metrics.compute_all_metrics()
    
    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = None, 
                         title: str = "Confusion Matrix", save_path: str = None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, 
                          title: str = "Regression Predictions", save_path: str = None):
    """Plot regression scatter plot."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    
    # Add correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                        train_metrics: List[float] = None, val_metrics: List[float] = None,
                        metric_name: str = "Accuracy", save_path: str = None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metric curves
    if train_metrics and val_metrics:
        axes[1].plot(train_metrics, label=f'Train {metric_name}', color='blue')
        axes[1].plot(val_metrics, label=f'Val {metric_name}', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'Training and Validation {metric_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show() 