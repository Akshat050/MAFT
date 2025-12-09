"""
Training Logger for MAFT

Logs training metrics, generates visualizations, and saves training history.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime


class TrainingLogger:
    """
    Comprehensive logger for training metrics and visualization.
    
    Args:
        log_dir: Directory to save logs and plots
        config: Training configuration dictionary
    """
    
    def __init__(self, log_dir, config=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train': {'loss': [], 'cls': [], 'reg': [], 'cons': []},
            'val': {'loss': [], 'cls': [], 'reg': [], 'cons': []},
            'lr': [],
            'epochs': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
        
        # Save config
        if config:
            with open(self.log_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
        
        # Start time
        self.start_time = datetime.now()
        
        print(f"📊 Logger initialized: {self.log_dir}")
    
    def log_epoch(self, epoch, train_logs, val_logs, lr):
        """
        Log metrics for one epoch.
        
        Args:
            epoch: Epoch number (0-indexed)
            train_logs: Dictionary with 'tot', 'cls', 'reg', 'cons'
            val_logs: Dictionary with 'tot', 'cls', 'reg', 'cons'
            lr: Current learning rate
        """
        # Store metrics
        self.history['epochs'].append(epoch + 1)
        
        self.history['train']['loss'].append(train_logs['tot'])
        self.history['train']['cls'].append(train_logs['cls'])
        self.history['train']['reg'].append(train_logs['reg'])
        self.history['train']['cons'].append(train_logs['cons'])
        
        self.history['val']['loss'].append(val_logs['tot'])
        self.history['val']['cls'].append(val_logs['cls'])
        self.history['val']['reg'].append(val_logs['reg'])
        self.history['val']['cons'].append(val_logs['cons'])
        
        self.history['lr'].append(lr)
        
        # Track best model
        if val_logs['tot'] < self.history['best_val_loss']:
            self.history['best_val_loss'] = val_logs['tot']
            self.history['best_epoch'] = epoch + 1
        
        # Save history to JSON
        self._save_history()
    
    def _save_history(self):
        """Save training history to JSON file."""
        with open(self.log_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self):
        """Generate comprehensive training curve visualizations."""
        if len(self.history['epochs']) == 0:
            print("⚠️  No data to plot yet")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = self.history['epochs']
        
        # 1. Total Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train']['loss'], 'b-o', label='Train', linewidth=2, markersize=4)
        ax.plot(epochs, self.history['val']['loss'], 'r-o', label='Val', linewidth=2, markersize=4)
        ax.axvline(x=self.history['best_epoch'], color='g', linestyle='--', 
                   label=f'Best (Epoch {self.history["best_epoch"]})', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Total Loss', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. Classification Loss
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train']['cls'], 'b-o', label='Train', linewidth=2, markersize=4)
        ax.plot(epochs, self.history['val']['cls'], 'r-o', label='Val', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Classification Loss', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. Regression Loss
        ax = axes[0, 2]
        ax.plot(epochs, self.history['train']['reg'], 'b-o', label='Train', linewidth=2, markersize=4)
        ax.plot(epochs, self.history['val']['reg'], 'r-o', label='Val', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Regression Loss', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 4. Consistency Loss
        ax = axes[1, 0]
        ax.plot(epochs, self.history['train']['cons'], 'b-o', label='Train', linewidth=2, markersize=4)
        ax.plot(epochs, self.history['val']['cons'], 'r-o', label='Val', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Consistency Loss', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 5. Learning Rate
        ax = axes[1, 1]
        ax.plot(epochs, self.history['lr'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 6. Overfitting Gap
        ax = axes[1, 2]
        train_loss = np.array(self.history['train']['loss'])
        val_loss = np.array(self.history['val']['loss'])
        gap = val_loss - train_loss
        
        ax.plot(epochs, gap, 'purple', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.fill_between(epochs, 0, gap, where=(gap>0), alpha=0.3, color='red', label='Overfitting')
        ax.fill_between(epochs, 0, gap, where=(gap<=0), alpha=0.3, color='green', label='Underfitting')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Val Loss - Train Loss', fontsize=11)
        ax.set_title('Generalization Gap', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.log_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved training curves: {save_path}")
    
    def print_summary(self):
        """Print training summary statistics."""
        if len(self.history['epochs']) == 0:
            return
        
        duration = datetime.now() - self.start_time
        
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Total epochs: {len(self.history['epochs'])}")
        print(f"Training time: {duration}")
        print(f"\nBest model:")
        print(f"  Epoch: {self.history['best_epoch']}")
        print(f"  Val loss: {self.history['best_val_loss']:.4f}")
        
        # Final metrics
        final_train = self.history['train']['loss'][-1]
        final_val = self.history['val']['loss'][-1]
        gap = final_val - final_train
        
        print(f"\nFinal metrics:")
        print(f"  Train loss: {final_train:.4f}")
        print(f"  Val loss: {final_val:.4f}")
        print(f"  Gap: {gap:.4f} ({'overfitting' if gap > 0.5 else 'good fit'})")
        
        # Loss breakdown
        print(f"\nFinal loss breakdown:")
        print(f"  Classification: {self.history['val']['cls'][-1]:.4f}")
        print(f"  Regression: {self.history['val']['reg'][-1]:.4f}")
        print(f"  Consistency: {self.history['val']['cons'][-1]:.4f}")
        
        print("="*70 + "\n")
    
    def save_checkpoint_info(self, epoch, checkpoint_path):
        """Log checkpoint save information."""
        info = {
            'epoch': epoch,
            'checkpoint_path': str(checkpoint_path),
            'timestamp': datetime.now().isoformat(),
            'val_loss': self.history['val']['loss'][-1] if self.history['val']['loss'] else None
        }
        
        checkpoints_log = self.log_dir / 'checkpoints.json'
        
        # Load existing checkpoints
        if checkpoints_log.exists():
            with open(checkpoints_log, 'r') as f:
                checkpoints = json.load(f)
        else:
            checkpoints = []
        
        checkpoints.append(info)
        
        # Save updated list
        with open(checkpoints_log, 'w') as f:
            json.dump(checkpoints, f, indent=2)


# Example usage
if __name__ == '__main__':
    print("="*70)
    print("TRAINING LOGGER - TEST")
    print("="*70)
    
    # Create logger
    logger = TrainingLogger('test_logs', config={'test': 'config'})
    
    # Simulate training
    print("\nSimulating 10 epochs of training...")
    for epoch in range(10):
        train_logs = {
            'tot': 2.0 - epoch * 0.15 + np.random.randn() * 0.1,
            'cls': 1.2 - epoch * 0.08,
            'reg': 0.7 - epoch * 0.05,
            'cons': 0.1 - epoch * 0.008
        }
        
        val_logs = {
            'tot': 2.2 - epoch * 0.12 + np.random.randn() * 0.15,
            'cls': 1.3 - epoch * 0.07,
            'reg': 0.8 - epoch * 0.04,
            'cons': 0.1 - epoch * 0.007
        }
        
        lr = 0.0001 * (0.95 ** epoch)
        
        logger.log_epoch(epoch, train_logs, val_logs, lr)
        print(f"Epoch {epoch+1}: Train={train_logs['tot']:.3f}, Val={val_logs['tot']:.3f}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    logger.plot_training_curves()
    
    # Print summary
    logger.print_summary()
    
    print("\n✅ Logger test complete! Check 'test_logs/' directory")