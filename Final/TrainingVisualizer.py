import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import warnings
from Config import DROConfig


class TrainingVisualizer:
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                pass 
    
    def plot_training_curves(self, history: Dict[str, List[float]]):

        if not history or not history.get('loss'):
            warnings.warn("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        epochs = range(1, len(history['loss']) + 1)
        
        # Loss plot
        ax = axes[0, 0]
        ax.plot(epochs, history['loss'], 'b-', linewidth=2, label='Train', alpha=0.8)
        if history.get('val_loss'):
            ax.plot(epochs, history['val_loss'], 'r--', linewidth=2, label='Val', alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax = axes[0, 1]
        if history.get('val_acc'):
            ax.plot(epochs, history['val_acc'], 'g-', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
        
        # Constraints plot
        ax = axes[0, 2]
        plotted = False
        if history.get('avg_constraint_dp'):
            ax.plot(epochs, history['avg_constraint_dp'], 'purple', 
                    linewidth=2, label='DP', alpha=0.8)
            plotted = True
        if history.get('avg_constraint_if'):
            ax.plot(epochs, history['avg_constraint_if'], 'orange', 
                    linewidth=2, label='IF', alpha=0.8)
            plotted = True
        if plotted:
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Constraint Value', fontsize=11)
            ax.set_title('Fairness Constraints', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Fairness Constraints', fontsize=12, fontweight='bold')
        
        # Lambdas plot
        ax = axes[1, 0]
        plotted = False
        if history.get('lambda_dp'):
            ax.plot(epochs, history['lambda_dp'], 'purple', 
                    linewidth=2, label='λ_DP', alpha=0.8)
            plotted = True
        if history.get('lambda_if'):
            ax.plot(epochs, history['lambda_if'], 'orange', 
                    linewidth=2, label='λ_IF', alpha=0.8)
            plotted = True
        if plotted:
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Lambda Value', fontsize=11)
            ax.set_title('Lagrange Multipliers', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Lagrange Multipliers', fontsize=12, fontweight='bold')
        
        # Parity plot
        ax = axes[1, 1]
        if history.get('val_parity'):
            ax.plot(epochs, history['val_parity'], 'red', linewidth=2, alpha=0.8)
            ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='5% threshold')
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Parity Gap', fontsize=11)
            ax.set_title('Demographic Parity Violation', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Demographic Parity Violation', fontsize=12, fontweight='bold')
        
        # IF plot
        ax = axes[1, 2]
        if history.get('val_if'):
            ax.plot(epochs, history['val_if'], 'brown', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('IF Violation Rate', fontsize=11)
            ax.set_title('Individual Fairness Violation', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Individual Fairness Violation', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.save_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {save_path}")
    
    def create_summary_report(
        self, 
        history: Dict[str, List[float]], 
        config: DROConfig,
        final_metrics: Optional[Dict[str, float]] = None
    ):
        """Generate comprehensive text summary report."""
        report_path = self.save_dir / "training_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DRO TRAINING SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            f.write("CONFIGURATION PARAMETERS:\n")
            f.write("-" * 80 + "\n")
            for key, value in sorted(config.to_dict().items()):
                f.write(f"  {key:.<35} {value}\n")
            f.write("\n")
            
            # Training progress
            f.write("TRAINING PROGRESS:\n")
            f.write("-" * 80 + "\n")
            if history.get('loss'):
                f.write(f"  Total Epochs Completed: {len(history['loss'])}\n")
                f.write(f"  Initial Training Loss: {history['loss'][0]:.6f}\n")
                f.write(f"  Final Training Loss: {history['loss'][-1]:.6f}\n")
                if len(history['loss']) > 1:
                    improvement = history['loss'][0] - history['loss'][-1]
                    f.write(f"  Loss Improvement: {improvement:.6f}\n")
            f.write("\n")
            
            # Final validation metrics
            f.write("FINAL VALIDATION METRICS:\n")
            f.write("-" * 80 + "\n")
            if history.get('val_loss'):
                f.write(f"  Validation Loss: {history['val_loss'][-1]:.6f}\n")
            if history.get('val_acc'):
                f.write(f"  Validation Accuracy: {history['val_acc'][-1]:.4f} ({history['val_acc'][-1]*100:.2f}%)\n")
            if history.get('val_parity'):
                f.write(f"  Demographic Parity Gap: {history['val_parity'][-1]:.6f}\n")
            if history.get('val_if'):
                f.write(f"  IF Violation Rate: {history['val_if'][-1]:.6f}\n")
            f.write("\n")
            
            # Test metrics if provided
            if final_metrics:
                f.write("FINAL TEST METRICS:\n")
                f.write("-" * 80 + "\n")
                for key, value in sorted(final_metrics.items()):
                    f.write(f"  {key:.<35} {value:.6f}\n")
                f.write("\n")
            
            # Lagrange multipliers
            if history.get('lambda_dp') and history.get('lambda_if'):
                f.write("FINAL LAGRANGE MULTIPLIERS:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  λ_DP (Demographic Parity): {history['lambda_dp'][-1]:.6f}\n")
                f.write(f"  λ_IF (Individual Fairness): {history['lambda_if'][-1]:.6f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"Summary report saved to: {report_path}")