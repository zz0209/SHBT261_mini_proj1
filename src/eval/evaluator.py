"""
Main Evaluator class that integrates metrics computation and visualization
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from .metrics import (
    compute_all_metrics,
    print_metrics_summary
)
from .visualization import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_training_curves,
    plot_top_confused_pairs
)


class Evaluator:
    """
    Unified evaluator for all models
    """
    
    def __init__(self, output_dir, class_names=None, num_classes=102):
        """
        Args:
            output_dir: Directory to save evaluation results
            class_names: List of class names (optional)
            num_classes: Number of classes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = class_names
        self.num_classes = num_classes
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
    
    def evaluate(self, y_true, y_pred, y_probs=None, verbose=True):
        """
        Compute all evaluation metrics
        
        Args:
            y_true: True labels (numpy array or list)
            y_pred: Predicted labels (numpy array or list)
            y_probs: Predicted probabilities (optional, for Top-K accuracy)
            verbose: Whether to print metrics summary
        
        Returns:
            dict: All computed metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_probs is not None:
            y_probs = np.array(y_probs)
        
        # Compute metrics
        metrics = compute_all_metrics(y_true, y_pred, y_probs, self.num_classes)
        
        # Print summary if requested
        if verbose:
            print_metrics_summary(metrics, self.class_names)
        
        return metrics
    
    def save_metrics(self, metrics, filename="metrics.json"):
        """
        Save metrics to JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, dict):
                # Check if dict has numeric keys (like per_class_accuracy)
                if all(isinstance(k, (int, np.integer)) for k in value.keys()):
                    metrics_serializable[key] = {int(k): float(v) for k, v in value.items()}
                else:
                    metrics_serializable[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value
        
        # Add metadata
        metrics_serializable['timestamp'] = datetime.now().isoformat()
        
        # Save to file
        save_path = self.output_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"\nMetrics saved to: {save_path}")
    
    def generate_plots(self, metrics, y_true=None, y_pred=None):
        """
        Generate all evaluation plots
        
        Args:
            metrics: Computed metrics dictionary
            y_true: True labels (needed for some plots)
            y_pred: Predicted labels (needed for some plots)
        """
        print("\nGenerating evaluation plots...")
        
        # 1. Confusion Matrix
        if 'confusion_matrix' in metrics:
            conf_mat = metrics['confusion_matrix']
            
            # Regular confusion matrix
            plot_confusion_matrix(
                conf_mat,
                class_names=self.class_names,
                save_path=self.plots_dir / "confusion_matrix.png",
                normalize=False
            )
            
            # Normalized confusion matrix
            plot_confusion_matrix(
                conf_mat,
                class_names=self.class_names,
                save_path=self.plots_dir / "confusion_matrix_normalized.png",
                normalize=True
            )
            
            # Top confused pairs
            plot_top_confused_pairs(
                conf_mat,
                class_names=self.class_names,
                save_path=self.plots_dir / "top_confused_pairs.png",
                top_n=15
            )
        
        # 2. Per-class Accuracy
        if 'per_class_accuracy' in metrics:
            # All classes
            plot_per_class_accuracy(
                metrics['per_class_accuracy'],
                class_names=self.class_names,
                save_path=self.plots_dir / "per_class_accuracy_all.png"
            )
            
            # Top/Bottom 20 classes
            plot_per_class_accuracy(
                metrics['per_class_accuracy'],
                class_names=self.class_names,
                save_path=self.plots_dir / "per_class_accuracy_top_bottom.png",
                top_n=20
            )
        
        print("All plots generated successfully!")
    
    def save_training_history(self, history, filename="training_history.json"):
        """
        Save training history and generate training curves
        
        Args:
            history: Dictionary with training history
                    (e.g., {'train_loss': [...], 'val_loss': [...], ...})
        """
        # Save history to JSON
        history_path = self.output_dir / filename
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to: {history_path}")
        
        # Generate training curves plot
        plot_training_curves(
            history,
            save_path=self.plots_dir / "training_curves.png"
        )
    
    def evaluate_and_save(self, y_true, y_pred, y_probs=None, history=None, 
                         model_name=None, config=None):
        """
        Complete evaluation pipeline: compute metrics, generate plots, and save results
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Predicted probabilities (optional)
            history: Training history (optional)
            model_name: Name of the model (optional)
            config: Configuration dictionary (optional)
        """
        print("\n" + "=" * 80)
        print(f"EVALUATING MODEL: {model_name if model_name else 'Unknown'}")
        print("=" * 80)
        
        # Compute metrics
        metrics = self.evaluate(y_true, y_pred, y_probs, verbose=True)
        
        # Add model name and config to metrics
        if model_name:
            metrics['model_name'] = model_name
        if config:
            metrics['config'] = config
        
        # Save metrics
        self.save_metrics(metrics)
        
        # Generate plots
        self.generate_plots(metrics, y_true, y_pred)
        
        # Save training history if provided
        if history:
            self.save_training_history(history)
        
        print("\n" + "=" * 80)
        print(f"EVALUATION COMPLETED!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80 + "\n")
        
        return metrics


if __name__ == "__main__":
    # Test the evaluator
    print("Testing Evaluator class...")
    
    # Create dummy data
    np.random.seed(42)
    num_samples = 1000
    num_classes = 102
    
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred = np.random.randint(0, num_classes, num_samples)
    y_probs = np.random.rand(num_samples, num_classes)
    
    # Normalize probabilities
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    
    # Create dummy training history
    history = {
        'train_loss': [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.7],
        'val_loss': [2.6, 2.1, 1.6, 1.4, 1.3, 1.2, 1.15],
        'train_acc': [0.3, 0.5, 0.65, 0.75, 0.82, 0.88, 0.91],
        'val_acc': [0.28, 0.48, 0.62, 0.70, 0.75, 0.78, 0.79]
    }
    
    # Load class names
    try:
        import json
        with open('data/splits/class_info.json', 'r') as f:
            class_info = json.load(f)
        class_names = [class_info['idx_to_class'][str(i)] for i in range(num_classes)]
    except:
        class_names = None
        print("Warning: Could not load class names, using indices instead")
    
    # Create evaluator
    evaluator = Evaluator(
        output_dir="results/test_evaluation",
        class_names=class_names,
        num_classes=num_classes
    )
    
    # Run complete evaluation
    config = {
        'model': 'test_model',
        'image_size': 128,
        'batch_size': 32,
        'optimizer': 'adam'
    }
    
    metrics = evaluator.evaluate_and_save(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        history=history,
        model_name="Test Model",
        config=config
    )
    
    print("\nEvaluator test completed successfully!")

