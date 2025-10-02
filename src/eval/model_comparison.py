"""
Model Comparison Tool

This script compares results from multiple models:
- HOG + SVM (classical ML)
- ResNet (deep learning)
- EfficientNet (deep learning)
- And any other models
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime


class ModelComparison:
    """
    Compare multiple models and generate comparison visualizations
    """
    
    def __init__(self, output_dir="results/comparison"):
        """
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.metrics_df = None
    
    def add_model(self, model_name, results_dir):
        """
        Add a model's results to the comparison
        
        Args:
            model_name: Display name for the model
            results_dir: Path to the model's results directory
        """
        results_dir = Path(results_dir)
        
        # Load metrics
        metrics_file = results_dir / "metrics.json"
        if not metrics_file.exists():
            print(f"Warning: metrics.json not found in {results_dir}")
            return
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Load config if available
        config_file = results_dir / "config.json"
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Load training history if available
        history_file = results_dir / "training_history.json"
        history = None
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        self.models[model_name] = {
            'metrics': metrics,
            'config': config,
            'history': history,
            'results_dir': results_dir
        }
        
        print(f"Added model: {model_name}")
        print(f"  Test Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    def create_metrics_dataframe(self):
        """
        Create a pandas DataFrame with all metrics for comparison
        """
        data = []
        
        for model_name, model_data in self.models.items():
            metrics = model_data['metrics']
            
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Mean Per-Class Acc': metrics.get('mean_per_class_accuracy', 0),
                'Precision (Macro)': metrics.get('precision_macro', 0),
                'Recall (Macro)': metrics.get('recall_macro', 0),
                'F1-Score (Macro)': metrics.get('f1_macro', 0),
                'Precision (Weighted)': metrics.get('precision_weighted', 0),
                'Recall (Weighted)': metrics.get('recall_weighted', 0),
                'F1-Score (Weighted)': metrics.get('f1_weighted', 0),
                'Top-5 Accuracy': metrics.get('top5_accuracy', 0) if metrics.get('top5_accuracy') is not None else 0,
            }
            
            data.append(row)
        
        self.metrics_df = pd.DataFrame(data)
        return self.metrics_df
    
    def save_metrics_table(self):
        """
        Save metrics comparison table to CSV and formatted text
        """
        if self.metrics_df is None:
            self.create_metrics_dataframe()
        
        # Save to CSV
        csv_path = self.output_dir / "metrics_comparison.csv"
        self.metrics_df.to_csv(csv_path, index=False)
        print(f"\nMetrics table saved to: {csv_path}")
        
        # Save formatted text version
        txt_path = self.output_dir / "metrics_comparison.txt"
        with open(txt_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("MODEL COMPARISON - EVALUATION METRICS\n")
            f.write("=" * 100 + "\n\n")
            f.write(self.metrics_df.to_string(index=False))
            f.write("\n\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n")
        
        print(f"Formatted table saved to: {txt_path}")
        
        # Print to console
        print("\n" + "=" * 100)
        print("MODEL COMPARISON - EVALUATION METRICS")
        print("=" * 100)
        print(self.metrics_df.to_string(index=False))
        print("=" * 100 + "\n")
    
    def plot_metrics_comparison(self):
        """
        Create bar charts comparing key metrics across models
        """
        if self.metrics_df is None:
            self.create_metrics_dataframe()
        
        # Metrics to compare
        metrics_to_plot = [
            ('Accuracy', 'Overall Accuracy'),
            ('Mean Per-Class Acc', 'Mean Per-Class Accuracy'),
            ('F1-Score (Macro)', 'F1-Score (Macro Average)'),
            ('Top-5 Accuracy', 'Top-5 Accuracy')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))
        
        for idx, (metric_col, metric_title) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            values = self.metrics_df[metric_col].values
            models = self.metrics_df['Model'].values
            
            bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}\n({val*100:.2f}%)',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title(metric_title, fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Highlight best model
            best_idx = np.argmax(values)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "metrics_comparison_bars.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison bar chart saved to: {save_path}")
        plt.close()
    
    def plot_detailed_metrics_heatmap(self):
        """
        Create a heatmap showing all metrics for all models
        """
        if self.metrics_df is None:
            self.create_metrics_dataframe()
        
        # Select numeric columns only
        numeric_cols = [col for col in self.metrics_df.columns if col != 'Model']
        
        # Create heatmap data
        heatmap_data = self.metrics_df[numeric_cols].values
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(self.models) * 1.5)))
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_yticks(np.arange(len(self.models)))
        
        # Set labels
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(self.metrics_df['Model'].values, fontsize=11, fontweight='bold')
        
        # Add text annotations
        for i in range(len(self.models)):
            for j in range(len(numeric_cols)):
                value = heatmap_data[i, j]
                text = ax.text(j, i, f'{value:.3f}',
                             ha="center", va="center", color="black" if value > 0.5 else "white",
                             fontsize=9, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        
        ax.set_title('Detailed Metrics Comparison Heatmap', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "metrics_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics heatmap saved to: {save_path}")
        plt.close()
    
    def plot_training_curves_comparison(self):
        """
        Compare training curves across models (only for deep learning models)
        """
        models_with_history = {name: data for name, data in self.models.items() 
                              if data['history'] is not None and 'train_loss' in data['history']}
        
        if not models_with_history:
            print("No training history available for comparison")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot loss curves
        ax = axes[0]
        for model_name, model_data in models_with_history.items():
            history = model_data['history']
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax.plot(epochs, history['train_loss'], 'o-', label=f'{model_name} (Train)', alpha=0.7)
            ax.plot(epochs, history['val_loss'], 's--', label=f'{model_name} (Val)', alpha=0.7)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot accuracy curves
        ax = axes[1]
        for model_name, model_data in models_with_history.items():
            history = model_data['history']
            epochs = range(1, len(history['train_acc']) + 1)
            
            ax.plot(epochs, history['train_acc'], 'o-', label=f'{model_name} (Train)', alpha=0.7)
            ax.plot(epochs, history['val_acc'], 's--', label=f'{model_name} (Val)', alpha=0.7)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "training_curves_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves comparison saved to: {save_path}")
        plt.close()
    
    def plot_per_class_accuracy_comparison(self):
        """
        Compare per-class accuracy across models
        """
        # Find common classes across all models
        all_per_class_acc = {}
        for model_name, model_data in self.models.items():
            metrics = model_data['metrics']
            if 'per_class_accuracy' in metrics:
                all_per_class_acc[model_name] = metrics['per_class_accuracy']
        
        if not all_per_class_acc:
            print("No per-class accuracy data available")
            return
        
        # Get all class indices
        all_classes = set()
        for per_class_acc in all_per_class_acc.values():
            all_classes.update(per_class_acc.keys())
        
        # Convert string keys to int and sort
        all_classes = sorted([int(k) if isinstance(k, str) else k for k in all_classes])
        
        # Create comparison data
        comparison_data = []
        for class_idx in all_classes:
            row = {'Class': class_idx}
            for model_name, per_class_acc in all_per_class_acc.items():
                # Handle both string and int keys
                key = str(class_idx) if str(class_idx) in per_class_acc else class_idx
                row[model_name] = per_class_acc.get(key, 0)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate average accuracy per class across models
        model_cols = [col for col in df.columns if col != 'Class']
        df['Average'] = df[model_cols].mean(axis=1)
        
        # Sort by average accuracy to show most/least difficult classes
        df = df.sort_values('Average', ascending=False)
        
        # Plot top 20 and bottom 20 classes
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Top 20 easiest classes
        top_20 = df.head(20)
        ax = axes[0]
        x = np.arange(len(top_20))
        width = 0.8 / len(model_cols)
        
        for idx, model_name in enumerate(model_cols):
            offset = (idx - len(model_cols)/2 + 0.5) * width
            ax.bar(x + offset, top_20[model_name].values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Class Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Easiest Classes (Highest Accuracy)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_20['Class'].values, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Bottom 20 hardest classes
        bottom_20 = df.tail(20)
        ax = axes[1]
        x = np.arange(len(bottom_20))
        
        for idx, model_name in enumerate(model_cols):
            offset = (idx - len(model_cols)/2 + 0.5) * width
            ax.bar(x + offset, bottom_20[model_name].values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Class Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Hardest Classes (Lowest Accuracy)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bottom_20['Class'].values, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "per_class_accuracy_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class accuracy comparison saved to: {save_path}")
        plt.close()
    
    def generate_summary_report(self):
        """
        Generate a comprehensive text summary report
        """
        if self.metrics_df is None:
            self.create_metrics_dataframe()
        
        report_path = self.output_dir / "comparison_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("MODEL COMPARISON SUMMARY REPORT\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of models compared: {len(self.models)}\n")
            f.write("=" * 100 + "\n\n")
            
            # Overall rankings
            f.write("OVERALL RANKINGS (by Test Accuracy)\n")
            f.write("-" * 100 + "\n")
            ranked = self.metrics_df.sort_values('Accuracy', ascending=False)
            for idx, row in ranked.iterrows():
                f.write(f"{idx+1}. {row['Model']:30s} - {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)\n")
            f.write("\n")
            
            # Best model in each metric
            f.write("BEST MODEL FOR EACH METRIC\n")
            f.write("-" * 100 + "\n")
            numeric_cols = [col for col in self.metrics_df.columns if col != 'Model']
            for metric in numeric_cols:
                best_idx = self.metrics_df[metric].idxmax()
                best_model = self.metrics_df.loc[best_idx, 'Model']
                best_value = self.metrics_df.loc[best_idx, metric]
                f.write(f"{metric:30s}: {best_model:20s} ({best_value:.4f})\n")
            f.write("\n")
            
            # Detailed metrics table
            f.write("DETAILED METRICS TABLE\n")
            f.write("-" * 100 + "\n")
            f.write(self.metrics_df.to_string(index=False))
            f.write("\n\n")
            
            # Model configurations
            f.write("MODEL CONFIGURATIONS\n")
            f.write("-" * 100 + "\n")
            for model_name, model_data in self.models.items():
                f.write(f"\n{model_name}:\n")
                config = model_data['config']
                if config:
                    for key, value in config.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write("  No configuration data available\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 100 + "\n")
        
        print(f"\nSummary report saved to: {report_path}")
    
    def run_full_comparison(self):
        """
        Run all comparison analyses and generate all visualizations
        """
        print("\n" + "=" * 100)
        print("RUNNING FULL MODEL COMPARISON")
        print("=" * 100 + "\n")
        
        # Create metrics dataframe
        self.create_metrics_dataframe()
        
        # Save metrics table
        self.save_metrics_table()
        
        # Generate all plots
        print("\nGenerating comparison visualizations...")
        self.plot_metrics_comparison()
        self.plot_detailed_metrics_heatmap()
        self.plot_training_curves_comparison()
        self.plot_per_class_accuracy_comparison()
        
        # Generate summary report
        self.generate_summary_report()
        
        print("\n" + "=" * 100)
        print("COMPARISON COMPLETED!")
        print(f"All results saved to: {self.output_dir}")
        print("=" * 100 + "\n")


def main():
    """
    Example usage of ModelComparison
    """
    print("=" * 100)
    print("MODEL COMPARISON TOOL")
    print("=" * 100)
    
    # Create comparison object
    comparison = ModelComparison(output_dir="results/comparison")
    
    # Add models (update these paths to your actual results directories)
    # Example paths - adjust based on your actual results
    
    # HOG + SVM baseline (try fast version first, then regular)
    hog_svm_fast_dir = "results/runs/hog_svm_fast"
    hog_svm_dir = "results/runs/hog_svm_baseline"
    if Path(hog_svm_fast_dir).exists() and (Path(hog_svm_fast_dir) / "metrics.json").exists():
        comparison.add_model("HOG + SVM", hog_svm_fast_dir)
    elif Path(hog_svm_dir).exists() and (Path(hog_svm_dir) / "metrics.json").exists():
        comparison.add_model("HOG + SVM", hog_svm_dir)
    
    # Find ResNet results (look for the most recent)
    resnet_dirs = list(Path("results/runs").glob("resnet_*"))
    if resnet_dirs:
        # Use the most recently modified directory
        latest_resnet = max(resnet_dirs, key=lambda p: p.stat().st_mtime)
        comparison.add_model("ResNet", latest_resnet)
    
    # Find EfficientNet results (look for the most recent)
    effnet_dirs = list(Path("results/runs").glob("efficientnet_*"))
    if effnet_dirs:
        # Use the most recently modified directory
        latest_effnet = max(effnet_dirs, key=lambda p: p.stat().st_mtime)
        comparison.add_model("EfficientNet", latest_effnet)
    
    # Run full comparison
    if len(comparison.models) >= 2:
        comparison.run_full_comparison()
    else:
        print("\nNeed at least 2 models for comparison!")
        print("Please train models first and ensure results are saved.")


if __name__ == "__main__":
    main()

