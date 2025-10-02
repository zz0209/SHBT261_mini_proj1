"""
Visualization utilities for model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_confusion_matrix(confusion_mat, class_names=None, save_path=None, 
                         figsize=(20, 18), normalize=False):
    """
    Plot confusion matrix as a heatmap
    
    Args:
        confusion_mat: Confusion matrix (num_classes, num_classes)
        class_names: List of class names (optional)
        save_path: Path to save the figure
        figsize: Figure size
        normalize: Whether to normalize by row (true labels)
    """
    if normalize:
        confusion_mat = confusion_mat.astype('float') / (confusion_mat.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    
    # Use class names if provided, otherwise use indices
    if class_names is None:
        class_names = [str(i) for i in range(len(confusion_mat))]
    
    # Plot heatmap
    sns.heatmap(
        confusion_mat,
        annot=False,  # Don't annotate with numbers (too many classes)
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()


def plot_per_class_accuracy(per_class_acc, class_names=None, save_path=None, 
                            figsize=(16, 10), top_n=None):
    """
    Plot per-class accuracy as a bar chart
    
    Args:
        per_class_acc: Dictionary {class_idx: accuracy}
        class_names: List of class names (optional)
        save_path: Path to save the figure
        figsize: Figure size
        top_n: If set, only show top/bottom N classes
    """
    # Sort by accuracy
    sorted_items = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    
    if top_n:
        # Show top N and bottom N
        selected_items = sorted_items[:top_n] + sorted_items[-top_n:]
    else:
        selected_items = sorted_items
    
    class_indices = [item[0] for item in selected_items]
    accuracies = [item[1] for item in selected_items]
    
    # Get class names
    if class_names is None:
        labels = [f"Class {idx}" for idx in class_indices]
    else:
        labels = [class_names[idx] for idx in class_indices]
    
    # Create plot
    plt.figure(figsize=figsize)
    colors = ['green' if acc > 0.7 else 'orange' if acc > 0.5 else 'red' for acc in accuracies]
    
    bars = plt.barh(range(len(labels)), accuracies, color=colors, alpha=0.7)
    
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3)
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to: {save_path}")
    
    plt.close()


def plot_training_curves(history, save_path=None, figsize=(14, 5)):
    """
    Plot training curves (loss and accuracy)
    
    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history:
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.close()


def plot_top_confused_pairs(confusion_mat, class_names=None, save_path=None, 
                           top_n=10, figsize=(12, 8)):
    """
    Plot the most confused class pairs
    
    Args:
        confusion_mat: Confusion matrix
        class_names: List of class names
        save_path: Path to save the figure
        top_n: Number of top confused pairs to show
    """
    # Get off-diagonal elements (misclassifications)
    num_classes = len(confusion_mat)
    confused_pairs = []
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:  # Skip diagonal
                count = confusion_mat[i, j]
                if count > 0:
                    confused_pairs.append((i, j, count))
    
    # Sort by count
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Get top N
    top_pairs = confused_pairs[:top_n]
    
    # Create labels
    if class_names is None:
        labels = [f"{i} -> {j}" for i, j, _ in top_pairs]
    else:
        labels = [f"{class_names[i]} -> {class_names[j]}" for i, j, _ in top_pairs]
    
    counts = [count for _, _, count in top_pairs]
    
    # Plot
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(labels)), counts, color='coral', alpha=0.7)
    
    plt.yticks(range(len(labels)), labels, fontsize=9)
    plt.xlabel('Number of Misclassifications', fontsize=12)
    plt.ylabel('Class Pair (True -> Predicted)', fontsize=12)
    plt.title(f'Top {top_n} Most Confused Class Pairs', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add counts on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(count + max(counts)*0.01, i, str(int(count)), va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confused pairs plot saved to: {save_path}")
    
    plt.close()


def plot_metrics_comparison(metrics_list, model_names, save_path=None, figsize=(12, 6)):
    """
    Plot comparison of metrics across multiple models
    
    Args:
        metrics_list: List of metric dictionaries
        model_names: List of model names
        save_path: Path to save the figure
    """
    metric_keys = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metric_labels = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
    
    x = np.arange(len(metric_keys))
    width = 0.8 / len(model_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (metrics, name) in enumerate(zip(metrics_list, model_names)):
        values = [metrics.get(key, 0) for key in metric_keys]
        offset = (i - len(model_names)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics comparison plot saved to: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization functions...")
    
    np.random.seed(42)
    num_classes = 10
    
    # Create dummy confusion matrix
    conf_mat = np.random.randint(0, 50, (num_classes, num_classes))
    np.fill_diagonal(conf_mat, np.random.randint(100, 200, num_classes))
    
    class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Test confusion matrix plot
    plot_confusion_matrix(conf_mat, class_names, save_path="test_confusion.png")
    
    # Test per-class accuracy plot
    per_class_acc = {i: np.random.rand() for i in range(num_classes)}
    plot_per_class_accuracy(per_class_acc, class_names, save_path="test_per_class.png")
    
    # Test training curves
    history = {
        'train_loss': [2.5, 2.0, 1.5, 1.2, 1.0, 0.8],
        'val_loss': [2.6, 2.1, 1.6, 1.4, 1.3, 1.2],
        'train_acc': [0.3, 0.5, 0.65, 0.75, 0.82, 0.88],
        'val_acc': [0.28, 0.48, 0.62, 0.70, 0.75, 0.78]
    }
    plot_training_curves(history, save_path="test_training_curves.png")
    
    # Test confused pairs
    plot_top_confused_pairs(conf_mat, class_names, save_path="test_confused_pairs.png")
    
    print("\nVisualization test completed!")
    print("Check generated PNG files.")

