"""
Metrics computation for model evaluation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def compute_accuracy(y_true, y_pred):
    """
    Compute overall accuracy
    """
    return accuracy_score(y_true, y_pred)


def compute_per_class_accuracy(y_true, y_pred, num_classes):
    """
    Compute accuracy for each class
    
    Returns:
        dict: {class_idx: accuracy}
    """
    per_class_acc = {}
    
    for class_idx in range(num_classes):
        # Find samples of this class
        mask = (y_true == class_idx)
        if mask.sum() == 0:
            per_class_acc[class_idx] = 0.0
            continue
        
        # Compute accuracy for this class
        class_correct = (y_pred[mask] == class_idx).sum()
        class_total = mask.sum()
        per_class_acc[class_idx] = class_correct / class_total
    
    return per_class_acc


def compute_precision_recall_f1(y_true, y_pred, average='macro'):
    """
    Compute precision, recall, and F1-score
    
    Args:
        average: 'macro' or 'weighted'
    
    Returns:
        tuple: (precision, recall, f1)
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    return precision, recall, f1


def compute_top_k_accuracy(y_true, y_probs, k=5):
    """
    Compute Top-K accuracy
    
    Args:
        y_true: True labels (N,)
        y_probs: Predicted probabilities (N, num_classes)
        k: Top K predictions to consider
    
    Returns:
        float: Top-K accuracy
    """
    if y_probs is None:
        return None
    
    # Get top K predictions
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    
    # Check if true label is in top K
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def compute_confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix
    """
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def compute_all_metrics(y_true, y_pred, y_probs=None, num_classes=102):
    """
    Compute all evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (optional, for Top-K)
        num_classes: Number of classes
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = compute_accuracy(y_true, y_pred)
    
    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(y_true, y_pred, num_classes)
    metrics['per_class_accuracy'] = per_class_acc
    metrics['mean_per_class_accuracy'] = np.mean(list(per_class_acc.values()))
    
    # Precision, Recall, F1 (macro)
    precision_macro, recall_macro, f1_macro = compute_precision_recall_f1(
        y_true, y_pred, average='macro'
    )
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    
    # Precision, Recall, F1 (weighted)
    precision_weighted, recall_weighted, f1_weighted = compute_precision_recall_f1(
        y_true, y_pred, average='weighted'
    )
    metrics['precision_weighted'] = precision_weighted
    metrics['recall_weighted'] = recall_weighted
    metrics['f1_weighted'] = f1_weighted
    
    # Top-5 accuracy (if probabilities provided)
    if y_probs is not None:
        metrics['top5_accuracy'] = compute_top_k_accuracy(y_true, y_probs, k=5)
    else:
        metrics['top5_accuracy'] = None
    
    # Confusion matrix
    metrics['confusion_matrix'] = compute_confusion_matrix(y_true, y_pred, num_classes)
    
    return metrics


def print_metrics_summary(metrics, class_names=None):
    """
    Print a summary of metrics
    """
    print("=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:              {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Mean Per-Class Acc:    {metrics['mean_per_class_accuracy']:.4f} ({metrics['mean_per_class_accuracy']*100:.2f}%)")
    
    print(f"\nMacro Average:")
    print(f"  Precision:             {metrics['precision_macro']:.4f}")
    print(f"  Recall:                {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:              {metrics['f1_macro']:.4f}")
    
    print(f"\nWeighted Average:")
    print(f"  Precision:             {metrics['precision_weighted']:.4f}")
    print(f"  Recall:                {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:              {metrics['f1_weighted']:.4f}")
    
    if metrics['top5_accuracy'] is not None:
        print(f"\nTop-5 Accuracy:          {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
    
    # Show best and worst classes
    per_class_acc = metrics['per_class_accuracy']
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 Best Performing Classes:")
    for i, (class_idx, acc) in enumerate(sorted_classes[:5]):
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        print(f"  {i+1}. {class_name:30s}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\nTop 5 Worst Performing Classes:")
    for i, (class_idx, acc) in enumerate(sorted_classes[-5:]):
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        print(f"  {i+1}. {class_name:30s}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("=" * 80)


if __name__ == "__main__":
    # Test with dummy data
    print("Testing metrics computation...")
    
    np.random.seed(42)
    num_samples = 1000
    num_classes = 102
    
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred = np.random.randint(0, num_classes, num_samples)
    y_probs = np.random.rand(num_samples, num_classes)
    
    metrics = compute_all_metrics(y_true, y_pred, y_probs, num_classes)
    print_metrics_summary(metrics)
    
    print("\nMetrics computation test completed!")

