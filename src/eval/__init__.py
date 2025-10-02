"""
Evaluation utilities for Caltech-101 classification
"""

from .evaluator import Evaluator
from .metrics import (
    compute_all_metrics,
    compute_accuracy,
    compute_per_class_accuracy,
    compute_top_k_accuracy,
    print_metrics_summary
)
from .visualization import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_training_curves,
    plot_top_confused_pairs,
    plot_metrics_comparison
)
from .model_comparison import ModelComparison

__all__ = [
    'Evaluator',
    'compute_all_metrics',
    'compute_accuracy',
    'compute_per_class_accuracy',
    'compute_top_k_accuracy',
    'print_metrics_summary',
    'plot_confusion_matrix',
    'plot_per_class_accuracy',
    'plot_training_curves',
    'plot_top_confused_pairs',
    'plot_metrics_comparison',
    'ModelComparison'
]

