"""
Fast HOG + SVM using SGDClassifier (much faster)
"""
import json
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time

from skimage.feature import hog
from skimage import color
from sklearn.linear_model import SGDClassifier  # Much faster!
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('.')
from src.eval import Evaluator
from src.models.hog_svm_baseline import HOGFeatureExtractor, load_split_data, load_class_info


def main():
    print("=" * 80)
    print("Fast HOG + SVM using SGDClassifier")
    print("=" * 80)
    
    # Same config as baseline
    config = {
        'image_size': (128, 128),
        'orientations': 9,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
        'svm_type': 'sgd',  # Using SGD for speed
        'alpha': 0.0001,    # L2 regularization
        'max_iter': 1000
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load class info
    class_names, num_classes = load_class_info()
    print(f"\nNumber of classes: {num_classes}")
    
    # Create output directory
    output_dir = Path("results/runs/hog_svm_fast")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize feature extractor
    extractor = HOGFeatureExtractor(
        image_size=config['image_size'],
        orientations=config['orientations'],
        pixels_per_cell=config['pixels_per_cell'],
        cells_per_block=config['cells_per_block']
    )
    
    # Load data splits
    print("\nLoading data splits...")
    train_paths, train_labels = load_split_data('train')
    val_paths, val_labels = load_split_data('val')
    test_paths, test_labels = load_split_data('test')
    
    print(f"  Train: {len(train_paths)} samples")
    print(f"  Val:   {len(val_paths)} samples")
    print(f"  Test:  {len(test_labels)} samples")
    
    # Extract HOG features
    print("\nExtracting HOG features...")
    start_time = time.time()
    
    X_train = extractor.extract_batch(train_paths, desc="Train set")
    X_val = extractor.extract_batch(val_paths, desc="Val set")
    X_test = extractor.extract_batch(test_paths, desc="Test set")
    
    extraction_time = time.time() - start_time
    print(f"\nFeature extraction completed in {extraction_time/60:.2f} minutes")
    print(f"  Train features shape: {X_train.shape}")
    print(f"  Val features shape:   {X_val.shape}")
    print(f"  Test features shape:  {X_test.shape}")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save scaler and extractor
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(extractor, output_dir / "hog_extractor.pkl")
    
    # Train fast SVM using SGDClassifier
    print("\n" + "=" * 80)
    print("Training SVM with SGDClassifier (FAST)")
    print("=" * 80)
    
    classifier = SGDClassifier(
        loss='hinge',  # Linear SVM
        alpha=config['alpha'],
        max_iter=config['max_iter'],
        random_state=42,
        verbose=1,  # Show progress
        n_jobs=-1   # Use all CPU cores
    )
    
    start_time = time.time()
    classifier.fit(X_train, train_labels)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    
    # Save model
    joblib.dump(classifier, output_dir / "svm_classifier.pkl")
    print("Classifier saved")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    
    print("\nPredicting on test set...")
    test_predictions = classifier.predict(X_test)
    
    # SGDClassifier has decision_function
    test_probs = classifier.decision_function(X_test)
    # Convert to probabilities using softmax
    exp_scores = np.exp(test_probs - np.max(test_probs, axis=1, keepdims=True))
    test_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Use evaluator
    evaluator = Evaluator(
        output_dir=output_dir,
        class_names=class_names,
        num_classes=num_classes
    )
    
    metrics = evaluator.evaluate_and_save(
        y_true=test_labels,
        y_pred=test_predictions,
        y_probs=test_probs,
        model_name="HOG + SVM (SGD)",
        config=config
    )
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Training time: {(extraction_time + training_time)/60:.2f} minutes total")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

