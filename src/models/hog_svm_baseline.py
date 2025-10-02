"""
HOG + SVM Baseline for Caltech-101 Classification

This script implements a traditional machine learning baseline using:
- HOG (Histogram of Oriented Gradients) for feature extraction
- SVM (Support Vector Machine) for classification
"""

import json
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time

from skimage.feature import hog
from skimage import color, transform
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append('.')
from src.eval import Evaluator


class HOGFeatureExtractor:
    """
    Extract HOG features from images
    """
    
    def __init__(self, image_size=(128, 128), orientations=9, 
                 pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Args:
            image_size: Resize images to this size
            orientations: Number of orientation bins
            pixels_per_cell: Size of a cell
            cells_per_block: Number of cells in each block
        """
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def extract_single(self, image_path):
        """
        Extract HOG features from a single image
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.image_size)
        img_array = np.array(img)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            img_gray = color.rgb2gray(img_array)
        else:
            img_gray = img_array
        
        # Extract HOG features
        features = hog(
            img_gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        
        return features
    
    def extract_batch(self, image_paths, desc="Extracting HOG features"):
        """
        Extract HOG features from multiple images
        """
        features_list = []
        
        for img_path in tqdm(image_paths, desc=desc):
            features = self.extract_single(img_path)
            features_list.append(features)
        
        return np.array(features_list)


def load_split_data(split_name):
    """
    Load image paths and labels from split file
    """
    split_file = f"data/splits/{split_name}.json"
    
    with open(split_file, 'r') as f:
        data = json.load(f)
    
    image_paths = [item['image_path'] for item in data]
    labels = [item['class_idx'] for item in data]
    
    return image_paths, np.array(labels)


def load_class_info():
    """
    Load class information
    """
    with open("data/splits/class_info.json", 'r') as f:
        class_info = json.load(f)
    
    class_names = [class_info['idx_to_class'][str(i)] for i in range(class_info['num_classes'])]
    
    return class_names, class_info['num_classes']


def main():
    print("=" * 80)
    print("HOG + SVM Baseline for Caltech-101")
    print("=" * 80)
    
    # Configuration
    config = {
        'image_size': (128, 128),
        'orientations': 9,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
        'svm_kernel': 'linear',  # 'linear' or 'rbf'
        'svm_C': 1.0,
        'grid_search': False  # Set to True for hyperparameter tuning (slower)
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load class info
    class_names, num_classes = load_class_info()
    print(f"\nNumber of classes: {num_classes}")
    
    # Create output directory
    output_dir = Path("results/runs/hog_svm_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize feature extractor
    print("\n" + "=" * 80)
    print("STEP 1: Feature Extraction")
    print("=" * 80)
    
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
    print("Scaler and extractor saved")
    
    # Train SVM
    print("\n" + "=" * 80)
    print("STEP 2: Training SVM Classifier")
    print("=" * 80)
    
    if config['grid_search']:
        print("\nPerforming grid search for hyperparameter tuning...")
        param_grid = {
            'C': [0.1, 1.0, 10.0],
        }
        
        if config['svm_kernel'] == 'linear':
            svm = LinearSVC(max_iter=5000, random_state=42)
        else:
            svm = SVC(kernel=config['svm_kernel'], random_state=42, probability=True)
            param_grid['gamma'] = ['scale', 'auto', 0.001, 0.01]
        
        grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, train_labels)
        
        classifier = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    else:
        print(f"\nTraining {config['svm_kernel']} SVM with C={config['svm_C']}...")
        
        if config['svm_kernel'] == 'linear':
            # dual=False is faster when n_samples > n_features
            classifier = LinearSVC(
                C=config['svm_C'], 
                max_iter=5000, 
                random_state=42,
                dual=False,  # Faster for large datasets
                verbose=1     # Show progress
            )
        else:
            classifier = SVC(
                kernel=config['svm_kernel'],
                C=config['svm_C'],
                random_state=42,
                probability=True  # Enable probability estimates
            )
        
        start_time = time.time()
        classifier.fit(X_train, train_labels)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time/60:.2f} minutes")
    
    # Save model
    joblib.dump(classifier, output_dir / "svm_classifier.pkl")
    print("Classifier saved")
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("STEP 3: Evaluation on Validation Set")
    print("=" * 80)
    
    print("\nPredicting on validation set...")
    val_predictions = classifier.predict(X_val)
    
    # Get probabilities if available
    try:
        val_probs = classifier.decision_function(X_val)
        # Convert decision function to probabilities (softmax)
        val_probs = np.exp(val_probs) / np.sum(np.exp(val_probs), axis=1, keepdims=True)
    except:
        val_probs = None
        print("Note: Probability estimates not available for this classifier")
    
    val_accuracy = (val_predictions == val_labels).mean()
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("STEP 4: Final Evaluation on Test Set")
    print("=" * 80)
    
    print("\nPredicting on test set...")
    test_predictions = classifier.predict(X_test)
    
    # Get probabilities
    try:
        if hasattr(classifier, 'predict_proba'):
            test_probs = classifier.predict_proba(X_test)
        else:
            test_probs = classifier.decision_function(X_test)
            test_probs = np.exp(test_probs) / np.sum(np.exp(test_probs), axis=1, keepdims=True)
    except:
        test_probs = None
    
    # Use evaluator for comprehensive metrics
    evaluator = Evaluator(
        output_dir=output_dir,
        class_names=class_names,
        num_classes=num_classes
    )
    
    # No training history for classical ML, but we can create a simple record
    history = {
        'extraction_time_minutes': extraction_time / 60,
        'training_time_minutes': training_time / 60 if 'training_time' in locals() else 0,
        'total_time_minutes': (extraction_time + (training_time if 'training_time' in locals() else 0)) / 60
    }
    
    with open(output_dir / "timing.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Complete evaluation
    metrics = evaluator.evaluate_and_save(
        y_true=test_labels,
        y_pred=test_predictions,
        y_probs=test_probs,
        model_name="HOG + SVM Baseline",
        config=config
    )
    
    print("\n" + "=" * 80)
    print("HOG + SVM Baseline Training Completed!")
    print("=" * 80)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

