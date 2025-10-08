"""
Prepare stratified train/val/test splits for Caltech-101 dataset.
Split ratio: 70% train, 15% validation, 15% test
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
import numpy as np

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
RAW_DATA_DIR = Path("data/raw/caltech-101")
SPLITS_DIR = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def get_class_images(data_dir):
    """
    Scan the dataset directory and return a dictionary mapping class names to image paths.
    """
    class_to_images = defaultdict(list)
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(class_dir.glob(ext))
        
        # Store relative paths from project root
        for img_path in image_files:
            # Path should be relative to project root (e.g., data/raw/caltech-101/...)
            relative_path = str(img_path)
            class_to_images[class_name].append(relative_path)
    
    return class_to_images

def stratified_split(class_to_images, train_ratio, val_ratio, test_ratio):
    """
    Perform stratified split on the dataset.
    Returns: train_data, val_data, test_data as lists of (image_path, class_name, class_idx)
    """
    train_data = []
    val_data = []
    test_data = []
    
    # Create class to index mapping
    class_names = sorted(class_to_images.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Total classes: {len(class_names)}")
    print(f"Performing stratified split (train: {train_ratio:.0%}, val: {val_ratio:.0%}, test: {test_ratio:.0%})")
    print("-" * 80)
    
    for class_name in class_names:
        images = class_to_images[class_name]
        class_idx = class_to_idx[class_name]
        
        # Shuffle images for this class
        random.shuffle(images)
        
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Split
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Add to respective lists
        for img in train_images:
            train_data.append({"image_path": img, "class_name": class_name, "class_idx": class_idx})
        for img in val_images:
            val_data.append({"image_path": img, "class_name": class_name, "class_idx": class_idx})
        for img in test_images:
            test_data.append({"image_path": img, "class_name": class_name, "class_idx": class_idx})
        
        print(f"{class_name:30s} | Total: {n_images:4d} | Train: {len(train_images):4d} | Val: {len(val_images):4d} | Test: {len(test_images):4d}")
    
    print("-" * 80)
    print(f"{'TOTAL':30s} | Total: {sum(len(v) for v in class_to_images.values()):4d} | "
          f"Train: {len(train_data):4d} | Val: {len(val_data):4d} | Test: {len(test_data):4d}")
    
    return train_data, val_data, test_data, class_to_idx

def save_split(data, split_name, splits_dir):
    """
    Save split data to JSON file.
    """
    output_path = splits_dir / f"{split_name}.json"
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved {split_name} split to: {output_path}")
    print(f"  - Number of samples: {len(data)}")

def main():
    print("=" * 80)
    print("Caltech-101 Dataset Preparation")
    print("=" * 80)
    print(f"Data directory: {RAW_DATA_DIR}")
    print(f"Output directory: {SPLITS_DIR}")
    print(f"Random seed: {SEED}")
    print()
    
    # Get all class images
    class_to_images = get_class_images(RAW_DATA_DIR)
    
    if not class_to_images:
        print("ERROR: No images found! Please check the data directory.")
        return
    
    # Perform stratified split
    train_data, val_data, test_data, class_to_idx = stratified_split(
        class_to_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )
    
    # Shuffle the splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # Save splits
    save_split(train_data, "train", SPLITS_DIR)
    save_split(val_data, "val", SPLITS_DIR)
    save_split(test_data, "test", SPLITS_DIR)
    
    # Save class mapping
    class_info = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {idx: name for name, idx in class_to_idx.items()},
        "num_classes": len(class_to_idx)
    }
    
    class_info_path = SPLITS_DIR / "class_info.json"
    with open(class_info_path, 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print(f"\nSaved class info to: {class_info_path}")
    print(f"  - Number of classes: {len(class_to_idx)}")
    
    print("\n" + "=" * 80)
    print("Dataset preparation completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()

