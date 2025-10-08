"""
PyTorch Dataset and DataLoader for Caltech-101
"""

import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Caltech101Dataset(Dataset):
    """
    Caltech-101 Dataset class
    """
    
    def __init__(self, split_file, root_dir=".", transform=None):
        """
        Args:
            split_file: Path to the JSON file containing split data
            root_dir: Root directory of the project
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Load split data
        with open(split_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {split_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        img_path = self.root_dir / item['image_path']
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = item['class_idx']
        
        return image, label


def get_transforms(image_size=128, augmentation=True, split='train'):
    """
    Get image transforms for different splits
    
    Args:
        image_size: Target image size (will create square images)
        augmentation: Whether to apply data augmentation (only for train)
        split: 'train', 'val', or 'test'
    
    Returns:
        torchvision.transforms.Compose object
    """
    
    if split == 'train' and augmentation:
        # Training with augmentation
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.1)),  # Slightly larger for random crop
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        # Validation/Test or training without augmentation
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def get_dataloader(split='train', batch_size=32, image_size=128, 
                   augmentation=True, num_workers=4, shuffle=None):
    """
    Create a DataLoader for the specified split
    
    Args:
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        image_size: Target image size
        augmentation: Whether to apply data augmentation (only affects train)
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data (default: True for train, False otherwise)
    
    Returns:
        DataLoader object
    """
    
    # Set default shuffle behavior
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Get transforms
    transform = get_transforms(image_size=image_size, 
                              augmentation=augmentation, 
                              split=split)
    
    # Create dataset
    split_file = f"data/splits/{split}.json"
    dataset = Caltech101Dataset(split_file=split_file, 
                               root_dir=".", 
                               transform=transform)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')  # Drop last incomplete batch for training
    )
    
    return dataloader


def get_class_info():
    """
    Load class information
    
    Returns:
        Dictionary with class_to_idx, idx_to_class, and num_classes
    """
    with open("data/splits/class_info.json", 'r') as f:
        class_info = json.load(f)
    
    # Convert idx_to_class keys back to integers
    class_info['idx_to_class'] = {int(k): v for k, v in class_info['idx_to_class'].items()}
    
    return class_info


if __name__ == "__main__":
    """
    Test the dataset and dataloader
    """
    print("=" * 80)
    print("Testing Caltech-101 Dataset and DataLoader")
    print("=" * 80)
    
    # Load class info
    class_info = get_class_info()
    print(f"\nNumber of classes: {class_info['num_classes']}")
    print(f"Sample classes: {list(class_info['class_to_idx'].keys())[:5]}")
    
    # Test different configurations
    configs = [
        {'split': 'train', 'image_size': 128, 'augmentation': True, 'batch_size': 16},
        {'split': 'train', 'image_size': 64, 'augmentation': False, 'batch_size': 32},
        {'split': 'val', 'image_size': 128, 'augmentation': False, 'batch_size': 32},
        {'split': 'test', 'image_size': 128, 'augmentation': False, 'batch_size': 32},
    ]
    
    for config in configs:
        print("\n" + "-" * 80)
        print(f"Config: {config}")
        
        dataloader = get_dataloader(**config, num_workers=0)  # num_workers=0 for testing
        
        # Get one batch
        images, labels = next(iter(dataloader))
        
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Sample labels: {labels[:5].tolist()}")
        print(f"Total batches: {len(dataloader)}")
    
    print("\n" + "=" * 80)
    print("Dataset test completed successfully!")
    print("=" * 80)

