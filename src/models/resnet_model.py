"""
ResNet Model for Caltech-101 Classification

This script implements ResNet-based deep learning models using transfer learning.
Supports ResNet18, ResNet34, ResNet50, etc.
"""

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import numpy as np

import sys
sys.path.append('.')
from src.data.dataset import get_dataloader, get_class_info
from src.eval import Evaluator


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier with transfer learning
    """
    
    def __init__(self, model_name='resnet18', num_classes=102, pretrained=True, freeze_backbone=False):
        """
        Args:
            model_name: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone layers (only train classifier)
        """
        super(ResNetClassifier, self).__init__()
        
        import torchvision.models as models
        
        # Get the ResNet model
        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        
        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(model_dict.keys())}")
        
        # Load pretrained model
        if pretrained:
            print(f"Loading pretrained {model_name}...")
            self.backbone = model_dict[model_name](weights='IMAGENET1K_V1')
        else:
            print(f"Initializing {model_name} from scratch...")
            self.backbone = model_dict[model_name](weights=None)
        
        # Freeze backbone if requested
        if freeze_backbone:
            print("Freezing backbone layers...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        self.model_name = model_name
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for fine-tuning"""
        print("Unfreezing backbone layers...")
        for param in self.backbone.parameters():
            param.requires_grad = True


class Trainer:
    """
    Trainer class for ResNet models
    """
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        """
        Args:
            model: ResNet model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
    
    def _get_optimizer(self):
        """Create optimizer based on config"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")
    
    def _get_scheduler(self):
        """Create learning rate scheduler based on config"""
        scheduler_name = self.config.get('scheduler', 'plateau').lower()
        
        if scheduler_name == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3
            )
        elif scheduler_name == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 50),
                eta_min=1e-6
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Update progress bar
            current_loss = running_loss / total_samples
            current_acc = running_corrects.double() / total_samples
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                # Update progress bar
                current_loss = running_loss / total_samples
                current_acc = running_corrects.double() / total_samples
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.4f}'
                })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def train(self, num_epochs, output_dir):
        """
        Complete training loop
        
        Args:
            num_epochs: Number of epochs to train
            output_dir: Directory to save checkpoints
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Optimizer: {self.config.get('optimizer', 'adam')}")
        print(f"Learning Rate: {self.config.get('learning_rate', 0.001)}")
        print(f"Batch Size: {self.config.get('batch_size', 32)}")
        print(f"Early Stopping Patience: {self.config.get('early_stopping_patience', 10)}")
        print("=" * 80 + "\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }
                torch.save(checkpoint, output_dir / 'best_model.pth')
                print(f"  [*] Best model saved! (Val Acc: {val_acc:.4f})")
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epochs (Best: {self.best_val_acc:.4f} at epoch {self.best_epoch+1})")
            
            # Early stopping
            early_stopping_patience = self.config.get('early_stopping_patience', 10)
            if self.epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                break
            
            print("-" * 80)
        
        training_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Total Training Time: {training_time/60:.2f} minutes")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch+1})")
        print("=" * 80 + "\n")
        
        # Save training history
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def predict(model, data_loader, device='cuda'):
    """
    Make predictions on a dataset
    
    Returns:
        y_true, y_pred, y_probs
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Predicting"):
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    """
    Main training function
    """
    print("=" * 80)
    print("ResNet Training for Caltech-101")
    print("=" * 80)
    
    # Configuration
    config = {
        # Model
        'model_name': 'resnet50',  # resnet18, resnet34, resnet50, resnet101, resnet152
        'pretrained': True,
        'freeze_backbone': False,  # Set to True for faster initial training
        
        # Data
        'image_size': 224,  # ResNet standard input size
        'batch_size': 32,
        'augmentation': True,
        'num_workers': 4,
        
        # Training
        'epochs': 50,
        'optimizer': 'adamw',  # adam, adamw, sgd
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'momentum': 0.9,  # for SGD
        
        # Scheduler
        'scheduler': 'plateau',  # plateau, cosine, none
        
        # Early stopping
        'early_stopping_patience': 10,
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load class info
    class_info = get_class_info()
    num_classes = class_info['num_classes']
    class_names = [class_info['idx_to_class'][i] for i in range(num_classes)]
    print(f"\nNumber of classes: {num_classes}")
    
    # Create data loaders
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    train_loader = get_dataloader(
        split='train',
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        augmentation=config['augmentation'],
        num_workers=config['num_workers']
    )
    
    val_loader = get_dataloader(
        split='val',
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        augmentation=False,
        num_workers=config['num_workers']
    )
    
    test_loader = get_dataloader(
        split='test',
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        augmentation=False,
        num_workers=config['num_workers']
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # Create model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    
    model = ResNetClassifier(
        model_name=config['model_name'],
        num_classes=num_classes,
        pretrained=config['pretrained'],
        freeze_backbone=config['freeze_backbone']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create output directory
    model_name_clean = config['model_name'].replace('/', '_')
    output_dir = Path(f"results/runs/resnet_{model_name_clean}_bs{config['batch_size']}_lr{config['learning_rate']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train model
    history = trainer.train(
        num_epochs=config['epochs'],
        output_dir=output_dir
    )
    
    # Load best model for evaluation
    print("\n" + "=" * 80)
    print("LOADING BEST MODEL FOR FINAL EVALUATION")
    print("=" * 80)
    
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded model from epoch {checkpoint['epoch']+1}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    
    y_true, y_pred, y_probs = predict(model, test_loader, device=device)
    
    # Use evaluator for comprehensive metrics
    evaluator = Evaluator(
        output_dir=output_dir,
        class_names=class_names,
        num_classes=num_classes
    )
    
    metrics = evaluator.evaluate_and_save(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        history=history,
        model_name=f"ResNet {config['model_name']}",
        config=config
    )
    
    print("\n" + "=" * 80)
    print("ALL DONE!")
    print("=" * 80)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Test Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

