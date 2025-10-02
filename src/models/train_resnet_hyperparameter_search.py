"""
Hyperparameter Search for ResNet Models

This script performs hyperparameter tuning for ResNet models.
You can customize the search space and run multiple experiments.
"""

import json
import torch
from pathlib import Path
from itertools import product
import sys
sys.path.append('.')

from src.models.resnet_model import ResNetClassifier, Trainer, predict
from src.data.dataset import get_dataloader, get_class_info
from src.eval import Evaluator


def run_experiment(config, experiment_name):
    """
    Run a single experiment with given configuration
    
    Args:
        config: Configuration dictionary
        experiment_name: Name of the experiment
    
    Returns:
        dict: Results including validation accuracy and test metrics
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load class info
    class_info = get_class_info()
    num_classes = class_info['num_classes']
    class_names = [class_info['idx_to_class'][i] for i in range(num_classes)]
    
    # Create data loaders
    train_loader = get_dataloader(
        split='train',
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        augmentation=config['augmentation'],
        num_workers=config.get('num_workers', 4)
    )
    
    val_loader = get_dataloader(
        split='val',
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        augmentation=False,
        num_workers=config.get('num_workers', 4)
    )
    
    test_loader = get_dataloader(
        split='test',
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        augmentation=False,
        num_workers=config.get('num_workers', 4)
    )
    
    # Create model
    model = ResNetClassifier(
        model_name=config['model_name'],
        num_classes=num_classes,
        pretrained=config['pretrained'],
        freeze_backbone=config.get('freeze_backbone', False)
    )
    
    # Create output directory
    output_dir = Path(f"results/runs/hyperparameter_search/{experiment_name}")
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
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    best_val_acc = checkpoint['val_acc']
    
    # Evaluate on test set
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
        model_name=f"ResNet {config['model_name']} - {experiment_name}",
        config=config
    )
    
    # Return results
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'best_val_acc': best_val_acc,
        'test_acc': metrics['accuracy'],
        'test_top5_acc': metrics['top5_accuracy'],
        'test_f1_macro': metrics['f1_macro'],
        'output_dir': str(output_dir)
    }
    
    return results


def main():
    """
    Main function for hyperparameter search
    """
    print("=" * 80)
    print("RESNET HYPERPARAMETER SEARCH")
    print("=" * 80)
    
    # Define search space
    # You can customize this based on what you want to explore
    
    search_space = {
        # Experiment 1: Compare different ResNet architectures
        'architecture_comparison': [
            {'model_name': 'resnet18', 'batch_size': 64, 'learning_rate': 0.001, 'optimizer': 'adamw'},
            {'model_name': 'resnet34', 'batch_size': 64, 'learning_rate': 0.001, 'optimizer': 'adamw'},
            {'model_name': 'resnet50', 'batch_size': 32, 'learning_rate': 0.001, 'optimizer': 'adamw'},
        ],
        
        # Experiment 2: Compare optimizers (using ResNet18 for speed)
        'optimizer_comparison': [
            {'model_name': 'resnet18', 'batch_size': 64, 'learning_rate': 0.001, 'optimizer': 'adam'},
            {'model_name': 'resnet18', 'batch_size': 64, 'learning_rate': 0.001, 'optimizer': 'adamw'},
            {'model_name': 'resnet18', 'batch_size': 64, 'learning_rate': 0.01, 'optimizer': 'sgd'},
        ],
        
        # Experiment 3: Learning rate tuning
        'learning_rate_tuning': [
            {'model_name': 'resnet50', 'batch_size': 32, 'learning_rate': 0.0001, 'optimizer': 'adamw'},
            {'model_name': 'resnet50', 'batch_size': 32, 'learning_rate': 0.001, 'optimizer': 'adamw'},
            {'model_name': 'resnet50', 'batch_size': 32, 'learning_rate': 0.01, 'optimizer': 'adamw'},
        ],
        
        # Experiment 4: Image size comparison (ablation study)
        'image_size_ablation': [
            {'model_name': 'resnet18', 'batch_size': 64, 'image_size': 64, 'learning_rate': 0.001, 'optimizer': 'adamw'},
            {'model_name': 'resnet18', 'batch_size': 64, 'image_size': 128, 'learning_rate': 0.001, 'optimizer': 'adamw'},
            {'model_name': 'resnet18', 'batch_size': 64, 'image_size': 224, 'learning_rate': 0.001, 'optimizer': 'adamw'},
        ],
        
        # Experiment 5: Data augmentation ablation
        'augmentation_ablation': [
            {'model_name': 'resnet18', 'batch_size': 64, 'learning_rate': 0.001, 'optimizer': 'adamw', 'augmentation': False},
            {'model_name': 'resnet18', 'batch_size': 64, 'learning_rate': 0.001, 'optimizer': 'adamw', 'augmentation': True},
        ],
    }
    
    # Select which experiments to run
    # You can comment out experiments you don't want to run
    experiments_to_run = [
        # 'architecture_comparison',  # Takes longer
        'optimizer_comparison',
        # 'learning_rate_tuning',
        'image_size_ablation',
        'augmentation_ablation',
    ]
    
    print("\nExperiments to run:")
    for exp in experiments_to_run:
        print(f"  - {exp}: {len(search_space[exp])} configurations")
    
    # Base configuration (defaults)
    base_config = {
        'pretrained': True,
        'freeze_backbone': False,
        'image_size': 224,
        'batch_size': 32,
        'augmentation': True,
        'num_workers': 4,
        'epochs': 30,  # Reduced for faster hyperparameter search
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'scheduler': 'plateau',
        'early_stopping_patience': 7,
    }
    
    # Run experiments
    all_results = []
    
    for experiment_group in experiments_to_run:
        configs = search_space[experiment_group]
        
        print(f"\n{'='*80}")
        print(f"Running Experiment Group: {experiment_group}")
        print(f"{'='*80}")
        
        for idx, config_overrides in enumerate(configs):
            # Merge base config with experiment-specific config
            config = base_config.copy()
            config.update(config_overrides)
            
            # Create experiment name
            experiment_name = f"{experiment_group}_{idx+1}"
            
            # Add descriptive suffix
            if 'model_name' in config_overrides:
                experiment_name += f"_{config_overrides['model_name']}"
            if 'optimizer' in config_overrides:
                experiment_name += f"_{config_overrides['optimizer']}"
            if 'learning_rate' in config_overrides:
                experiment_name += f"_lr{config_overrides['learning_rate']}"
            if 'image_size' in config_overrides:
                experiment_name += f"_img{config_overrides['image_size']}"
            if 'augmentation' in config_overrides:
                aug_str = "aug" if config_overrides['augmentation'] else "noaug"
                experiment_name += f"_{aug_str}"
            
            try:
                # Run experiment
                results = run_experiment(config, experiment_name)
                all_results.append(results)
                
                print(f"\n✓ Experiment completed: {experiment_name}")
                print(f"  Best Val Acc: {results['best_val_acc']:.4f}")
                print(f"  Test Acc: {results['test_acc']:.4f}")
                print(f"  Test Top-5 Acc: {results['test_top5_acc']:.4f}")
                
            except Exception as e:
                print(f"\n✗ Experiment failed: {experiment_name}")
                print(f"  Error: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save all results
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETED")
    print("=" * 80)
    
    results_dir = Path("results/runs/hyperparameter_search")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\nSummary of All Experiments:")
    print("-" * 80)
    print(f"{'Experiment':<50} {'Val Acc':<10} {'Test Acc':<10} {'Top-5 Acc':<10}")
    print("-" * 80)
    
    # Sort by test accuracy
    all_results_sorted = sorted(all_results, key=lambda x: x['test_acc'], reverse=True)
    
    for result in all_results_sorted:
        print(f"{result['experiment_name']:<50} "
              f"{result['best_val_acc']:<10.4f} "
              f"{result['test_acc']:<10.4f} "
              f"{result['test_top5_acc']:<10.4f}")
    
    print("-" * 80)
    print(f"\nBest configuration: {all_results_sorted[0]['experiment_name']}")
    print(f"Test Accuracy: {all_results_sorted[0]['test_acc']:.4f}")
    print(f"\nAll results saved to: {results_dir / 'all_results.json'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

