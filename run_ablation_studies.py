"""
Minimal Ablation Studies for Project Requirements

Only runs the essential ablation experiments:
1. Image Size Ablation (64x64, 128x128, 224x224)
2. Data Augmentation Ablation (with/without)
"""

import json
import torch
from pathlib import Path
import sys
sys.path.append('.')

from src.models.resnet_model import ResNetClassifier, Trainer, predict
from src.data.dataset import get_dataloader, get_class_info
from src.eval import Evaluator


def run_experiment(config, experiment_name):
    """Run a single experiment"""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)
    
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
    output_dir = Path(f"results/runs/ablation/{experiment_name}")
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
    
    # Evaluate on test set
    y_true, y_pred, y_probs = predict(model, test_loader, device=device)
    
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
        model_name=experiment_name,
        config=config
    )
    
    return {
        'experiment_name': experiment_name,
        'config': config,
        'test_acc': metrics['accuracy'],
        'test_f1_macro': metrics['f1_macro'],
        'best_epoch': checkpoint['epoch']
    }


def main():
    print("=" * 80)
    print("MINIMAL ABLATION STUDIES")
    print("=" * 80)
    
    # Base configuration
    base_config = {
        'model_name': 'resnet18',  # Use ResNet18 for speed
        'pretrained': True,
        'freeze_backbone': False,
        'batch_size': 64,
        'num_workers': 4,
        'epochs': 20,  # Reduced to 20 for faster results
        'optimizer': 'adamw',
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'scheduler': 'plateau',
        'early_stopping_patience': 5,
    }
    
    # Define ablation experiments
    experiments = [
        # Image Size Ablation
        {'name': '1_img64_aug', 'image_size': 64, 'augmentation': True},
        {'name': '2_img128_aug', 'image_size': 128, 'augmentation': True},
        {'name': '3_img224_aug', 'image_size': 224, 'augmentation': True},
        
        # Data Augmentation Ablation (using 224x224)
        {'name': '4_img224_noaug', 'image_size': 224, 'augmentation': False},
        # Already have img224_aug from above
    ]
    
    print(f"\nTotal experiments to run: {len(experiments)}")
    print("Estimated total time: ~2 hours\n")
    
    results = []
    
    for exp in experiments:
        config = base_config.copy()
        config['image_size'] = exp['image_size']
        config['augmentation'] = exp['augmentation']
        
        try:
            result = run_experiment(config, exp['name'])
            results.append(result)
            
            print(f"\n[SUCCESS] {exp['name']}")
            print(f"  Test Accuracy: {result['test_acc']:.4f}")
            
        except Exception as e:
            print(f"\n[FAILED] {exp['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_path = Path("results/runs/ablation/ablation_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION STUDIES SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<30} {'Test Accuracy':<15} {'F1-Score':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['experiment_name']:<30} {r['test_acc']:<15.4f} {r['test_f1_macro']:<10.4f}")
    print("=" * 80)
    
    print(f"\nAll results saved to: results/runs/ablation/")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

