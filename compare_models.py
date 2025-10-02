"""
Model Comparison Script

Run this after training all models to generate comprehensive comparisons.
"""

import sys
sys.path.append('.')

from src.eval.model_comparison import main

if __name__ == "__main__":
    """
    对比所有已训练的模型
    
    自动查找并对比：
    - HOG + SVM
    - ResNet
    - EfficientNet
    - 其他任何已训练的模型
    """
    print("\n正在对比所有模型...")
    print("=" * 80)
    main()

