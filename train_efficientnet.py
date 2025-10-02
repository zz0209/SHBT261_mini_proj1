"""
Quick Start Script for EfficientNet Training

This is a convenience script to quickly train EfficientNet models.
"""

import sys
sys.path.append('.')

from src.models.efficientnet_model import main

if __name__ == "__main__":
    """
    运行EfficientNet训练
    
    配置说明：
    - 编辑 src/models/efficientnet_model.py 中的 config 字典来调整超参数
    """
    main()

