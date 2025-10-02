"""
Quick Start Script for ResNet Training

This is a convenience script to quickly train ResNet models.
You can customize the configuration below and run directly.
"""

import sys
sys.path.append('.')

from src.models.resnet_model import main

if __name__ == "__main__":
    """
    运行ResNet训练
    
    配置说明：
    - 编辑 src/models/resnet_model.py 中的 config 字典来调整超参数
    - 或者直接在这里修改配置
    """
    main()

