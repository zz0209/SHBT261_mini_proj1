# ResNet 训练指南

## 📚 项目结构

```
Proj1/
├── src/
│   ├── models/
│   │   ├── resnet_model.py                    # ResNet主训练脚本
│   │   ├── train_resnet_hyperparameter_search.py  # 超参数搜索脚本
│   │   └── hog_svm_baseline.py                # HOG+SVM基线
│   ├── data/
│   │   └── dataset.py                         # 数据加载器
│   └── eval/
│       ├── evaluator.py                       # 评估器
│       ├── metrics.py                         # 评估指标
│       └── visualization.py                   # 可视化工具
├── train_resnet.py                            # 快速启动脚本
└── requirements.txt                           # 依赖包
```

## 🚀 快速开始

### 1. 激活虚拟环境

```powershell
# Windows PowerShell
.\miniproj1\Scripts\Activate.ps1

# 或者 CMD
miniproj1\Scripts\activate.bat
```

### 2. 检查GPU可用性

```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### 3. 训练ResNet模型

#### 方法一：使用默认配置快速训练

```powershell
python train_resnet.py
```

#### 方法二：直接运行主脚本

```powershell
python src/models/resnet_model.py
```

#### 方法三：使用超参数搜索（推荐用于调参和消融实验）

```powershell
python src/models/train_resnet_hyperparameter_search.py
```

## ⚙️ 配置说明

### 主要超参数

编辑 `src/models/resnet_model.py` 中的 `config` 字典：

```python
config = {
    # 模型配置
    'model_name': 'resnet50',        # 可选: resnet18, resnet34, resnet50, resnet101, resnet152
    'pretrained': True,               # 是否使用预训练权重（ImageNet）
    'freeze_backbone': False,         # 是否冻结主干网络（只训练分类器）
    
    # 数据配置
    'image_size': 224,                # 图像尺寸（ResNet标准为224）
    'batch_size': 32,                 # 批次大小（根据GPU显存调整）
    'augmentation': True,             # 是否使用数据增强
    'num_workers': 4,                 # 数据加载线程数
    
    # 训练配置
    'epochs': 50,                     # 训练轮数
    'optimizer': 'adamw',             # 优化器：adam, adamw, sgd
    'learning_rate': 0.001,           # 学习率
    'weight_decay': 1e-4,             # 权重衰减（L2正则化）
    'momentum': 0.9,                  # SGD动量（仅用于SGD）
    
    # 学习率调度
    'scheduler': 'plateau',           # 调度器：plateau, cosine, none
    
    # 早停
    'early_stopping_patience': 10,    # 多少轮无提升后停止
}
```

### 批次大小建议

根据GPU显存调整：

| GPU显存 | ResNet18/34 | ResNet50 | ResNet101/152 |
|---------|-------------|----------|---------------|
| 4GB     | 64          | 16       | 8             |
| 6GB     | 96          | 32       | 16            |
| 8GB     | 128         | 48       | 24            |
| 12GB+   | 256         | 64       | 32            |

如果遇到 `CUDA out of memory` 错误，请降低 `batch_size`。

## 🔬 超参数调优实验

`train_resnet_hyperparameter_search.py` 脚本支持以下实验：

### 1. 架构对比实验
比较不同ResNet架构（ResNet18, ResNet34, ResNet50）

### 2. 优化器对比实验（消融研究）
比较 Adam, AdamW, SGD 优化器

### 3. 学习率调优
测试不同学习率（0.0001, 0.001, 0.01）

### 4. 图像尺寸消融实验
比较 64×64, 128×128, 224×224 图像尺寸

### 5. 数据增强消融实验
比较有无数据增强的效果

### 自定义实验

编辑 `train_resnet_hyperparameter_search.py` 中的 `experiments_to_run` 列表：

```python
experiments_to_run = [
    'optimizer_comparison',      # 优化器对比
    'image_size_ablation',       # 图像尺寸消融
    'augmentation_ablation',     # 数据增强消融
]
```

运行：

```powershell
python src/models/train_resnet_hyperparameter_search.py
```

## 📊 结果查看

训练完成后，结果保存在：

```
results/runs/
├── resnet_resnet50_bs32_lr0.001/     # 单次训练结果
│   ├── config.json                     # 配置文件
│   ├── best_model.pth                  # 最佳模型权重
│   ├── metrics.json                    # 评估指标
│   ├── training_history.json           # 训练历史
│   └── plots/                          # 可视化图表
│       ├── confusion_matrix.png
│       ├── confusion_matrix_normalized.png
│       ├── per_class_accuracy_all.png
│       ├── per_class_accuracy_top_bottom.png
│       ├── top_confused_pairs.png
│       └── training_curves.png
│
└── hyperparameter_search/              # 超参数搜索结果
    ├── all_results.json                # 所有实验结果汇总
    └── [各个实验文件夹]/
```

### 关键指标

- **accuracy**: 整体准确率
- **top5_accuracy**: Top-5准确率
- **f1_macro**: 宏平均F1分数（适合不平衡数据）
- **f1_weighted**: 加权平均F1分数
- **per_class_accuracy**: 每个类别的准确率

## 💡 训练建议

### 快速测试（约30分钟）

```python
config = {
    'model_name': 'resnet18',
    'batch_size': 64,
    'epochs': 20,
    'early_stopping_patience': 5,
}
```

### 高精度训练（约2-3小时）

```python
config = {
    'model_name': 'resnet50',
    'batch_size': 32,
    'epochs': 50,
    'optimizer': 'adamw',
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
}
```

### 迁移学习最佳实践

1. **阶段一：冻结主干，训练分类器**
   ```python
   config = {
       'freeze_backbone': True,
       'epochs': 10,
       'learning_rate': 0.001,
   }
   ```

2. **阶段二：解冻主干，微调整个网络**
   ```python
   config = {
       'freeze_backbone': False,
       'epochs': 40,
       'learning_rate': 0.0001,  # 更小的学习率
   }
   ```

## 🐛 常见问题

### 1. CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决方法**：
- 减小 `batch_size`
- 减小 `image_size`
- 使用更小的模型（如ResNet18）

### 2. 训练过慢

**解决方法**：
- 增加 `num_workers`（数据加载线程）
- 减小 `image_size`
- 使用更小的模型

### 3. 验证集准确率不提升

**解决方法**：
- 降低学习率
- 增加数据增强
- 增加权重衰减（防止过拟合）
- 使用预训练权重

### 4. 过拟合（训练准确率高，验证准确率低）

**解决方法**：
- 增加 `weight_decay`
- 启用 `augmentation`
- 使用 Dropout（需修改模型）
- 减少训练轮数

## 📈 实验记录建议

为了完成报告，建议记录以下信息：

| 实验 | 模型 | 优化器 | 学习率 | 批次大小 | 图像尺寸 | 数据增强 | 验证准确率 | 测试准确率 | Top-5准确率 |
|------|------|--------|--------|----------|----------|----------|------------|------------|-------------|
| 1    |      |        |        |          |          |          |            |            |             |
| 2    |      |        |        |          |          |          |            |            |             |

所有实验结果都会自动保存在 `results/runs/hyperparameter_search/all_results.json`。

## 🎯 完成任务检查清单

- [ ] 训练至少3种不同的ResNet配置
- [ ] 完成优化器对比实验（Adam vs SGD）
- [ ] 完成图像尺寸消融实验（64 vs 128 vs 224）
- [ ] 完成数据增强消融实验（有 vs 无）
- [ ] 保存所有实验结果和可视化图表
- [ ] 分析混淆矩阵，找出容易混淆的类别
- [ ] 分析每类准确率，找出困难类别
- [ ] 记录训练时间和资源使用

## 📝 报告建议

### 方法部分
- ResNet架构说明
- 迁移学习策略
- 数据增强方法
- 训练超参数

### 结果部分
- 不同模型性能对比表格
- 混淆矩阵可视化
- 训练曲线
- 每类准确率分析

### 消融研究
- 图像尺寸影响
- 数据增强影响
- 优化器影响

### 经验总结
- 哪个配置效果最好
- 哪些类别最难分类
- 改进建议

祝训练顺利！🚀

