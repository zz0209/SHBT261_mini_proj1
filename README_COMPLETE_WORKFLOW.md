# Caltech-101 图像分类完整工作流程

## 📚 项目概览

本项目实现了三种不同的图像分类方法：
1. **HOG + SVM** (经典机器学习)
2. **ResNet** (深度学习 - CNN)
3. **EfficientNet** (深度学习 - 先进CNN)

所有模型都在**Caltech-101数据集**上训练和评估（102类，约9000张图像）。

---

## 🗂️ 项目结构

```
Proj1/
├── data/
│   ├── raw/caltech-101/          # 原始图像数据
│   └── splits/                    # 训练/验证/测试分割
│
├── src/
│   ├── data/
│   │   └── dataset.py             # PyTorch数据加载器
│   ├── eval/
│   │   ├── evaluator.py           # 评估器
│   │   ├── metrics.py             # 评估指标
│   │   ├── visualization.py       # 可视化工具
│   │   └── model_comparison.py    # 模型对比工具
│   └── models/
│       ├── hog_svm_baseline.py    # HOG+SVM模型
│       ├── resnet_model.py        # ResNet模型
│       └── efficientnet_model.py  # EfficientNet模型
│
├── results/
│   ├── runs/                      # 各个模型的训练结果
│   └── comparison/                # 模型对比结果
│
├── train_resnet.py                # ResNet快速启动
├── train_efficientnet.py          # EfficientNet快速启动
├── compare_models.py              # 模型对比脚本
│
├── run_resnet_training.bat        # Windows批处理
├── run_efficientnet_training.bat  # Windows批处理
└── run_comparison.bat             # Windows批处理
```

---

## 🚀 完整训练流程

### 步骤1：训练HOG+SVM基线模型

```powershell
.\miniproj1\Scripts\Activate.ps1
python src/models/hog_svm_baseline.py
```

**时间**: 约10-15分钟  
**输出**: `results/runs/hog_svm_baseline/`

### 步骤2：训练ResNet模型

```powershell
.\miniproj1\Scripts\Activate.ps1
python train_resnet.py
```

或者双击运行: `run_resnet_training.bat`

**时间**: 约1-2小时（50轮）  
**输出**: `results/runs/resnet_resnet50_bs32_lr0.001/`

### 步骤3：训练EfficientNet模型

```powershell
.\miniproj1\Scripts\Activate.ps1
python train_efficientnet.py
```

或者双击运行: `run_efficientnet_training.bat`

**时间**: 约1-2小时（50轮）  
**输出**: `results/runs/efficientnet_efficientnet_b0_bs32_lr0.001/`

### 步骤4：对比所有模型

```powershell
.\miniproj1\Scripts\Activate.ps1
python compare_models.py
```

或者双击运行: `run_comparison.bat`

**输出**: `results/comparison/`
- 对比表格 (CSV + TXT)
- 对比可视化图表
- 详细分析报告

---

## 📊 评估指标

所有模型都会自动生成以下评估指标：

### 核心指标
- ✅ **Accuracy** - 整体准确率
- ✅ **Mean Per-Class Accuracy** - 每类平均准确率（处理类别不平衡）
- ✅ **Precision (Macro & Weighted)** - 精确率
- ✅ **Recall (Macro & Weighted)** - 召回率
- ✅ **F1-Score (Macro & Weighted)** - F1分数
- ✅ **Top-5 Accuracy** - Top-5准确率（深度学习模型）

### 可视化
- ✅ **Confusion Matrix** - 混淆矩阵（原始 + 归一化）
- ✅ **Per-Class Accuracy** - 每类准确率柱状图
- ✅ **Training Curves** - 训练/验证损失和准确率曲线
- ✅ **Top Confused Pairs** - 最容易混淆的类别对
- ✅ **Model Comparison Charts** - 模型对比图表

---

## 🎯 模型配置

### ResNet50
```python
config = {
    'model_name': 'resnet50',
    'pretrained': True,          # 使用ImageNet预训练权重
    'image_size': 224,
    'batch_size': 32,
    'augmentation': True,
    'epochs': 50,
    'optimizer': 'adamw',
    'learning_rate': 0.001,
    'scheduler': 'plateau',
    'early_stopping_patience': 10,
}
```

### EfficientNet-B0
```python
config = {
    'model_name': 'efficientnet_b0',
    'pretrained': True,          # 使用ImageNet预训练权重
    'image_size': 224,
    'batch_size': 32,
    'augmentation': True,
    'epochs': 50,
    'optimizer': 'adamw',
    'learning_rate': 0.001,
    'scheduler': 'plateau',
    'early_stopping_patience': 10,
}
```

### HOG+SVM
```python
config = {
    'image_size': (128, 128),
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'svm_kernel': 'linear',
    'svm_C': 1.0,
}
```

---

## 📈 结果文件说明

### 单个模型结果 (`results/runs/[model_name]/`)

```
model_name/
├── config.json                    # 模型配置
├── best_model.pth                 # 最佳模型权重（深度学习）
├── metrics.json                   # 所有评估指标
├── training_history.json          # 训练历史
└── plots/                         # 可视化图表
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── per_class_accuracy_all.png
    ├── per_class_accuracy_top_bottom.png
    ├── top_confused_pairs.png
    └── training_curves.png
```

### 模型对比结果 (`results/comparison/`)

```
comparison/
├── metrics_comparison.csv         # 指标对比表格（CSV）
├── metrics_comparison.txt         # 指标对比表格（文本）
├── comparison_summary.txt         # 详细分析报告
├── metrics_comparison_bars.png    # 关键指标对比柱状图
├── metrics_heatmap.png           # 所有指标热力图
├── training_curves_comparison.png # 训练曲线对比
└── per_class_accuracy_comparison.png  # 每类准确率对比
```

---

## 🔧 自定义训练

### 修改ResNet配置

编辑 `src/models/resnet_model.py` 第115-136行：

```python
config = {
    'model_name': 'resnet18',      # 改为更小的模型以加快训练
    'batch_size': 64,              # 增大批次大小
    'epochs': 30,                  # 减少训练轮数
    # ... 其他配置
}
```

### 修改EfficientNet配置

编辑 `src/models/efficientnet_model.py` 第515-536行：

```python
config = {
    'model_name': 'efficientnet_b1',  # 使用更大的EfficientNet
    'batch_size': 16,                 # 减小批次大小
    # ... 其他配置
}
```

---

## 🐛 常见问题

### 1. GPU内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减小 `batch_size` (如改为16或8)
- 使用更小的模型 (ResNet18 instead of ResNet50)
- 减小 `image_size`

### 2. 训练过慢

**解决方案**:
- 增加 `num_workers` (数据加载线程数)
- 减小 `image_size`
- 使用更小的模型

### 3. 模型过拟合

**症状**: 训练准确率高，验证准确率低

**解决方案**:
- 增加 `weight_decay`
- 启用 `augmentation=True`
- 减少 `epochs`
- 使用 `early_stopping`

---

## 📝 报告建议

### 方法部分
- 描述三种方法的原理
- 说明数据预处理和增强策略
- 列出超参数配置

### 结果部分
- 展示模型对比表格
- 包含关键可视化图表
- 分析每类准确率

### 讨论部分
- 分析哪个模型效果最好，为什么
- 讨论最难分类的类别
- 提出改进建议

### 附录
- 完整的评估指标表格
- 混淆矩阵
- 训练曲线

---

## 📊 预期结果

基于快速测试，预期性能：

| 模型 | 预期准确率 | 训练时间 |
|------|-----------|---------|
| HOG + SVM | 50-60% | 10-15分钟 |
| ResNet50 | 80-85% | 1-2小时 |
| EfficientNet-B0 | 82-87% | 1-2小时 |

*注: 实际结果可能因超参数和训练轮数而异*

---

## 🎓 学习要点

1. **迁移学习的威力**: 使用预训练权重显著提升性能
2. **数据增强的重要性**: 有效防止过拟合
3. **架构选择**: EfficientNet通常比ResNet更高效
4. **经典ML vs 深度学习**: 深度学习在图像分类上明显优于传统方法

---

## 📞 需要帮助？

查看以下文件获取更多信息：
- `README_RESNET.md` - ResNet详细指南
- `src/models/resnet_model.py` - ResNet实现
- `src/models/efficientnet_model.py` - EfficientNet实现
- `src/eval/model_comparison.py` - 模型对比工具

祝训练顺利！ 🚀

