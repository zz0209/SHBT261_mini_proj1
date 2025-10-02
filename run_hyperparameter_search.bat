@echo off
REM ResNet超参数搜索启动脚本 (Windows)
echo ========================================
echo ResNet Hyperparameter Search
echo ========================================
echo.
echo 警告: 这将运行多个实验，可能需要数小时！
echo.
pause

REM 激活虚拟环境
echo.
echo [1/3] 激活虚拟环境...
call miniproj1\Scripts\activate.bat

REM 检查GPU
echo.
echo [2/3] 检查GPU状态...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

REM 开始超参数搜索
echo.
echo [3/3] 开始超参数搜索...
echo.
python src/models/train_resnet_hyperparameter_search.py

echo.
echo ========================================
echo 超参数搜索完成！
echo 结果保存在 results/runs/hyperparameter_search/ 目录
echo ========================================
pause

