@echo off
REM ResNet训练启动脚本 (Windows)
echo ========================================
echo ResNet Training for Caltech-101
echo ========================================
echo.

REM 激活虚拟环境
echo [1/3] 激活虚拟环境...
call miniproj1\Scripts\activate.bat

REM 检查GPU
echo.
echo [2/3] 检查GPU状态...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

REM 开始训练
echo.
echo [3/3] 开始训练ResNet...
echo.
python train_resnet.py

echo.
echo ========================================
echo 训练完成！
echo 结果保存在 results/runs/ 目录
echo ========================================
pause

