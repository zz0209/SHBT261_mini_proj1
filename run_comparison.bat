@echo off
REM 模型对比脚本 (Windows)
echo ========================================
echo Model Comparison Tool
echo ========================================
echo.

REM 激活虚拟环境
echo [1/2] 激活虚拟环境...
call miniproj1\Scripts\activate.bat

REM 运行对比
echo.
echo [2/2] 对比所有模型结果...
echo.
python compare_models.py

echo.
echo ========================================
echo 对比完成！
echo 结果保存在 results/comparison/ 目录
echo ========================================
pause

