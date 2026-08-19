# build.bat - 用 PyInstaller 把 AutoMask 打包成独立 exe (CPU 版)
#
# 前提：在 goal 环境中执行本脚本。
#   conda activate goal
#   build.bat
#
# 说明：
#   - ultralytics / torch 大量使用动态导入且带数据文件（yaml 配置等），
#     必须用 --collect-all（含子模块+数据+二进制），否则打包后可能起不来。
#   - 打包后的 exe 体积较大（torch 本身就很重）。若想显著缩小体积，
#     可在【仅含 CPU 版 torch】的干净环境中打包（见 README"打包"一节）。
#   - --windowed 表示无控制台窗口（GUI 程序）。

pyinstaller ^
  --name AutoMask ^
  --windowed ^
  --noconfirm ^
  --collect-all ultralytics ^
  --collect-all torch ^
  --hidden-import cv2 ^
  --hidden-import numpy ^
  --hidden-import PIL ^
  --hidden-import yaml ^
  main.py

echo.
echo 打包完成，exe 位于 dist\AutoMask\AutoMask.exe
pause
