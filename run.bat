@echo off
REM AutoMask 启动脚本：免激活 goal 环境，路径跟随脚本所在目录（可整体移动文件夹）
cd /d "%~dp0"

REM 1) 优先用本机 goal 环境的 python（绝对路径）
set "GOAL_PY=C:\Users\11137\miniconda3\envs\goal\python.exe"
if exist "%GOAL_PY%" (
    "%GOAL_PY%" "%~dp0main.py"
    goto :end
)

REM 2) 找不到时回退到 PATH 里的 python（需已 conda activate goal 或系统默认）
where python >nul 2>nul
if %errorlevel%==0 (
    python "%~dp0main.py"
    goto :end
)

echo [错误] 找不到 goal 环境或 python，请检查 conda 环境后手动运行：
echo   conda activate goal
echo   cd /d "%~dp0"
echo   python main.py

:end
pause
