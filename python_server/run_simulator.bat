@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo.
echo ============================================
echo   AI Glasses - Video Simulator
echo ============================================
echo.
echo Starting...
echo.
py video_simulator.py
pause
