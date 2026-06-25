@echo off
title MT5 Quant OS
cd /d "%~dp0"
echo.
echo  Starting MT5 Quant OS...
echo.
python start.py
if errorlevel 1 (
    echo.
    echo  Agent exited with an error.
    pause
)