@echo off
echo ============================================================
echo   Vegetable Freshness Scanner Backend
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo Dependencies OK
)

echo.
echo [2/3] Starting FastAPI server...
echo.
echo Server will run on: http://0.0.0.0:9055
echo WebSocket endpoint: ws://0.0.0.0:9055/ws/detect
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

python main.py

pause
