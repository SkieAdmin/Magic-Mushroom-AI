#!/bin/bash

echo "============================================================"
echo "  Vegetable Freshness Scanner Backend"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

echo "[1/3] Checking dependencies..."
if ! pip show fastapi &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Dependencies OK"
fi

echo ""
echo "[2/3] Starting FastAPI server..."
echo ""
echo "Server will run on: http://0.0.0.0:9055"
echo "WebSocket endpoint: ws://0.0.0.0:9055/ws/detect"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

python3 main.py
