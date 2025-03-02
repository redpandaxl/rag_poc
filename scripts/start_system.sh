#!/bin/bash

# Kill all existing ragstack processes
echo "Stopping any existing ragstack processes..."
pkill -f "python.*ragstack" || true
pkill -f "streamlit run" || true
pkill -f "uvicorn" || true
pkill -f "run_dashboard.py" || true

# Wait for processes to terminate
sleep 2

# Start the system using run.py
echo "Starting RAGStack system..."
cd "$(dirname "$0")/.."
python run.py