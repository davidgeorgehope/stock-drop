#!/bin/bash

# This script automates the startup of the local development environment for the
# AI-Powered Observability Demo Generator on macOS.
#
# It performs the following steps:
# 1. Sets up and starts the Python/FastAPI backend in the background.
# 2. Sets up and starts the Node.js/React frontend in the foreground.
# 3. Sets up a trap to automatically shut down the backend when you stop the script (Ctrl+C).
#
# Prerequisites:
# - Node.js and npm
# - Python 3.9+ and pip

set -e # Exit immediately if a command fails.

# --- Cleanup Function ---
# This function is triggered on script exit to ensure the backend process is terminated.
cleanup() {
    echo -e "\n\nGracefully shutting down services..."
    if [ -n "$BACKEND_PID" ]; then
        echo "Stopping backend server (PID: $BACKEND_PID)..."
        # Kill the process group to ensure all child processes (like reload workers) are terminated.
        kill -9 -$BACKEND_PID > /dev/null 2>&1
    fi
    echo "Shutdown complete."
    exit 0
}

# Trap signals to ensure cleanup runs
trap cleanup SIGINT SIGTERM EXIT

# --- Backend ---
println() { printf "%s\n" "$*"; }

echo "--- Setting up Python backend ---"
cd backend

echo "Loading environment from ../.env if present..."
if [ -f ../.env ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' ../.env | xargs -I{} echo {})
fi

# Ensure SQLite local path for development
export SQLITE_DB_PATH=${SQLITE_DB_PATH:-"$(pwd)/data/stockdrop.db"}
mkdir -p "$(dirname "$SQLITE_DB_PATH")"

# Enable dev signals UI for frontend
export VITE_DEV_SIGNALS=1

# Allow cache pre-population but with smart rate limiting
# Commenting out to allow SQLite population on startup
# export SKIP_CACHE_PREPOPULATION=1

echo "Activating virtual environment..."
if [ -f ../.venv/bin/activate ]; then
    source ../.venv/bin/activate
elif [ -f ../venv/bin/activate ]; then
    source ../venv/bin/activate
elif [ -f ../myenv/bin/activate ]; then
    source ../myenv/bin/activate
else
    echo "No virtual environment found. Creating one..."
    python3 -m venv ../.venv
    source ../.venv/bin/activate
fi

echo "Installing Python dependencies from requirements.txt..."
python -m pip install --disable-pip-version-check --quiet -r requirements.txt

echo "Starting backend server in the background..."
# set -m allows job control, which is important for killing the process group
set -m
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
set +m
cd ..

# Give the backend a moment to initialize
sleep 3
echo "Backend server started with PID: $BACKEND_PID. API is available at http://localhost:8000"


# --- Frontend ---
echo -e "\n--- Setting up React frontend ---"
cd frontend

echo "Installing Node.js dependencies from package.json..."
npm install --silent

echo -e "\nStarting frontend dev server... (Press Ctrl+C to stop everything)"
# The 'trap' will handle shutting down the backend when this command exits.
VITE_DEV_SIGNALS=1 npm run dev 