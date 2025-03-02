#!/bin/bash
set -e

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies using uv
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt')"

echo "Setup complete! Activate the environment with: source venv/bin/activate"
