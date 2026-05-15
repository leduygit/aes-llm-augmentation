#!/bin/bash
set -e

echo "=================================="
echo "IELTS AES Environment Setup"
echo "=================================="

PROJECT_DIR="/kaggle/working/llm-aes-aug"

echo ""
echo "[1/5] Checking Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 not found. Installing..."
    apt update
    apt install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt update
    apt install -y python3.11 python3.11-venv python3.11-dev
    echo "Python 3.11 installed"
else
    echo "Python 3.11 already installed"
fi

echo "Using: $(python3.11 --version)"

echo ""
echo "[2/5] Setting up project directory..."
cd "$PROJECT_DIR"

echo ""
echo "[3/5] Cleaning up old virtual environment..."
if [ -d "venv" ]; then
    rm -rf venv
    echo "Old venv removed"
else
    echo "No existing venv found"
fi

echo ""
echo "[4/5] Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate
echo "Virtual environment created and activated"

echo ""
echo "[5/5] Installing packages..."
python -m pip install --upgrade pip==24.1.2 setuptools==75.2.0 wheel==0.45.1

echo ""
echo "Installing PyTorch 2.6.0 with CUDA 12.4 support..."
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import openai; print(f'OpenAI: {openai.__version__}')"
echo ""
echo "Environment setup successful."
echo ""
echo "To activate the environment later, run:"
echo "  cd $PROJECT_DIR && source venv/bin/activate"
echo ""
