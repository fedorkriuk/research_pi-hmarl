#\!/bin/bash
# Setup script for PI-HMARL development environment

set -e

echo "=========================================="
echo "PI-HMARL Environment Setup"
echo "=========================================="

# Check if running in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: Not running in a virtual environment."
    echo "It's recommended to use a virtual environment."
    echo "Create one with: python3 -m venv venv && source venv/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ \! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
REQUIRED_VERSION="3.8"

if (( $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) )); then
    echo "Error: Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version: $PYTHON_VERSION"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing PI-HMARL in development mode..."
pip install -e .

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/real_parameters
mkdir -p data/synthetic
mkdir -p experiments
mkdir -p logs
mkdir -p models

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Download real parameter data (placeholder)
echo "Setting up real parameter data directory..."
cat > data/real_parameters/README.md << 'INNEREOF'
# Real Parameter Data

This directory contains real-world parameter data for synthetic data generation:

- `drone_specs/`: Manufacturer specifications for various drones
- `battery_data/`: Battery discharge curves and specifications  
- `communication_data/`: Network latency and bandwidth measurements
- `weather_data/`: Wind patterns and environmental conditions

The data in this directory is used to generate physics-accurate synthetic training data.
INNEREOF

# Setup pre-commit hooks if available
if command -v pre-commit &> /dev/null; then
    echo "Setting up pre-commit hooks..."
    pre-commit install
fi

echo ""
echo "=========================================="
echo "✓ Environment setup complete\!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Configure your experiment in configs/default_config.yaml"
echo "2. Set up Weights & Biases (optional): wandb login"
echo "3. Test the installation: python -m pytest tests/"
echo "4. Start training: python src/train.py"
echo ""
