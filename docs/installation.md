# Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ / macOS 11+ / Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 8GB
- **Storage**: 10GB free space
- **CPU**: 4 cores

### Recommended Requirements
- **OS**: Ubuntu 22.04 / macOS 13+
- **Python**: 3.9 or 3.10
- **RAM**: 16GB or more
- **Storage**: 20GB free space
- **CPU**: 8+ cores
- **GPU**: NVIDIA GPU with CUDA 11.0+ support

## Installation Methods

### 1. Quick Install (Recommended)

#### Using pip

```bash
# Install from PyPI (when available)
pip install pi-hmarl

# Or install from GitHub
pip install git+https://github.com/your-org/pi-hmarl.git
```

#### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/pi-hmarl.git
cd pi-hmarl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### 2. Docker Installation

#### Pre-built Image

```bash
# Pull the latest image
docker pull pi-hmarl/pi-hmarl:latest

# Run container
docker run -it --rm \
    --gpus all \
    -p 5000:5000 \
    -p 8080:8080 \
    -v $(pwd)/data:/app/data \
    pi-hmarl/pi-hmarl:latest
```

#### Build from Source

```bash
# Clone repository
git clone https://github.com/your-org/pi-hmarl.git
cd pi-hmarl

# Build Docker image
docker build -t pi-hmarl:dev .

# Run with GPU support
docker run -it --rm \
    --gpus all \
    -p 5000:5000 \
    -v $(pwd)/data:/app/data \
    pi-hmarl:dev
```

### 3. Conda Installation

```bash
# Create conda environment
conda create -n pi-hmarl python=3.9
conda activate pi-hmarl

# Install PyTorch (with CUDA support)
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch

# Clone and install PI-HMARL
git clone https://github.com/your-org/pi-hmarl.git
cd pi-hmarl
pip install -e .
```

### 4. Development Installation

For contributors and developers:

```bash
# Clone with submodules
git clone --recursive https://github.com/your-org/pi-hmarl.git
cd pi-hmarl

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify installation
pytest tests/
```

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    python3-venv \
    git

# Install additional dependencies for visualization
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Continue with standard installation
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.9
brew install git

# Install additional tools
brew install cmake
brew install pkg-config

# Continue with standard installation
```

### Windows

```powershell
# Install Python from python.org or Microsoft Store
# Ensure Python is added to PATH

# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install Git
# Download from: https://git-scm.com/download/win

# Open PowerShell as Administrator
pip install --upgrade pip setuptools wheel

# Continue with standard installation
```

## GPU Support

### NVIDIA CUDA

```bash
# Check CUDA version
nvidia-smi

# Install appropriate PyTorch version
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

### AMD ROCm (Experimental)

```bash
# Install ROCm-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Verify GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

### Apple Silicon (M1/M2)

```bash
# Install Metal-accelerated PyTorch
pip install torch torchvision torchaudio

# Verify MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Dependencies

### Core Dependencies

```txt
# requirements.txt
torch>=1.13.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pandas>=1.3.0
gym>=0.21.0
stable-baselines3>=1.6.0
```

### Additional Dependencies

```txt
# Visualization
plotly>=5.0.0
dash>=2.0.0
flask>=2.0.0
flask-socketio>=5.0.0

# Communication
aiohttp>=3.8.0
websockets>=10.0
pyzmq>=22.0.0

# Deployment
docker>=5.0.0
kubernetes>=18.0.0
prometheus-client>=0.12.0

# Development
pytest>=6.0.0
pytest-asyncio>=0.18.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
```

## Verification

### Basic Verification

```python
# verify_installation.py
import sys
import importlib

def check_module(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} installed")
        return True
    except ImportError:
        print(f"✗ {module_name} not found")
        return False

# Check core modules
modules = [
    'pi_hmarl',
    'torch',
    'numpy',
    'gym',
    'flask',
    'plotly'
]

all_installed = all(check_module(m) for m in modules)

if all_installed:
    print("\n✓ All core modules installed successfully!")
else:
    print("\n✗ Some modules are missing. Please check installation.")
    sys.exit(1)

# Check GPU support
import torch
if torch.cuda.is_available():
    print(f"✓ GPU support enabled ({torch.cuda.get_device_name(0)})")
else:
    print("ℹ GPU support not available (CPU mode)")
```

### Full System Test

```bash
# Run comprehensive tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_environment.py -v
python -m pytest tests/test_agents.py -v
python -m pytest tests/test_physics.py -v
```

### Performance Benchmark

```bash
# Run performance benchmarks
python -m pi_hmarl.benchmarks

# Expected output:
# Environment Step Time: ~5ms
# Agent Decision Time: ~2ms
# Physics Update Time: ~1ms
# Total FPS: ~60
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'pi_hmarl'

```bash
# Ensure you're in the correct directory
cd pi-hmarl

# Install in development mode
pip install -e .
```

#### CUDA out of memory

```python
# Reduce batch size in configuration
config['training']['batch_size'] = 32  # Instead of 64

# Enable gradient accumulation
config['training']['gradient_accumulation_steps'] = 2
```

#### Slow performance

```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# Set device explicitly
export CUDA_VISIBLE_DEVICES=0
```

#### Missing system dependencies

```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# macOS
brew install python-tk

# Windows
# Install Visual C++ Redistributable
```

### Getting Help

1. Check the [FAQ](faq.md)
2. Search [GitHub Issues](https://github.com/your-org/pi-hmarl/issues)
3. Join our [Discord Server](https://discord.gg/pi-hmarl)
4. Post on [Stack Overflow](https://stackoverflow.com/questions/tagged/pi-hmarl) with tag `pi-hmarl`

## Next Steps

After successful installation:

1. Follow the [Getting Started Guide](getting_started.md)
2. Try the [Example Scripts](examples.md)
3. Read the [API Documentation](api_reference.md)
4. Join the community!

## Updating

### Update to Latest Version

```bash
# Using pip
pip install --upgrade pi-hmarl

# From source
cd pi-hmarl
git pull origin main
pip install -e . --upgrade
```

### Migration Guide

When updating between major versions, check the [Migration Guide](migration.md) for breaking changes and update instructions.