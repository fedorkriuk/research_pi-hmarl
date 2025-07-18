#!/bin/bash

# Setup Environment Script for PI-HMARL
# This script sets up the complete development environment for Physics-Informed 
# Hierarchical Multi-Agent Reinforcement Learning

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
PYTHON_VERSION="3.10"
CONDA_ENV_NAME="pi-hmarl"
DOCKER_IMAGE_NAME="pi-hmarl"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_system_requirements() {
    print_status "Checking system requirements..."
    
    # Check operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Detected Linux system"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "Detected macOS system"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check Python
    if command_exists python3; then
        PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        print_status "Found Python $PYTHON_VER"
        if [[ "$PYTHON_VER" < "3.8" ]]; then
            print_error "Python 3.8 or higher required. Found: $PYTHON_VER"
            exit 1
        fi
    else
        print_error "Python3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check Git
    if ! command_exists git; then
        print_error "Git not found. Please install Git."
        exit 1
    fi
    
    # Check CUDA (optional)
    if command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        print_status "Found CUDA $CUDA_VERSION"
        CUDA_AVAILABLE=true
    else
        print_warning "CUDA not found. GPU acceleration will not be available."
        CUDA_AVAILABLE=false
    fi
    
    print_success "System requirements check completed"
}

# Function to setup Python environment
setup_python_environment() {
    print_status "Setting up Python environment..."
    
    cd "$PROJECT_DIR"
    
    # Check if conda/mamba is available
    if command_exists conda; then
        CONDA_CMD="conda"
        print_status "Found conda"
    elif command_exists mamba; then
        CONDA_CMD="mamba"
        print_status "Found mamba"
    else
        print_status "Conda/Mamba not found. Using pip with virtual environment..."
        setup_pip_environment
        return
    fi
    
    # Check if environment already exists
    if $CONDA_CMD env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_warning "Environment $CONDA_ENV_NAME already exists. Removing..."
        $CONDA_CMD env remove -n $CONDA_ENV_NAME -y
    fi
    
    # Create conda environment
    print_status "Creating conda environment: $CONDA_ENV_NAME"
    $CONDA_CMD create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
    
    # Activate environment
    print_status "Activating environment: $CONDA_ENV_NAME"
    source "$($CONDA_CMD info --base)/etc/profile.d/conda.sh"
    $CONDA_CMD activate $CONDA_ENV_NAME
    
    # Install PyTorch with CUDA support if available
    if [[ "$CUDA_AVAILABLE" == true ]]; then
        print_status "Installing PyTorch with CUDA support..."
        $CONDA_CMD install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        print_status "Installing PyTorch (CPU only)..."
        $CONDA_CMD install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Install the package in development mode
    print_status "Installing PI-HMARL package in development mode..."
    pip install -e .
    
    print_success "Python environment setup completed"
}

# Function to setup pip virtual environment
setup_pip_environment() {
    print_status "Setting up pip virtual environment..."
    
    cd "$PROJECT_DIR"
    
    # Create virtual environment
    if [[ -d "venv" ]]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch
    if [[ "$CUDA_AVAILABLE" == true ]]; then
        print_status "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "Installing PyTorch (CPU only)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Install the package in development mode
    print_status "Installing PI-HMARL package in development mode..."
    pip install -e .
    
    print_success "Pip virtual environment setup completed"
}

# Function to setup Docker environment
setup_docker_environment() {
    print_status "Setting up Docker environment..."
    
    cd "$PROJECT_DIR"
    
    # Check if Docker is installed
    if ! command_exists docker; then
        print_error "Docker not found. Please install Docker."
        return 1
    fi
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker."
        return 1
    fi
    
    # Check for NVIDIA Docker support
    if [[ "$CUDA_AVAILABLE" == true ]]; then
        if command_exists nvidia-docker; then
            print_status "Found nvidia-docker"
            DOCKER_CMD="nvidia-docker"
        elif docker info 2>/dev/null | grep -q "nvidia"; then
            print_status "Found Docker with NVIDIA GPU support"
            DOCKER_CMD="docker"
        else
            print_warning "NVIDIA Docker support not found. GPU acceleration will not be available in Docker."
            DOCKER_CMD="docker"
        fi
    else
        DOCKER_CMD="docker"
    fi
    
    # Build Docker image
    print_status "Building Docker image: $DOCKER_IMAGE_NAME"
    $DOCKER_CMD build -t $DOCKER_IMAGE_NAME .
    
    # Test Docker image
    print_status "Testing Docker image..."
    $DOCKER_CMD run --rm $DOCKER_IMAGE_NAME python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    
    print_success "Docker environment setup completed"
}

# Function to setup development tools
setup_development_tools() {
    print_status "Setting up development tools..."
    
    cd "$PROJECT_DIR"
    
    # Install pre-commit hooks
    if command_exists pre-commit; then
        print_status "Installing pre-commit hooks..."
        pre-commit install
    else
        print_warning "pre-commit not found. Skipping pre-commit hooks setup."
    fi
    
    # Create necessary directories
    print_status "Creating project directories..."
    mkdir -p logs data/raw data/processed data/results experiments
    
    # Create .gitignore if it doesn't exist
    if [[ ! -f ".gitignore" ]]; then
        print_status "Creating .gitignore file..."
        cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Project specific
logs/
data/
experiments/
checkpoints/
models/
*.pkl
*.h5
*.hdf5
.DS_Store
EOF
    fi
    
    print_success "Development tools setup completed"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    cd "$PROJECT_DIR"
    
    # Activate environment if using conda
    if command_exists conda && conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $CONDA_ENV_NAME
    elif [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi
    
    # Run tests
    if [[ -f "tests/test_setup.py" ]]; then
        python -m pytest tests/test_setup.py -v
    else
        print_warning "Test file not found. Creating basic tests..."
        python -c "
import torch
import numpy as np
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger
from src.utils.gpu_utils import get_gpu_manager

print('Testing imports...')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'NumPy version: {np.__version__}')

print('Testing ConfigManager...')
config = ConfigManager()
print('ConfigManager initialized successfully')

print('Testing Logger...')
logger = get_logger()
logger.info('Logger test successful')

print('Testing GPU utilities...')
gpu_manager = get_gpu_manager()
print('GPU manager initialized successfully')

print('All tests passed!')
"
    fi
    
    print_success "Tests completed"
}

# Function to display usage information
display_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -p, --python-only       Setup Python environment only"
    echo "  -d, --docker-only       Setup Docker environment only"
    echo "  -t, --test              Run tests after setup"
    echo "  --no-cuda               Disable CUDA support"
    echo "  --conda-env NAME        Specify conda environment name (default: pi-hmarl)"
    echo ""
    echo "Examples:"
    echo "  $0                      Full setup (Python + Docker + dev tools)"
    echo "  $0 -p                   Python environment only"
    echo "  $0 -d                   Docker environment only"
    echo "  $0 -t                   Full setup with tests"
    echo ""
}

# Parse command line arguments
PYTHON_ONLY=false
DOCKER_ONLY=false
RUN_TESTS=false
NO_CUDA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            display_usage
            exit 0
            ;;
        -p|--python-only)
            PYTHON_ONLY=true
            shift
            ;;
        -d|--docker-only)
            DOCKER_ONLY=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        --no-cuda)
            NO_CUDA=true
            shift
            ;;
        --conda-env)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            display_usage
            exit 1
            ;;
    esac
done

# Override CUDA detection if --no-cuda is specified
if [[ "$NO_CUDA" == true ]]; then
    CUDA_AVAILABLE=false
    print_status "CUDA support disabled by user"
fi

# Main setup process
main() {
    print_status "Starting PI-HMARL environment setup..."
    print_status "Project directory: $PROJECT_DIR"
    
    # Check system requirements
    check_system_requirements
    
    # Setup based on options
    if [[ "$DOCKER_ONLY" == true ]]; then
        setup_docker_environment
    elif [[ "$PYTHON_ONLY" == true ]]; then
        setup_python_environment
        setup_development_tools
    else
        # Full setup
        setup_python_environment
        setup_docker_environment
        setup_development_tools
    fi
    
    # Run tests if requested
    if [[ "$RUN_TESTS" == true ]]; then
        run_tests
    fi
    
    print_success "PI-HMARL environment setup completed successfully!"
    
    # Display next steps
    echo ""
    print_status "Next steps:"
    
    if [[ "$DOCKER_ONLY" != true ]]; then
        if command_exists conda && conda env list | grep -q "^${CONDA_ENV_NAME} "; then
            echo "  1. Activate conda environment: conda activate $CONDA_ENV_NAME"
        elif [[ -f "venv/bin/activate" ]]; then
            echo "  1. Activate virtual environment: source venv/bin/activate"
        fi
        echo "  2. Start development: python -m src.main"
        echo "  3. Run tests: python -m pytest tests/"
    fi
    
    if [[ "$PYTHON_ONLY" != true ]]; then
        echo "  4. Start Docker container: docker-compose up"
        echo "  5. Access Jupyter notebook: http://localhost:8888"
        echo "  6. Access TensorBoard: http://localhost:6006"
    fi
    
    echo ""
    print_status "Happy coding! 🚀"
}

# Run main function
main "$@"