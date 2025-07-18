#!/bin/bash
# Docker setup script for PI-HMARL

set -e

echo "=========================================="
echo "PI-HMARL Docker Setup"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    echo "Please install Docker from https://www.docker.com/"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed."
    echo "Please install docker-compose from https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q nvidia; then
    echo "Warning: NVIDIA Docker runtime not found."
    echo "GPU support will not be available."
    echo "Install nvidia-docker2 for GPU support: https://github.com/NVIDIA/nvidia-docker"
    read -p "Continue without GPU support? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Weights & Biases API Key (optional)
WANDB_API_KEY=

# Experiment name
EXPERIMENT_NAME=pi_hmarl_experiment

# Number of GPUs to use
NUM_GPUS=1
EOF
    echo "✓ Created .env file. Please add your WANDB_API_KEY if using W&B."
fi

# Build Docker image
echo "Building Docker image..."
docker-compose build pi-hmarl

# Create Docker volumes
echo "Creating Docker volumes..."
docker volume create pi-hmarl-experiments
docker volume create pi-hmarl-data

echo ""
echo "=========================================="
echo "✓ Docker setup complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  Start development container:  docker-compose up -d pi-hmarl"
echo "  Enter container:             docker-compose exec pi-hmarl bash"
echo "  Start TensorBoard:           docker-compose up -d tensorboard"
echo "  Start Jupyter:               docker-compose up -d jupyter"
echo "  Stop all services:           docker-compose down"
echo ""
echo "The Jupyter notebook will be available at: http://localhost:8889"
echo "TensorBoard will be available at: http://localhost:6007"
echo ""