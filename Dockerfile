# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    patchelf \
    cmake \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace/pi-hmarl

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /workspace/pi-hmarl/experiments \
    /workspace/pi-hmarl/data/real_parameters \
    /workspace/pi-hmarl/data/synthetic \
    /workspace/pi-hmarl/logs

# Set up environment for MuJoCo
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/root/.mujoco/mujoco210/bin"
ENV MUJOCO_GL=egl

# Expose ports for TensorBoard and other services
EXPOSE 6006 8265 8888

# Set up entrypoint
ENTRYPOINT ["/bin/bash"]
EOF < /dev/null