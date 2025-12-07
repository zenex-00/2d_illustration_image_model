# Optimized Dockerfile for serverless deployment
# Base: PyTorch 2.4.0 with CUDA 12.1 and cuDNN 8 runtime (RTX 4090 compatible)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and install VTracer binary
# VTracer is a Rust binary for vectorization
RUN VTracer_VERSION="0.6.1" && \
    VTracer_URL="https://github.com/visioncortex/vtracer/releases/download/v${VTracer_VERSION}/vtracer-linux-x64" && \
    wget -q ${VTracer_URL} -O /usr/local/bin/vtracer && \
    chmod +x /usr/local/bin/vtracer && \
    vtracer --version || echo "VTracer installed"

# Suppress root user warnings
ENV PIP_ROOT_USER_ACTION=ignore

# Copy requirements first for better caching
COPY requirements.txt .

# Copy installation script
COPY scripts/install_dependencies.sh /tmp/install_dependencies.sh
RUN chmod +x /tmp/install_dependencies.sh

# Install Python dependencies using the script (handles lama-cleaner isolation)
RUN /tmp/install_dependencies.sh

# Set environment variable for lama-cleaner venv location
ENV LAMA_VENV_DIR=/opt/lama-cleaner-venv

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV MODEL_CACHE_DIR=/models

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]






