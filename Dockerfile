# Optimized Dockerfile for serverless deployment
# Base: PyTorch with CUDA 11.8 and cuDNN 8 runtime
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

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

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

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






