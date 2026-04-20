# IatrogeniX Production Inference Dockerfile
# Optimized for CPU-based Clinical Validation

FROM python:3.11-slim

# Install system dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy project files
COPY inference/ ./inference/
COPY safety/ ./safety/
COPY models/ ./models/

# Environment variables
ENV IATROGENIX_MODEL=models/iatrogenix-q5_k_m.gguf
ENV N_CTX=4096
ENV N_GPU_LAYERS=0

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "inference.engine:app", "--host", "0.0.0.0", "--port", "8000"]
