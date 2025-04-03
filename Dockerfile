FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config \
    python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install setuptools wheel

# Install Python dependencies for FLUX models
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install -U transformers==4.36.2 accelerate==0.25.0 bitsandbytes==0.41.1 && \
    pip3 install -U safetensors sentencepiece protobuf flask

# Install FLUX-specific dependencies
RUN pip3 install -U --no-deps huggingface_hub

EXPOSE 8484

CMD ["python3", "app.py"]