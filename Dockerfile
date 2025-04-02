FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config \
    python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Install Python dependencies in fewer layers
RUN pip3 install --upgrade pip && \
    pip3 install -U diffusers[torch] torch torchvision torchaudio transformers \
    flask protobuf sentencepiece --upgrade accelerate

EXPOSE 8484

CMD ["python3", "app.py"]