FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Install Python dependencies in fewer layers
RUN pip install --upgrade pip && \
    pip install -U diffusers[torch] torch torchvision torchaudio transformers \
    flask protobuf sentencepiece --upgrade accelerate huggingface_hub

# # Set HuggingFace token as build argument
# ARG HF_TOKEN
# RUN huggingface-cli login --token $HF_TOKEN --add-to-git-credential

EXPOSE 8484

CMD ["python", "app.py"]