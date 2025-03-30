FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install protobuf && \
    pip install -U diffusers && \
    pip install -U "diffusers[torch]" && \
    pip install flask transformers sentencepiece && \
    pip install --upgrade accelerate && \
    pip install torch torchvision torchaudio transformers && \
    pip install -U xformers --index-url https://download.pytorch.org/whl/cu124 && \
    pip install ninja


EXPOSE 8484

CMD ["python", "app.py"]