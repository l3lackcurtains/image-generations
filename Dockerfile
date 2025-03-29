FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

RUN apt-get install -y cmake gcc-c++ protobuf-devel

COPY ./local_models/ /app/local_models/

COPY . .

RUN pip install --upgrade pip && \
    pip install -U diffusers && \
    pip install --upgrade accelerate && \
    pip install sentencepiece && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

RUN mkdir -p inputs results

EXPOSE 8484

CMD ["python", "app.py"]