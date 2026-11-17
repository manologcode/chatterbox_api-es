FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    VOICES_DIR=/app/voices \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TQDM_DISABLE=1 \
    TTS_DEVICE=auto \
    TTS_WARMUP=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/ /app/

RUN pip install --no-cache-dir -r requirements.txt \
    && mkdir -p /app/voices /models/huggingface

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
