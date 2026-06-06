# ─── Base: CUDA 12.8 + cuDNN (Blackwell-ready) ───────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# ─── Sistem ───────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        ffmpeg \
        libgl1-mesa-glx libglib2.0-0 \
        git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python

# ─── PyTorch 2.7 + CUDA 12.8 (Blackwell sm_120 nativ) ───────────────────────
RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ─── Transformers (latest) + deps Florence-2 ─────────────────────────────────
# Folosim florence-community/Florence-2-large = checkpoint oficial convertit
# cu integrare nativa in transformers (fara trust_remote_code).
# Compatibil cu transformers 5.x.
RUN pip install --no-cache-dir \
    transformers \
    timm \
    einops \
    sentencepiece

# flash-attn optional (~20% speedup) — daca compilarea esueaza, sarim
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \\
    echo "[INFO] flash-attn skip — Florence ruleaza si fara"

# ─── ProPainter (video inpainting) ───────────────────────────────────────────
RUN git clone --depth=1 https://github.com/sczhou/ProPainter.git /app/ProPainter && \
    pip install --no-cache-dir -r /app/ProPainter/requirements.txt

# ─── Utilitare generale ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    runpod \
    requests \
    opencv-python-headless \
    Pillow \
    numpy \
    huggingface_hub \
    accelerate

# ─── Pre-download weights Florence-2 (florence-community — nativ transformers) 
# Nu mai are nevoie de trust_remote_code → compatibil cu orice versiune transformers
RUN python -c "\
from transformers import AutoProcessor, Florence2ForConditionalGeneration; \
import torch; \
print('Downloading florence-community/Florence-2-large...'); \
AutoProcessor.from_pretrained('florence-community/Florence-2-large'); \
Florence2ForConditionalGeneration.from_pretrained( \
    'florence-community/Florence-2-large', \
    torch_dtype=torch.float32); \
print('Florence-2 weights OK')"

# ─── Pre-download weights ProPainter ─────────────────────────────────────────
RUN mkdir -p /app/ProPainter/weights && \
    wget -q -O /app/ProPainter/weights/ProPainter.pth \
        https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth && \
    wget -q -O /app/ProPainter/weights/recurrent_flow_completion.pth \
        https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth && \
    wget -q -O /app/ProPainter/weights/raft-things.pth \
        https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth && \
    echo "ProPainter weights OK"

COPY handler.py .

CMD ["python", "-u", "handler.py"]