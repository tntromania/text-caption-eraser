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

# ─── Florence-2 (detectie text/logo) ─────────────────────────────────────────
RUN pip install --no-cache-dir \
    transformers>=4.41.0 \
    timm \
    einops \
    flash-attn --no-build-isolation || \
    pip install --no-cache-dir transformers>=4.41.0 timm einops
# flash-attn e optional — daca compilarea esueaza, Florence ruleaza si fara

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

# ─── Pre-download weights Florence-2 (evitam cold-start download) ────────────
RUN python -c "\
from transformers import AutoProcessor, AutoModelForCausalLM; \
import torch; \
print('Downloading Florence-2-large...'); \
AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True); \
AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', \
    torch_dtype=torch.float16, trust_remote_code=True); \
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
