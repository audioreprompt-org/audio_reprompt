FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf \
    WEIGHTS_DIR=/opt/weights/tasty \
    MUSICGEN_MODEL_PATH=/opt/weights/tasty \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MODEL_DIR=/opt/weights/tasty

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
      libsndfile1 ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --no-cache-dir "torch==2.3.0" --index-url https://download.pytorch.org/whl/cu121

COPY app/musicgen/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip cache purge

WORKDIR /app

COPY app/musicgen/handler.py /app/handler.py
COPY models/musicgen /app/models/musicgen
COPY models/tasty-musicgen-small /opt/weights/tasty

CMD ["python", "-u", "handler.py"]
