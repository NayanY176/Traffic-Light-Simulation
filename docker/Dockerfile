FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN useradd --create-home --shell /bin/bash appuser

COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
COPY samples ./samples
COPY Makefile .

RUN mkdir -p /app/data /app/weights /app/outputs /app/runs \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch, ultralytics, cv2; print('ok')" || exit 1

ENTRYPOINT ["python"]
CMD ["scripts/infer.py", "--source", "samples", "--no-show", "--save", "outputs/infer"]

FROM base AS train
CMD ["scripts/train.py", "--config", "configs/train.yaml"]

FROM base AS infer
CMD ["scripts/infer.py", "--source", "samples", "--no-show", "--save", "outputs/infer"]
