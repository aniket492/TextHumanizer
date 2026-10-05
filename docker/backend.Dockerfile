# FastAPI backend (CPU). Build context: project root.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=120

WORKDIR /app

# CPU-only PyTorch first so the large CUDA build is never pulled in.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

COPY ml /app/ml
COPY backend /app/backend
COPY scripts /app/scripts

# Run as an unprivileged user; the HF cache dir is pre-created so the named volume inherits its owner.
RUN useradd --create-home --uid 1000 app     && mkdir -p /home/app/.cache/huggingface     && chown -R app:app /app /home/app/.cache
ENV HF_HOME=/home/app/.cache/huggingface
USER app

WORKDIR /app/backend
EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'"]
