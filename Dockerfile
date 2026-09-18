# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/huggingface \
    HF_HUB_CACHE=/opt/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface/sentence-transformers

WORKDIR /app

# Build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python virtual environment
RUN python -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip setuptools wheel

# Install Python dependencies
#
# pyproject.toml is the single source of truth for DocsQuery
# production dependencies.
COPY pyproject.toml README.md ./
COPY app ./app
COPY scripts ./scripts

# Production BM25 retrieval artifact
COPY data/index ./data/index

# CPU-only PyTorch
RUN pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch

# Install DocsQuery and its production dependencies
RUN pip install .

# ============================================================
# Download ML models during image build
# ============================================================
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print(f"Loading embedding model: {embedding_model}")
embedding = SentenceTransformer(embedding_model)

print(
    "Embedding dimension:",
    embedding.get_sentence_embedding_dimension(),
)

print(f"Loading reranker model: {reranker_model}")
CrossEncoder(reranker_model)

print("ML models successfully cached.")
PY

# ============================================================
# Verify models work completely offline
# ============================================================
RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

embedding = SentenceTransformer(embedding_model)
CrossEncoder(reranker_model)

print("Offline embedding model load: OK")
print(
    "Embedding dimension:",
    embedding.get_sentence_embedding_dimension(),
)
print("Offline reranker model load: OK")
print("Offline model verification: OK")
PY


# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/huggingface \
    HF_HUB_CACHE=/opt/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface/sentence-transformers \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Runtime certificate support
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python environment
COPY --from=builder /opt/venv /opt/venv

# Baked Hugging Face model cache
COPY --from=builder /opt/huggingface /opt/huggingface

# Application
COPY --from=builder /app/app ./app
COPY --from=builder /app/scripts ./scripts
COPY --from=builder /app/pyproject.toml ./

# Immutable BM25 retrieval artifact
COPY --from=builder /app/data/index ./data/index

RUN test -s /app/data/index/bm25.json \
    && echo "BM25 artifact present: OK" \
    || (echo "BM25 artifact missing or empty" && exit 1)


# Non-root runtime user
RUN useradd \
        --create-home \
        --shell /usr/sbin/nologin \
        --uid 10001 \
        docsquery \
    && mkdir -p /app/data \
    && chown -R docsquery:docsquery /app /opt/huggingface

USER docsquery

# Container healthcheck
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=30s \
            --retries=3 \
    CMD python -c \
        "import os, urllib.request; port=os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health').read()"
EXPOSE 8000

# FastAPI application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]