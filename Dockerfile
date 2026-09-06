
# syntax=docker/dockerfile:1

# ============================================================
# DocsQuery Production Image
#
# Architecture:
#
#   Builder
#       ↓
#   CPU-only PyTorch
#       ↓
#   Runtime Python dependencies
#       ↓
#   DocsQuery application
#       ↓
#   Minimal runtime image
#
# The production API uses:
#   - Sentence Transformers
#   - Transformers
#   - PyTorch
#   - Qdrant client
#   - FastAPI
#   - Gemini SDK
#
# PyTorch is installed explicitly from the official CPU wheel
# index so Docker does not pull CUDA/NVIDIA packages.
# ============================================================


# ============================================================
# Stage 1: Builder
# ============================================================

FROM python:3.12-slim AS builder

WORKDIR /build

# ------------------------------------------------------------
# Native build tools
#
# Some Python packages may require compilation during
# installation.
# ------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Create isolated Python environment.
# ------------------------------------------------------------
RUN python -m venv /venv

ENV PATH="/venv/bin:$PATH"

# ------------------------------------------------------------
# Upgrade packaging tools.
# ------------------------------------------------------------
RUN python -m pip install --upgrade \
    pip \
    setuptools \
    wheel


# ============================================================
# PyTorch
# ============================================================

# ------------------------------------------------------------
# Install CPU-only PyTorch.
#
# This is intentionally installed before the other packages.
#
# The CPU wheel index prevents pip from selecting a CUDA/NVIDIA
# PyTorch distribution.
# ------------------------------------------------------------
RUN pip install \
    --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.0,<3.0"


# ============================================================
# DocsQuery Runtime Dependencies
# ============================================================

# ------------------------------------------------------------
# Copy project metadata.
#
# This layer changes only when project dependency metadata
# changes.
# ------------------------------------------------------------
COPY pyproject.toml ./

# ------------------------------------------------------------
# Install runtime dependencies explicitly.
#
# We intentionally do NOT run:
#
#     pip install .
#
# at this point because that would make pip resolve the complete
# dependency graph again.
#
# Torch is already installed above.
# ------------------------------------------------------------
RUN pip install \
    --no-cache-dir \
    --prefer-binary \
    "fastapi>=0.115,<1.0" \
    "uvicorn[standard]>=0.30,<1.0" \
    "pydantic>=2.0,<3.0" \
    "pydantic-settings>=2.0,<3.0" \
    "python-dotenv>=1.0,<2.0" \
    "pyyaml>=6.0,<7.0" \
    "pypdf>=5.0,<7.0" \
    "rank-bm25>=0.2,<1.0" \
    "sentence-transformers>=3.0,<6.0" \
    "transformers>=4.0,<6.0" \
    "qdrant-client>=1.12,<2.0" \
    "google-genai>=1.0,<2.0" \
    "httpx>=0.27,<1.0"


# ============================================================
# Application
# ============================================================

# ------------------------------------------------------------
# Copy source code.
# ------------------------------------------------------------
COPY app ./app
COPY scripts ./scripts

# ------------------------------------------------------------
# Install DocsQuery itself WITHOUT installing dependencies.
#
# All runtime dependencies have already been installed above.
# ------------------------------------------------------------
RUN pip install \
    --no-cache-dir \
    --no-deps \
    .


# ============================================================
# Build-time verification
# ============================================================

# ------------------------------------------------------------
# Verify that the main runtime packages can be imported.
#
# qdrant-client does not reliably expose a __version__
# attribute, so we only verify that the module imports.
# ------------------------------------------------------------
RUN python -c "\
import torch; \
import transformers; \
import sentence_transformers; \
import fastapi; \
import qdrant_client; \
import google.genai; \
print('Torch:', torch.__version__); \
print('Transformers:', transformers.__version__); \
print('Sentence Transformers:', sentence_transformers.__version__); \
print('FastAPI:', fastapi.__version__); \
print('Qdrant Client: import OK'); \
print('Google GenAI SDK:', google.genai.__version__) \
"


# ============================================================
# Stage 2: Runtime
# ============================================================

FROM python:3.12-slim AS runtime

WORKDIR /app

# ------------------------------------------------------------
# Python runtime configuration.
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH"

# ------------------------------------------------------------
# Copy the prepared virtual environment from the builder.
# ------------------------------------------------------------
COPY --from=builder /venv /venv

# ------------------------------------------------------------
# Copy only the application source needed at runtime.
#
# We intentionally do not copy:
#   - tests
#   - raw PDFs
#   - local evaluation artifacts
#   - .env
#   - .git
# ------------------------------------------------------------
COPY --from=builder /build/app ./app

# ------------------------------------------------------------
# Create a dedicated non-root user.
# ------------------------------------------------------------
RUN groupadd --system docsquery \
    && useradd \
        --system \
        --gid docsquery \
        --create-home \
        --home-dir /home/docsquery \
        docsquery

# ------------------------------------------------------------
# Create runtime data directory and assign ownership.
# ------------------------------------------------------------
RUN mkdir -p /app/data \
    && chown -R docsquery:docsquery \
        /app \
        /home/docsquery

# Never run the API as root.
USER docsquery

EXPOSE 8000


# ============================================================
# Container Healthcheck
# ============================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=60s \
    --retries=3 \
    CMD python -c "\
import urllib.request; \
urllib.request.urlopen( \
    'http://127.0.0.1:8000/health', \
    timeout=3 \
)"


# ============================================================
# Application Startup
# ============================================================

# One worker for now.
#
# Sentence Transformers and the CrossEncoder load ML models
# into process memory, so multiple workers would multiply the
# memory footprint.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

