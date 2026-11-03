# =====================================================
# Grammar Correction API - Backend Dockerfile
# =====================================================
# Uses Python 3.11 slim base. PyTorch wheel includes
# CUDA 12.x runtime — no separate CUDA base needed.
# Run with NVIDIA Container Toolkit for GPU inference.
# =====================================================

FROM python:3.11-slim

WORKDIR /app

# System deps: git (for peft/transformers installs), libgomp (OpenMP for torch)
RUN apt-get update && apt-get install -y \
        git \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (inference-only, no training tools)
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-api.txt

# Copy application source (checkpoints are volume-mounted at runtime)
COPY src/ ./src/
COPY pyproject.toml .

# Health check — polls /api/v1/health every 30s
# First check delayed 90s to allow models to load (T5 + Llama ~ 60-90s)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

EXPOSE 8000

# Start API (no --reload in production)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
