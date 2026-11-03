# Phase 7: Containerization and Deployment

**Project:** Grammar Error Correction (GEC) System
**Phase:** 7 — Docker Containerization, Compose Orchestration, and Local Deployment
**Date:** February 2026
**Status:** Complete (local Docker deployment operational)

---

## Table of Contents

1. [Overview and Goals](#1-overview-and-goals)
2. [Docker Architecture](#2-docker-architecture)
3. [Files Created](#3-files-created)
4. [Bug Fixed](#4-bug-fixed)
5. [Prerequisites and Setup](#5-prerequisites-and-setup)
6. [Running with Docker Compose](#6-running-with-docker-compose)
7. [Running Locally (Dev Mode)](#7-running-locally-dev-mode)
8. [Port Configuration](#8-port-configuration)
9. [Environment Variables](#9-environment-variables)
10. [Next Steps](#10-next-steps)
11. [Lessons Learned](#11-lessons-learned)

---

## 1. Overview and Goals

Phase 7 wraps the GEC system into Docker containers so that the backend inference service and the React frontend can be deployed together as a reproducible stack, isolated from the host environment.

### The System

The GEC system exposes three fine-tuned models through a single FastAPI backend:

| Model | Parameters | F0.5 Score | Notes |
|---|---|---|---|
| FLAN-T5-Large | 780 M | 0.32 | Best performing model |
| CoEdIT | 770 M | — | Instruction-tuned GEC specialist |
| Llama 3.2-3B | 3 B | 0.03 | Experimental; underperforms |

The frontend is a React 18 + TypeScript + Vite + TailwindCSS single-page application that communicates with the FastAPI backend through an Nginx reverse proxy when running in Docker.

### Phase 7 Goals

- Package the FastAPI backend into a self-contained Docker image with GPU support.
- Package the React frontend into an Nginx-served Docker image.
- Wire both services together with Docker Compose.
- Avoid baking environment-specific URLs into the Vite build (handled by Nginx proxy).
- Resolve port conflicts introduced by other services already running on the host.
- Fix a dtype kwarg bug in the T5 model loader that was previously masked during local dev.

---

## 2. Docker Architecture

### Service Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Host Machine                                           │
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │  Frontend Container  │  │  Backend Container       │ │
│  │                      │  │                          │ │
│  │  Nginx :80           │  │  FastAPI (uvicorn) :8000 │ │
│  │  ├─ Serves React SPA │  │  ├─ /correct             │ │
│  │  └─ /api/* ──────────┼──►  ├─ /health              │ │
│  │       proxy_pass     │  │  ├─ /models               │ │
│  │                      │  │  └─ /docs (Swagger)      │ │
│  └──────────┬───────────┘  └──────────────────────────┘ │
│             │                         ▲                  │
│    Host port 3000              Host port 9000            │
│             │                         │                  │
└─────────────┼─────────────────────────┼─────────────────┘
              │                         │
         Browser /                 Direct API
         Frontend UI               access / dev
```

### How the Services Connect

1. **Browser → Host port 3000** — The user opens `http://localhost:3000`. The host maps this to the frontend container's Nginx on port 80.
2. **Nginx → /api/\* proxy** — Any request with an `/api/` prefix is forwarded by Nginx inside the container to `http://backend:8000`. Docker Compose's internal DNS resolves the `backend` hostname.
3. **Browser → Host port 9000** — Direct access to the FastAPI backend (Swagger UI, raw API calls during development, and health checks). The host maps this to the backend container's port 8000.
4. **GPU passthrough** — The backend container reaches the host GPU through NVIDIA Container Toolkit. PyTorch inside the container sees the GPU as if it were running natively.
5. **Model checkpoints** — Mounted read-only from the host into the backend container. The container never writes to the checkpoint directory.
6. **Hugging Face cache** — A named Docker volume (`hf_cache`) persists the HF model cache across container restarts, avoiding repeated downloads.

### Internal Docker Network

Both services are placed on a Compose-managed bridge network. The frontend container refers to the backend container as `backend` (the Compose service name). No host ports need to be specified for inter-service communication.

---

## 3. Files Created

### 3.1 `Dockerfile` (Backend)

**Location:** `d:\humanizeAI\Dockerfile`

**Base image:** `python:3.11-slim`

**Key decisions:**

| Decision | Rationale |
|---|---|
| `python:3.11-slim` base | Keeps the image small; PyTorch CUDA wheel brings its own CUDA runtime libraries, so a full `nvidia/cuda` base is not required |
| PyTorch installed via CUDA wheel URL | Ensures the correct CUDA-enabled build is installed; the default PyPI wheel is CPU-only |
| `libgomp1` system package | OpenMP runtime required by PyTorch CPU operations and some HuggingFace internals; without it the container crashes at import time |
| `requirements-api.txt` (not `requirements.txt`) | The full `requirements.txt` contains training and evaluation tools (datasets, evaluate, seqeval, etc.) that are not needed at inference time and would bloat the image |
| Health check (`/health` endpoint) | Allows Compose to delay starting the frontend until the backend is genuinely ready to handle requests, not just running a process |
| Non-root user | The container drops to a non-root user after setup for basic security hygiene |

**Health check configuration:**

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

The `--start-period=120s` accounts for model loading time. FLAN-T5-Large and CoEdIT each take 30–60 seconds to load from disk on first startup.

---

### 3.2 `requirements-api.txt`

**Location:** `d:\humanizeAI\requirements-api.txt`

Inference-only dependency list. Intentionally excludes all training and evaluation tooling.

**Included packages:**

```
torch                  # Core tensor operations and GPU inference
transformers           # Model classes (T5ForConditionalGeneration, etc.)
peft                   # LoRA adapter loading (used by Llama model)
accelerate             # Device mapping and mixed-precision utilities
safetensors            # Loading .safetensors checkpoint files
fastapi                # REST API framework
uvicorn[standard]      # ASGI server
pydantic               # Request/response validation and serialization
python-dotenv          # .env file loading
loguru                 # Structured logging
numpy                  # Array utilities
```

**Explicitly excluded (present in full requirements.txt):**

```
datasets               # HuggingFace dataset loading — training only
evaluate               # Metric computation — evaluation only
seqeval                # Sequence labeling metrics — evaluation only
scikit-learn           # ML utilities — not needed at inference time
wandb                  # Experiment tracking — training only
jupyter / notebook     # Development tools — not needed in production
```

---

### 3.3 `.dockerignore` (Backend)

**Location:** `d:\humanizeAI\.dockerignore`

Controls what is sent to the Docker build daemon as the build context. Excluding large directories dramatically reduces build time and image size.

**Key exclusions:**

| Path excluded | Reason |
|---|---|
| `checkpoints/` | Model weights are mounted as a volume at runtime; baking them into the image would produce an enormous (10+ GB) image |
| `data/` | Training datasets are not needed at inference time |
| `frontend/` | The frontend has its own separate Dockerfile; including it here would bloat the backend build context |
| `__pycache__/`, `*.pyc` | Compiled Python bytecode regenerated inside the container |
| `*.log` | Log files from local development |
| `.env` | Environment secrets must never be baked into an image |
| `notebooks/` | Jupyter notebooks — development artifacts |
| `scripts/` | Training and utility scripts not needed at runtime |

---

### 3.4 `frontend/.dockerignore`

**Location:** `d:\humanizeAI\frontend\.dockerignore`

**Key exclusions:**

| Path excluded | Reason |
|---|---|
| `.env`, `.env.local` | These may contain `VITE_API_URL=http://localhost:9000` or similar. If included in the Vite build, the hardcoded localhost URL would be embedded in the compiled JS bundle and would not work inside Docker. Nginx proxy routing handles the `/api/` prefix instead. |
| `node_modules/` | Installed fresh inside the container; including host `node_modules` can cause architecture-specific binary conflicts |
| `dist/` | Previous local build artifacts; the container builds from source |

**Why this matters for Vite:** Vite inlines environment variables (prefixed `VITE_`) at build time using `import.meta.env`. If `.env.local` is present in the build context and contains `VITE_API_URL=http://localhost:9000`, that URL gets compiled into the React bundle. Inside Docker, the frontend container cannot reach `localhost:9000` — it must go through the Nginx proxy to `http://backend:8000`. By excluding `.env` files from the build context, the Vite build falls back to using relative paths (e.g., `/api/correct`), which Nginx then proxies correctly.

---

### 3.5 `docker-compose.yml`

**Location:** `d:\humanizeAI\docker-compose.yml`

Orchestrates both services, their port mappings, GPU configuration, volumes, and startup order.

**Backend service configuration:**

```yaml
backend:
  build:
    context: .
    dockerfile: Dockerfile
  ports:
    - "9000:8000"           # Host 9000 → Container 8000 (port conflict workaround)
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  volumes:
    - ./checkpoints:/app/checkpoints:ro    # Read-only model weights
    - hf_cache:/root/.cache/huggingface    # Persistent HF cache
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    start_period: 120s
    retries: 3
```

**Frontend service configuration:**

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
  ports:
    - "3000:80"             # Host 3000 → Nginx :80
  depends_on:
    backend:
      condition: service_healthy   # Wait for backend health check to pass
```

**Named volumes:**

```yaml
volumes:
  hf_cache:    # Survives container removal; avoids re-downloading model files on restart
```

---

## 4. Bug Fixed

### `src/models/t5_gec.py` — Wrong `dtype` Keyword Argument

**File:** `d:\humanizeAI\src\models\t5_gec.py`

**Symptom:** The backend would crash at startup when loading the FLAN-T5-Large model with a `TypeError` similar to:

```
TypeError: T5ForConditionalGeneration.from_pretrained() got an unexpected keyword argument 'dtype'
```

**Root cause:** The model loader was building a `load_kwargs` dictionary and setting:

```python
# BEFORE (incorrect)
load_kwargs["dtype"] = torch.float16
```

The `transformers` `from_pretrained()` API does not accept a `dtype` keyword argument. The correct parameter name is `torch_dtype`.

**Fix:**

```python
# AFTER (correct)
load_kwargs["torch_dtype"] = torch.float16
```

**Why it was not caught earlier:** During local development, the model was either loaded in full float32 (no dtype kwarg needed) or the code path that set `dtype` was not exercised in typical dev runs. The Dockerfile's clean environment and explicit startup sequence exposed the bug during container testing.

**Impact:** Without this fix, the backend container would fail to start with the T5 model loaded, causing the health check to fail and the frontend to never start (due to `depends_on: condition: service_healthy`).

**Diff:**

```diff
- load_kwargs["dtype"] = torch.float16
+ load_kwargs["torch_dtype"] = torch.float16
```

---

## 5. Prerequisites and Setup

### System Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Windows 10/11, Ubuntu 20.04+ | WSL2 required on Windows for Docker GPU |
| Docker | 24.0+ | Earlier versions may not support `deploy.resources.reservations` syntax |
| Docker Compose | v2.20+ | Use `docker compose` (v2), not `docker-compose` (v1) |
| NVIDIA GPU | CUDA Compute 7.0+ (Volta+) | Required for GPU inference |
| NVIDIA Driver | 525+ | Must support CUDA 12.x |
| NVIDIA Container Toolkit | Latest | Enables GPU passthrough into containers |
| Disk space | 25 GB free | ~10 GB for Docker image layers, ~10 GB for checkpoints, ~5 GB for HF cache |
| RAM | 16 GB | 32 GB recommended if loading all three models simultaneously |
| VRAM | 8 GB | 16 GB recommended for Llama 3.2-3B in float16 |

### Installing NVIDIA Container Toolkit (Ubuntu/WSL2)

```bash
# Add the NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify GPU is accessible inside Docker:**

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed in the output. If this command fails, Docker Compose GPU passthrough will not work.

### Windows + WSL2 Notes

On Windows, Docker Desktop with WSL2 backend automatically bridges NVIDIA drivers to WSL2. NVIDIA Container Toolkit installation is handled by Docker Desktop. Verify GPU access with:

```powershell
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Project Directory Structure (Expected Before Running)

```
d:\humanizeAI\
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements-api.txt
├── .env                         # Not committed; see Environment Variables section
├── checkpoints/
│   ├── flan-t5-large-gec/       # FLAN-T5 fine-tuned checkpoint
│   ├── coedit-gec/              # CoEdIT fine-tuned checkpoint
│   └── llama-3.2-3b-gec/       # Llama fine-tuned checkpoint (LoRA)
├── src/
│   ├── api/
│   └── models/
│       ├── t5_gec.py
│       ├── coedit_gec.py
│       └── llama_gec.py
└── frontend/
    ├── Dockerfile
    ├── .dockerignore
    ├── package.json
    └── src/
```

---

## 6. Running with Docker Compose

### Step 1 — Create the environment file

Create `d:\humanizeAI\.env` with the required variables (see [Section 9](#9-environment-variables) for the full reference):

```bash
# d:\humanizeAI\.env
MODEL_CHECKPOINT_DIR=/app/checkpoints
DEFAULT_MODEL=flan-t5-large
LOG_LEVEL=INFO
```

Do not commit `.env` to version control.

### Step 2 — Build both images

```bash
cd d:\humanizeAI

# Build without cache to ensure a clean image (recommended first time)
docker compose build --no-cache

# Subsequent builds (uses layer cache where possible)
docker compose build
```

Build time: approximately 5–15 minutes depending on network speed (PyTorch CUDA wheel is ~2 GB).

### Step 3 — Start the stack

```bash
docker compose up
```

Or run in detached mode (no log streaming):

```bash
docker compose up -d
```

### Step 4 — Watch the startup sequence

With `docker compose up` (non-detached), you will see interleaved logs from both containers. The expected sequence is:

```
backend   | Loading FLAN-T5-Large from /app/checkpoints/flan-t5-large-gec ...
backend   | Loading CoEdIT from /app/checkpoints/coedit-gec ...
backend   | Loading Llama 3.2-3B from /app/checkpoints/llama-3.2-3b-gec ...
backend   | Application startup complete.
backend   | Uvicorn running on http://0.0.0.0:8000
frontend  | Starting Nginx ...
```

The frontend container will not start until the backend health check passes. On first startup, allow 2–3 minutes for model loading before the health check succeeds.

### Step 5 — Verify the deployment

**Backend health check:**

```bash
curl http://localhost:9000/health
# Expected: {"status": "ok", "models_loaded": ["flan-t5-large", "coedit", "llama-3.2-3b"]}
```

**Backend API docs:**

Open `http://localhost:9000/docs` in a browser. The FastAPI Swagger UI lists all available endpoints.

**Frontend:**

Open `http://localhost:3000` in a browser. The React SPA should load and be able to submit text for correction.

**Test a correction request:**

```bash
curl -X POST http://localhost:9000/correct \
  -H "Content-Type: application/json" \
  -d '{"text": "She go to school every day.", "model": "flan-t5-large"}'
```

Expected response:

```json
{
  "corrected": "She goes to school every day.",
  "model": "flan-t5-large",
  "processing_time_ms": 312
}
```

### Step 6 — View logs

```bash
# All services
docker compose logs -f

# Backend only
docker compose logs -f backend

# Frontend only
docker compose logs -f frontend
```

### Step 7 — Stop the stack

```bash
# Stop and remove containers (volumes are preserved)
docker compose down

# Stop and remove containers AND the hf_cache volume (forces HF model re-download on next start)
docker compose down -v
```

### Rebuilding After Code Changes

```bash
# Rebuild and restart a single service
docker compose up --build backend

# Rebuild the frontend after React code changes
docker compose up --build frontend
```

---

## 7. Running Locally (Dev Mode)

For active development, running without Docker eliminates the image rebuild cycle.

### Backend (FastAPI with uvicorn)

**Prerequisites:** Python 3.11, CUDA-enabled PyTorch installed in your virtual environment.

```bash
cd d:\humanizeAI

# Activate virtual environment
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install inference dependencies
pip install -r requirements-api.txt

# Start the API server on port 9000
uvicorn src.api.main:app --host 0.0.0.0 --port 9000 --reload
```

The `--reload` flag enables hot-reloading on source file changes. Do not use `--reload` in production.

**Verify:**

```bash
curl http://localhost:9000/health
```

### Frontend (Vite dev server)

**Prerequisites:** Node.js 18+ and npm.

```bash
cd d:\humanizeAI\frontend

# Install dependencies (only needed once or after package.json changes)
npm install

# Start the Vite dev server on port 5173
npm run dev
```

Open `http://localhost:5173` in a browser.

**Setting the API URL for local dev:**

When running the frontend locally (outside Docker), the Nginx proxy is not present. Set the backend URL in `.env.local`:

```bash
# d:\humanizeAI\frontend\.env.local
VITE_API_URL=http://localhost:9000
```

Vite reads `.env.local` automatically. Do not commit this file.

### Mixed Mode (local frontend + Docker backend)

You can run the Docker backend only and point your local Vite dev server at it:

```bash
# Start only the backend container
docker compose up backend

# In another terminal, start the local frontend
cd d:\humanizeAI\frontend
VITE_API_URL=http://localhost:9000 npm run dev
```

This is useful for frontend development when you do not want to rebuild the frontend Docker image on every change.

---

## 8. Port Configuration

### Why Port 9000?

The host machine running this project also runs an Oracle RAG service that automatically occupies ports 8000–8004. The RAG service is configured as a system service that restarts automatically, making those ports permanently unavailable.

| Port | Status | Occupied by |
|---|---|---|
| 8000 | Occupied | Oracle RAG service (primary) |
| 8001 | Occupied | Oracle RAG service (secondary) |
| 8002 | Occupied | Oracle RAG service (auxiliary) |
| 8003 | Occupied | Oracle RAG service (auxiliary) |
| 8004 | Occupied | Oracle RAG service (auxiliary) |
| 9000 | **Clear** | **GEC backend (this project)** |
| 3000 | Clear | GEC frontend (this project) |
| 5173 | Clear | GEC frontend (Vite dev server) |

Port 9000 was selected after confirming it was not in use. The backend FastAPI process inside the container still binds to its default port 8000 — only the Docker host-to-container port mapping is adjusted.

### Port Mapping Summary

| Service | Host Port | Container Port | Access URL |
|---|---|---|---|
| FastAPI backend | 9000 | 8000 | `http://localhost:9000` |
| React frontend (Docker) | 3000 | 80 (Nginx) | `http://localhost:3000` |
| React frontend (dev) | 5173 | 5173 (Vite) | `http://localhost:5173` |

### Internal Docker Network Ports

Within the Compose bridge network, services communicate on container ports (not host ports):

```
frontend container → http://backend:8000/api/correct
                                  ^^^^
                      Service name from docker-compose.yml
                      Resolves to backend container's internal IP
```

The Nginx configuration inside the frontend container uses `proxy_pass http://backend:8000;` — this never touches the host network.

---

## 9. Environment Variables

### Backend Environment Variables

Create `d:\humanizeAI\.env`. This file is read by `python-dotenv` at startup and is excluded from Docker build context by `.dockerignore`.

| Variable | Required | Default | Description |
|---|---|---|---|
| `MODEL_CHECKPOINT_DIR` | Yes | `/app/checkpoints` | Absolute path inside the container where model checkpoints are mounted |
| `DEFAULT_MODEL` | No | `flan-t5-large` | Model used when no model is specified in the request. Options: `flan-t5-large`, `coedit`, `llama-3.2-3b` |
| `LOG_LEVEL` | No | `INFO` | Loguru log level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MAX_INPUT_LENGTH` | No | `512` | Maximum number of tokens accepted in a single correction request |
| `DEVICE` | No | `cuda` | PyTorch device. Set to `cpu` to force CPU inference (much slower) |
| `TORCH_DTYPE` | No | `float16` | Model weight precision. Options: `float16`, `bfloat16`, `float32` |

**Example `.env`:**

```bash
MODEL_CHECKPOINT_DIR=/app/checkpoints
DEFAULT_MODEL=flan-t5-large
LOG_LEVEL=INFO
MAX_INPUT_LENGTH=512
DEVICE=cuda
TORCH_DTYPE=float16
```

### Frontend Environment Variables

Create `d:\humanizeAI\frontend\.env.local` for local development. This file is excluded from the Docker build context by `frontend/.dockerignore`.

| Variable | Required | Default | Description |
|---|---|---|---|
| `VITE_API_URL` | Dev only | (empty) | Backend API base URL. Only used during local development. In Docker, Nginx proxy handles routing — do not set this in the Docker build. |

**Example `frontend/.env.local` (local dev only):**

```bash
VITE_API_URL=http://localhost:9000
```

### Docker Compose Environment Variable Passthrough

Variables defined in `.env` at the project root are automatically read by Docker Compose and can be passed to containers via the `environment` or `env_file` directive in `docker-compose.yml`:

```yaml
backend:
  env_file:
    - .env
```

---

## 10. Next Steps

The following improvements are planned for Phase 8 and beyond.

### 10.1 Hugging Face Spaces Deployment

Deploy the GEC system to Hugging Face Spaces for public access without managing infrastructure.

**Planned approach:**

- Use a `DOCKER` Space (not `GRADIO` or `STREAMLIT`) to upload the existing Docker images.
- The backend Space will use a GPU-enabled tier (A10G recommended for 3B model).
- The frontend can be served from the same Space or as a separate static Space.
- HF Spaces does not support `docker-compose.yml` directly; services will need to be combined into a single container or deployed as separate Spaces with CORS configured.

**Key blockers to resolve:**
- HF Spaces GPU tier pricing and availability.
- Static frontend Space communicating with backend Space across domains (CORS headers on FastAPI).
- Checkpoint upload via `huggingface-hub` CLI or Git LFS (checkpoints are currently local only).

### 10.2 Quantization (GPTQ / GGUF / bitsandbytes)

Reduce VRAM requirements and improve inference throughput by quantizing the models.

| Method | Target Model | Expected Benefit |
|---|---|---|
| `bitsandbytes` int8 | Llama 3.2-3B | ~50% VRAM reduction, minimal quality loss |
| `bitsandbytes` int4 (NF4) | Llama 3.2-3B | ~75% VRAM reduction, some quality loss |
| GPTQ | Llama 3.2-3B | Fast GPU inference, requires calibration data |
| Dynamic quantization | FLAN-T5-Large | Small VRAM win; encoder-decoder models benefit less |

Quantization would allow the Llama 3.2-3B model to fit in 6–8 GB VRAM instead of 12–16 GB, making it viable on consumer GPUs.

**Implementation path:** Add `load_in_8bit=True` or `load_in_4bit=True` to the `from_pretrained()` call in `src/models/llama_gec.py`. Requires `bitsandbytes` added to `requirements-api.txt`.

### 10.3 ONNX Export and ONNX Runtime Inference

Export models to ONNX format for faster inference and broader hardware compatibility.

**Benefits:**
- 2–4x inference speedup for encoder models (FLAN-T5, CoEdIT) using ONNX Runtime with CUDA execution provider.
- CPU inference becomes viable for lower-end deployments.
- Enables TensorRT optimization for maximum GPU throughput.

**Implementation path:**

```python
from transformers.onnx import export
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# Export
model = ORTModelForSeq2SeqLM.from_pretrained(
    "checkpoints/flan-t5-large-gec",
    export=True
)
model.save_pretrained("checkpoints/flan-t5-large-gec-onnx")
```

Requires `optimum[onnxruntime-gpu]` added to `requirements-api.txt`.

### 10.4 Model Serving Optimizations

- **Batching:** Implement dynamic batching in the FastAPI endpoint to process multiple requests in parallel on the GPU.
- **Caching:** Add a Redis-backed response cache for identical input strings (useful in demo scenarios).
- **Async loading:** Load models asynchronously on startup rather than blocking the event loop.
- **Model selection UI:** Add a model selector dropdown in the frontend so users can switch between models interactively.

### 10.5 Continuous Integration

- Add a GitHub Actions workflow that builds Docker images on push and runs a smoke test against the `/health` and `/correct` endpoints.
- Add automated F0.5 score regression tests against a fixed test set to catch model quality regressions.

---

## 11. Lessons Learned

### 11.1 PyTorch CUDA Wheels Must Be Explicitly Specified

The default `pip install torch` installs a CPU-only wheel. In the Dockerfile, the CUDA wheel must be specified via the `--index-url` flag:

```dockerfile
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Forgetting this produces a container that appears to work but silently falls back to CPU, causing 10–100x slower inference with no error message.

### 11.2 `libgomp1` Is a Silent Dependency

PyTorch on Linux requires the OpenMP shared library (`libgomp.so.1`). On a full desktop Linux system this is always present. On `python:3.x-slim` it is not. The symptom is an `ImportError` mentioning `libgomp.so.1: cannot open shared object file` that only appears when the container is running, not during the build. Adding `RUN apt-get install -y libgomp1` to the Dockerfile fixes it.

### 11.3 The `dtype` vs `torch_dtype` Bug Illustrates the Value of Clean Environments

The kwarg inconsistency (`dtype` vs `torch_dtype`) had been in the codebase since the model was first written. It was never caught because:

- Local development ran with float32 (the else branch that set `dtype` was never reached).
- Test scripts called the model directly without going through the loader's dtype logic.

The containerized environment forced a full startup with all code paths exercised, revealing the bug immediately. This is one of the primary values of containerization for ML systems: it eliminates "works on my machine" conditions.

### 11.4 Vite Environment Variables Are Baked Into the Bundle at Build Time

Vite inlines `import.meta.env.VITE_*` values during the build step — they are not read at runtime. This means:

- A Vite build that includes `VITE_API_URL=http://localhost:9000` will have that URL hardcoded in the output JavaScript.
- That hardcoded URL will not work inside Docker where `localhost` refers to the container's own loopback, not the host.
- Solution: exclude `.env` files from the Docker build context so Vite uses relative paths, then use Nginx to proxy `/api/*` to the backend service.

This is a common point of confusion when dockerizing Vite applications.

### 11.5 Health Check `start_period` Must Account for Model Load Time

Docker's default health check `start_period` is 0 seconds, meaning the first health check runs immediately. ML models can take 30–120 seconds to load from disk. Without a sufficient `start_period`, the container appears unhealthy before it has had a chance to finish initializing, and `depends_on: condition: service_healthy` causes the frontend to abort before the backend is ready. Setting `start_period: 120s` resolved this without requiring changes to the application code.

### 11.6 Read-Only Volume Mounts Prevent Accidental Checkpoint Corruption

Mounting checkpoints as `:ro` (read-only) ensures the container cannot accidentally overwrite or corrupt model weights. This is particularly important for fine-tuned checkpoints that cannot be trivially recovered. If a bug in the inference code attempted to write to the checkpoint directory, the container would get a permission error rather than silently destroying the weights.

### 11.7 Named Volumes Survive `docker compose down`

Docker Compose named volumes (like `hf_cache`) persist even when containers are removed with `docker compose down`. They are only deleted with `docker compose down -v`. This behavior is intentional — it prevents repeated multi-gigabyte downloads of HuggingFace model files on each deployment. However, it means stale cache entries can accumulate over time. Periodically inspect the volume with `docker volume inspect humanizeai_hf_cache` if unexpected behavior is observed.

---

*Document generated for Phase 7 of the HumanizeAI GEC project. For Phase 6 (model training and evaluation), see `d:\humanizeAI\documentation\phase_6_training\README.md`.*
