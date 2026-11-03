# Grammar Correction System

A production-grade Grammar Error Correction (GEC) system with two fine-tuned models, a FastAPI backend, and a React web interface. Built end-to-end as a learning project covering data engineering, model training, optimization, evaluation, and deployment.

## Models

| Model | Parameters | Adapter Size | F0.5 | GLEU | Inference |
|-------|-----------|--------------|------|------|-----------|
| FLAN-T5-Large + LoRA | 780M | 18MB | 0.3201 | 0.9245 | ~500ms |
| **Llama 3.2-3B + LoRA** | 3B | ~97MB | 0.0303 | 0.7431 | ~1300ms |

> Llama 3.2-3B has high recall (0.95) but over-corrects — a known issue with chat-format LLMs on GEC tasks. T5 is the stronger GEC model. Llama improvement is planned via prompt tuning.

## Architecture

```
User → React Frontend (localhost:5173)
            |
            | Axios HTTP (VITE_API_BASE_URL)
            v
   FastAPI Backend (localhost:9000)
   ├── POST /api/v1/correct         ← single sentence
   ├── POST /api/v1/correct/batch   ← multiple sentences
   ├── GET  /api/v1/health          ← model status, uptime
   └── GET  /api/v1/model/info      ← model metadata
            |
            ├── llama  → Llama 3.2-3B-Instruct + LoRA (3B, chat format)
            ├── t5     → FLAN-T5-Large + LoRA (780M, seq2seq)
            └── coedit → Grammarly CoEdIT-Large (770M, seq2seq)
            |
            v
   Response: { corrected_text, corrections[], confidence_score, processing_time_ms }
```

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── download.py          # HuggingFace dataset downloader
│   │   ├── preprocess.py        # Tokenization, DataLoaders, dynamic padding collate_fn
│   │   └── augmentation.py      # Synthetic error injection
│   ├── models/
│   │   ├── t5_gec.py            # T5GEC wrapper (FLAN-T5 + LoRA)
│   │   └── llama_gec.py         # LlamaGEC wrapper (Llama 3.2-3B + LoRA + SDPA)
│   ├── training/
│   │   ├── train.py             # Training loop, resume logic, dynamic padding wiring
│   │   ├── evaluate.py          # F0.5 (ERRANT), GLEU scorers
│   │   └── utils.py             # Checkpointing, LR scheduler, step checkpoint save/find
│   ├── api/
│   │   ├── main.py              # FastAPI app, model loading on startup
│   │   ├── routes.py            # Endpoints, sentence splitting, diff extraction
│   │   └── models.py            # Pydantic request/response schemas
├── frontend/                    # React 18 + TypeScript + TailwindCSS + Vite
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── TextEditor.tsx   # Input + model selector dropdown
│   │   │   └── CorrectionPanel.tsx
│   │   ├── services/api.ts      # Axios client (points to localhost:9000)
│   │   └── types/index.ts
│   └── .env                     # VITE_API_BASE_URL=http://localhost:9000/api/v1
├── checkpoints/
│   ├── flan_t5_large_bea2019/   # T5 checkpoint + LoRA adapter (18MB)
│   └── llama32_bea2019/         # Llama checkpoint + LoRA adapter (97MB)
│       ├── llama_gec_lora/      # Final adapter (loaded by API)
│       └── step_checkpoints/    # 14 crash-recovery checkpoints (every 500 steps)
├── documentation/               # Phase-by-phase technical documentation
├── Dockerfile                   # Backend image (python:3.11-slim + CUDA torch wheel)
├── docker-compose.yml           # Backend + frontend services, GPU, volume mounts
├── requirements-api.txt         # Inference-only deps (used in Docker, smaller than requirements.txt)
├── train_bea2019.py             # T5 training entry point
├── train_llama32_bea2019.py     # Llama training entry point
├── evaluate_bea2019.py          # T5 evaluation script
└── evaluate_llama32.py          # Llama evaluation script
```

## Quick Start

### Option A — Docker (Recommended)

Requires: [Docker](https://docs.docker.com/get-docker/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU)

```bash
# Clone and enter project directory
git clone <your-repo> && cd humanizeAI

# (Optional) provide HuggingFace token for Llama gated model access
export HF_TOKEN=your_hf_token    # Linux/Mac
$env:HF_TOKEN="your_hf_token"   # PowerShell

# Build images and start services
docker compose up --build

# Open browser
# Frontend: http://localhost:3000
# API docs: http://localhost:9000/docs
```

> Checkpoints are volume-mounted from `./checkpoints` — no model weights are baked into the image.
> First startup takes ~90s while T5 + Llama load into GPU.

### Option B — Local Development

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
cd frontend && npm install
```

#### 2. Start the API

```bash
cd d:\humanizeAI
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 9000
```

The server loads the Llama 3.2-3B model on startup (~5 seconds). Available at:
- API: http://localhost:9000
- Swagger docs: http://localhost:9000/docs

#### 3. Start the Frontend

```bash
cd frontend && npm run dev
# http://localhost:5173
```

#### 4. API Usage

```bash
# Correct with Llama (fine-tuned)
curl -X POST http://localhost:9000/api/v1/correct \
  -H "Content-Type: application/json" \
  -d '{"text": "She go to school yesterday.", "model": "llama"}'

# Correct with T5 (fine-tuned)
curl -X POST http://localhost:9000/api/v1/correct \
  -H "Content-Type: application/json" \
  -d '{"text": "She go to school yesterday.", "model": "t5"}'

# Health check
curl http://localhost:9000/api/v1/health
```

#### 5. Training

```bash
# Train FLAN-T5-Large + LoRA on BEA 2019
python train_bea2019.py

# Train Llama 3.2-3B + LoRA on BEA 2019 (with all optimizations)
python train_llama32_bea2019.py
```

#### 6. Evaluation

```bash
python evaluate_bea2019.py      # T5 evaluation
python evaluate_llama32.py      # Llama evaluation
```

## Training Optimizations (Llama)

All four techniques applied to reduce training time from ~30h to ~5h (6x speedup):

| Technique | What it does | Benefit |
|-----------|-------------|---------|
| **SDPA** | PyTorch built-in FlashAttention kernels (`attn_implementation="sdpa"`) | Faster attention, fixes batch_size=8 GQA bottleneck |
| **Dynamic Padding** | Pad each batch to its longest sequence, not global max_length | ~2.9x less wasted attention compute |
| **Gradient Checkpointing** | Recompute activations during backward pass | 40-50% VRAM savings, enables larger batch |
| **Step Checkpointing** | Save every 500 optimizer steps | Crash recovery without losing progress |

**Benchmark (RTX 4080 SUPER 16GB):**

| Config | Speed | Throughput | VRAM |
|--------|-------|-----------|------|
| Old (bs=4, seq=256, no SDPA) | 1355ms/batch | 3.0 samp/s | 15.4GB |
| New (bs=8, SDPA+dynpad+gradckpt) | 521ms/batch | 15.3 samp/s | 8.1GB |

## Hardware

- CPU: Intel i9-14900K
- RAM: 64GB DDR5
- GPU: NVIDIA RTX 4080 SUPER 16GB
- OS: Windows 11 Pro

## Documentation

- `documentation/PROJECT_STATUS.md` — Full phase-by-phase status
- `documentation/phase_1_planning/` — Architecture decisions and theory
- `documentation/phase_2_data_engineering/README.md` — Data pipeline
- `documentation/phase_3_model_development/README.md` — T5 training
- `documentation/phase_4_evaluation/README.md` — Evaluation metrics
- `documentation/phase_5_api_frontend/README.md` — API and frontend
- `documentation/phase_6_llama_training/README.md` — Llama training and optimizations

## License

MIT
