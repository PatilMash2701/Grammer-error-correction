# Grammar Correction System - Project Status

**Last Updated:** February 22, 2026
**Project Status:** Phase 7 In Progress - Docker containerization

---

## Quick Overview

Production-grade Grammar Error Correction system with two fine-tuned models: FLAN-T5-Large (best GEC performance) and Llama 3.2-3B-Instruct (largest model, 3B params), both served via FastAPI backend and React frontend.

**Goal:** Build a Grammarly-like system for correcting grammatical errors in English text.

**Model Performance:**

| Model | F0.5 | Precision | Recall | GLEU | Inference |
|-------|------|-----------|--------|------|-----------|
| FLAN-T5-Large + LoRA | 0.3201 | 0.2744 | 0.9588 | 0.9245 | ~500ms |
| Llama 3.2-3B + LoRA | 0.0303 | 0.0244 | 0.9515 | 0.7431 | ~1300ms |

---

## Development Progress

### Phase 1: Planning & Setup (Complete)
- [x] Project scaffold (73 files)
- [x] Configuration files
- [x] Model architecture analysis (Llama, T5, BART)
- [x] Plan data pipeline and training strategy

**Documentation:** `documentation/phase_1_planning/`

---

### Phase 2: Data Engineering (Complete)
- [x] BEA 2019 dataset from HuggingFace (juancavallotti/bea-19-corruption)
- [x] 84,106 total samples (75,691 train / 4,203 dev / 4,206 test)
- [x] Synthetic GEC dataset generator (6 error types, 7K samples)
- [x] Preprocessing pipeline with tokenization
- [x] PyTorch DataLoader creation

**Key Stats:**
- BEA 2019: 84,106 samples (primary dataset)
- Synthetic: 8,418 samples (supplementary)
- Data cleaning: Removed null/NaN rows

**Documentation:** `documentation/phase_2_data_engineering/README.md`

**Files Implemented:**
- `src/data/download.py` - Dataset downloading
- `src/data/preprocess.py` - Tokenization & DataLoaders
- `src/data/augmentation.py` - Synthetic error injection
- `scripts/generate_synthetic_gec_dataset.py` - Synthetic dataset generator
- `scripts/convert_synthetic_to_csv.py` - Format converter
- `scripts/download_bea2019_hf.py` - BEA 2019 HuggingFace downloader

---

### Phase 3: Model Development & Training (Complete)
- [x] T5GEC wrapper class with LoRA support
- [x] LlamaGEC wrapper class with LoRA support
- [x] Training pipeline with gradient accumulation
- [x] Mixed precision training (bfloat16 for T5 stability)
- [x] Learning rate scheduler with warmup
- [x] Evaluation and checkpointing
- [x] Full training on BEA 2019 (3 epochs, completed)

**Training Optimization Techniques Used:**
1. LoRA (r=16, alpha=32) - 4.7M trainable params (0.6% of 780M)
2. Mixed Precision (bfloat16) - half memory, GPU acceleration
3. Gradient Accumulation (steps=2, effective batch=8)
4. Gradient Clipping (max_norm=1.0)
5. Linear LR Warmup + Decay (10% warmup)
6. AdamW Optimizer (lr=1e-5, weight_decay=0.01)
7. Early Stopping (patience=2)
8. Pin Memory for faster CPU-to-GPU transfer
9. Seed Management (seed=42) for reproducibility

**Trained Model:**
- Model: google/flan-t5-large (780M parameters)
- Dataset: BEA 2019 (75,691 training samples)
- Epochs: 3 (completed in ~5.5 hours on RTX 4080 SUPER)
- Checkpoints: `checkpoints/flan_t5_large_bea2019/`
- LoRA adapter: `checkpoints/flan_t5_large_bea2019/llama_gec_lora/` (18MB)

**Documentation:** `documentation/phase_3_model_development/README.md`

---

### Phase 4: Evaluation (Complete)
- [x] Implemented F0.5 scorer using ERRANT framework
- [x] Implemented GLEU scorer
- [x] Full evaluation on BEA 2019 test set (4,206 samples)
- [x] Evaluated Grammarly CoEdIT-large as alternative model
- [x] Model comparison and selection

**Evaluation Results:**

| Model | F0.5 | Precision | Recall | GLEU | Correction Rate |
|-------|------|-----------|--------|------|-----------------|
| **FLAN-T5-Large + LoRA** | **0.3201** | **0.2744** | **0.9588** | **0.9245** | **12.8%** |
| Grammarly CoEdIT-large | 0.0548 | 0.0443 | 0.9792 | 0.6849 | 98.15% |

**Key Findings:**
- Fine-tuned model is 6x better than Grammarly CoEdIT on F0.5
- CoEdIT over-corrects massively (16,232 false positives vs 431 for our model)
- Our model has high recall (0.96) but lower precision (0.27)
- Decision: Keep fine-tuned FLAN-T5-Large as production model

**Key Files:**
- `src/training/evaluate.py` - F0.5, GLEU, per-error-type analysis
- `evaluate_bea2019.py` - Full evaluation script
- `test_grammarly_batched.py` - Grammarly CoEdIT evaluation
- `checkpoints/flan_t5_large_bea2019/evaluation_results.txt` - Results
- `checkpoints/grammarly_coedit_results.txt` - Grammarly results

**Documentation:** `documentation/phase_4_evaluation/README.md`

---

### Phase 5: API & Frontend (Complete)
- [x] FastAPI backend with 4 endpoints
- [x] Pydantic v2 request/response schemas
- [x] CORS middleware for frontend communication
- [x] Word-level diff extraction using difflib
- [x] T5 output post-processing (tokenization artifact cleanup)
- [x] React 18 + TypeScript 5 + TailwindCSS 3 frontend
- [x] Vite dev server with API proxy
- [x] End-to-end testing verified

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/correct` | Correct single text |
| POST | `/api/v1/correct/batch` | Correct multiple texts |
| GET | `/api/v1/health` | Health check with uptime |
| GET | `/api/v1/model/info` | Model information |

**Key Files:**
- `src/api/main.py` - FastAPI app with T5GEC model loading
- `src/api/routes.py` - Route handlers with diff extraction
- `src/api/models.py` - Pydantic schemas
- `frontend/src/App.tsx` - Main React app
- `frontend/src/components/TextEditor.tsx` - Text input component
- `frontend/src/components/CorrectionPanel.tsx` - Corrections display
- `frontend/src/services/api.ts` - API client (Axios)

**Documentation:** `documentation/phase_5_api_frontend/README.md`

---

### Phase 6: Llama 3.2-3B Fine-tuning (Complete)
- [x] LlamaGEC model wrapper with LoRA, SDPA attention, gradient checkpointing
- [x] Dynamic padding collate function (per-batch padding, not global max_length)
- [x] Step-based crash recovery checkpointing (every 500 optimizer steps)
- [x] Resume-from-checkpoint with LoRA weight restoration
- [x] Full training on BEA 2019 (3 epochs, ~11 hours total on RTX 4080 SUPER)
- [x] Evaluation on BEA 2019 test set (4,206 samples)
- [x] API updated to support all 3 models (llama, t5, coedit)
- [x] Frontend updated with model selector (Llama as default)

**Why 30h → 11h (Training Optimizations):**

| Technique | Implementation | Benefit |
|-----------|---------------|---------|
| SDPA | `attn_implementation="sdpa"` in from_pretrained | Built-in FlashAttention, fixed GQA batch_size=8 bottleneck |
| Dynamic Padding | `padding=False` in tokenizer + `make_collate_fn()` | ~2.9x less wasted attention compute per batch |
| Gradient Checkpointing | `model.gradient_checkpointing_enable(use_reentrant=False)` + `enable_input_require_grads()` | 40-50% VRAM savings → doubled batch size |
| Larger Batch | bs=4 → bs=8 (ga=8 → ga=4, effective batch=32 unchanged) | 2x samples per forward pass |

**Benchmark Results (RTX 4080 SUPER 16GB):**

| Config | ms/batch | Throughput | VRAM |
|--------|----------|-----------|------|
| Old: bs=4, seq=256, no SDPA | 1355ms | 3.0 samp/s | 15.4GB |
| New: bs=8, SDPA+dynpad+gradckpt | 521ms | 15.3 samp/s | 8.1GB |
| **Speedup** | **2.6x** | **5.1x** | **47% less** |

**Training Configuration:**
- Model: meta-llama/Llama-3.2-3B-Instruct (3B parameters)
- Trainable: 24,313,856 parameters (0.75% — LoRA r=16, alpha=32)
- Dataset: BEA 2019 (75,691 train / 4,203 val / 4,206 test)
- Batch size: 8 (effective: 32 with gradient_accumulation_steps=4)
- Learning rate: 2e-4 with linear warmup (709 steps) + decay
- Mixed precision: bfloat16
- Epochs: 3 (7,096 total optimizer steps)
- Step checkpoints: every 500 steps (14 checkpoints saved)

**Training Loss Progression:**
- Step 500 (epoch 1, batch 2000): loss = 0.1671
- Step 1000 (epoch 1, batch 4000): loss = 0.1457
- Step 2000 (epoch 1, end): loss = 0.1324
- Step 2500 (epoch 2, batch 540): loss = 0.0912
- Step 4000 (epoch 2, batch 6540): loss = 0.0941
- Step 5000 (epoch 3, batch 1080): loss = 0.0656
- Step 6000 (epoch 3, batch 5080): loss = 0.0671
- Step 7095 (epoch 3, end): loss = 0.1313
- **Best validation loss: 0.1417**

**Evaluation Results (BEA 2019 test set, 4206 samples):**
- F0.5: 0.0303 | Precision: 0.0244 | Recall: 0.9515 | GLEU: 0.7431
- Correction rate: 98.9% (over-correcting — rewrites sentences)
- TP/FP/FN: 255 / 10,210 / 13
- Root cause: Chat-format LLMs paraphrase instead of minimally editing
- Fix planned: Constrained decoding or DPO fine-tuning

**Key Files Modified:**
- `src/models/llama_gec.py` — SDPA + gradient checkpointing + LoRA
- `src/data/preprocess.py` — Dynamic padding (no-pad tokenize + collate_fn)
- `src/training/train.py` — Resume logic, collate_fn wiring, step checkpointing
- `src/training/utils.py` — `save_step_checkpoint()`, `find_latest_step_checkpoint()`
- `src/api/main.py` — Llama model loading on startup (path: checkpoints/llama32_bea2019/llama_gec_lora)
- `src/api/models.py` — Pattern updated to allow "llama" model name
- `frontend/src/components/TextEditor.tsx` — Added Llama option to dropdown
- `frontend/src/App.tsx` — Default model changed to "llama"
- `frontend/src/services/api.ts` — Default model "llama", timeout 300s
- `frontend/.env` — VITE_API_BASE_URL=http://localhost:9000/api/v1
- `train_llama32_bea2019.py` — Updated bs=8, ga=4, save_steps=500
- `watch_training.py` — Fixed Unicode (Windows cp1252), fixed tqdm regex

**Checkpoints:**
```
checkpoints/llama32_bea2019/
  best_model.pt                    — Best validation checkpoint
  checkpoint_epoch_1.pt            — End of epoch 1
  checkpoint_epoch_2.pt            — End of epoch 2
  checkpoint_epoch_3.pt            — End of epoch 3
  llama_gec_lora/                  — Final LoRA adapter (loaded by API)
    adapter_config.json
    adapter_model.safetensors      — ~97MB
    tokenizer.json / tokenizer_config.json
  step_checkpoints/
    step_0000500/  step_0001000/  step_0001500/  step_0002000/
    step_0002500/  step_0003000/  step_0003500/  step_0004000/
    step_0004500/  step_0005000/  step_0005500/  step_0006000/
    step_0006500/  step_0007000/
    (each contains: lora_adapter/, optimizer.pt, scheduler.pt, meta.json)
  evaluation_predictions.csv       — Full predictions (4206 samples)
  evaluation_results.txt           — F0.5 / GLEU scores
```

**Documentation:** `documentation/phase_6_llama_training/README.md`

---

### Phase 7: Deployment (In Progress)
- [x] Backend `Dockerfile` (python:3.11-slim, CUDA via PyTorch wheel, `/api` health check)
- [x] `requirements-api.txt` — inference-only deps (no training/eval tools, ~60% smaller)
- [x] `.dockerignore` — excludes checkpoints, data, logs from build context (fast builds)
- [x] `frontend/.dockerignore` — excludes `.env` so localhost URL not baked in; Nginx proxy handles routing
- [x] `docker-compose.yml` — backend + frontend services, GPU via NVIDIA Container Toolkit, volume mounts for checkpoints + HuggingFace cache
- [ ] Hugging Face Spaces deployment
- [ ] Inference optimization (quantization, ONNX)

**Docker Architecture:**
```
Host port 3000 → [Nginx :80] → serves React SPA
                              → /api/* → [FastAPI :8000] (backend container)
Host port 9000 → [FastAPI :8000] (direct API access)
```

**Key Design Decisions:**
- Checkpoints mounted as read-only volumes (not baked into image — too large: ~2GB+)
- HuggingFace cache in named Docker volume (persists CoEdIT download across restarts)
- `frontend/.dockerignore` excludes `.env` → Vite uses fallback `"/api/v1"` → Nginx proxies to backend
- Backend needs NVIDIA Container Toolkit (`--gpus all`) — CPU-only fallback available but slow
- `HF_TOKEN` passed via env for gated model downloads (Llama 3.2)

**New Files:**
- `Dockerfile` — Backend image
- `docker-compose.yml` — Multi-service orchestration
- `requirements-api.txt` — Inference-only dependencies
- `.dockerignore` — Backend build context exclusions
- `frontend/.dockerignore` — Frontend build context exclusions

**Run Instructions:**
```bash
# One-time setup: export HuggingFace token (for Llama download if needed)
export HF_TOKEN=your_token_here   # Linux/Mac
$env:HF_TOKEN="your_token"        # PowerShell

# Build and start all services
docker compose up --build

# Detached mode
docker compose up -d --build

# View logs
docker compose logs -f backend

# Access
# Frontend: http://localhost:3000
# API docs: http://localhost:9000/docs
# Health:   http://localhost:9000/api/v1/health
```

---

## System Architecture

```
User → React Frontend (localhost:5173)
           |
           | Vite Proxy (/api → :8000)
           v
    FastAPI Backend (localhost:8000)
    ├── POST /api/v1/correct
    ├── POST /api/v1/correct/batch
    ├── GET  /api/v1/health
    └── GET  /api/v1/model/info
           |
           v
    FLAN-T5-Large + LoRA (780M params, 18MB adapter)
    - bfloat16 precision
    - Beam search (num_beams=4)
    - @torch.no_grad() for inference
    - clean_t5_output() post-processing
           |
           v
    Response: {
        corrected_text, corrections[],
        confidence_score, processing_time_ms
    }
```

---

## Hardware Specifications

- CPU: Intel i9-14900K (24 cores)
- RAM: 64GB DDR5
- GPU: NVIDIA RTX 4080 SUPER 16GB
- Storage: NVMe SSD

**Actual Performance:**
- Training: ~5.5 hours (3 epochs, 75K samples, ~5 it/s)
- VRAM usage: ~8-10GB during training
- Inference: ~500ms per sentence (GPU)

---

## Checkpoint Files

```
checkpoints/flan_t5_large_bea2019/
  best_model.pt              (1.9 GB) - Best validation loss
  checkpoint_epoch_1.pt      (1.9 GB) - After epoch 1
  checkpoint_epoch_2.pt      (1.9 GB) - After epoch 2
  checkpoint_epoch_3.pt      (1.9 GB) - After epoch 3
  evaluation_results.txt     - F0.5 and GLEU scores
  evaluation_predictions.csv - Full predictions (4,206 samples)
  llama_gec_lora/
    adapter_config.json      - LoRA configuration
    adapter_model.safetensors (18 MB) - LoRA weights
    tokenizer.json           - Tokenizer
    tokenizer_config.json    - Tokenizer config
```
