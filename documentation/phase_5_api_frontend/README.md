# Phase 5: API & Frontend - Documentation

**Status:** Complete
**Date:** February 2026

---

## Overview

Built a complete web application for the grammar correction system:
- FastAPI REST backend serving the fine-tuned FLAN-T5-Large model
- React + TypeScript frontend with text editing and correction display
- End-to-end tested with Vite dev proxy

---

## Backend: FastAPI

### Architecture

```
FastAPI App (src/api/main.py)
├── Lifespan: Loads T5GEC model on startup
├── CORS: localhost:3000, localhost:5173
├── Router: /api/v1 prefix
│   ├── POST /correct        → correct_text()
│   ├── POST /correct/batch  → correct_batch()
│   ├── GET  /health         → health_check()
│   └── GET  /model/info     → model_info()
└── Root: GET / → API info
```

### Model Loading

On startup, the API loads the T5GEC model from the LoRA checkpoint:
```python
from src.models.t5_gec import T5GEC
model = T5GEC.from_pretrained("checkpoints/flan_t5_large_bea2019/llama_gec_lora")
```

- Model loaded once during lifespan context manager
- Stored as global variable for request handlers
- bfloat16 precision on CUDA
- Startup time tracked for uptime reporting

### API Endpoints

#### POST `/api/v1/correct`

Correct a single text.

**Request:**
```json
{
  "text": "She go to school yesterday.",
  "num_beams": 4
}
```

**Response:**
```json
{
  "original_text": "She go to school yesterday.",
  "corrected_text": "She went to school yesterday.",
  "corrections": [
    {
      "original": "go",
      "corrected": "went",
      "error_type": "grammar",
      "position": {"start": 4, "end": 6}
    }
  ],
  "confidence_score": 0.9,
  "processing_time_ms": 500.0
}
```

#### POST `/api/v1/correct/batch`

Correct multiple texts in one request.

**Request:**
```json
{
  "texts": [
    "She go to school.",
    "They was playing.",
    "This is correct."
  ],
  "num_beams": 4
}
```

**Response:**
```json
{
  "results": [
    {"original_text": "She go to school.", "corrected_text": "She went to school.", ...},
    {"original_text": "They was playing.", "corrected_text": "They were playing.", ...},
    {"original_text": "This is correct.", "corrected_text": "This is correct.", "corrections": [], ...}
  ],
  "total_processing_time_ms": 548.2
}
```

#### GET `/api/v1/health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "checkpoints/flan_t5_large_bea2019/llama_gec_lora",
  "version": "0.1.0",
  "uptime_seconds": 120.5
}
```

#### GET `/api/v1/model/info`

**Response:**
```json
{
  "model_name": "checkpoints/flan_t5_large_bea2019/llama_gec_lora",
  "model_type": "T5ForConditionalGeneration (LoRA)",
  "parameters": 780000000,
  "max_length": 256,
  "device": "cuda"
}
```

### Pydantic Schemas

**File:** `src/api/models.py`

| Schema | Fields |
|--------|--------|
| `CorrectionRequest` | `text` (1-5000 chars), `num_beams` (1-10, default 4) |
| `CorrectionResponse` | `original_text`, `corrected_text`, `corrections[]`, `confidence_score`, `processing_time_ms` |
| `CorrectionDiff` | `original`, `corrected`, `error_type`, `position: {start, end}` |
| `PositionSpan` | `start: int`, `end: int` |
| `BatchCorrectionRequest` | `texts[]` (max 50), `num_beams` |
| `BatchCorrectionResponse` | `results[]`, `total_processing_time_ms` |
| `HealthResponse` | `status`, `model_loaded`, `model_name`, `version`, `uptime_seconds` |
| `ModelInfoResponse` | `model_name`, `model_type`, `parameters`, `max_length`, `device` |

### Word-Level Diff Extraction

**File:** `src/api/routes.py` → `extract_corrections()`

Since T5GEC only returns corrected text (not individual edits), diffs are extracted at the API layer:

1. Split original and corrected text into words
2. Use `difflib.SequenceMatcher` to find edit operations
3. Map each edit to character positions in the original text
4. Return as `CorrectionDiff` list with `{original, corrected, error_type, position}`

### T5 Output Post-Processing

**File:** `src/api/routes.py` → `clean_t5_output()`

T5 tokenizer sometimes introduces spaces before punctuation (e.g., "yesterday ." instead of "yesterday."). The cleanup function:
- Removes spaces before `.,!?;:'")}]`
- Removes spaces after opening brackets/quotes
- Applied before diff extraction to avoid spurious corrections

---

## Frontend: React + TypeScript

### Tech Stack
- React 18
- TypeScript 5
- TailwindCSS 3
- Vite 5 (build tool and dev server)
- Axios (HTTP client)

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `App` | `src/App.tsx` | Main layout, state management, API calls |
| `TextEditor` | `src/components/TextEditor.tsx` | Text input area |
| `CorrectionPanel` | `src/components/CorrectionPanel.tsx` | Displays corrections with error type badges |
| `Header` | `src/components/Header.tsx` | App header |

### API Client

**File:** `frontend/src/services/api.ts`

Axios instance configured with:
- `baseURL: "/api/v1"` — all requests go through Vite proxy
- Methods: `correctText()`, `correctBatch()`, `checkHealth()`

### TypeScript Types

**File:** `frontend/src/types/index.ts`

Frontend types mirror backend Pydantic schemas:
- `CorrectionDiff` — `{original, corrected, error_type, position: {start, end}}`
- `CorrectionRequest` — `{text, num_beams?}`
- `CorrectionResponse` — `{original_text, corrected_text, corrections[], confidence_score, processing_time_ms}`
- `BatchCorrectionRequest/Response` — batch variants
- `HealthResponse` — `{status, model_loaded, version, uptime_seconds}`

### Vite Proxy Configuration

**File:** `frontend/vite.config.ts`

```typescript
server: {
  proxy: {
    "/api": {
      target: "http://localhost:8000",
      changeOrigin: true,
    }
  }
}
```

Routes `/api/*` from Vite dev server (`:5173`) to FastAPI backend (`:8000`).

---

## Running the System

### Start Backend
```bash
python -m src.api.main
# Loads model, starts on http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Start Frontend
```bash
cd frontend && npm run dev
# Vite dev server on http://localhost:5173
# Proxies API calls to :8000
```

### Build Frontend for Production
```bash
cd frontend && npm run build
# Output in frontend/dist/
```

---

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `src/api/models.py` | Rewritten | Aligned Pydantic schemas with frontend types |
| `src/api/main.py` | Rewritten | Switched BartGEC → T5GEC, added CORS, uptime |
| `src/api/routes.py` | Rewritten | Added diff extraction, timing, T5 cleanup |
| `src/models/t5_gec.py` | Edited | Added `num_beams` to `correct_batch()` |
| `frontend/src/services/api.ts` | Edited | Changed baseURL to `/api/v1` |
| `frontend/src/components/CorrectionPanel.tsx` | Edited | Fixed quoted object key syntax |
| `frontend/tsconfig.node.json` | Rewritten | Added `composite: true` for build |
| `frontend/src/vite-env.d.ts` | Created | Vite type declarations |

---

## Issues Encountered & Fixed

1. **Schema mismatch** — Backend `confidence` vs frontend `confidence_score` → Rewrote Pydantic schemas
2. **Wrong model** — API loaded BartGEC (not implemented) → Switched to T5GEC
3. **Empty corrections** — T5GEC only returns text → Implemented difflib-based extraction
4. **URL mismatch** — Frontend called `/api/correct`, backend at `/api/v1/correct` → Fixed baseURL
5. **TypeScript build errors** — Missing `composite: true`, unquoted key, missing `vite-env.d.ts` → All fixed
6. **T5 tokenization artifacts** — "yesterday ." → Added `clean_t5_output()` post-processing
7. **`correct_batch` missing `num_beams`** — Added parameter to T5GEC method signature

---

## Test Results

### API Endpoints (Verified)

```bash
# Single correction
POST /api/v1/correct {"text": "She go to school yesterday."}
→ 200 OK: "She went to school yesterday." (1 correction, 500ms)

# Batch correction
POST /api/v1/correct/batch {"texts": ["She go to school.", "They was playing.", "This is correct."]}
→ 200 OK: 3 results (2 corrections, 548ms total)

# Health check
GET /api/v1/health
→ 200 OK: {"status": "healthy", "model_loaded": true, ...}

# Model info
GET /api/v1/model/info
→ 200 OK: {"model_type": "T5ForConditionalGeneration (LoRA)", "parameters": 780000000, ...}

# Frontend proxy
POST http://localhost:5173/api/v1/correct → proxied to :8000 successfully
```

### Frontend Build
```bash
npm run build → Success (no TypeScript errors)
npm run dev → Vite dev server running on :5173
```
