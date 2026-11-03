# Project Overview and Specifications

**Status:** Complete
**Date:** February 2026

---

## Project Goal

Build a production-grade Grammar Error Correction (GEC) system that detects and corrects grammatical errors in English text, similar to Grammarly. The system includes a fine-tuned language model, REST API, and web interface.

---

## Specifications

### Functional Requirements
1. Accept English text input (1-5000 characters)
2. Return corrected text with word-level change annotations
3. Support single and batch correction requests
4. Provide confidence scores for corrections
5. Track processing time per request
6. Expose health check and model info endpoints

### Non-Functional Requirements
- Inference latency: < 500ms per sentence on GPU
- Model size: < 500MB (LoRA adapter: 18MB actual)
- API uptime: Health endpoint for monitoring
- Frontend: Responsive web interface

### Target Metrics
| Metric | Target | Achieved |
|--------|--------|----------|
| F0.5 (BEA 2019) | > 0.60 | 0.3201 |
| GLEU | > 0.55 | 0.9245 |
| Precision | > 0.50 | 0.2744 |
| Recall | > 0.50 | 0.9588 |
| Inference Speed | < 500ms | ~500ms |

---

## Architecture Decisions

### Model Selection
- **Chosen:** FLAN-T5-Large (780M params) with LoRA fine-tuning
- **Alternatives considered:** Llama 3.2-3B (decoder-only, slower inference), BART-base (140M, less capacity), Grammarly CoEdIT-large (over-corrects)
- **Rationale:** Encoder-decoder architecture is natural fit for GEC (seq2seq), moderate size for 16GB GPU, strong baseline from FLAN instruction tuning

### Training Strategy
- **Method:** LoRA (Low-Rank Adaptation) - trains only 0.6% of model parameters
- **Dataset:** BEA 2019 (84K samples)
- **Precision:** bfloat16 (avoids NaN issues seen with float16 on T5)

### Tech Stack
- **Backend:** FastAPI + Uvicorn
- **Frontend:** React 18 + TypeScript 5 + TailwindCSS 3 + Vite 5
- **ML:** PyTorch + Transformers + PEFT
- **Evaluation:** ERRANT (F0.5), NLTK (GLEU)

---

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Planning & Setup | Complete |
| 2 | Data Engineering | Complete |
| 3 | Model Development & Training | Complete |
| 4 | Evaluation | Complete |
| 5 | API & Frontend | Complete |
| 6 | Deployment | Next |

---

## Hardware

- CPU: Intel i9-14900K (24 cores)
- RAM: 64GB DDR5
- GPU: NVIDIA RTX 4080 SUPER 16GB
- Storage: NVMe SSD
