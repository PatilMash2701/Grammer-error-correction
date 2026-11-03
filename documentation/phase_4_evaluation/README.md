# Phase 4: Evaluation - Documentation

**Status:** Complete
**Date:** February 2026

---

## Overview

Evaluated the fine-tuned FLAN-T5-Large model on the BEA 2019 test set using industry-standard GEC metrics. Also tested Grammarly's CoEdIT-large as an alternative model and performed comparative analysis.

---

## Evaluation Setup

### Test Dataset
- **Dataset:** BEA 2019 test split
- **Samples:** 4,206 sentence pairs
- **Format:** CSV with `source` and `target` columns

### Model Under Test
- **Model:** FLAN-T5-Large (780M params) with LoRA adapters
- **Checkpoint:** `checkpoints/flan_t5_large_bea2019/llama_gec_lora/`
- **Adapter size:** 18MB
- **Precision:** bfloat16

### Alternative Model Tested
- **Model:** Grammarly CoEdIT-large (770M params)
- **Source:** `grammarly/coedit-large` on HuggingFace
- **Type:** Multi-task instruction-tuned T5 model

---

## Metrics Implementation

### F0.5 Score
**File:** `src/training/evaluate.py` → `compute_f05()`

Uses the ERRANT framework to:
1. Parse source, prediction, and reference into spaCy docs
2. Extract minimal edit annotations between sentence pairs
3. Key edits by (start_position, end_position, correction_string)
4. Compare prediction edits against reference edits
5. Compute precision, recall, and F0.5 (beta=0.5)

### GLEU Score
**File:** `src/training/evaluate.py` → `compute_gleu()`

N-gram based fluency metric:
1. Extract 1-gram through 4-gram from source, prediction, and reference
2. Count matching n-grams between prediction and reference
3. Divide by total prediction n-grams
4. Average across all sentences

### Per-Error-Type Analysis
**File:** `src/training/evaluate.py` → `per_error_analysis()`

Breaks down results by ERRANT error categories (R:VERB:SVA, M:DET, R:PREP, etc.) with individual precision/recall/F0.5 per category.

---

## Results

### FLAN-T5-Large + LoRA (Our Model)

```
Test samples:     4,206
F0.5 Score:       0.3201
Precision:        0.2744
Recall:           0.9588
TP:               163
FP:               431
FN:               7
GLEU Score:       0.9245
Avg Latency:      2,060ms (batched over 4K samples)
Correction Rate:  12.8%
Exact Match:      29.9%
```

**Interpretation:**
- High recall (0.96): Model catches most errors that need correction
- Low precision (0.27): Many false positives (corrections where none needed)
- Conservative correction rate (12.8%): Only modifies sentences where confident
- High GLEU (0.92): Corrected text is fluent and natural

### Grammarly CoEdIT-large

```
Test samples:     4,206
F0.5 Score:       0.0548
Precision:        0.0443
Recall:           0.9792
TP:               753
FP:               16,232
FN:               16
GLEU Score:       0.6849
Correction Rate:  98.15%
Evaluation Time:  6.8 minutes
```

**Interpretation:**
- Near-total correction rate (98%): Modifies almost every sentence
- Massive false positives (16,232): Over-corrects aggressively
- Very low precision (0.04): Only 4% of corrections are correct
- Multi-task training dilutes GEC-specific accuracy

---

## Comparative Analysis

| Metric | FLAN-T5-Large + LoRA | Grammarly CoEdIT | Winner |
|--------|---------------------|------------------|--------|
| F0.5 | **0.3201** | 0.0548 | Ours (6x better) |
| Precision | **0.2744** | 0.0443 | Ours (6x better) |
| Recall | 0.9588 | **0.9792** | CoEdIT (marginal) |
| GLEU | **0.9245** | 0.6849 | Ours |
| FP count | **431** | 16,232 | Ours (38x fewer) |
| Correction Rate | **12.8%** | 98.15% | Ours (conservative) |

### Key Findings

1. **Our model is 6x better on F0.5** — the primary GEC metric
2. **CoEdIT produces 38x more false positives** — unacceptable for production
3. **Both models have high recall** — they catch errors well, but CoEdIT also "corrects" things that aren't errors
4. **Our GLEU is higher** — our corrections are more fluent and natural
5. **Conservative is better** — the 12.8% correction rate means the model only acts when it's reasonably confident

### Decision
**Keep FLAN-T5-Large + LoRA as the production model.** CoEdIT is rejected due to catastrophic over-correction.

---

## Known Limitations

1. **F0.5 of 0.32 is below target (0.60)** — model needs improvement
2. **Precision-recall imbalance** — too many false positives relative to true positives
3. **Some error types not well handled** — complex rewriting, word order changes
4. **Tokenization artifacts** — T5 sometimes adds spaces before punctuation (fixed with `clean_t5_output()` post-processing)

### Potential Improvements
- Expand LoRA config (r=32, target all projection layers)
- Train on larger combined datasets (BEA 2019 + Lang-8)
- Switch to BART-Large (purpose-built denoising architecture)
- Confidence-based filtering (only apply corrections above threshold)

---

## Files

| File | Description |
|------|-------------|
| `src/training/evaluate.py` | F0.5, GLEU, per-error-type metric implementations |
| `evaluate_bea2019.py` | Main evaluation script for fine-tuned model |
| `test_grammarly_batched.py` | Grammarly CoEdIT evaluation with batched inference |
| `test_grammarly_model.py` | Grammarly CoEdIT single-sample evaluation |
| `checkpoints/flan_t5_large_bea2019/evaluation_results.txt` | Numeric results |
| `checkpoints/flan_t5_large_bea2019/evaluation_predictions.csv` | Full predictions |
| `checkpoints/grammarly_coedit_results.txt` | Grammarly results |
| `checkpoints/grammarly_coedit_evaluation.csv` | Grammarly full predictions |

---

## References

- ERRANT: Bryant et al., "Automatic Annotation and Evaluation of Error Types for Grammatical Error Correction" (2017)
- GLEU: Napoles et al., "Ground Truth for Grammatical Error Correction Metrics" (2015)
- BEA-2019: https://www.cl.cam.ac.uk/research/nl/bea2019st/
- Grammarly CoEdIT: https://huggingface.co/grammarly/coedit-large
