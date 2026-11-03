# Evaluation Metrics

**Status:** Complete
**Date:** February 2026

---

## Primary Metrics

### 1. F0.5 Score (Primary)

**Standard for:** GEC evaluation (CoNLL-2014 benchmark)
**Implementation:** `src/training/evaluate.py` → `compute_f05()`

**Formula:**
```
F0.5 = (1 + 0.5^2) * (Precision * Recall) / (0.5^2 * Precision + Recall)
```

**Why F0.5 (not F1)?**
- GEC values precision over recall (beta=0.5 weights precision 2x)
- A false correction (FP) is worse than a missed error (FN)
- Users prefer fewer but accurate corrections
- Standard adopted by CoNLL-2013/2014 shared tasks

**How it works:**
1. Use ERRANT to annotate edits between source → prediction and source → reference
2. Edits are keyed by (start_position, end_position, correction_string)
3. **TP:** Edit present in both prediction and reference
4. **FP:** Edit in prediction but not reference (wrong correction)
5. **FN:** Edit in reference but not prediction (missed error)

**Our Results:**
| Metric | Value |
|--------|-------|
| F0.5 | 0.3201 |
| Precision | 0.2744 |
| Recall | 0.9588 |
| TP | 163 |
| FP | 431 |
| FN | 7 |

**Interpretation:** High recall (catches 96% of errors) but low precision (only 27% of corrections are correct). The model makes many false positive corrections.

---

### 2. GLEU Score

**Standard for:** Fluency evaluation (JFLEG benchmark)
**Implementation:** `src/training/evaluate.py` → `compute_gleu()`

**How it works:**
1. Compare n-grams (1-gram through 4-gram) across source, prediction, and reference
2. Count prediction n-grams that match reference (correct changes)
3. GLEU = matched_ngrams / total_prediction_ngrams
4. Averaged across all sentences

**Our Results:** GLEU = 0.9245

**Interpretation:** High GLEU indicates the model produces fluent output that closely matches reference corrections at the n-gram level.

---

### 3. Per-Error-Type Analysis

**Implementation:** `src/training/evaluate.py` → `per_error_analysis()`

Uses ERRANT error type codes to break down performance by category:
- `R:VERB:SVA` — Subject-verb agreement replacements
- `M:DET` — Missing determiners
- `R:PREP` — Preposition replacements
- `R:SPELL` — Spelling corrections
- `U:PUNCT` — Unnecessary punctuation
- etc.

---

## Support Metrics

### Correction Rate
- Percentage of input sentences that the model modifies
- Our model: 12.8% (conservative, only corrects when confident)
- Grammarly CoEdIT: 98.15% (over-corrects nearly everything)

### Exact Match
- Percentage of predictions that match the reference exactly
- Our model: 29.9%

### Latency
- Time to process a single sentence end-to-end
- Our model: ~500ms on GPU (RTX 4080 SUPER)

---

## ERRANT Framework

**ERRANT** (ERRor ANnotation Toolkit) is the standard tool for GEC evaluation.

**How it works:**
1. Parse source and hypothesis/reference with spaCy
2. Extract minimal edits between the two
3. Classify each edit into error types (55+ categories)
4. Compare hypothesis edits to reference edits for TP/FP/FN

**Usage in our system:**
```python
import errant
annotator = errant.load("en")
src_parsed = annotator.parse(source_text)
pred_parsed = annotator.parse(predicted_text)
edits = annotator.annotate(src_parsed, pred_parsed)
```

---

## Model Comparison Results

| Model | F0.5 | Precision | Recall | GLEU | Corr. Rate |
|-------|------|-----------|--------|------|------------|
| FLAN-T5-Large + LoRA | 0.3201 | 0.2744 | 0.9588 | 0.9245 | 12.8% |
| Grammarly CoEdIT-large | 0.0548 | 0.0443 | 0.9792 | 0.6849 | 98.15% |

**Conclusion:** Our fine-tuned model is the better choice despite the lower absolute F0.5, because it achieves 6x better precision-weighted performance than the off-the-shelf alternative.

---

## References

- CoNLL-2014 Shared Task: Ng et al., 2014
- ERRANT: Bryant et al., "Automatic Annotation and Evaluation of Error Types for Grammatical Error Correction" (2017)
- GLEU: Napoles et al., "Ground Truth for Grammatical Error Correction Metrics" (2015)
- JFLEG: Napoles et al., "JFLEG: A Fluency Corpus and Benchmark for Grammatical Error Correction" (2017)
