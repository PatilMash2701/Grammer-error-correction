# Model Architecture Deep Dive

**Status:** Complete
**Date:** February 2026

---

## Models Evaluated

### 1. FLAN-T5-Large (Chosen)

**Architecture:** Encoder-decoder transformer (T5 family)
**Parameters:** 780M total, 4.7M trainable with LoRA (0.6%)
**Base:** google/flan-t5-large

**Key Properties:**
- Encoder-decoder: natural fit for seq2seq tasks like GEC
- Input: `grammar: {text}` prefix format
- Output: corrected text directly
- Pre-trained with instruction tuning (FLAN collection)
- Supports bfloat16 natively

**Architecture Details:**
- Encoder: 24 layers, 16 attention heads, 1024 hidden dim
- Decoder: 24 layers, 16 attention heads, 1024 hidden dim
- Vocabulary: 32,128 tokens (SentencePiece)
- Max sequence length: 512 tokens (using 256 for training)
- Relative positional encoding (T5-style)

**LoRA Configuration:**
```
r=16, alpha=32, dropout=0.05
Target modules: q, v (attention projections)
Trainable: 4.7M / 780M (0.6%)
Adapter size: 18MB
```

**Inference:**
- Beam search: num_beams=4, early_stopping=True
- Generation: max_new_tokens=128
- Post-processing: `clean_t5_output()` fixes tokenization artifacts

---

### 2. Llama 3.2-3B-Instruct (Evaluated, Not Used in Production)

**Architecture:** Decoder-only transformer
**Parameters:** 3.21B total

**Why not chosen for production:**
- Decoder-only models are suboptimal for seq2seq GEC tasks
- 3-5x slower inference (autoregressive generation of full output)
- Heavier memory footprint
- T5 encoder-decoder processes input and output in parallel

**Architecture Details:**
- 32 layers, 32 attention heads, 3072 hidden dim
- Grouped-query attention (GQA)
- RoPE positional embeddings
- SwiGLU activation
- Vocabulary: 128,256 tokens
- Chat template with system/user/assistant roles

---

### 3. BART-base (Originally Planned, Replaced by T5)

**Architecture:** Encoder-decoder transformer
**Parameters:** 140M

**Why replaced:**
- Too small for complex grammar corrections
- T5-Large offers significantly more capacity (780M vs 140M)
- FLAN instruction tuning provides better zero-shot starting point

---

### 4. Grammarly CoEdIT-large (Evaluated as Alternative)

**Architecture:** T5-based (770M params)
**From:** Grammarly official multi-task model

**Evaluation Results:**
- F0.5: 0.0548 (vs 0.3201 for our model)
- Massively over-corrects (98% correction rate, 16K false positives)
- Multi-task design dilutes GEC-specific performance

---

## Comparison Summary

| Model | Params | Type | F0.5 | Inference | Status |
|-------|--------|------|------|-----------|--------|
| FLAN-T5-Large + LoRA | 780M | Enc-Dec | 0.3201 | ~500ms | Production |
| Llama 3.2-3B | 3.21B | Dec-only | N/A | ~1500ms | Dev only |
| BART-base | 140M | Enc-Dec | N/A | ~200ms | Replaced |
| CoEdIT-large | 770M | Enc-Dec | 0.0548 | ~500ms | Rejected |

---

## Key Insight: Encoder-Decoder vs Decoder-Only for GEC

Encoder-decoder models (T5, BART) are preferred for GEC because:
1. The encoder processes the full input in parallel, understanding context
2. The decoder generates corrections while attending to the full input
3. Cross-attention between encoder and decoder allows precise word-level corrections
4. More efficient than decoder-only models that must regenerate the entire sequence

Decoder-only models (Llama, GPT) have disadvantages for GEC:
1. Must generate the entire output autoregressively
2. No bidirectional encoding of the input
3. Slower inference (3-5x) due to sequential generation
4. More prone to hallucination on short correction tasks
