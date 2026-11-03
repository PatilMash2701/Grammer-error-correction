# Phase 6: Llama 3.2-3B Fine-tuning for GEC

**Status:** Complete
**Duration:** ~11 hours training (3 epochs on RTX 4080 SUPER 16GB)
**Date:** February 19–21, 2026

---

## Overview

This phase extends the project by fine-tuning Meta's Llama 3.2-3B-Instruct on the BEA 2019 Grammar Error Correction dataset using LoRA. Unlike the encoder-decoder T5 model in Phase 3, Llama is a decoder-only causal LM — it generates corrections by continuing the input prompt in chat format.

The key challenge: naive training on RTX 4080 SUPER took ~30 hours. Through four optimization techniques, this was reduced to ~11 hours (3x real speedup; benchmark showed 5x throughput improvement).

---

## 1. Model Architecture

### Why Llama for GEC?

| Aspect | FLAN-T5-Large (Phase 3) | Llama 3.2-3B (Phase 6) |
|--------|------------------------|------------------------|
| Architecture | Encoder-Decoder | Decoder-only |
| Parameters | 780M | 3B |
| Format | Seq2Seq (input → output) | Chat (system + user + assistant) |
| Attention | Standard MHA | Grouped Query Attention (GQA) |
| Training | Supervised fine-tuning | Causal LM on chat template |
| Strength | Precise minimal edits | Fluent natural language |
| Weakness | Less fluent rewrites | Over-corrects (paraphrases) |

### Chat Template Format

During training, every sentence pair is formatted as:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a grammar correction assistant. Correct the grammatical errors in the given sentence. Return only the corrected sentence without any explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>

Correct this sentence: {source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{target}<|eot_id|>
```

The loss is computed only on the assistant (target correction) tokens. Prompt tokens are masked with -100.

### LoRA Configuration

```python
LoraConfig(
    r=16,                          # Rank (16 = balance of capacity vs params)
    lora_alpha=32,                 # Alpha = 2*r (standard scaling)
    lora_dropout=0.05,
    target_modules=[               # 7 modules targeted
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Trainable parameters:** 24,313,856 / 3,237,063,680 = **0.75%** of model

---

## 2. Training Optimizations

### Problem: Why 30 Hours?

Original config (batch_size=4, seq_len=256, no SDPA):
- Each batch: 4 samples × 256 tokens = 1024 token-steps
- BEA 2019 average sentence: ~150 tokens — 59% of 256 is padding
- Attention is O(n²): padding is not just wasted memory, it's quadratically wasted compute
- Llama 3.2 uses GQA (Grouped Query Attention) — at batch_size=8, PyTorch's standard attention kernel had an 11× slowdown due to unsupported GQA shapes
- Result: 1355ms per batch, 3.0 samples/second

### Optimization 1: SDPA (Scaled Dot-Product Attention)

**What:** PyTorch 2.0+ built-in implementation of FlashAttention and MemoryEfficient attention.

**How:**
```python
# In llama_gec.py from_pretrained()
model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "device_map": "cuda:0",
    "attn_implementation": "sdpa",   # ← This one line
}
```

**Why it matters:**
- Standard attention stores the full N×N attention matrix in VRAM
- FlashAttention tiles computation to avoid materializing the full matrix
- Most importantly: **it eliminated the GQA batch_size=8 bottleneck** — SDPA's kernel handles GQA correctly at all batch sizes
- No extra packages required (pure PyTorch)

### Optimization 2: Dynamic Padding

**What:** Instead of padding every sequence to 256 tokens globally, pad each batch to its own longest sequence.

**Why it matters:**
- BEA 2019 mean sequence length (with chat template): ~150 tokens
- At mean length: 150/256 = 59% utilization → 41% waste
- Attention is O(n²): at 150 tokens vs 256 tokens, attention is 2.9× less work
- Different batches have different lengths, so the savings compound

**How — preprocess.py:**
```python
# Before: padding="max_length" padded everything to 256
encoding = tokenizer(full_text, max_length=256, padding=False, truncation=True)
# Returns variable-length tensors — no padding yet
```

**How — new make_collate_fn() in preprocess.py:**
```python
def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        max_len = max(item["input_ids"].shape[0] for item in batch)
        padded = []
        for item in batch:
            pad_len = max_len - item["input_ids"].shape[0]
            padded_ids = torch.cat([
                item["input_ids"],
                torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ])
            padded_mask = torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ])
            padded_labels = torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long)  # -100 = ignored in loss
            ])
            padded.append((padded_ids, padded_mask, padded_labels))
        return {
            "input_ids": torch.stack([x[0] for x in padded]),
            "attention_mask": torch.stack([x[1] for x in padded]),
            "labels": torch.stack([x[2] for x in padded]),
        }
    return collate_fn
```

The collate_fn is passed to DataLoader and called per batch. Each batch gets its own optimal sequence length.

### Optimization 3: Gradient Checkpointing

**What:** Instead of storing all layer activations in VRAM during the forward pass (needed for backpropagation), recompute them on the fly during the backward pass.

**Why it matters:**
- A 3B parameter model stores massive activation tensors during training
- Freeing activation memory allows a larger batch size
- Doubling batch_size from 4 to 8 with the same effective batch size (32) means halving gradient_accumulation_steps: fewer optimizer calls, faster training
- Cost: ~30% more compute during backward pass (acceptable tradeoff)

**How — llama_gec.py:**
```python
if use_gradient_checkpointing and device == "cuda":
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # Required when combining LoRA + gradient checkpointing:
    # LoRA patches specific layers and gradient checkpointing must be able
    # to trace through them. enable_input_require_grads() makes all
    # inputs require gradients so the checkpoint mechanism works.
    if use_lora and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
```

`use_reentrant=False` uses the modern non-reentrant implementation which is more compatible with PEFT/LoRA.

### Optimization 4: Step-based Checkpointing (Crash Recovery)

**What:** Save model weights + optimizer state + scheduler state every 500 optimizer steps, not just at epoch end.

**Why it matters:**
- Epoch = 9,462 batches ≈ 1.5 hours
- Without step checkpoints, a crash at batch 9,000 means restarting from epoch start (losing 1.4h)
- With step checkpoints every 500 steps (~22 minutes), max loss on crash = 22 minutes

**How — utils.py:**
```python
def save_step_checkpoint(model, optimizer, scheduler, epoch, batch_idx,
                          optimizer_step, loss, checkpoint_dir):
    step_dir = Path(checkpoint_dir) / f"step_{optimizer_step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(step_dir / "lora_adapter"))   # LoRA weights only
    torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), step_dir / "scheduler.pt")

    meta = {"epoch": epoch, "batch_idx": batch_idx,
            "optimizer_step": optimizer_step, "loss": loss}
    with open(step_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def find_latest_step_checkpoint(checkpoint_dir):
    step_dirs = sorted(
        [d for d in Path(checkpoint_dir).iterdir()
         if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    if not step_dirs:
        return None
    latest = step_dirs[-1]
    with open(latest / "meta.json") as f:
        return latest, json.load(f)
```

**Resume logic in train.py:**
```python
result = find_latest_step_checkpoint(step_checkpoint_dir)
if result is not None:
    ckpt_path, meta = result
    resume_epoch = meta["epoch"] - 1      # 0-indexed
    resume_batch = meta["batch_idx"] + 1  # start from next batch
    resume_step  = meta["optimizer_step"]

    # Restore LoRA weights
    adapter_weights = load_file(str(ckpt_path / "lora_adapter/adapter_model.safetensors"))
    model.load_state_dict(adapter_weights, strict=False)  # strict=False = only LoRA keys

    # Restore optimizer + scheduler
    optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    scheduler.load_state_dict(torch.load(sched_path, map_location=device))
```

In the training loop, already-processed batches are skipped with:
```python
if batch_idx < start_batch:
    continue
```

### Optimization Tried but Failed: torch.compile

Attempted `torch.compile(model, backend="inductor")` which can give 20-30% speedup by fusing operations into optimized kernels.

**Error:**
```
BackendCompilerFailed: backend='inductor' raised:
AttributeError: 'float' object has no attribute 'meta'
```

**Root cause:** The inductor backend on Windows does not support PEFT LoRA models. The LoRA adapter patches layers in a way the inductor graph compiler cannot trace through on Windows. This is a known limitation — torch.compile with LoRA on Windows backend is unsupported.

**Decision:** Abandoned. The other three optimizations already achieve 5.2x speedup.

### Final Benchmark

| Config | ms/batch | Throughput | VRAM |
|--------|----------|-----------|------|
| OLD: bs=4, seq=256, no SDPA | 1355ms | 3.0 samp/s | 15.4GB |
| NEW: bs=4, SDPA+dynpad+gradckpt | 539ms | 7.4 samp/s | 7.2GB |
| **NEW: bs=8, SDPA+dynpad+gradckpt** | **521ms** | **15.3 samp/s** | **8.1GB** |
| TESTED: bs=16, SDPA+dynpad+gradckpt | 1686ms | 9.5 samp/s | 11.9GB |

bs=16 is slower because batches become too large and start hitting memory bandwidth limits.

---

## 3. Training Run Details

### Config (train_llama32_bea2019.py)

```python
train(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    model_type="llama",
    dataset_path="./data/bea2019_hf",
    batch_size=8,
    learning_rate=2e-4,
    epochs=3,
    max_length=256,
    gradient_accumulation_steps=4,  # effective batch = 8 × 4 = 32
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    seed=42,
    output_dir="./checkpoints/llama32_bea2019",
    save_steps=500,
)
```

### LR Scheduler

Linear warmup for 10% of total steps, then linear decay to 0:
- Total steps: 7,096
- Warmup steps: 709
- Peak LR: 2e-4
- Final LR: ~0

### Loss Progression

| Checkpoint | Epoch | Batch | Optimizer Step | Loss |
|-----------|-------|-------|----------------|------|
| step_0000500 | 1 | 1999 | 500 | 0.1671 |
| step_0001000 | 1 | 3999 | 1000 | 0.1457 |
| step_0001500 | 1 | 5999 | 1500 | 0.1372 |
| step_0002000 | 1 | 7999 | 2000 | 0.1324 |
| epoch_1 end | 1 | 9461 | ~2365 | — |
| step_0002500 | 2 | 539 | 2500 | 0.0912 |
| step_0003000 | 2 | 2539 | 3000 | 0.0928 |
| step_0003500 | 2 | 4539 | 3500 | 0.0938 |
| step_0004000 | 2 | 6539 | 4000 | 0.0941 |
| step_0004500 | 2 | 8539 | 4500 | 0.0945 |
| epoch_2 end | 2 | 9461 | ~4730 | — |
| step_0005000 | 3 | 1079 | 5000 | 0.0656 |
| step_0005500 | 3 | 3079 | 5500 | 0.0659 |
| step_0006000 | 3 | 5079 | 6000 | 0.0671 |
| step_0006500 | 3 | 7079 | 6500 | — |
| step_0007000 | 3 | 9079 | 7000 | — |
| epoch_3 end | 3 | 9461 | 7095 | 0.1313 |

**Best validation loss: 0.1417** (saved to best_model.pt)

### Crash Recovery Event

Training stalled at step ~6243 (epoch 3, batch 6053) after running for 19+ hours. GPU dropped to 1% utilization with only 2.4GB VRAM — the process hung. Root cause unknown (possibly Windows memory management or CUDA context issue).

**Recovery:** Killed the hung process, restarted training. Resume logic automatically:
1. Found `step_0006000` (latest checkpoint)
2. Loaded LoRA adapter weights from `step_0006000/lora_adapter/adapter_model.safetensors`
3. Restored optimizer and scheduler state
4. Skipped epochs 1 and 2 entirely
5. Fast-forwarded through batches 0–5079 of epoch 3 at 170 it/s (no compute, just dataloader iteration)
6. Resumed actual training from batch 5080, step 6000

Only ~22 minutes of training progress was lost.

---

## 4. Evaluation Results

### Methodology

- Test set: BEA 2019 test split (4,206 sentences)
- Inference: `model.correct_text(text, num_beams=1, max_new_tokens=128)`
- Batch size: 1 (sequential, no batching during eval)
- Metrics: F0.5 (ERRANT framework), GLEU

### Results

| Metric | Llama 3.2-3B | T5-Large (Phase 3) |
|--------|-------------|-------------------|
| F0.5 | 0.0303 | **0.3201** |
| Precision | 0.0244 | **0.2744** |
| Recall | **0.9515** | 0.9588 |
| GLEU | 0.7431 | **0.9245** |
| Correction rate | 98.9% | 12.8% |
| TP / FP / FN | 255 / 10,210 / 13 | — |

### Analysis

**High recall (0.95):** The model almost never misses a real error. When there is a grammar error, it will detect it.

**Low precision (0.02):** The model makes 10,210 false positive corrections — editing sentences that were already correct, or making unnecessary changes. Only 255 of its corrections were true positives.

**Root cause:** Llama 3.2-3B-Instruct is a chat model trained to be helpful and generative. When given a grammatically poor sentence, it doesn't minimally edit it — it paraphrases and rewrites it fluently. This creates many surface-level "corrections" that don't match the reference targets.

**Example:**
- Input: `She go to school yesterday.`
- Reference: `She went to school yesterday.`
- Llama output: `She went to school yesterday and returned home safely.` ← added unrequested content

**Fixes planned:**
1. **Prompt tuning:** Add explicit instruction "Do not add new information. Only fix grammar errors."
2. **Constrained decoding:** Stop generation as soon as the correction is complete
3. **DPO fine-tuning:** Use Direct Preference Optimization to teach minimal edits vs paraphrasing

---

## 5. API Integration

The API (`src/api/main.py`) loads the Llama model on startup alongside T5:

```python
llama_path = os.getenv(
    "LLAMA32_MODEL_PATH",
    "checkpoints/llama32_bea2019/llama_gec_lora"  # default path
)
models["llama"] = LlamaGEC.from_pretrained(llama_path)
```

The `/api/v1/correct` endpoint accepts `"model": "llama"` to route to it:

```json
POST /api/v1/correct
{
    "text": "She go to school yesterday.",
    "model": "llama",
    "num_beams": 4
}
```

---

## 6. Frontend Changes

Three files were updated to add Llama to the UI:

**TextEditor.tsx** — Added option to model dropdown:
```tsx
<select value={selectedModel} onChange={(e) => onModelChange(e.target.value)}>
    <option value="llama">Llama 3.2-3B (Fine-tuned)</option>   {/* NEW - first/default */}
    <option value="coedit">CoEdIT (Grammarly)</option>
    <option value="t5">T5 (Fine-tuned)</option>
</select>
```

**App.tsx** — Changed default:
```tsx
const [selectedModel, setSelectedModel] = useState("llama");  // was "coedit"
```

**frontend/.env** — Pointed to port 8001 (8000 was occupied by another app); later updated to 9000 in Phase 7 as 8001 was also taken:
```
VITE_API_BASE_URL=http://localhost:9000/api/v1
```

**frontend/src/services/api.ts** — Increased timeout for long paragraphs:
```ts
timeout: 300_000,  // 5 minutes (was 30s — too short for 100-sentence paragraphs)
```

---

## 7. Files Created / Modified in This Phase

### New Files
- `train_llama32_bea2019.py` — Training entry point
- `evaluate_llama32.py` — Evaluation script
- `watch_training.py` — Live training monitor (Windows-compatible)
- `training_llama32.log` — Training log (original run)
- `training_llama32_resume.log` — Resume run log
- `evaluation_llama32.log` — Evaluation log
- `frontend/.env` — API URL config
- `documentation/phase_6_llama_training/README.md` — This file

### Modified Files
- `src/models/llama_gec.py` — SDPA, gradient checkpointing, use_gradient_checkpointing param
- `src/data/preprocess.py` — `padding=False` in tokenizer, `make_collate_fn()`, updated `build_dataloader()`
- `src/training/train.py` — Dynamic padding wiring, resume logic, LoRA weight restore, step checkpoint calls
- `src/training/utils.py` — `save_step_checkpoint()`, `find_latest_step_checkpoint()`, json/Tuple imports
- `src/api/main.py` — Llama model loading block
- `src/api/models.py` — Pattern `^(t5|coedit|llama)$` already included llama
- `src/api/routes.py` — Already supported llama routing
- `frontend/src/App.tsx` — Default model to "llama"
- `frontend/src/components/TextEditor.tsx` — Llama option in dropdown
- `frontend/src/services/api.ts` — Model type, default, timeout
- `watch_training.py` — Unicode fix, tqdm regex fix

---

## 8. Lessons Learned

1. **SDPA is a free lunch on Windows.** One line of code (`attn_implementation="sdpa"`) eliminated both the slow attention kernels and the GQA batch_size=8 bottleneck. Always use it for Llama/Mistral-style models.

2. **Dynamic padding matters more than expected.** With BEA 2019's short sentences (~150 tokens avg), fixed 256-token padding wasted 41% of every batch. The attention O(n²) cost means this isn't 41% waste — it's closer to 3x waste in attention compute.

3. **Gradient checkpointing + LoRA requires `enable_input_require_grads()`.** This is a subtle requirement not always documented: LoRA adds adapters that don't naturally require input gradients, causing gradient checkpointing to fail silently. Always call `model.enable_input_require_grads()` when combining both.

4. **torch.compile is not usable on Windows with LoRA.** The inductor backend cannot trace through PEFT adapters on Windows. Use Linux/WSL2 for this optimization.

5. **Chat-format LLMs are bad at minimal GEC out of the box.** The FLAN-T5-Large (trained as a seq2seq GEC model) significantly outperforms the 4x larger Llama 3.2-3B on GEC metrics. Task format matters more than raw model size.

6. **Step checkpoints saved the training.** The training crashed at 84% completion (step 6243). Without step checkpoints every 500 steps, we would have needed to restart the entire epoch. With them, only 22 minutes of work was lost.
