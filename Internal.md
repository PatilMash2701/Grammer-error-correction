# Internal Project Handbook

> **Scope:** This is the consolidated technical context for the Grammar Error Correction (GEC) application in `Grammar_Corrector-main/`. It describes the implementation and the project’s recorded experiments, including discrepancies that should not be hidden. It is not a claim that every reported experiment can be reproduced from the current checkout.
>
> **Evidence labels used below**
> - **In source:** directly visible in checked-in code or configuration.
> - **Reported:** recorded in project documentation or interview notes; raw run artifacts were not available to independently recalculate it during this review.
> - **Planned/scaffold:** proposed, partially implemented, or not yet demonstrated as a working end-to-end feature.

## Contents

1. [Executive summary](#1-executive-summary)
2. [Architecture and request flow](#2-architecture-and-request-flow)
3. [Models and why they were tried](#3-models-and-why-they-were-tried)
4. [Data and preprocessing](#4-data-and-preprocessing)
5. [Training approaches and configurations](#5-training-approaches-and-configurations)
6. [Optimization work and measured benchmarks](#6-optimization-work-and-measured-benchmarks)
7. [Evaluation methodology and recorded results](#7-evaluation-methodology-and-recorded-results)
8. [Backend behavior and API contract](#8-backend-behavior-and-api-contract)
9. [Frontend behavior](#9-frontend-behavior)
10. [Deployment and operations](#10-deployment-and-operations)
11. [Model artifacts, commands, and reproducibility](#11-model-artifacts-commands-and-reproducibility)
12. [Known gaps, inconsistencies, and cautions](#12-known-gaps-inconsistencies-and-cautions)
13. [Project map and further reading](#13-project-map-and-further-reading)

---

## 1. Executive summary

This project is an English grammar-error-correction system with a React web interface, a FastAPI service, and several Hugging Face model integrations. It is also an ML engineering/learning project: the repository contains data-generation and download scripts, LoRA fine-tuning code, evaluation scripts, model wrappers, and Docker configuration.

The principal task is **minimal grammatical correction**: preserve the author’s meaning and wording wherever possible, and change only what is needed. The task is sequence-to-sequence (erroneous text → corrected text), and evaluating unnecessary edits matters as much as detecting errors.

The main reported experimental comparison is:

| Model / run | Parameters | Fine-tuning | Reported F0.5 | Precision | Recall | Reported GLEU | Reported correction rate | Selection |
|---|---:|---|---:|---:|---:|---:|---:|---|
| FLAN-T5-Large + LoRA | ~780M base | BEA-2019, 3 epochs | 0.3201 | 0.2744 | 0.9588 | 0.9245 | 12.8% | Best reported GEC result; intended primary model |
| Llama 3.2-3B-Instruct + LoRA | ~3.2B base | BEA-2019, 3 epochs | 0.0303 | 0.0244 | 0.9515 | 0.7431 | 98.9% | Comparative experiment; severe over-editing |
| Grammarly CoEdIT-Large, zero-shot | ~770M | No project fine-tuning for this comparison | 0.0548 | 0.0443 | 0.9792 | 0.6849 | 98.15% | Tested baseline; rejected for over-editing |
| Grammarly CoEdIT-Large + LoRA | ~770M | A separate training script exists | No verified result found | — | — | — | — | Training experiment, result not documented here |

**Decision rationale:** the smaller encoder-decoder T5 is a better fit for producing localized corrections and was much better on the recorded precision-weighted metric. Llama has very high recall but frequently paraphrases or adds content, creating many false-positive edits. The project therefore treats the T5 run as the best reported GEC model, while retaining Llama and CoEdIT as selectable experiments in the API.

**Important current-checkout limitation:** the application source is present under `Grammar_Corrector/`, but the `src/data/` package, `data/` directory, and `checkpoints/` directory are absent from this checkout. Consequently, the documented full training and model-backed inference runs cannot be reproduced from this checkout alone. Several reported numbers below are project-recorded results, not independently rerun or verified from raw prediction files.

---

## 2. Architecture and request flow

```text
Browser
  React 18 + TypeScript + TailwindCSS + Vite
       │ Axios JSON request (/api/v1)
       ▼
  FastAPI + Pydantic
       ├── validate input and model choice
       ├── select loaded model
       ├── sentence split for paragraph requests
       ├── model inference
       ├── output cleanup
       └── word-level diff extraction (difflib)
       ▼
  PyTorch / Transformers / PEFT
       ├── "t5"     FLAN-T5-Large + LoRA
       ├── "llama"  Llama 3.2-3B-Instruct + LoRA
       └── "coedit" Grammarly CoEdIT-Large, no LoRA in API loader
```

### Main layers

1. **Frontend:** `Grammar_Corrector/frontend/`. The React app keeps the text, selected model, loading/error state, and latest result. Axios sends requests to `VITE_API_BASE_URL`, falling back to `/api/v1`.
2. **API:** `Grammar_Corrector/src/api/`. `main.py` creates the FastAPI application, installs CORS, and loads the models in the application lifespan. `routes.py` handles correction, health, and model metadata.
3. **Model wrappers:** `Grammar_Corrector/src/models/`. `T5GEC` and `LlamaGEC` provide load, generate, single-correction, and batch-correction methods. BART, GECToR, and ensemble modules are incomplete scaffolds.
4. **Training/evaluation:** `Grammar_Corrector/src/training/` and top-level scripts. The common training loop handles LoRA, optimization, validation, and checkpointing; model-specific scripts provide the intended hyperparameters.
5. **Deployment:** Docker Compose starts the API and an Nginx-served frontend. The frontend’s Nginx config proxies `/api` requests to the backend container.

The API attempts to load all three serving models at startup. Loading failures are logged as warnings rather than preventing the service process from starting; health is reported as `degraded` if no models loaded. A request to an unloaded model returns HTTP 503.

---

## 3. Models and why they were tried

### 3.1 FLAN-T5-Large + LoRA — task model

- **Base:** `google/flan-t5-large`, about 780M parameters.
- **Architecture:** encoder-decoder / sequence-to-sequence. The encoder reads the full source; the decoder generates a corrected target.
- **Training input convention:** task prefix `grammar: ` followed by the erroneous sentence.
- **Training target:** corrected sentence.
- **LoRA in the wrapper:** rank 16, alpha 32, dropout 0.05, target modules `q` and `v`, no bias adaptation.
- **Reported adapter:** ~18 MB; documentation reports 4,718,592 trainable parameters (~4.7M).
- **Inference:** beam search; wrapper defaults to 4 beams and max 128 new tokens. CUDA loading uses bfloat16.
- **Output cleanup:** whitespace before punctuation and after opening punctuation is normalized in the API.

**Why it was selected:** GEC is a constrained transformation task. An encoder-decoder model has a natural source-to-target interface and, in the reported comparison, made fewer unnecessary changes than Llama or the CoEdIT zero-shot baseline.

### 3.2 Llama 3.2-3B-Instruct + LoRA — comparative decoder-only model

- **Base:** `meta-llama/Llama-3.2-3B-Instruct`; documentation records 3,237,063,680 parameters.
- **Architecture:** decoder-only causal transformer with grouped-query attention (GQA), RoPE, and SwiGLU. The model is used with a chat template.
- **Training approach:** supervised causal language modeling on prompt/assistant examples; documentation says prompt tokens are masked from the loss and target assistant tokens are trained. The preprocessing implementation that would verify this is missing from this checkout.
- **LoRA:** rank 16, alpha 32, dropout 0.05, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`; documentation records 24,313,856 trainable parameters (~0.75%).
- **Inference wrapper prompt:** system instruction asks for grammar correction only; user message is `Correct this: {text}`. The training documentation describes `Correct this sentence: {source}`, so the exact train/inference prompt match should be checked when recovering the training preprocessor.
- **Inference defaults:** greedy decoding in the wrapper (`num_beams=1`) unless a caller overrides it; max 128 new tokens. The API schema defaults `num_beams` to 4 and forwards that value, so API-driven Llama inference may not match the recorded evaluation setup (which used 1 beam).
- **Reported adapter:** ~97 MB.

**Observed issue and interpretation:** the project’s recorded results show high recall but extremely low precision. The documented explanation is that an instruction-tuned conversational model tends to rewrite or embellish rather than minimally edit, which ERRANT counts as false-positive edits. A project example describes Llama adding an unrequested continuation to a corrected sentence. Prompt tuning, constrained decoding, and DPO are described as future ideas, not completed fixes.

### 3.3 Grammarly CoEdIT-Large — baseline and separate fine-tuning experiment

- **Base:** `grammarly/coedit-large`, documented as ~770M parameters and a multi-task instruction-tuned encoder-decoder model.
- **API loading:** `T5GEC.from_pretrained(..., use_lora=False)` with task prefix `Fix grammatical errors in this sentence: `.
- **Zero-shot result:** reported F0.5 0.0548 on the project’s BEA test evaluation; very high correction rate and false-positive count.
- **Separate training script:** `train_coedit_bea2019.py` fine-tunes CoEdIT with LoRA. This is a different run from the zero-shot baseline and has no final evaluation score documented in the checked-in project notes.

**Why try it:** it is a strong, readily available instruction-tuned correction baseline and allows comparison against the project-specific T5 fine-tuning. The recorded zero-shot result was not competitive under the project’s precision-weighted metric.

### 3.4 BART, GECToR, and ensembles — scaffolding / discarded direction

- **BART:** `src/models/bart_gec.py` wraps `BartForConditionalGeneration`, defaults to `facebook/bart-base` (140M parameters), and implements basic loading/forward/generation. `correct_text()` and `correct_batch()` raise `NotImplementedError`. Documentation says the original BART-base direction was replaced by FLAN-T5-Large. The BART-Large performance figures in `MODEL_COMPARISON.md` are estimates, not measured project results.
- **GECToR:** `src/models/gector.py` outlines a RoBERTa token-tagging approach (KEEP/DELETE/REPLACE/APPEND-style tags), but loading, inference, and forward methods are unimplemented.
- **Ensemble:** `src/models/ensemble.py` lists vote/rank/cascade strategies, but correction/voting methods are unimplemented.

These are not functional serving alternatives in the current code.

### 3.5 LoRA rationale

LoRA freezes pre-trained weights and learns low-rank updates:

```text
W = W0 + (alpha / r) * (B @ A)
```

Here `W0` is the frozen pre-trained matrix, while `A` and `B` are the trainable low-rank factors. This reduces optimizer/gradient memory and allows small adapter files instead of saving a full fine-tuned base model. The project uses `r=16`, `alpha=32` for T5 and Llama; the CoEdIT training script uses `r=32`, `alpha=64`.

---

## 4. Data and preprocessing

### 4.1 Primary dataset: BEA-2019 corruption pairs

The project’s downloader targets Hugging Face dataset `juancavallotti/bea-19-corruption` and writes CSVs under `data/bea2019_hf/`. Its intended fields are:

| Field | Meaning |
|---|---|
| `source` | Erroneous learner sentence |
| `target` | Corrected sentence |

The downloader maps the dataset’s `broken` text to `source` and `sentence` to `target`; it renames a `validation` split to `dev.csv`. Training/evaluation scripts expect `train.csv`, `dev.csv`, and `test.csv`.

The data includes learner errors such as verb form/agreement, determiners/articles, prepositions, spelling, punctuation, and word order. The intended pipeline removes invalid rows, tokenizes with the selected model tokenizer, truncates at a configured maximum length, and batches examples. Exact split counts are inconsistent in existing notes; see [count and evidence cautions](#121-recorded-inconsistencies).

### 4.2 Synthetic data and augmentation

- `scripts/generate_synthetic_gec_dataset.py` injects rule-based errors into clean example sentences. Its error families include subject-verb agreement, articles, pluralization, tense, prepositions, and contractions.
- Project notes report a generated supplementary dataset of 8,418 pairs; no run artifact was present to verify that count.
- `scripts/convert_synthetic_to_csv.py` converts parallel `source.txt` / `target.txt` files into CSVs.
- `src/data/augmentation.py` is referenced by training code and docs for BEA-derived error injection, but that module is absent in this checkout.
- **Important run setting:** the checked-in T5 and Llama entry-point scripts pass `augmentation_factor=0`. Thus, despite general docs describing augmentation, those two stated full BEA runs are configured with augmentation disabled. The CoEdIT-specific script also loads BEA directly without the shared augmentation path.

Synthetic data is useful for controlled error coverage, but errors generated by simple rules may be less natural than learner errors and can create distribution mismatch. The checked-in training entry points choose the real BEA pairs for their reported main runs.

### 4.3 Other evaluation datasets mentioned

- **CoNLL-2014:** described in planning notes as a 1,312-sentence benchmark. The main recorded model numbers in this handbook are explicitly reported on the BEA test split, not established here as CoNLL-2014 results.
- **JFLEG:** described as a fluency benchmark with multiple references. Project notes associate GLEU with JFLEG, but the available reported scores are presented alongside BEA evaluations. Do not claim a standard JFLEG evaluation unless the matching dataset, script invocation, and outputs are available.

### 4.4 Preprocessing status

The shared training loop imports `GECDataset`, `build_dataloader`, `create_data_splits`, and `make_collate_fn` from `src/data/preprocess.py`; these files are not included in the current checkout. Therefore, exact tokenization details, Llama label masking, and padding behavior are described in project notes/configuration but cannot be fully audited from source here.

---

## 5. Training approaches and configurations

### 5.1 FLAN-T5-Large run

Entry point: `train_bea2019.py`, which calls the shared `src.training.train.train()` loop.

| Setting | Configured value |
|---|---|
| Base model | `google/flan-t5-large` |
| Dataset directory | `./data/bea2019_hf` |
| Epochs | 3 |
| Per-device batch | 4 |
| Gradient accumulation | 2; effective batch 8 |
| Learning rate | `1e-5` |
| Max sequence length | 256 |
| LoRA | enabled; rank 16, alpha 32 |
| LoRA target layers | `q`, `v` in `T5GEC` |
| Augmentation | disabled (`augmentation_factor=0`) |
| Seed | 42 |
| Precision | bfloat16 on CUDA |
| Optimizer | AdamW, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01` |
| Gradient clipping | max norm 1.0 |
| Learning-rate schedule | linear warmup for 10% of calculated steps, then linear decay |
| Early stopping | patience 2, minimum delta 0.001 |
| Output path | `./checkpoints/flan_t5_large_bea2019` |

The T5 path uses the conventional seq2seq loss on target tokens. The wrapper can load either a base model (optionally adding fresh LoRA) or an existing PEFT adapter checkpoint.

Project status notes report ~5.5 hours for this run on the listed RTX 4080 SUPER, but the absence of data, checkpoints, and logs prevents independent verification.

### 5.2 Llama 3.2-3B run

Entry point: `train_llama32_bea2019.py`; it allows `LLAMA32_MODEL_PATH` to override the Hugging Face model name and optionally reads `WANDB_PROJECT`.

| Setting | Configured value |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` by default |
| Dataset directory | `./data/bea2019_hf` |
| Epochs | 3 |
| Per-device batch | 8 |
| Gradient accumulation | 4; effective batch 32 |
| Learning rate | `2e-4` |
| Max sequence length | 256 |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| LoRA targets | Seven attention/MLP projections listed in §3.2 |
| Augmentation | disabled (`augmentation_factor=0`) |
| Seed | 42 |
| Precision | bfloat16 on CUDA |
| Optimizer | AdamW; weight decay 0.01 |
| Gradient clipping | max norm 1.0 |
| Learning-rate schedule | linear warmup (10%) then linear decay |
| Mid-epoch saves | every 500 optimizer steps |
| Output path | `./checkpoints/llama32_bea2019` |

The wrapper uses explicit CUDA placement for fresh training to avoid an `auto` device-map placement that project notes associate with adapters being placed on CPU and causing excessive CPU/GPU transfers. It enables SDPA and gradient checkpointing for CUDA training.

### 5.3 CoEdIT-Large fine-tuning run

Entry point: `train_coedit_bea2019.py`. This is separate from zero-shot CoEdIT evaluation.

| Setting | Configured value |
|---|---|
| Base model | `grammarly/coedit-large` |
| Task prefix | `Fix grammatical errors in this sentence: ` |
| Dataset directory | `./data/bea2019_hf` |
| Epochs | 3 |
| Batch / accumulation | 8 × 8 = effective 64 |
| Learning rate | `2e-5` |
| Max length | 128 |
| LoRA | rank 32, alpha 64, dropout 0.05 |
| LoRA target layers | `q`, `v`, `k` |
| Precision | bfloat16 |
| Batching | approximate character-length-sorted sampler; dynamic batch padding |
| Scheduler | cosine with 10% warmup |
| Gradient clipping | 1.0 |
| Output path | `./checkpoints/coedit_large_bea2019/coedit_gec_lora` |

The script saves the best adapter by validation loss. No final F0.5/GLEU result for this fine-tuned run was found in the project notes.

### 5.4 Shared training loop

`src/training/train.py` provides:

- deterministic seeding across Python, NumPy, and PyTorch;
- model-specific wrappers for T5 and Llama;
- CSV loading and optional augmentation;
- train/dev DataLoaders;
- AdamW, linear warmup/decay, gradient accumulation, and gradient clipping;
- validation loss/perplexity;
- early stopping and best/regular checkpoints;
- optional Weights & Biases logging;
- Llama step-checkpoint resume support.

The shared loop’s module docstring mentions BART/GECToR, but `setup_training()` currently supports Llama and T5 and raises `NotImplementedError` for other model types. Its generic CLI defaults and Makefile targets still refer to BART-era paths/configuration; use the dedicated scripts for the documented BEA runs rather than assuming the generic CLI is current.

---

## 6. Optimization work and measured benchmarks

The main optimization work addressed Llama training on an NVIDIA RTX 4080 SUPER with 16 GB VRAM.

### 6.1 Techniques and reasons

1. **Scaled Dot-Product Attention (SDPA):** `attn_implementation="sdpa"` selects PyTorch’s optimized attention kernels. Project notes say this fixed a severe batched Grouped-Query-Attention bottleneck. SDPA is not the same as claiming a particular FlashAttention version; it delegates kernel choice to PyTorch/runtime.
2. **Dynamic padding:** Llama input sequences are tokenized without global fixed-length padding and padded per batch. Attention work grows roughly quadratically with sequence length, so avoiding padding reduces wasted compute and memory. The shared loop wires `make_collate_fn()` for Llama; its implementation is absent here.
3. **Gradient checkpointing:** recomputes selected activations during backward instead of retaining all activations. It trades compute for memory and enables larger microbatches. The wrapper uses `use_reentrant=False` and calls `enable_input_require_grads()` when combining checkpointing with LoRA.
4. **Larger microbatch:** increasing microbatch from 4 to 8, while using accumulation 4, gives effective batch 32. The stated goal was more throughput without changing effective batch.
5. **Step checkpoints:** save LoRA adapter, optimizer, scheduler, and epoch/batch/step metadata every 500 optimizer updates to limit lost progress after a crash. Resume logic restores these and skips prior batches.
6. **bfloat16:** used for T5 and Llama CUDA runs. Its exponent range matches float32 more closely than float16, and the project chose it for training stability and hardware support. The shared training loop only enables GradScaler for model types other than T5/Llama.
7. **Reproducibility/optimization basics:** seed 42, AdamW, 10% warmup, linear decay for shared T5/Llama runs, and max gradient norm 1.0.

### 6.2 Reported benchmark

Reported baseline and optimized measurements:

| Configuration | Time per batch | Throughput | VRAM |
|---|---:|---:|---:|
| Baseline: batch 4, sequence 256, no SDPA | 1,355 ms | 3.0 samples/s | 15.4 GB |
| SDPA + dynamic padding + checkpointing, batch 4 | 539 ms | 7.4 samples/s | 7.2 GB |
| SDPA + dynamic padding + checkpointing, batch 8 | **521 ms** | **15.3 samples/s** | **8.1 GB** |
| Same optimizations, batch 16 | 1,686 ms | 9.5 samples/s | 11.9 GB |

The project reports batch-8 throughput as **5.1×** baseline and VRAM use as **47% lower**. Batch 16 was slower, illustrating that larger batches are not automatically faster on this hardware. Project status/interview notes report overall Llama training time improving from ~30 hours to ~11 hours.

### 6.3 Failed optimization

The project notes record an attempt to use `torch.compile(..., backend="inductor")` that failed on Windows/PEFT with `BackendCompilerFailed` and an `'float' object has no attribute 'meta'` error. It was abandoned rather than included in the successful benchmark.

### 6.4 Training loss notes (Llama)

The Llama training notes report these intermediate training-loss values:

| Optimizer step | Reported loss |
|---:|---:|
| 500 | 0.1671 |
| 1,000 | 0.1457 |
| 1,500 | 0.1372 |
| 2,000 | 0.1324 |
| 2,500 | 0.0912 |
| 3,000 | 0.0928 |
| 3,500 | 0.0938 |
| 4,000 | 0.0941 |
| 4,500 | 0.0945 |
| 5,000 | 0.0656 |
| 5,500 | 0.0659 |
| 6,000 | 0.0671 |
| End of epoch 3 / step ~7,095 | 0.1313 |

Best validation loss is recorded as **0.1417**. The project also reports an interruption after 19+ hours, followed by resuming from the 6,000-step checkpoint with about 22 minutes of progress lost. This incident and the separate ~11-hour “total training” claim are not reconciled in the existing notes; treat the total wall-clock time as approximate.

---

## 7. Evaluation methodology and recorded results

### 7.1 F0.5 and edit accounting

The primary metric is precision-weighted F-score:

```text
F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
F0.5   = 1.25 * P * R / (0.25 * P + R)
```

For GEC, beta 0.5 weights precision more heavily than recall. An unnecessary correction can damage meaning or style, while a missed correction leaves the original wording intact; this makes F0.5 more appropriate than raw token accuracy for the intended product.

`src/training/evaluate.py` uses ERRANT to annotate edits from source to prediction and source to reference, keys an edit by source span and correction string, then aggregates true positives, false positives, and false negatives. Where multiple references are passed, the implementation chooses the reference with the best per-sentence F0.5. The main BEA evaluation scripts appear to wrap each target as a single reference.

### 7.2 Recorded model scores

These figures are in project documentation and scripts. The raw checkpoint/prediction/result artifacts were not in the current checkout, so they have not been independently recalculated.

#### FLAN-T5-Large + LoRA (BEA-2019 test, reported 4,206 samples)

| Metric | Reported value |
|---|---:|
| F0.5 | **0.3201** |
| Precision | 0.2744 |
| Recall | 0.9588 |
| TP / FP / FN | 163 / 431 / 7 |
| GLEU (project implementation) | 0.9245 |
| Correction rate | 12.8% |
| Exact match | 29.9% |
| Latency | ~500 ms per sentence in project summary |

Precision and recall are arithmetically consistent with the recorded edit counts: `163/(163+431) ≈ 0.2744`; `163/(163+7) ≈ 0.9588`. The high recall and substantially lower precision mean many edits did not match reference edits, despite the relatively low correction rate.

A separate phase evaluation document says “Avg Latency: 2,060ms (batched over 4K samples)”. This is ambiguous and does not agree with the summary’s ~500 ms per sentence; see [inconsistencies](#121-recorded-inconsistencies).

#### Llama 3.2-3B-Instruct + LoRA (BEA-2019 test, reported 4,206 samples)

| Metric | Reported value |
|---|---:|
| F0.5 | **0.0303** |
| Precision | 0.0244 |
| Recall | 0.9515 |
| TP / FP / FN | 255 / 10,210 / 13 |
| GLEU (project implementation) | 0.7431 |
| Correction rate | 98.9% |
| Latency | ~1,300 ms in summary; another planning note estimates ~1,500 ms |

The documented diagnosis is over-correction/paraphrasing. The raw false-positive count (10,210) is dramatically higher than the T5 run’s 431. The model almost always changed text, while the reference-based scorer considered very few of those changes correct.

#### Grammarly CoEdIT-Large zero-shot baseline (reported BEA-2019 test, 4,206 samples)

| Metric | Reported value |
|---|---:|
| F0.5 | 0.0548 |
| Precision | 0.0443 |
| Recall | 0.9792 |
| TP / FP / FN | 753 / 16,232 / 16 |
| GLEU (project implementation) | 0.6849 |
| Correction rate | 98.15% |
| Evaluation time | 6.8 minutes (reported) |

This zero-shot baseline has more true positives than T5 but far more false positives; the precision-weighted total is worse. Do not confuse it with the separate CoEdIT-LoRA training script.

### 7.3 GLEU implementation caveat

The project calls its metric `compute_gleu()`, but the current implementation is a simplified project-specific n-gram overlap score:

- lowercases and whitespace-tokenizes text;
- builds n-grams through order 4;
- takes the best score over references;
- returns matched prediction/reference n-grams divided by total prediction n-grams.

Although the function/docstring describes a penalty involving source n-grams, the implementation computes source-related counters but does not use them in the final score. Consequently, the recorded 0.9245 / 0.7431 / 0.6849 values should be described as **the project’s reported GLEU-like implementation scores**, not assumed to be directly comparable with published/official GLEU results.

### 7.4 What the results mean

- **F0.5 is the main decision metric:** it reflects the product’s preference for not making unsupported changes.
- **Recall alone is misleading:** all three reported models have recall near 0.95–0.98, but the false-positive totals differ greatly.
- **Correction rate is diagnostic, not a quality metric:** very high modification rates can signal an aggressive rewriting style.
- **Exact match is strict:** valid corrections can differ from a single reference and still be counted as non-matches.
- **Dataset and metric implementation matter:** numbers from this repository are not automatically benchmark-comparable without the exact split, references, scorer version, and inference configuration.

---

## 8. Backend behavior and API contract

### 8.1 Routes

The application mounts the router at `/api/v1`.

| Method and route | Purpose |
|---|---|
| `POST /api/v1/correct` | Correct one text; paragraphs are split into sentences first |
| `POST /api/v1/correct/batch` | Correct a list of texts |
| `GET /api/v1/health` | Report uptime and which expected models loaded |
| `GET /api/v1/model/info?model=t5\|coedit\|llama` | Return model type, parameter estimate, max length, and device |
| `GET /` | Basic API name/version/docs link |

### 8.2 Validation and defaults

From `src/api/models.py`:

- Text is required and constrained to 1–5,000 characters.
- Model must be one of `t5`, `coedit`, or `llama`.
- `num_beams` is an integer from 1 to 10, default 4.
- Batch accepts 1–50 texts.
- The response includes original/corrected text, diffs, confidence score, model name, and processing time.

The Pydantic request default model is **`coedit`**. The React app and Axios helper use **`llama`** as their default. The API and frontend are therefore not using one consistent default.

### 8.3 Single-text and paragraph processing

For a single sentence, the route calls `correct_text()`. For multiple sentence segments it calls `correct_batch()`, then rejoins corrected sentences with whitespace captured by `split_into_sentences()`. The splitter uses punctuation followed by whitespace as its boundary. It begins with `text.strip()`, so outer whitespace is not preserved exactly despite the helper’s stated intent; abbreviations and other ambiguous punctuation are not linguistically handled.

### 8.4 Diff and output normalization

- `clean_t5_output()` removes spaces before punctuation and after opening brackets/quotes. Despite the name, the route applies this cleanup to outputs from any selected model.
- `extract_corrections()` whitespace-splits source/output and uses `difflib.SequenceMatcher` to group `equal`, `replace`, `insert`, and `delete` operations.
- It maps source word indices to character offsets and returns `{original, corrected, error_type, position:{start,end}}`.
- `error_type` is always the generic string `"grammar"`; this is not ERRANT error categorization.
- The word-level alignment and reconstructed spans normalize whitespace and can be approximate for repeated words, unusual whitespace, and insertions/deletions.
- Model wrappers return their own confidence estimate; this is not a calibrated probability of correctness.

### 8.5 Model startup and operational behavior

`src/api/main.py` reads:

| Environment variable | Default |
|---|---|
| `MODEL_PATH` | `checkpoints/flan_t5_large_bea2019/llama_gec_lora` |
| `LLAMA32_MODEL_PATH` | `checkpoints/llama32_bea2019/llama_gec_lora` |
| `COEDIT_PATH` | `grammarly/coedit-large` |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` |

Model load exceptions are logged and swallowed per-model so other models may still load; unavailable model requests receive 503. On shutdown the global model dictionary is cleared. The default T5 adapter folder is named `llama_gec_lora`, which is a confusing historical path but is what the API currently expects.

---

## 9. Frontend behavior

### Stack and components

- React 18, TypeScript, TailwindCSS 3, Vite 5, Axios.
- `App.tsx`: stores input, selected model, result, loading and error state; calls `correctText()`.
- `TextEditor.tsx`: text area, 5,000-character limit, model selector, submit button, Ctrl/Cmd+Enter shortcut.
- `CorrectionPanel.tsx`: displays confidence percentage, correction count, corrected output, processing time, and individual word-diff rows.
- `services/api.ts`: JSON API client, 300,000 ms (5-minute) timeout, base URL from `VITE_API_BASE_URL` or `/api/v1`.

The UI defaults to Llama and presents the options “Llama 3.2-3B (Fine-tuned)”, “CoEdIT (Grammarly)”, and “T5 (Fine-tuned)”.

**Feature boundary:** the current panel displays corrected text and a list of changes. It does not implement the accept/reject controls or inline positional highlights described in the interview notes. Those should not be presented as shipped behavior.

**Type-contract mismatch:** `frontend/src/types/index.ts` defines the request `model` as `"t5" | "coedit"` (omitting Llama), while `App.tsx` and `api.ts` pass/use a union that includes `"llama"`. The health response interface also declares `model_loaded: boolean`, while the backend schema returns a `models_loaded` map. These are drift between frontend and backend contracts and should be reconciled if frontend type/build validation is part of future work.

---

## 10. Deployment and operations

### Docker Compose topology

`docker-compose.yml` defines:

- **Backend:** Python 3.11 slim image, Uvicorn on container port 8000, host port 9000. It mounts `./checkpoints` read-only and a named Hugging Face cache volume. GPU access is configured through NVIDIA Container Toolkit.
- **Frontend:** multi-stage Node 18 build followed by Nginx, container port 80, host port 3000. Nginx serves the SPA and proxies `/api` to `http://backend:8000`.
- **Startup dependency:** frontend waits for backend health. Compose health check polls `/api/v1/health`; model loading can make initial startup take roughly 60–120 seconds according to comments/configuration.
- **Secrets:** `HF_TOKEN` may be supplied through environment/`.env` for access to the gated Llama base model. Never commit a real token.

`requirements-api.txt` is intended to keep the runtime image smaller by excluding training/evaluation/testing dependencies. Model weights and adapters are not copied into the Docker image; they must exist in the mounted host `checkpoints/` path. CoEdIT may be downloaded into the persistent Hugging Face cache.

### Local development commands

Run commands from the `Grammar_Corrector/` application directory:

```powershell
# Install Python and frontend dependencies
pip install -r requirements.txt
Set-Location frontend
npm install

# In a second terminal, from the application directory:
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 9000

# In another terminal:
Set-Location frontend
npm run dev
```

The API module’s direct-run default port is 8000; the README/Compose development mapping uses host port 9000 → container 8000. Keep the chosen local port and frontend `VITE_API_BASE_URL`/proxy configuration aligned.

For Docker:

```powershell
$env:HF_TOKEN = "your Hugging Face token"
docker compose up --build
```

Expected Compose URLs: frontend `http://localhost:3000`, API `http://localhost:9000`, docs `http://localhost:9000/docs`, health `http://localhost:9000/api/v1/health`.

---

## 11. Model artifacts, commands, and reproducibility

### Expected local assets

The documented expected data/checkpoint layout is:

```text
data/
  bea2019_hf/
    train.csv
    dev.csv
    test.csv
checkpoints/
  flan_t5_large_bea2019/
    llama_gec_lora/
  llama32_bea2019/
    llama_gec_lora/
    step_checkpoints/
  coedit_large_bea2019/
    coedit_gec_lora/
```

These paths are **expected locations**, not assets present in the current checkout inspected for this handbook.

### Download / train / evaluate

From `Grammar_Corrector/`, after restoring dependencies and required data:

```powershell
python scripts/download_bea2019_hf.py
python train_bea2019.py
python evaluate_bea2019.py

# Llama requires access to the gated base model unless cached or pointed
# to a local model directory.
$env:HF_TOKEN = "your Hugging Face token"
python train_llama32_bea2019.py
python evaluate_llama32.py

# Separate CoEdIT-LoRA training experiment; not the zero-shot baseline.
python train_coedit_bea2019.py
```

Evaluation scripts expect the data and checkpoint paths described above. Llama evaluation loops over test sentences and uses `num_beams=1`, `max_new_tokens=128`; it writes predictions/results under the checkpoint parent directory. No checkpoint is shipped in this checkout.

### Reproducibility checklist

To turn the recorded metrics into independently reproducible results, preserve and record:

1. The exact downloaded CSVs and row counts for each split.
2. Commit/version of the dataset and dataset mapping.
3. Base model revision and tokenizer revision.
4. Final adapter files and adapter config.
5. Exact prompt text, label masking and truncation rules.
6. Generation settings, including beams and max new tokens.
7. ERRANT/spaCy/NLTK versions and the exact scorer implementation.
8. Raw prediction/reference CSVs and the full result log.
9. Hardware, precision, batch settings, and timing methodology.

Without these, results remain historical project reports rather than a fully reproducible benchmark.

---

## 12. Known gaps, inconsistencies, and cautions

### 12.1 Recorded inconsistencies

| Topic | Conflicting records | How to interpret |
|---|---|---|
| BEA split sizes | Dataset notes say 75,691 train / 4,203 dev / 4,206 test (sum 84,100) and total 84,106. `train_bea2019.py` prints 75,695 / 4,205. Other docs say ~75K / ~4K. | Counts cannot all describe the same exact files. Check actual CSVs before claiming exact split sizes. |
| Augmentation | General phase docs describe 2× augmentation; the dedicated T5 and Llama training scripts pass `augmentation_factor=0`. | For the documented dedicated runs, the executable script config is the stronger evidence: no augmentation. |
| CoEdIT results | CoEdIT is evaluated as a zero-shot baseline; a separate LoRA training script also exists. | The reported 0.0548 result belongs to zero-shot evaluation unless a result artifact proves otherwise. No CoEdIT-LoRA score was found. |
| T5 latency | Summary says ~500 ms/sentence; evaluation notes say 2,060 ms “batched over 4K samples”. | Timing scope is ambiguous; report both with the caveat, not one as a definitive benchmark. |
| Llama runtime | Phase summary says ~11 hours total; same training notes describe a process stalled after 19+ hours and a resumed run. | Wall-clock accounting is inconsistent; quote ~11 hours only as the project’s reported optimized-run estimate. |
| Llama optimizer steps | One project-status summary says 7,096; detailed notes end at ~7,095. | Likely rounding/partial-final-batch accounting; verify from run logs before asserting an exact count. |
| GLEU | Docs call the scores GLEU, but the implementation’s final formula ignores the computed source penalty. | Label them project implementation scores; do not compare directly with official GLEU leaderboards. |
| Frontend/API defaults | Backend Pydantic default is `coedit`; UI/helper default is `llama`. | Explicitly pass model in clients; unify defaults if desired. |
| Frontend types | Model type omits Llama; health schema field names differ from backend. | Type definitions are stale relative to backend/UI behavior. |
| Interview descriptions | Interview notes describe accept/reject and inline highlighting; current React components show a result panel and diff list only. | Document the actual UI as implemented, not the interview narrative. |
| Generic training paths | Makefile and generic CLI still mention BART and legacy dataset options. | Dedicated entry points are the clearest expression of the intended T5/Llama BEA runs. |
| BART performance | `MODEL_COMPARISON.md` gives F0.5 and runtime estimates for possible options. | Treat these as hypotheses/proposals, not project measurements. |

### 12.2 Missing in this checkout

The reviewed application tree does not contain:

- `src/data/` even though `src/training/train.py` imports `src.data.*`;
- `data/` CSV datasets;
- `checkpoints/` trained model adapters or model evaluation files.

As a result, training imports and model-backed inference are not end-to-end runnable from the reviewed checkout without restoring the missing package/assets. The project documentation describes them, but the descriptions cannot substitute for the absent implementation and artifacts.

### 12.3 Additional technical cautions

- The API loads all three models eagerly, which can create significant GPU-memory pressure; some models may fail to load, leaving the service degraded.
- The T5 and Llama checkpoint directory defaults are relative paths and must resolve from the application working directory.
- The API’s default `num_beams=4` differs from Llama’s reported evaluation setting of 1 beam.
- Sentence splitting is punctuation/whitespace based, not a full sentence-boundary detector; outer whitespace is stripped.
- Word-level diff spans are generated with whitespace tokens and generic error labels, not linguistically categorized edits.
- Confidence values come from generation-score transformations/token probabilities and should not be interpreted as calibrated correctness likelihoods.
- The BART/GECToR/ensemble classes are incomplete; their existence does not mean they are supported by the API or training flow.
- Planning-document targets (for example F0.5 > 0.60) are goals, not achieved scores. Reported T5 F0.5 is 0.3201.

---

## 13. Project map and further reading

Paths below are relative to the application folder `Grammar_Corrector/`:

| Path | Role |
|---|---|
| `README.md` | User-facing overview, quick start, reported headline results |
| `MODEL_COMPARISON.md` | Model alternatives and recommendations; includes unverified estimates |
| `documentation/PROJECT_STATUS.md` | Phase status and detailed historical run notes |
| `documentation/phase_1_planning/theory/01_PROJECT_OVERVIEW_AND_SPECIFICATIONS.md` | Requirements and architecture rationale |
| `documentation/phase_1_planning/theory/02_MODEL_ARCHITECTURE_DEEP_DIVE.md` | Model architecture comparison |
| `documentation/phase_1_planning/theory/03_DATASET_ANALYSIS.md` | Dataset notes and described preprocessing |
| `documentation/phase_1_planning/theory/04_EVALUATION_METRICS.md` | Metric definitions and historical results |
| `documentation/phase_2_data_engineering/README.md` | Data pipeline phase notes |
| `documentation/phase_3_model_development/README.md` | Training/model-development phase notes |
| `documentation/phase_4_evaluation/README.md` | Detailed reported metric tables |
| `documentation/phase_5_api_frontend/README.md` | API schemas and UI notes |
| `documentation/phase_6_llama_training/README.md` | Llama optimization, training, recovery, and evaluation notes |
| `documentation/phase_7_deployment/README.md` | Deployment phase notes |
| `train_bea2019.py` | FLAN-T5 BEA training configuration |
| `train_llama32_bea2019.py` | Llama BEA training configuration |
| `train_coedit_bea2019.py` | Separate CoEdIT-LoRA training experiment |
| `evaluate_bea2019.py`, `evaluate_llama32.py` | Evaluation entry points |
| `src/models/t5_gec.py`, `src/models/llama_gec.py` | Inference/model wrappers |
| `src/training/train.py`, `src/training/utils.py`, `src/training/evaluate.py` | Shared loop, checkpointing, and scorer |
| `src/api/main.py`, `src/api/routes.py`, `src/api/models.py` | Backend app, handlers, schemas |
| `frontend/src/` | React UI, API client, and TypeScript interfaces |
| `Dockerfile`, `docker-compose.yml`, `requirements-api.txt` | Containerized serving setup |

### Bottom line

The project demonstrates the full intended GEC workflow: paired learner data, parameter-efficient fine-tuning, optimization for a 16 GB GPU, edit-based evaluation, and a web-serving stack. The most defensible recorded model conclusion is that FLAN-T5-Large + LoRA outperformed the tested Llama and zero-shot CoEdIT configurations on the project’s precision-weighted evaluation. The primary next step for rigorous technical claims is to restore the missing preprocessing/data/checkpoint assets and rerun evaluation with a pinned scorer, explicit generation settings, and saved raw outputs.
