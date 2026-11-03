# Grammar Error Correction (GEC) System - Technical Interview Preparation Guide

This document is designed to prepare you for your technical interview on the **Grammar Error Correction (GEC)** project. It covers key conceptual, architectural, mathematical, and implementation-level questions that an interviewer is likely to ask, complete with clear, professional, and detailed answers.

---

## 📋 Resume Internship Description (4 Bullet Points)

Here is your updated metric-driven 4-line description to add to your resume under your internship experience:

* **System & Models:** Engineered an end-to-end Grammar Error Correction (GEC) system, fine-tuning FLAN-T5-Large (780M) and Llama 3.2-3B models on the BEA 2019 dataset using LoRA adapters.
* **Model Performance:** Established FLAN-T5-Large as the superior GEC model, achieving a top $F_{0.5}$ score of 0.3201 and a GLEU score of 0.9245, while building custom token post-processing and edit alignment pipelines.
* **Optimizations:** Reduced VRAM usage by 47% and training time from 30 to 11 hours (5.1x throughput speedup) by implementing **Scaled Dot-Product Attention (SDPA)**, dynamic batch padding, and gradient checkpointing.
* **Conclusion:** Concluded that task-specific encoder-decoder architectures (T5) significantly outperform larger, decoder-only chat models (Llama 3.2) in both precision and latency for GEC, verifying them as the optimal production choice.

---

## Table of Contents
1. [Project Overview & System Architecture](#1-project-overview--system-architecture)
2. [Model Architecture & Fine-Tuning (T5 vs. Llama vs. BART)](#2-model-architecture--fine-tuning-t5-vs-llama-vs-bart)
3. [PEFT & LoRA (Parameter-Efficient Fine-Tuning)](#3-peft--lora-parameter-efficient-fine-tuning)
4. [Training Optimizations & Performance Engineering](#4-training-optimizations--performance-engineering)
5. [NLP Evaluation Metrics (Precision, Recall, F0.5, GLEU, GLUE, ERRANT)](#5-nlp-evaluation-metrics-precision-recall-f05-gleu-glue-errant)
6. [API Design, Diff Extraction, & Post-Processing](#6-api-design-diff-extraction--post-processing)
7. [Frontend & Deployment (Docker, React, Nginx)](#7-frontend--deployment-docker-react-nginx)
8. [Conceptual & Advanced Machine Learning Questions](#8-conceptual--advanced-machine-learning-questions)

---

## 1. Project Overview & System Architecture

### Q1. Can you walk me through the high-level architecture of your project?
**Answer:**
The system is built as a production-grade, end-to-end web application consisting of three main tiers:
1. **React Frontend:** A modern single-page application (SPA) built using React 18, TypeScript, TailwindCSS, and Vite. It provides an interactive text editor where users input text, select which ML model to use from a dropdown menu, and view word-level grammar corrections with an interface to accept/reject edits.
2. **FastAPI Backend:** A lightweight, high-performance API built with FastAPI. It handles CORS, Pydantic data validation, sentence-level splitting, post-processing of model outputs, and word-level diff extraction using Python's `difflib`. It serves models in a stateless REST fashion.
3. **ML Inference Engine:** A PyTorch-based model runner hosting three distinct models:
   * **FLAN-T5-Large + LoRA:** Encoder-Decoder seq2seq model (780M base parameters, 18MB adapter), which represents our best-performing, most precise model.
   * **Llama 3.2-3B-Instruct + LoRA:** Decoder-only causal language model (3B parameters, 97MB adapter) fine-tuned in a chat-prompt format.
   * **Grammarly CoEdIT-Large:** A 770M parameter zero-shot instruction-tuned baseline model.

```
[User Interface (React 18 + TS)] 
              │
              │ Axios HTTP Requests
              ▼
    [FastAPI Web Server] 
      ├── Sentence Splitting & Reassembly
      ├── difflib (Word-Level Diff Extraction)
      └── Model Routing (llama, t5, or coedit)
              │
              ▼
[PyTorch ML Engine (CUDA / bfloat16)]
  ├── FLAN-T5-Large + LoRA Adapter (780M) 
  ├── Llama 3.2-3B-Instruct + LoRA Adapter (3B)
  └── CoEdIT-Large (770M)
```

### Q2. Why did you design a dual-model system featuring both T5 and Llama 3.2?
**Answer:**
The dual-model setup serves a comparative research purpose and highlights different LLM paradigms:
* **T5 (Encoder-Decoder):** Seq2Seq models are structurally suited for translation-like tasks. In GEC, T5 acts as a translator from "broken English" to "correct English". It has a strong inductive bias for preserving sequence structure and is highly efficient at making precise, localized edits (high Precision).
* **Llama 3.2 (Decoder-Only Causal LM):** Standard autoregressive models generate tokens left-to-right. Fine-tuning a 3B parameter chat model on GEC tests how well a general-purpose LLM can follow correction instructions. It has high fluency and vocabulary capacity (high Recall) but suffers from a tendency to over-correct and paraphrase rather than minimally editing.
* **Result:** Serving both allows us to contrast a lightweight, task-specific encoder-decoder model (~500ms latency) against a heavy, generative decoder-only model (~1300ms latency).

---

## 2. Model Architecture & Fine-Tuning (T5 vs. Llama vs. BART)

### Q3. How does the fine-tuning process differ between T5 (Encoder-Decoder) and Llama 3.2 (Decoder-Only)?
**Answer:**
The formatting and training objectives are radically different:
1. **T5 Fine-tuning:** 
   * **Input Format:** We prepend a task-specific prefix to the erroneous sentence, e.g., `"grammar: She go to school yesterday."`.
   * **Target Format:** The target sequence is just the clean label, e.g., `"She went to school yesterday."`.
   * **Objective:** Standard sequence-to-sequence cross-entropy loss. The encoder processes the full source sentence, and the decoder autoregressively generates the corrected sentence.
2. **Llama 3.2 Fine-tuning:** 
   * **Input Format:** Formatted using Llama's specific chat template:
     ```
     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
     You are a grammar correction assistant. Correct the grammatical errors in the given sentence. Return only the corrected sentence without any explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
     Correct this sentence: She go to school yesterday.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
     She went to school yesterday.<|eot_id|>
     ```
   * **Objective:** Autoregressive causal language modeling. Crucially, to prevent the model from learning to predict its own prompt instructions, we use **masked loss** (using PyTorch/PEFT's label masking). The labels corresponding to the system and user prompt are set to `-100` so that cross-entropy loss is computed **only** on the target assistant tokens (`"She went to school yesterday."`).

### Q4. Your evaluation notes that Llama 3.2-3B has high recall but extremely low precision (0.0244) and F0.5 score (0.0303). What is the root cause?
**Answer:**
The root cause is the **generative bias** of instruction-tuned causal LLMs. 
* Llama 3.2-3B-Instruct was pre-trained to be conversational, helpful, and highly fluent. When fine-tuned on GEC, instead of performing a *minimal edit* (changing "go" to "went"), it often paraphrases or expands the sentence to make it sound more sophisticated (e.g., rewriting `"She go to school yesterday."` to `"She went to school yesterday to attend her classes."`).
* Because the standard evaluation framework (ERRANT / CoNLL-2014) evaluates GEC by comparing exact word spans against a reference edit, any stylistic rewrite counts as a massive sequence of **False Positives** (unnecessary changes).
* This leads to 10,210 false positives on Llama vs. only a few hundred on T5. Llama's high recall (0.95) shows it catches almost all errors, but its low precision (0.02) shows it ruins its score by over-correcting.

### Q5. How would you resolve this over-correction/paraphrasing issue in Llama 3.2?
**Answer:**
I would implement three techniques to constrain the decoder-only model:
1. **DPO (Direct Preference Optimization) or RLHF:** Train the model using preference pairs where the preferred output is the *minimal edit* correction and the dispreferred output is the *paraphrased* correction.
2. **Prompt Engineering / Few-Shot Examples:** Modify the system instructions to explicitly include few-shot examples showing that stylistic rewrites are forbidden (e.g., `"Input: He walked slow. Output: He walked slowly. (Not: He walked at a slow pace)"`).
3. **Constrained Decoding / Levenshtein Edit Constraints:** Utilize decoding techniques that restrict the output vocabulary or force the generation to align closely with the source tokens, allowing only insertions, deletions, or substitutions.

---

## 3. PEFT & LoRA (Parameter-Efficient Fine-Tuning)

### Q6. What is LoRA (Low-Rank Adaptation) and why did you use it?
**Answer:**
LoRA is a Parameter-Efficient Fine-Tuning (PEFT) method that freezes the pre-trained model weights and injects trainable rank decomposition matrices into the attention layers.
* Instead of updating a massive weight matrix $W_0 \in \mathbb{R}^{d \times k}$ (which is computationally expensive and requires huge optimizer memory), LoRA parameterizes the weight update $\Delta W$ as the product of two low-rank matrices $A$ and $B$:
  $$\Delta W = B \cdot A$$
  where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with a rank $r \ll \min(d, k)$.
* **Benefits:** 
  * **Memory Savings:** Reduces optimizer state VRAM requirements (since we only calculate gradients and track optimizer states for $A$ and $B$).
  * **Storage Efficiency:** For Llama 3.2-3B, instead of saving a 6GB model checkpoint, we only save the 97MB adapter weights. For T5-Large, the adapter is only 18MB.
  * **No Inference Latency:** At inference, the adapter weights can be permanently merged back into the base model weights ($W = W_0 + \frac{\alpha}{r} BA$), introducing zero extra computational overhead.

```
Input Vector (x)
  ├──► [Frozen Pre-trained Weights W0] ────► Output (W0 * x)
  │                                           ▲
  └──► [Trainable Matrix A (down-proj)]       │ (+)
         ▼                                    │
       [Trainable Matrix B (up-proj)]  ──────► Output (ΔW * x)
```

### Q7. How did you choose your LoRA parameters ($r=16$, $\alpha=32$), and which modules did you target?
**Answer:**
* **Rank ($r=16$):** The rank determines the bottleneck size of the low-rank adapters. A rank of 16 represents a standard sweet spot for task-specific adaptation, providing enough capacity to capture grammar correction patterns without overfitting.
* **Alpha ($\alpha=32$):** Alpha is a scaling factor for the LoRA weights. The update is scaled by $\frac{\alpha}{r}$. Setting $\alpha = 2 \cdot r$ is a standard convention that stabilizes the learning rate behavior when adjusting the rank.
* **Target Modules:**
  * For **T5**, we targeted the query (`q`) and value (`v`) projections in the attention mechanism, which is standard for seq2seq models.
  * For **Llama 3.2-3B**, we targeted **all** linear layers: `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention projections) and `gate_proj`, `up_proj`, `down_proj` (MLP layers). Targeting all modules has been shown in papers (like QLoRA) to match or exceed full fine-tuning performance and prevent catastrophic forgetting.

---

## 4. Training Optimizations & Performance Engineering

### Q8. How did you reduce the training time of Llama 3.2-3B from ~30 hours to 11 hours (a ~3x overall speedup, with 5x throughput improvements)?
**Answer:**
We implemented four main optimization techniques:
1. **SDPA (Scaled Dot-Product Attention):** Enabled PyTorch’s native FlashAttention-2 kernels. This bypassed standard attention's memory bottleneck and resolved a critical 11x performance slowdown associated with Grouped Query Attention (GQA) when batching.
2. **Dynamic Padding (Collate Function):** Instead of padding all sentences to a static `max_length=256` globally, we wrote a custom collate function to pad each mini-batch only to the length of its longest sequence. Because attention compute scales quadratically $O(N^2)$, reducing average sequence length from 256 to ~150 tokens saved ~2.9x attention compute.
3. **Gradient Checkpointing:** Recomputed activations on the fly during the backward pass instead of storing them all in VRAM. This saved 40-50% memory, allowing us to double our batch size from 4 to 8, which halved gradient accumulation steps and reduced optimizer overhead.
4. **Step-based Checkpointing & Warm-up:** Avoided training crashes causing full epoch restarts. We used a linear learning rate warmup (10% of steps) and bfloat16 mixed-precision training.

### Q9. Can you explain the difference between standard padding and your dynamic padding implementation?
**Answer:**
* **Standard Padding:** In a typical Hugging Face pipeline, tokenization is done with `padding="max_length"` and `max_length=256`. Even if a batch contains only short sentences of 30 tokens, every sample is padded with 226 `<pad>` tokens. Since attention complexity is $O(N^2)$ (where $N$ is sequence length), we perform calculations on large matrices where 85% of the data is useless padding.
* **Dynamic Padding:** We tokenize with `padding=False`. Then, we feed these variable-length arrays into the PyTorch `DataLoader` with a custom `collate_fn`. The collate function inspects the current batch, finds the local maximum sequence length ($L_{max}$), pads all samples in that batch to $L_{max}$, and sets the target labels for those pads to `-100` (so they are ignored by PyTorch's cross-entropy loss).
* **Code Implementation:**
  ```python
  def make_collate_fn(pad_token_id: int):
      def collate_fn(batch):
          max_len = max(item["input_ids"].shape[0] for item in batch)
          padded = []
          for item in batch:
              pad_len = max_len - item["input_ids"].shape[0]
              padded_ids = torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
              padded_mask = torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
              padded_labels = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
              padded.append((padded_ids, padded_mask, padded_labels))
          return {
              "input_ids": torch.stack([x[0] for x in padded]),
              "attention_mask": torch.stack([x[1] for x in padded]),
              "labels": torch.stack([x[2] for x in padded]),
          }
      return collate_fn
  ```

### Q10. What is Gradient Checkpointing, and what are its trade-offs? What is `model.enable_input_require_grads()`?
**Answer:**
* **Gradient Checkpointing:** Intermediate activation tensors are recomputed during the backward pass instead of storing them all in VRAM. This saved 40-50% memory, allowing us to double our batch size (bs=4 to bs=8), which increased GPU throughput and ultimately made training faster.
* **Trade-off:** Trades compute for memory. It introduces a ~30% computational overhead (performing part of the forward pass twice), but saves ~40-50% VRAM.
* **`enable_input_require_grads()`:** In PyTorch, if a model’s input does not have `requires_grad=True`, the autograd graph will not compute gradients for the inputs. When we use LoRA, the base model weights are frozen; only the tiny adapter matrices require gradients. When combined with gradient checkpointing, this lack of gradients on input features causes PyTorch to throw an error since it cannot trace the gradient flow back through the layers. Calling `model.enable_input_require_grads()` forces the input embedding layer to require gradients, solving this compatibility bug.

---

## 5. NLP Evaluation Metrics (Precision, Recall, F0.5, GLEU, GLUE, ERRANT)

### Q11. Can you explain what Precision and Recall mean specifically in the context of this Grammar Corrector project?
**Answer:**
In this project, evaluation metrics are based on the **ERRANT alignment framework**, which compares the edits proposed by the model against the ground-truth edits proposed by human annotators.
1. **Precision in GEC:** 
   * **Definition:** Out of all the grammatical corrections proposed by our model, what percentage was actually correct (i.e. matched the human reference edits)?
   * **Formula:** 
     $$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$
   * **True Positive (TP):** The model suggested a correction (e.g. changing `"go"` to `"went"`), and this edit matched the reference correction.
   * **False Positive (FP):** The model made a change that was *not* in the reference. This includes:
     * Flagging a grammatically correct sentence and changing it unnecessarily.
     * Rewriting or paraphrasing a sentence (which is what Llama 3.2-3B did, leading to **10,210 FP**).
     * Correcting an actual error but suggesting the wrong replacement word.
2. **Recall in GEC:**
   * **Definition:** Out of all the actual grammatical errors present in the source text, what percentage did our model successfully identify and correct?
   * **Formula:**
     $$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$
   * **False Negative (FN):** The model missed an error entirely, leaving grammatically incorrect text untouched (e.g., leaving `"She go to school"` as `"She go to school"`).

### Q12. Explain the $F_{\text{score}}$ (specifically the $F_{0.5}$ score used here) and why we use it instead of $F_1$ or Accuracy.
**Answer:**
* **$F_{\text{score}}$ ($F_{\beta}$):** The weighted harmonic mean of Precision and Recall. The parameter $\beta$ determines the relative weight given to Precision vs Recall:
  $$F_{\beta} = (1 + \beta^2) \frac{\text{Precision} \times \text{Recall}}{(\beta^2 \times \text{Precision}) + \text{Recall}}$$
* **Why $F_{0.5}$?** In a standard $F_1$ score, Precision and Recall are weighted equally ($\beta=1$). In GEC, we set $\beta=0.5$, which weights **Precision twice as heavily as Recall**:
  $$F_{0.5} = 1.25 \frac{\text{Precision} \times \text{Recall}}{(0.25 \times \text{Precision}) + \text{Recall}}$$
* **GEC Preference for Precision:**
  * If a grammar checker misses a small mistake (**False Negative / Lower Recall**), the text remains slightly grammatically flawed, but is still readable.
  * If a grammar checker changes correct text into something wrong, or alters the user's meaning through paraphrasing (**False Positive / Lower Precision**), it actively damages the user's work, leading to frustration and loss of trust.
  * Therefore, GEC systems must be conservative. We prefer a system that corrects only what it is confident about (high Precision) over a system that attempts to catch everything but makes many incorrect modifications (high Recall, low Precision).
* **Why not Accuracy?** Accuracy measures the percentage of correct predictions. In a GEC dataset, 90%+ of the words in a sentence are already grammatically correct. A dummy model that does nothing (copies input to output) would achieve over 90% accuracy. This is the **class imbalance** problem, which makes accuracy a useless metric for GEC.

### Q13. What is the GLEU score, how is it calculated, and why is it used for GEC evaluation?
**Answer:**
* **GLEU (Generalized Language Evaluation Understanding):** A metric specifically designed for GEC, adapted from BLEU (BiLingual Evaluation Understudy) used in Machine Translation.
* **The Problem with BLEU in GEC:** BLEU only counts n-gram overlap between the model prediction and the reference. Because GEC source and target sentences are highly similar (often sharing 90% of the same words), a model that makes zero edits gets a deceptively high BLEU score.
* **How GLEU Works:** GLEU resolves this by evaluating n-grams (up to 4-grams) in the prediction, comparing them to both the **source** sentence and the **reference** sentence.
  * **Rewards:** It rewards n-grams in the model prediction that match the reference and were *not* present in the original source (meaning the model made a successful correction).
  * **Penalizes:** It penalizes n-grams in the prediction that do *not* match the reference but *were* in the source (meaning the model made an incorrect edit or corrupted a correct part of the sentence).
* **Formula Concept:**
  $$\text{GLEU} = \frac{\text{Matched Output n-grams} - \text{Penalty for bad changes}}{\text{Total n-grams in Prediction}}$$
* **Significance:** GLEU evaluates the **overall fluency and grammatical correctness** of the entire sentence in a single 0-to-1 score, making it a great companion to the span-based $F_{0.5}$ metric.

### Q14. What is GLUE, and how does it differ from GLEU? (Interviewer Trick Question)
**Answer:**
It is crucial not to confuse these two similarly-named terms, as they represent entirely different concepts in NLP:

| Feature | **GLEU** (Generalized Language Evaluation Understanding) | **GLUE** (General Language Understanding Evaluation) |
| :--- | :--- | :--- |
| **Type** | **Evaluation Metric** (Single mathematical score) | **Benchmark Dataset** (A collection of 9 NLP tasks) |
| **Purpose** | Measures the n-gram level quality and fluency of corrected sentences in GEC. | Evaluates how well an NLP model performs across a variety of general language understanding tasks. |
| **Origin** | Adapted from BLEU specifically for grammar error correction. | Developed by NYU, UW, and DeepMind as a standardized model leaderboard. |
| **Composition** | Formulated mathematically based on n-gram precision and editing penalties. | Includes tasks for Sentiment Analysis (SST-2), Natural Language Inference (MNLI, RTE, WNLI), Semantic Similarity (MRPC, QQP, STS-B), and Linguistic Acceptability (CoLA). |

> [!TIP]
> If an interviewer asks: *"Did you use GLUE in your project?"*
> **Correct Answer:** *"We used **GLEU** (with an 'E') as an evaluation metric to measure the fluency of our grammar corrections. We did not use the **GLUE** benchmark suite, as our project focuses specifically on the sequence-to-sequence task of Grammar Error Correction (GEC) evaluated on the BEA 2019 dataset, whereas GLUE is a multi-task benchmark for general classification and inference."*

### Q15. How does the ERRANT framework compute edits under the hood?
**Answer:**
ERRANT (Evaluation and Estimation of Random and Systematic Error) works in three stages:
1. **Linguistic Alignment:** It aligns the source sentence and the model hypothesis sentence. It uses a Levenshtein distance algorithm, but assigns different edit costs based on word similarity, POS (Part of Speech) tags, and lemma matches to ensure alignment is linguistically logical.
2. **Edit Extraction:** It merges adjacent insertions, deletions, and substitutions to extract complete edit spans.
3. **Category Classification:** It runs the edits through a rule-based classifier that uses POS tags, dependency parses, and morphological analysis to assign a specific category label to each edit, such as `R:VERB:SVA` (Subject-Verb Agreement replacement) or `M:DET` (Missing Determiner insertion).

---

## 6. API Design, Diff Extraction, & Post-Processing

### Q16. How does your backend handle paragraph-level text inputs if the underlying models are trained on single sentences?
**Answer:**
We implement a robust **split-and-reassemble** pipeline:
1. **Sentence Splitting:** When a paragraph is received at `/api/v1/correct`, we use a regular expression (`re.split`) to split the text into sentences while capturing the trailing whitespaces:
   ```python
   # Match sentences ending with punctuation followed by space
   parts = re.split(r'(?<=[.!?])\s+', text.strip())
   ```
2. **Preserving Layout:** We map these into `(sentence, trailing_whitespace)` tuples. This is critical because it preserves the exact formatting (newlines, double spaces) of the original paragraph.
3. **Batched Inference:** We collect the sentences and feed them into the model's `correct_batch()` method, processing them in parallel on the GPU.
4. **Reassembly:** We combine the corrected sentences with their original trailing whitespaces:
   ```python
   corrected_text = "".join(corrected_sent + trailing_ws for corrected_sent, trailing_ws in results)
   ```

### Q17. How did you implement the word-level diff extraction (highlighting edits) in the API?
**Answer:**
We use Python's built-in `difflib.SequenceMatcher` to perform a word-level comparison:
1. We split the original and corrected sentences into arrays of words.
2. We run the sequence matcher and retrieve the opcodes (`equal`, `replace`, `insert`, `delete`).
3. We map the word indexes back to character offsets (`PositionSpan`) in the original text so the frontend can highlight the exact character coordinates.
4. **Code Implementation:**
  ```python
  def extract_corrections(original: str, corrected: str) -> List[CorrectionDiff]:
      orig_words = original.split()
      corr_words = corrected.split()
      matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
      corrections = []
      
      # Map word indices to character positions
      char_positions, pos = [], 0
      for word in orig_words:
          idx = original.index(word, pos)
          char_positions.append(idx)
          pos = idx + len(word)
          
      for op, i1, i2, j1, j2 in matcher.get_opcodes():
          if op == "equal": continue
          orig_span = " ".join(orig_words[i1:i2])
          corr_span = " ".join(corr_words[j1:j2])
          start_pos = char_positions[i1] if i1 < len(char_positions) else len(original)
          end_pos = start_pos + len(orig_span)
          
          corrections.append(CorrectionDiff(
              original=orig_span if orig_span else "(missing)",
              corrected=corr_span if corr_span else "(removed)",
              error_type="grammar",
              position=PositionSpan(start=start_pos, end=end_pos)
          ))
      return corrections
  ```

---

## 7. Frontend & Deployment (Docker, React, Nginx)

### Q18. Can you describe how the frontend processes and renders the corrections returned by the FastAPI server?
**Answer:**
* The FastAPI endpoint returns a list of corrections containing `original`, `corrected`, and a `position` object `{start: int, end: int}`.
* In the React frontend, we render the original text. However, we intercept the text matching the coordinates of the edits and wrap them in custom styled HTML tags (e.g., with a red strike-through for deletion and green background for insertion).
* When a user hovers over a highlighted error, a popover/tooltip component displays the correction.
* The React state maintains the text. If the user clicks "Accept", we replace the original substring with the corrected substring at those exact character coordinates, adjusting the offsets of any subsequent corrections dynamically.

### Q19. How did you structure your Docker configuration to ensure optimized builds?
**Answer:**
We implemented two major optimizations in our `Dockerfile` and `docker-compose.yml`:
1. **Separation of Dependencies:** We created two requirements files: `requirements.txt` (full training dependencies, including heavy libraries like HuggingFace `evaluate`, `wandb`, `pytest`) and `requirements-api.txt` (inference-only dependencies, which omit training and testing tools). The FastAPI Docker image is built using `requirements-api.txt`, reducing the final image size by ~60%.
2. **Checkpoint Volume Mounting:** We do **not** bake model weights or fine-tuned LoRA checkpoints (which are ~2GB+ for T5 and ~6GB+ for Llama) directly into the Docker image. Instead, we configure `docker-compose.yml` to mount the local `./checkpoints` directory as a read-only volume. This keeps our Docker images lightweight (under 1.5GB) and allows us to update model weights on the host machine without rebuilding the container.
3. **Named Cache Volumes:** We mount a named volume for the Hugging Face cache directory (`~/.cache/huggingface`) to persist downloaded base models across container restarts.

---

## 8. Conceptual & Advanced Machine Learning Questions

### Q20. Why are encoder-decoder architectures (T5, BART) generally preferred over decoder-only models (Llama, Mistral) for task-oriented sequence-to-sequence NLP tasks like GEC?
**Answer:**
* **Bidirectional Context:** Encoder-Decoder models use a bidirectional encoder. Every input token attends to every other input token. This allows the model to form a holistic understanding of the erroneous sentence before generating the correction. Decoder-only models are causal; they can only attend to previous tokens.
* **Separation of Representations:** The encoder maps the source sentence to a dense, continuous representation space, and the cross-attention layers in the decoder query this representation. This cleanly separates "understanding the source" from "generating the target". In contrast, a decoder-only model must pack prompt parsing, instructions, and generation history into a single active key-value (KV) cache, leading to representation overlap.
* **Minimal Edit Bias:** Encoder-Decoder models trained on seq2seq tasks naturally align their cross-attention matrices to perform token substitutions. Autoregressive decoder-only models generate sequences token-by-token and are structurally biased to generate long, conversational continuations rather than strict translations.
* **Inference Efficiency:** T5-Large has 780M parameters, whereas Llama 3.2 has 3B parameters. T5 is 3-4x faster and uses a fraction of the memory, making it far more practical for high-throughput production deployment.

### Q21. What is Mixed Precision Training, and why did you use `bfloat16` instead of `float16`?
**Answer:**
* **Mixed Precision Training:** Uses both 16-bit and 32-bit floating-point formats during training. It keeps model weights in 16-bit for fast matrix multiplication (tensor cores) and lower memory, but computes loss and certain gradient updates in 32-bit to avoid numerical instability.
* **Why `bfloat16`?** 
  * `float16` has 5 exponent bits and 10 mantissa (significand) bits. Because of its narrow exponent range, it is prone to underflow/overflow, which requires using a `GradScaler` to scale loss values. T5 is notorious for gradient overflows and training instability under `float16`.
  * `bfloat16` has 8 exponent bits and 7 mantissa bits—the exact same exponent range as `float32`. This means it can represent the same range of errors as float32, completely eliminating underflow/overflow issues. It does not require a `GradScaler` and provides stable training for both T5 and Llama models.

```
float32:  [1 sign bit] [8 exponent bits] [23 mantissa bits]
float16:  [1 sign bit] [5 exponent bits] [10 mantissa bits]  (Prone to overflow)
bfloat16: [1 sign bit] [8 exponent bits] [7 mantissa bits]   (Stable range)
```

### Q22. Can you explain the GQA (Grouped Query Attention) architecture in Llama 3.2 and its advantage over MHA (Multi-Head Attention)?
**Answer:**
* **Multi-Head Attention (MHA):** Every query head has a corresponding Key ($K$) and Value ($V$) head. During generation, we store these $K$ and $V$ states for all tokens in the KV Cache. For large batch sizes and sequence lengths, the KV Cache becomes extremely large, bottlenecking GPU memory bandwidth.
* **Multi-Query Attention (MQA):** All query heads share a single $K$ and $V$ head. This minimizes KV cache size but degrades model capacity and quality.
* **Grouped Query Attention (GQA):** Llama 3.2 groups query heads into $N$ groups (e.g., 8 groups). Each group shares a single $K$ and $V$ head. GQA acts as a middle ground: it provides almost the same performance as MHA while reducing the KV cache footprint by 8x. This allows for much larger batch sizes during both training and inference.

```
Multi-Head Attention (MHA)        Grouped Query Attention (GQA)       Multi-Query Attention (MQA)
    Q Q Q Q Q Q Q Q                   Q Q Q Q   Q Q Q Q                   Q Q Q Q Q Q Q Q
    │ │ │ │ │ │ │ │                    \ \ / /   \ \ / /                    \ \ \ / / / /
    ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼                     ▼   ▼     ▼   ▼                       ▼       ▼
    K K K K K K K K                     K   K     K   K                       K       K
    V V V V V V V V                     V   V     V   V                       V       V
```

---

## 9. Key Performance Benchmarks

Be prepared to quote these numbers in your interview to show you are highly data-driven:

### Model Performance (BEA 2019 Test Set)
* **FLAN-T5-Large + LoRA (Our Best Model):**
  * **$F_{0.5}$ Score:** **0.3201**
  * **Precision:** **0.2744** (conservatively corrects actual errors)
  * **Recall:** **0.9588** (detects almost all errors)
  * **GLEU Score:** **0.9245** (fluent output)
  * **Latency:** **~500ms** per sentence
* **Llama 3.2-3B + LoRA (Generative Model):**
  * **$F_{0.5}$ Score:** **0.0303** (low due to massive paraphrasing false positives)
  * **Precision:** **0.0244**
  * **Recall:** **0.9515**
  * **GLEU Score:** **0.7431**
  * **Latency:** **~1300ms** per sentence

### Llama 3.2 Training Speedups (RTX 4080 SUPER 16GB)
* **Baseline (No optimizations, Batch Size = 4):** 1355ms/batch | 3.0 samples/sec | 15.4GB VRAM
* **Optimized (SDPA + Dynamic Padding + Grad Checkpointing + Batch Size = 8):** **521ms/batch** | **15.3 samples/sec** | **8.1GB VRAM**
* **Net Result:** **5.1x throughput improvement** and **47% VRAM reduction**, cutting training time from ~30 hours down to **11 hours**.
