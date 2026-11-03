"""
Llama 3.2-3B Grammar Error Correction model with LoRA.

Wraps Meta's Llama-3.2-3B-Instruct model with a domain-specific
interface for grammar correction tasks using LoRA fine-tuning.

Architecture:
    - Base: Llama-3.2-3B-Instruct (3 billion parameters)
    - Fine-tuning: LoRA adapters (trains only ~0.1% of parameters)
    - Training method: Causal language modeling with chat format
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CorrectionOutput:
    """Output from the grammar correction model.

    Attributes:
        corrected_text: The corrected version of the input text.
        confidence: Model confidence score (0-1).
        corrections: List of individual corrections made.
    """

    corrected_text: str
    confidence: float
    corrections: List[Dict[str, str]]


class LlamaGEC(nn.Module):
    """
    Llama 3.2-3B for Grammar Error Correction with LoRA.

    Uses instruction-following format with chat template for
    natural grammar correction interactions.

    Attributes:
        model: Underlying Llama model (with optional LoRA adapters).
        tokenizer: Llama tokenizer.
        max_length: Maximum sequence length for generation.
        device: Device the model is on (cpu/cuda).
        use_lora: Whether LoRA adapters are attached.

    Example:
        >>> # Load pre-trained model with LoRA
        >>> gec = LlamaGEC.from_pretrained(
        ...     "meta-llama/Llama-3.2-3B-Instruct",
        ...     use_lora=True,
        ...     lora_r=16
        ... )
        >>>
        >>> # Correct text
        >>> result = gec.correct_text("She go to school yesterday.")
        >>> print(result.corrected_text)
        "She went to school yesterday."
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        use_lora: bool = False,
    ) -> None:
        """
        Initialize LlamaGEC.

        Args:
            model: Pre-trained or fine-tuned Llama model.
            tokenizer: Corresponding tokenizer.
            max_length: Maximum generation length.
            use_lora: Whether LoRA adapters are attached.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_lora = use_lora
        self.device = next(model.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_8bit: bool = False,
        device: Optional[str] = None,
        max_length: int = 512,
        use_gradient_checkpointing: bool = False,
    ) -> "LlamaGEC":
        """
        Load a Llama model for grammar correction.

        Handles two cases automatically:
        1. Existing PEFT/LoRA checkpoint (has adapter_config.json) -> loads
           the base model and applies saved LoRA weights.
        2. Base model name (HuggingFace hub or local) -> loads base model
           and optionally attaches fresh LoRA adapters for training.

        Args:
            model_name_or_path: HuggingFace model name, local LoRA checkpoint
                path, or base model path.
            use_lora: Whether to add fresh LoRA adapters (ignored if loading
                an existing LoRA checkpoint).
            lora_r: LoRA rank.
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout for LoRA layers.
            load_in_8bit: Use 8-bit quantization to reduce VRAM.
            device: Device to load model on. Auto-detects if None.
            max_length: Maximum generation length.

        Returns:
            Initialized LlamaGEC instance.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Detect if this is a saved PEFT checkpoint
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        is_peft_checkpoint = os.path.isfile(adapter_config_path)

        if is_peft_checkpoint:
            logger.info(f"Detected PEFT/LoRA checkpoint: {model_name_or_path}")
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path", "meta-llama/Llama-3.2-3B-Instruct")
            logger.info(f"  Base model: {base_model_name}")
        else:
            base_model_name = model_name_or_path
            logger.info(f"Loading base model: {model_name_or_path}")

        logger.info(f"  Device: {device}")
        logger.info(f"  8-bit: {load_in_8bit}")

        # Load tokenizer from checkpoint or base model
        tokenizer_path = model_name_or_path if is_peft_checkpoint else base_model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Llama does not have a pad token by default
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Left-pad for causal LM generation
        tokenizer.padding_side = "left"

        # Build kwargs for base model loading.
        # For inference from a saved checkpoint use device_map="auto" so
        # multi-GPU or CPU-offload works transparently.
        # For training we must NOT use device_map="auto" because accelerate
        # will place LoRA adapter parameters on CPU while the base model sits
        # on the GPU, causing a ~80× slowdown from CPU↔GPU transfers on every
        # forward pass.  Use the explicit device string instead.
        if is_peft_checkpoint:
            # Inference path: "auto" is fine, adapters load onto the same
            # device as the base layer they belong to.
            map_device: Union[str, None] = "auto" if device == "cuda" else None
        else:
            # Training path: load everything onto one explicit device so LoRA
            # parameters end up on the GPU after get_peft_model().
            map_device = device if device == "cuda" else None

        model_kwargs: Dict = {
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
            "device_map": map_device,
            # Use PyTorch SDPA attention (built-in FlashAttention + MemEfficient
            # kernels). Requires PyTorch >= 2.0. Gives ~1.5-2x speedup over
            # standard attention for typical sequence lengths.
            "attn_implementation": "sdpa",
        }
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

        if device == "cpu" and not load_in_8bit:
            model = model.to(device)

        if is_peft_checkpoint:
            # Apply saved LoRA weights
            try:
                from peft import PeftModel
                logger.info("Loading LoRA adapter weights...")
                model = PeftModel.from_pretrained(model, model_name_or_path)
                logger.info("LoRA adapters loaded successfully")
                use_lora = True
            except ImportError:
                logger.error("PEFT library not installed: pip install peft")
                use_lora = False

        elif use_lora:
            # Attach fresh LoRA adapters for training
            try:
                from peft import LoraConfig, get_peft_model, TaskType

                logger.info("Adding fresh LoRA adapters for training...")
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()

            except ImportError:
                logger.error("PEFT library not installed: pip install peft")
                use_lora = False

        # Gradient checkpointing: recomputes activations during backward pass
        # instead of storing them in VRAM. Trades ~30% extra compute for ~40-50%
        # VRAM savings on activations. Frees enough memory to potentially double
        # the batch size (which more than compensates for the compute overhead).
        if use_gradient_checkpointing and device == "cuda":
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # Required for gradient checkpointing with LoRA
            if use_lora and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            logger.info("Gradient checkpointing enabled (saves ~40-50% activation VRAM)")

        logger.info("Model loaded successfully")

        return cls(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            use_lora=use_lora,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Tokenized input sequence [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            labels: Target token IDs for training [batch_size, seq_len].
                Padding tokens and prompt tokens should be -100 (ignored by loss).

        Returns:
            Dictionary with 'loss' (if labels provided) and 'logits'.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss

        return result

    @torch.no_grad()
    def _generate_with_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        """
        Run generation and return (sequences, per-token probabilities).

        Returns a tuple of:
            sequences: generated token IDs including prompt [batch, seq]
            token_probs: mean softmax probability of generated tokens per item
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        prompt_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        token_probs = []

        for b in range(batch_size):
            generated = outputs.sequences[b][prompt_len:]
            if len(outputs.scores) == 0 or len(generated) == 0:
                token_probs.append(0.8)
                continue

            probs_per_token = []
            for step_idx, step_scores in enumerate(outputs.scores):
                if step_idx >= len(generated):
                    break
                token_id = generated[step_idx].item()
                if token_id == self.tokenizer.eos_token_id:
                    break
                # step_scores shape: [beam_batch, vocab] — take beam 0 for item b
                # When num_beams > 1, scores are for beam-0 sequence
                score_row = step_scores[b] if step_scores.shape[0] > b else step_scores[0]
                prob = F.softmax(score_row, dim=-1)[token_id].item()
                probs_per_token.append(prob)

            if probs_per_token:
                # Geometric mean gives a useful "overall confidence"
                import math
                log_sum = sum(math.log(max(p, 1e-9)) for p in probs_per_token)
                geo_mean = math.exp(log_sum / len(probs_per_token))
                token_probs.append(round(geo_mean, 4))
            else:
                token_probs.append(0.8)

        return outputs.sequences, token_probs

    def _build_chat_prompt(self, text: str) -> str:
        """Format a single text as a Llama chat prompt for inference."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a grammar correction assistant. "
                    "Correct grammatical errors in the given text. "
                    "Respond ONLY with the corrected text, without explanations."
                ),
            },
            {
                "role": "user",
                "content": f"Correct this: {text}",
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def correct_text(
        self,
        text: str,
        num_beams: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> CorrectionOutput:
        """
        Correct grammatical errors in input text.

        Args:
            text: Input text with potential grammatical errors.
            num_beams: Number of beams for beam search (1 = greedy).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (only used if do_sample=True).

        Returns:
            CorrectionOutput with corrected text and confidence score.

        Example:
            >>> result = gec.correct_text("She go to school yesterday.")
            >>> print(result.corrected_text)
            "She went to school yesterday."
        """
        prompt = self._build_chat_prompt(text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        sequences, token_probs = self._generate_with_scores(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = sequences[0][prompt_length:]
        corrected = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return CorrectionOutput(
            corrected_text=corrected,
            confidence=token_probs[0],
            corrections=[],
        )

    @torch.no_grad()
    def correct_batch(
        self,
        texts: List[str],
        num_beams: int = 1,
        max_new_tokens: int = 128,
        batch_size: int = 4,
    ) -> List[CorrectionOutput]:
        """
        Correct grammatical errors in a batch of texts.

        Processes texts in mini-batches using true batched inference with
        left-padding for efficient parallel generation.

        Args:
            texts: List of input texts.
            num_beams: Number of beams for beam search (1 = greedy).
            max_new_tokens: Maximum tokens to generate per text.
            batch_size: Number of texts per GPU mini-batch.

        Returns:
            List of CorrectionOutput for each input text.
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            prompts = [self._build_chat_prompt(t) for t in batch_texts]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            prompt_length = inputs["input_ids"].shape[1]

            sequences, token_probs = self._generate_with_scores(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )

            for b, conf in enumerate(token_probs):
                generated_ids = sequences[b][prompt_length:]
                corrected = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
                results.append(
                    CorrectionOutput(
                        corrected_text=corrected,
                        confidence=conf,
                        corrections=[],
                    )
                )

        return results

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model and tokenizer to a directory.

        If using LoRA, saves only the adapter weights (~20-30 MB).
        Otherwise saves the full model.

        Args:
            save_directory: Path to save directory.
        """
        logger.info(f"Saving model to: {save_directory}")
        self.tokenizer.save_pretrained(save_directory)
        self.model.save_pretrained(save_directory)
        logger.info("Model saved successfully")
