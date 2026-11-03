"""
T5-based Grammar Error Correction model with LoRA support.

Wraps Hugging Face's T5ForConditionalGeneration (FLAN-T5-Large)
with LoRA adapters for parameter-efficient fine-tuning.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizer


@dataclass
class CorrectionOutput:
    """Output from the grammar correction model."""

    corrected_text: str
    confidence: float
    corrections: List[Dict[str, str]]


class T5GEC(nn.Module):
    """
    T5-based Grammar Error Correction with LoRA.

    Wraps FLAN-T5-Large (780M parameters) with LoRA adapters
    for efficient grammar correction fine-tuning.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        max_length: int = 256,
        task_prefix: str = "grammar: ",
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.task_prefix = task_prefix

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "google/flan-t5-large",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_8bit: bool = False,
        device: Optional[str] = None,
        max_length: int = 256,
        task_prefix: str = "grammar: ",
    ) -> "T5GEC":
        """
        Load T5 model with optional LoRA adapters.

        If model_name_or_path points to a PEFT adapter checkpoint (contains
        adapter_config.json), loads the base model and applies the saved
        adapter weights. Otherwise loads the model directly and optionally
        adds fresh LoRA adapters for training.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            use_lora: Whether to add LoRA adapters.
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha scaling factor.
            lora_dropout: Dropout for LoRA layers.
            load_in_8bit: Use 8-bit quantization.
            device: Device to load model on.
            max_length: Maximum sequence length.

        Returns:
            T5GEC instance with loaded model.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading T5 model from: {model_name_or_path}")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - LoRA: {use_lora} (r={lora_r})")
        logger.info(f"  - 8-bit: {load_in_8bit}")

        # Check if this is a saved PEFT adapter checkpoint
        adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
        is_peft_checkpoint = adapter_config_path.exists()

        # Load tokenizer from the provided path
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Build model loading kwargs
        load_kwargs = {
            "device_map": "auto" if device == "cuda" else None,
        }
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["torch_dtype"] = torch.float16
        elif device == "cuda":
            load_kwargs["torch_dtype"] = torch.bfloat16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        if is_peft_checkpoint:
            # ── Loading a saved PEFT adapter ────────────────────────────
            # Read the adapter config to find the base model
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get(
                "base_model_name_or_path", "google/flan-t5-large"
            )
            logger.info(f"Detected PEFT checkpoint. Base model: {base_model_name}")

            # Load the base model (without any adapters)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name, **load_kwargs
            )

            # Load trained LoRA adapter weights onto the base model
            from peft import PeftModel

            logger.info("Loading trained LoRA adapter weights...")
            model = PeftModel.from_pretrained(base_model, model_name_or_path)
            model.print_trainable_parameters()
        else:
            # ── Loading a base model (for training or non-LoRA use) ─────
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path, **load_kwargs
            )

            if use_lora:
                try:
                    from peft import LoraConfig, TaskType, get_peft_model

                    logger.info("Adding fresh LoRA adapters for training...")
                    lora_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=["q", "v"],
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type=TaskType.SEQ_2_SEQ_LM,
                    )
                    model = get_peft_model(model, lora_config)
                    model.print_trainable_parameters()
                except ImportError:
                    logger.warning("PEFT not installed. Skipping LoRA.")

        if device == "cuda" and not load_in_8bit:
            model = model.to(device)

        logger.info(f"✓ T5 model loaded successfully")

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            task_prefix=task_prefix,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            labels: Target token IDs [batch, seq_len].

        Returns:
            Dictionary with 'loss' and 'logits'.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        temperature: float = 1.0,
    ):
        """
        Generate corrected text with sequence scores.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            max_new_tokens: Maximum tokens to generate.
            num_beams: Number of beams for beam search.
            temperature: Sampling temperature.

        Returns:
            GenerateOutput with .sequences and .sequences_scores.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    @staticmethod
    def _score_to_confidence(score: float) -> float:
        """Convert a log-probability sequence score to a 0-1 confidence value.

        Beam search scores are length-normalized log-probabilities (negative).
        We map them through a sigmoid-like transformation so that typical
        high-quality outputs (score ~ -0.1 to -0.5) land in the 0.7-0.95
        range, while poor outputs (score < -2) fall below 0.5.
        """
        import math
        # score is negative log-prob; closer to 0 = more confident
        # sigmoid(score * 3 + 2) maps: -0.1 → 0.85, -0.5 → 0.62, -1.0 → 0.27
        # We use a gentler curve: sigmoid(score + 1) * scale
        return round(1.0 / (1.0 + math.exp(-(score + 1.0) * 2.0)), 4)

    def correct_text(
        self,
        text: str,
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ) -> CorrectionOutput:
        """
        Correct a single text.

        Args:
            text: Erroneous text to correct.
            max_new_tokens: Maximum tokens to generate.
            num_beams: Number of beams.

        Returns:
            CorrectionOutput with corrected text.
        """
        # Format input with task prefix
        prompt = f"{self.task_prefix}{text}"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        ).to(self.device)

        # Generate
        gen_output = self.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        # Decode
        corrected = self.tokenizer.decode(
            gen_output.sequences[0], skip_special_tokens=True
        )

        # Compute confidence from beam search sequence score
        if gen_output.sequences_scores is not None:
            confidence = self._score_to_confidence(
                gen_output.sequences_scores[0].item()
            )
        else:
            confidence = 0.5

        return CorrectionOutput(
            corrected_text=corrected.strip(),
            confidence=confidence,
            corrections=[],
        )

    def correct_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ) -> List[CorrectionOutput]:
        """
        Correct a batch of texts.

        Args:
            texts: List of texts to correct.
            batch_size: Batch size for processing.
            max_new_tokens: Max tokens to generate.
            num_beams: Number of beams for beam search.

        Returns:
            List of CorrectionOutputs.
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Format with task prefix
            prompts = [f"{self.task_prefix}{text}" for text in batch]

            # Tokenize batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            ).to(self.device)

            # Generate
            gen_output = self.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

            # Decode batch
            corrected_texts = self.tokenizer.batch_decode(
                gen_output.sequences, skip_special_tokens=True
            )

            # Extract per-sequence confidence scores
            if gen_output.sequences_scores is not None:
                scores = gen_output.sequences_scores.tolist()
            else:
                scores = [0.0] * len(corrected_texts)

            for text, score in zip(corrected_texts, scores):
                results.append(
                    CorrectionOutput(
                        corrected_text=text.strip(),
                        confidence=self._score_to_confidence(score),
                        corrections=[],
                    )
                )

        return results

    def save_pretrained(self, output_dir: str) -> None:
        """
        Save model and tokenizer.

        Args:
            output_dir: Directory to save to.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model (LoRA adapters if applicable)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"Model saved to: {output_path}")
