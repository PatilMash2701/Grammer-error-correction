"""
BART-based Grammar Error Correction model.

Wraps Hugging Face's BartForConditionalGeneration with a
domain-specific interface for grammar correction tasks.

Architecture:
    - Encoder: 6 transformer layers, processes erroneous input
    - Decoder: 6 transformer layers, generates corrected output
    - Parameters: 140M (bart-base)
    - Pre-training: Denoising autoencoder (text reconstruction)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    PreTrainedTokenizer,
)


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


class BartGEC(nn.Module):
    """
    BART-based Grammar Error Correction model.

    Wraps BartForConditionalGeneration with convenience methods
    for grammar correction inference and training.

    Attributes:
        model: Underlying BART model from Hugging Face.
        tokenizer: BART tokenizer.
        max_length: Maximum sequence length for generation.
        device: Device to run the model on (cpu/cuda).

    Example:
        >>> gec = BartGEC.from_pretrained("facebook/bart-base")
        >>> result = gec.correct_text("She go to school yesterday.")
        >>> print(result.corrected_text)
        "She went to school yesterday."
    """

    def __init__(
        self,
        model: BartForConditionalGeneration,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ) -> None:
        """
        Initialize BartGEC.

        Args:
            model: Pre-trained BART model.
            tokenizer: Corresponding tokenizer.
            max_length: Maximum generation length.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = next(model.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "facebook/bart-base",
        max_length: int = 128,
        device: Optional[str] = None,
    ) -> "BartGEC":
        """
        Load a pre-trained BART model for grammar correction.

        Args:
            model_name_or_path: Hugging Face model name or local path.
            max_length: Maximum generation length.
            device: Device to load model on. Auto-detects if None.

        Returns:
            Initialized BartGEC instance.

        Example:
            >>> gec = BartGEC.from_pretrained("facebook/bart-base")
            >>> gec = BartGEC.from_pretrained("./checkpoints/best_model")
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        model = model.to(device)

        return cls(model=model, tokenizer=tokenizer, max_length=max_length)

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

        Returns:
            Dictionary with 'loss' (if labels provided) and 'logits'.

        Example:
            >>> outputs = gec(input_ids, attention_mask, labels=labels)
            >>> loss = outputs['loss']
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
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_beams: int = 5,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate corrected token sequences using beam search.

        Args:
            input_ids: Tokenized input [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            num_beams: Number of beams for beam search decoding.
            max_length: Maximum generation length (overrides default).

        Returns:
            Generated token IDs [batch_size, gen_seq_len].

        Example:
            >>> generated_ids = gec.generate(input_ids, attention_mask)
            >>> text = gec.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length or self.max_length,
            early_stopping=True,
        )

    def correct_text(
        self,
        text: str,
        num_beams: int = 5,
    ) -> CorrectionOutput:
        """
        Correct grammatical errors in the input text.

        End-to-end method: tokenizes input, runs generation, decodes
        output, and computes a diff of corrections.

        Args:
            text: Input text with potential grammatical errors.
            num_beams: Number of beams for beam search.

        Returns:
            CorrectionOutput with corrected text and metadata.

        Example:
            >>> gec = BartGEC.from_pretrained("facebook/bart-base")
            >>> result = gec.correct_text("She go to school yesterday.")
            >>> print(result.corrected_text)
            >>> print(result.corrections)
        """
        raise NotImplementedError("correct_text not yet implemented")

    def correct_batch(
        self,
        texts: List[str],
        num_beams: int = 5,
        batch_size: int = 8,
    ) -> List[CorrectionOutput]:
        """
        Correct grammatical errors in a batch of texts.

        Processes texts in batches for efficiency.

        Args:
            texts: List of input texts.
            num_beams: Number of beams for beam search.
            batch_size: Number of texts to process at once.

        Returns:
            List of CorrectionOutput for each input text.

        Example:
            >>> results = gec.correct_batch([
            ...     "She go to school.",
            ...     "He have a cat.",
            ... ])
        """
        raise NotImplementedError("correct_batch not yet implemented")
