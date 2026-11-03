"""Generate synthetic GEC dataset using error injection.

This script creates a synthetic grammatical error correction dataset by:
1. Using clean English sentences from various sources
2. Applying rule-based error injection (verb agreement, articles, plurals, etc.)
3. Creating parallel source (erroneous) and target (correct) pairs
"""

import random
import re
from pathlib import Path
from typing import List, Tuple
import json

# Clean sentence sources (can be expanded with real text files)
SAMPLE_SENTENCES = [
    "The student goes to school every day.",
    "They were playing basketball in the park.",
    "She has three books on her desk.",
    "I am going to the store tomorrow.",
    "He doesn't like pizza very much.",
    "The cats are sleeping on the couch.",
    "We have been studying for two hours.",
    "The teacher explains the lesson clearly.",
    "My friends and I went to the movies.",
    "The children play in the garden.",
    "She writes a letter to her grandmother.",
    "They don't understand the question.",
    "The dog runs very fast.",
    "I have seen that movie before.",
    "The flowers bloom in spring.",
    "He reads books every night.",
    "We are learning English grammar.",
    "The bird flies over the house.",
    "She cooks dinner for her family.",
    "They study mathematics at university.",
    "The sun rises in the east.",
    "I work at a technology company.",
    "The students do their homework.",
    "She speaks three languages fluently.",
    "We eat breakfast at seven o'clock.",
    "The car needs a new battery.",
    "He teaches English at the school.",
    "They build houses for a living.",
    "The baby cries when hungry.",
    "I meet my friends every weekend.",
    "The train arrives at noon.",
    "She knows the answer to the question.",
    "We travel to Europe every summer.",
    "The phone rings constantly.",
    "He fixes computers professionally.",
    "They grow vegetables in their garden.",
    "The wind blows strongly today.",
    "I buy groceries every week.",
    "The river flows through the city.",
    "She thinks about the problem carefully.",
    "We listen to music while working.",
    "The scientists conduct important research.",
    "He drives to work every morning.",
    "They sell handmade jewelry online.",
    "The clock shows the correct time.",
    "I remember my childhood clearly.",
    "The athletes train hard every day.",
    "She manages a large team.",
    "We discuss important topics regularly.",
]

class GrammarErrorInjector:
    """Inject various types of grammatical errors into clean sentences."""

    def __init__(self, error_probability: float = 1.0):
        """Initialize with probability of injecting errors."""
        self.error_probability = error_probability

    def inject_verb_agreement_error(self, sentence: str) -> Tuple[str, bool]:
        """Inject subject-verb agreement errors."""
        # Pattern: third person singular present tense
        patterns = [
            (r'\b(He|She|It|The \w+)\s+(goes|does|has|is|was|writes|reads|teaches|knows)\b',
             lambda m: f"{m.group(1)} {self._wrong_verb_form(m.group(2))}"),
            (r'\b(I|You|We|They|The \w+s)\s+(go|do|have|are|were|write|read|teach|know)\b',
             lambda m: f"{m.group(1)} {self._wrong_verb_form_plural(m.group(2))}"),
        ]

        for pattern, replacement in patterns:
            if re.search(pattern, sentence):
                modified = re.sub(pattern, replacement, sentence, count=1)
                if modified != sentence:
                    return modified, True
        return sentence, False

    def _wrong_verb_form(self, verb: str) -> str:
        """Convert singular verb to incorrect form."""
        mapping = {
            "goes": "go", "does": "do", "has": "have",
            "is": "are", "was": "were", "writes": "write",
            "reads": "read", "teaches": "teach", "knows": "know"
        }
        return mapping.get(verb, verb.rstrip('s'))

    def _wrong_verb_form_plural(self, verb: str) -> str:
        """Convert plural verb to incorrect singular form."""
        mapping = {
            "go": "goes", "do": "does", "have": "has",
            "are": "is", "were": "was", "write": "writes",
            "read": "reads", "teach": "teaches", "know": "knows"
        }
        return mapping.get(verb, verb + 's')

    def inject_article_error(self, sentence: str) -> Tuple[str, bool]:
        """Inject article errors (a/an/the)."""
        modifications = []

        # Remove articles
        modified = re.sub(r'\b(a|an|the)\s+', '', sentence, count=1)
        if modified != sentence:
            modifications.append(modified)

        # Wrong article choice
        modified = re.sub(r'\ba\s+([aeiou])', r'a \1', sentence, count=1)
        if modified != sentence:
            modifications.append(modified)

        modified = re.sub(r'\ban\s+([^aeiou])', r'an \1', sentence, count=1)
        if modified != sentence:
            modifications.append(modified)

        if modifications:
            return random.choice(modifications), True
        return sentence, False

    def inject_plural_error(self, sentence: str) -> Tuple[str, bool]:
        """Inject singular/plural errors."""
        # Pattern: number + plural noun -> number + singular noun
        modified = re.sub(r'\b(two|three|four|five|many|several)\s+(\w+s)\b',
                         lambda m: f"{m.group(1)} {m.group(2).rstrip('s')}",
                         sentence, count=1)
        if modified != sentence:
            return modified, True

        # Pattern: singular subject + plural verb
        modified = re.sub(r'\bThe (\w+)\s+(are|were|have)\b',
                         lambda m: f"The {m.group(1)} {m.group(2)}",
                         sentence, count=1)
        if modified != sentence:
            return modified, True

        return sentence, False

    def inject_tense_error(self, sentence: str) -> Tuple[str, bool]:
        """Inject verb tense errors."""
        # Pattern: auxiliary + verb
        patterns = [
            (r'\b(have|has)\s+(go|see|be|do)\b',
             lambda m: f"{m.group(1)} {m.group(2)}"),
            (r'\b(will|would|can|could)\s+(\w+ed)\b',
             lambda m: f"{m.group(1)} {m.group(2)}"),
        ]

        for pattern, replacement in patterns:
            modified = re.sub(pattern, replacement, sentence, count=1)
            if modified != sentence:
                return modified, True
        return sentence, False

    def inject_preposition_error(self, sentence: str) -> Tuple[str, bool]:
        """Inject preposition errors."""
        prep_swaps = {
            " in ": " on ", " on ": " in ",
            " at ": " in ", " to ": " for ",
            " for ": " to ", " with ": " by "
        }

        for correct, wrong in prep_swaps.items():
            if correct in sentence:
                modified = sentence.replace(correct, wrong, 1)
                return modified, True
        return sentence, False

    def inject_contraction_error(self, sentence: str) -> Tuple[str, bool]:
        """Inject contraction errors (don't vs doesn't)."""
        patterns = [
            (r"\b(He|She|It)\s+don't\b", r"\1 doesn't"),
            (r"\b(He|She|It)\s+doesn't\b", r"\1 don't"),
            (r"\b(I|You|We|They)\s+doesn't\b", r"\1 don't"),
        ]

        for pattern, replacement in patterns:
            modified = re.sub(pattern, replacement, sentence)
            if modified != sentence:
                return modified, True
        return sentence, False

    def inject_random_errors(self, sentence: str, num_errors: int = 1) -> str:
        """Inject random grammatical errors into a sentence."""
        if random.random() > self.error_probability:
            return sentence

        error_types = [
            self.inject_verb_agreement_error,
            self.inject_article_error,
            self.inject_plural_error,
            self.inject_tense_error,
            self.inject_preposition_error,
            self.inject_contraction_error,
        ]

        modified_sentence = sentence
        errors_injected = 0
        attempts = 0
        max_attempts = len(error_types) * 2

        while errors_injected < num_errors and attempts < max_attempts:
            error_func = random.choice(error_types)
            new_sentence, success = error_func(modified_sentence)

            if success:
                modified_sentence = new_sentence
                errors_injected += 1

            attempts += 1

        # Return original if no errors could be injected
        return modified_sentence if errors_injected > 0 else sentence


def generate_dataset(
    num_samples: int,
    output_dir: Path,
    base_sentences: List[str] = None,
    error_probability: float = 1.0,
    errors_per_sentence: int = 1
) -> None:
    """Generate synthetic GEC dataset.

    Args:
        num_samples: Number of training samples to generate
        output_dir: Directory to save dataset files
        base_sentences: List of clean sentences (uses SAMPLE_SENTENCES if None)
        error_probability: Probability of injecting errors (0.0 to 1.0)
        errors_per_sentence: Number of errors to inject per sentence
    """
    if base_sentences is None:
        base_sentences = SAMPLE_SENTENCES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    injector = GrammarErrorInjector(error_probability=error_probability)

    # Generate samples
    source_sentences = []
    target_sentences = []

    for i in range(num_samples):
        # Cycle through base sentences
        clean_sentence = base_sentences[i % len(base_sentences)]

        # Inject errors
        erroneous_sentence = injector.inject_random_errors(
            clean_sentence,
            num_errors=errors_per_sentence
        )

        # Only include if error was successfully injected
        if erroneous_sentence != clean_sentence:
            source_sentences.append(erroneous_sentence)
            target_sentences.append(clean_sentence)

    # Save in BEA-2019 format (one sentence per line, parallel files)
    with open(output_dir / "source.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(source_sentences))

    with open(output_dir / "target.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(target_sentences))

    # Save metadata
    metadata = {
        "num_samples": len(source_sentences),
        "base_sentences_count": len(base_sentences),
        "error_probability": error_probability,
        "errors_per_sentence": errors_per_sentence,
        "format": "BEA-2019 compatible (parallel source/target files)"
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Generated {len(source_sentences)} samples")
    print(f"[OK] Saved to: {output_dir}")
    print(f"  - source.txt: {len(source_sentences)} erroneous sentences")
    print(f"  - target.txt: {len(target_sentences)} correct sentences")
    print(f"  - metadata.json: dataset information")

    # Print sample
    print("\n" + "="*60)
    print("SAMPLE DATA (first 5 pairs):")
    print("="*60)
    for i in range(min(5, len(source_sentences))):
        print(f"\n[{i+1}] Erroneous: {source_sentences[i]}")
        print(f"    Correct:   {target_sentences[i]}")


def load_external_sentences(file_path: Path) -> List[str]:
    """Load sentences from external text file (one per line)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic GEC dataset")
    parser.add_argument("--num_train", type=int, default=10000,
                       help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=1000,
                       help="Number of validation samples")
    parser.add_argument("--num_test", type=int, default=1000,
                       help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="./data/synthetic_gec",
                       help="Output directory for dataset")
    parser.add_argument("--sentences_file", type=str, default=None,
                       help="Optional: Path to file with clean sentences (one per line)")
    parser.add_argument("--errors_per_sentence", type=int, default=1,
                       help="Number of errors to inject per sentence")

    args = parser.parse_args()

    # Load custom sentences if provided
    base_sentences = SAMPLE_SENTENCES
    if args.sentences_file:
        base_sentences = load_external_sentences(Path(args.sentences_file))
        print(f"[OK] Loaded {len(base_sentences)} base sentences from {args.sentences_file}")
    else:
        print(f"[OK] Using {len(SAMPLE_SENTENCES)} built-in sample sentences")

    output_base = Path(args.output_dir)

    # Generate train set
    print("\n" + "="*60)
    print("GENERATING TRAINING SET")
    print("="*60)
    generate_dataset(
        num_samples=args.num_train,
        output_dir=output_base / "train",
        base_sentences=base_sentences,
        errors_per_sentence=args.errors_per_sentence
    )

    # Generate validation set
    print("\n" + "="*60)
    print("GENERATING VALIDATION SET")
    print("="*60)
    generate_dataset(
        num_samples=args.num_val,
        output_dir=output_base / "val",
        base_sentences=base_sentences,
        errors_per_sentence=args.errors_per_sentence
    )

    # Generate test set
    print("\n" + "="*60)
    print("GENERATING TEST SET")
    print("="*60)
    generate_dataset(
        num_samples=args.num_test,
        output_dir=output_base / "test",
        base_sentences=base_sentences,
        errors_per_sentence=args.errors_per_sentence
    )

    print("\n" + "="*60)
    print("[OK] DATASET GENERATION COMPLETE!")
    print("="*60)
    print(f"\nDataset saved to: {output_base}")
    print(f"  - train/: {args.num_train} samples")
    print(f"  - val/:   {args.num_val} samples")
    print(f"  - test/:  {args.num_test} samples")
