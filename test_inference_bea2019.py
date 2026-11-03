"""Test inference with the trained FLAN-T5-Large model on BEA 2019."""

import time
import torch
from src.models.t5_gec import T5GEC

print("=" * 60)
print("PHASE 4: INFERENCE TEST - FLAN-T5-LARGE (BEA 2019)")
print("=" * 60)

# Load trained model with optimizations
print("\nLoading trained model...")
print("  - Flash Attention (SDPA): Enabled")
print("  - Precision: bfloat16")

start = time.time()
model = T5GEC.from_pretrained(
    model_name_or_path="checkpoints/flan_t5_large_bea2019/llama_gec_lora",
    use_lora=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
load_time = time.time() - start
print(f"  Model loaded in {load_time:.1f}s")

# Test sentences with various error types
test_sentences = [
    # Subject-verb agreement
    "She go to school every day.",
    "They was playing football in the park.",
    "He don't like pizza.",
    # Tense errors
    "Yesterday, I go to the store and buy some groceries.",
    "She has went to the library last week.",
    # Article errors
    "I saw a elephant at zoo yesterday.",
    "She is best student in the class.",
    # Plural errors
    "I have three book on my desk.",
    "The childs were playing outside.",
    # Preposition errors
    "She is good in mathematics.",
    "I arrived to the airport on time.",
    # Complex errors (multiple)
    "Me and him goes to same school since five year.",
    "Their going to there house for they're party.",
    # Already correct (should not change)
    "The weather is beautiful today.",
    "She graduated from university last year.",
]

print(f"\n{'='*60}")
print(f"TESTING {len(test_sentences)} SENTENCES")
print(f"{'='*60}")

total_time = 0
for i, sentence in enumerate(test_sentences):
    start = time.time()
    result = model.correct_text(sentence, max_new_tokens=128, num_beams=4)
    elapsed = time.time() - start
    total_time += elapsed

    changed = sentence.strip() != result.corrected_text.strip()
    status = "CORRECTED" if changed else "UNCHANGED"

    print(f"\n[{i+1}] {status} ({elapsed*1000:.0f}ms)")
    print(f"  Input:  {sentence}")
    print(f"  Output: {result.corrected_text}")

avg_ms = (total_time / len(test_sentences)) * 1000
print(f"\n{'='*60}")
print(f"INFERENCE STATS")
print(f"{'='*60}")
print(f"  Total sentences: {len(test_sentences)}")
print(f"  Total time: {total_time:.2f}s")
print(f"  Average latency: {avg_ms:.0f}ms per sentence")
print(f"  Device: {model.device}")
print(f"{'='*60}")
