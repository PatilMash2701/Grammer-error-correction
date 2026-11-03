"""Test inference with trained T5 model."""

from src.models.t5_gec import T5GEC

print("=" * 60)
print("TESTING TRAINED T5 MODEL INFERENCE")
print("=" * 60)

# Load trained model
print("\nLoading trained model...")
model = T5GEC.from_pretrained(
    model_name_or_path="checkpoints/test_run_t5/llama_gec_lora",
    use_lora=True,
    device="cpu"
)

# Test with some examples
test_sentences = [
    "She go to school yesterday.",
    "I has three apple.",
    "They was playing football.",
    "He don't likes pizza.",
    "The cat are sleeping."
]

print("\n" + "=" * 60)
print("CORRECTIONS:")
print("=" * 60)

for sentence in test_sentences:
    print(f"\nOriginal: {sentence}")
    result = model.correct_text(sentence, max_new_tokens=64, num_beams=2)
    print(f"Corrected: {result.corrected_text}")

print("\n" + "=" * 60)
print("INFERENCE TEST COMPLETE!")
print("=" * 60)
