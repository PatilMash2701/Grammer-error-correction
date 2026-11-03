"""Download BEA 2019 dataset from Hugging Face.

Dataset: juancavallotti/bea-19-corruption (81k rows)
Prepares it in the format expected by our training pipeline.
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path

print("=" * 60)
print("DOWNLOADING BEA 2019 FROM HUGGING FACE")
print("=" * 60)

# Download dataset
print("\nDownloading dataset (this may take a few minutes)...")
dataset = load_dataset("juancavallotti/bea-19-corruption")

print(f"\nDownloaded successfully!")
print(f"  Splits available: {list(dataset.keys())}")

# Display dataset info
for split_name, split_data in dataset.items():
    print(f"  {split_name}: {len(split_data)} samples")
    if len(split_data) > 0:
        print(f"    Columns: {split_data.column_names}")
        print(f"    Example: {split_data[0]}")

# Prepare output directory
output_dir = Path("./data/bea2019_hf")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nSaving to CSV format in: {output_dir}")

# Convert each split to CSV
for split_name, split_data in dataset.items():
    # Determine column names (common variations)
    source_col = None
    target_col = None

    for col in split_data.column_names:
        col_lower = col.lower()
        if 'source' in col_lower or 'incorrect' in col_lower or 'error' in col_lower or 'input' in col_lower:
            source_col = col
        elif 'target' in col_lower or 'correct' in col_lower or 'output' in col_lower or 'reference' in col_lower:
            target_col = col

    if not source_col or not target_col:
        # Try to infer from first two columns
        cols = split_data.column_names
        if len(cols) >= 2:
            source_col = cols[0]
            target_col = cols[1]
            print(f"  Assuming: source='{source_col}', target='{target_col}'")

    # Convert to DataFrame with correct mapping
    # 'broken' = erroneous input (source)
    # 'sentence' = correct output (target)
    df = pd.DataFrame({
        "source": split_data["broken"],  # Erroneous sentence
        "target": split_data["sentence"]  # Correct sentence
    })

    # Save to CSV
    csv_name = "dev.csv" if split_name == "validation" else f"{split_name}.csv"
    csv_path = output_dir / csv_name
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"  OK {split_name} -> {csv_name} ({len(df)} samples)")

print("\n" + "=" * 60)
print("BEA 2019 DATASET READY!")
print("=" * 60)
print(f"Location: {output_dir.absolute()}")
print("\nYou can now train with:")
print(f'  dataset_path="{output_dir}"')
print("=" * 60)
