"""Convert synthetic dataset format to CSV format expected by training pipeline."""

import pandas as pd
from pathlib import Path

def convert_synthetic_to_csv(input_dir: Path, output_dir: Path, split_name: str):
    """Convert source.txt + target.txt to CSV format.

    Args:
        input_dir: Directory containing source.txt and target.txt
        output_dir: Directory to save CSV file
        split_name: Name of split (train, dev, test)
    """
    source_file = input_dir / "source.txt"
    target_file = input_dir / "target.txt"

    # Read parallel files
    with open(source_file, "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f if line.strip()]

    with open(target_file, "r", encoding="utf-8") as f:
        targets = [line.strip() for line in f if line.strip()]

    # Ensure same length
    assert len(sources) == len(targets), f"Mismatch: {len(sources)} sources vs {len(targets)} targets"

    # Create DataFrame
    df = pd.DataFrame({
        "source": sources,
        "target": targets
    })

    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_name = "dev.csv" if split_name == "val" else f"{split_name}.csv"
    output_path = output_dir / csv_name
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[OK] Converted {split_name}: {len(df)} samples -> {output_path}")
    return output_path

if __name__ == "__main__":
    synthetic_base = Path("data/synthetic_gec")
    output_base = Path("data/synthetic_gec_csv")

    print("="*60)
    print("CONVERTING SYNTHETIC DATASET TO CSV FORMAT")
    print("="*60)

    # Convert all splits
    convert_synthetic_to_csv(synthetic_base / "train", output_base, "train")
    convert_synthetic_to_csv(synthetic_base / "val", output_base, "val")
    convert_synthetic_to_csv(synthetic_base / "test", output_base, "test")

    print("\n" + "="*60)
    print("[OK] CONVERSION COMPLETE!")
    print("="*60)
    print(f"Dataset ready at: {output_base}")
