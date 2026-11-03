"""Create small synthetic dataset for testing the training pipeline."""

import pandas as pd
from pathlib import Path

# Sample error patterns
incorrect_samples = [
    ("She go to school yesterday.", "She went to school yesterday."),
    ("He don't like apples.", "He doesn't like apples."),
    ("They was playing outside.", "They were playing outside."),
    ("I has a cat and a dog.", "I have a cat and a dog."),
    ("She have been working here.", "She has been working here."),
    ("We was surprised by the news.", "We were surprised by the news."),
    ("He do his homework every day.", "He does his homework every day."),
    ("The cat walk on the street.", "The cat walks on the street."),
    ("She study English yesterday.", "She studied English yesterday."),
    ("They is coming to the party.", "They are coming to the party."),
    ("I seen him at the park.", "I saw him at the park."),
    ("He go swimming every week.", "He goes swimming every week."),
    ("She have three brother.", "She has three brothers."),
    ("We was watching a movie.", "We were watching a movie."),
    ("The dog bark loudly.", "The dog barks loudly."),
    ("She write a letter yesterday.", "She wrote a letter yesterday."),
    ("They doesn't understand it.", "They don't understand it."),
    ("He walk to school every day.", "He walks to school every day."),
    ("I has to go now.", "I have to go now."),
    ("She don't know the answer.", "She doesn't know the answer."),
    ("We goes to the beach.", "We go to the beach."),
    ("He have a new car.", "He has a new car."),
    ("They was very happy.", "They were very happy."),
    ("She like ice cream.", "She likes ice cream."),
    ("I doesn't want to go.", "I don't want to go."),
    ("He run very fast.", "He runs very fast."),
    ("They has many friends.", "They have many friends."),
    ("She were at the store.", "She was at the store."),
    ("I goes there often.", "I go there often."),
    ("He don't care about it.", "He doesn't care about it"),
    ("She have to finish this.", "She has to finish this."),
    ("We was there last week.", "We were there last week."),
    ("They is nice people.", "They are nice people."),
    ("He like playing football.", "He likes playing football."),
    ("I was running when he call.", "I was running when he called."),
    ("She go shopping yesterday.", "She went shopping yesterday."),
    ("They doesn't have time.", "They don't have time."),
    ("He were very tired.", "He was very tired."),
    ("I has been there before.", "I have been there before."),
    ("She don't believe me.", "She doesn't believe me."),
    ("We goes to school together.", "We go to school together."),
    ("He have many books.", "He has many books."),
    ("They was playing games.", "They were playing games."),
    ("She study hard for test.", "She studied hard for the test."),
    ("I doesn't like vegetables.", "I don't like vegetables."),
    ("He walk his dog daily.", "He walks his dog daily."),
    ("They is my best friends.", "They are my best friends."),
    ("She write in her diary.", "She writes in her diary."),
    ("I has three cats.", "I have three cats."),
    ("He don't want to leave.", "He doesn't want to leave."),
]

# Create more samples by repeating and slightly modifying
train_samples = []
for _ in range(4):  # Create 200 samples total
    for src, tgt in incorrect_samples:
        train_samples.append({"source": src, "target": tgt})

# Create validation set (same samples for now)
val_samples = incorrect_samples[:15]  # 15 validation samples

# Save to CSV
output_dir = Path("./data/raw/bea2019_test")
output_dir.mkdir(parents=True, exist_ok=True)

train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples, columns=["source", "target"])

train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "dev.csv", index=False)

print(f"Created test dataset:")
print(f"  Train: {len(train_df)} samples -> {output_dir / 'train.csv'}")
print(f"  Val: {len(val_df)} samples -> {output_dir / 'dev.csv'}")
