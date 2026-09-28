from pathlib import Path
import pandas as pd
import numpy as np
from typing import cast

from . import INITIAL_METADATA_FILE, OUTPUT_DIR_INITIAL_CSV_SPLITS


def filter_and_split_metadata(
    input_csv: Path = Path(INITIAL_METADATA_FILE),
    output_dir: Path = Path(OUTPUT_DIR_INITIAL_CSV_SPLITS),
    min_recordings_per_species: int = 3,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    # 1. Quality Filter: Only 'A' and 'B' recordings
    df = df[df["quality"].isin(["a", "b", "A", "B", "c", "C"])].copy()

    # 2. Duration Filter: 3.0s to 300.0s
    lengths = df["length_seconds"].astype(str)
    normalized = np.where(lengths.str.count(":") == 1, "00:" + lengths, lengths)
    df["length_seconds"] = pd.to_timedelta(normalized).total_seconds()
    df = df[(df["length_seconds"] >= 3.0) & (df["length_seconds"] <= 300.0)].copy()

    # 3. Minimum Recordings per species
    species_counts = df["scientific_name"].value_counts()
    valid_species = species_counts[species_counts >= min_recordings_per_species].index
    df = df[df["scientific_name"].isin(valid_species.tolist())].copy()
    print(species_counts)
    # 4. Train, Val, Test split
    train_list, val_list, test_list = [], [], []

    for species, grp in df.groupby("scientific_name"):
        shuffled = grp.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(shuffled)

        if n == 3:
            train_list.append(shuffled.iloc[[0]])
            val_list.append(shuffled.iloc[[1]])
            test_list.append(shuffled.iloc[[2]])
        else:
            n_test = max(1, int(0.10 * n))
            n_val = max(1, int(0.10 * n))
            n_train = n - (n_val + n_test)

            train_list.append(shuffled.iloc[:n_train])
            test_list.append(shuffled.iloc[n_train : n_train + n_test])
            val_list.append(shuffled.iloc[n_train + n_test :])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(
        f"✓ Splits created: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}"
    )


if __name__ == "__main__":
    filter_and_split_metadata()
