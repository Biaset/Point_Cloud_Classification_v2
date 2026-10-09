"""
Utilities for loading and preparing point-cloud CSV files.
"""

from pathlib import Path

import pandas as pd

from config import CLASS_REMAP, CSV_SEPARATOR, DROP_COLUMNS


def load_point_cloud_files(file_specs, root="."):
    """Load multiple point-cloud CSVs and attach their cluster IDs."""
    frames = []

    for spec in file_specs:
        path = Path(root) / spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"Point-cloud file not found: {path}")

        df = pd.read_csv(path, sep=CSV_SEPARATOR)

        if "//X" in df.columns and "X" not in df.columns:
            df = df.rename(columns={"//X": "X"})

        df["Cluster"] = spec["cluster"]

        for column in DROP_COLUMNS:
            if column in df.columns:
                df = df.drop(columns=column)

        frames.append(df)

    if not frames:
        raise ValueError("No point-cloud files were provided.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna().reset_index(drop=True)

    if "Classification" in combined.columns:
        combined["Classification"] = combined["Classification"].replace(CLASS_REMAP)

    return combined


def split_by_cluster(df, train_clusters, test_clusters):
    """Split a prepared DataFrame using whole point-cloud clusters."""
    train_mask = df["Cluster"].isin(train_clusters).to_numpy()
    test_mask = df["Cluster"].isin(test_clusters).to_numpy()

    if not train_mask.any():
        raise ValueError("No training points found for the selected clusters.")
    if not test_mask.any():
        raise ValueError("No test points found for the selected clusters.")

    return train_mask, test_mask
