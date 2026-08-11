#!/usr/bin/env python3
"""Validate the expected LMVD release layout without modifying the dataset."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--max_files", type=int, default=0, help="0 checks every sample; positive values check only that many")
    args = parser.parse_args()

    root = args.data_dir.expanduser().resolve()
    labels_path = root / "lmvd_labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)

    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = [row for row in reader if len(row) >= 3]

    if not rows:
        raise ValueError("lmvd_labels.csv contains no usable rows")
    invalid_folds = sorted({row[2] for row in rows} - {"train", "valid", "test"})
    if invalid_folds:
        raise ValueError(f"Unexpected fold values: {invalid_folds}")

    selected = rows if args.max_files <= 0 else rows[: args.max_files]
    missing = []
    bad_shapes = []
    for sample_id, _, _ in selected:
        visual_path = root / "visual" / f"{sample_id}_visual.npy"
        audio_path = root / "audio" / f"{sample_id}.npy"
        if not visual_path.is_file() or not audio_path.is_file():
            missing.append(sample_id)
            continue
        visual = np.load(visual_path, mmap_mode="r")
        audio = np.load(audio_path, mmap_mode="r")
        if visual.ndim != 2 or visual.shape[1] != 136 or audio.ndim != 2 or audio.shape[1] != 128:
            bad_shapes.append((sample_id, visual.shape, audio.shape))

    if missing:
        raise FileNotFoundError(f"Missing visual/audio arrays for {len(missing)} samples; first: {missing[:5]}")
    if bad_shapes:
        raise ValueError(f"Unexpected feature shapes; first: {bad_shapes[:5]}")

    folds = Counter(row[2] for row in rows)
    labels = Counter(row[1] for row in rows)
    print(f"Header: {header}")
    print(f"Rows: {len(rows)} | folds: {dict(folds)} | labels: {dict(labels)}")
    print(f"Checked feature arrays: {len(selected)} | expected dimensions: visual=136, audio=128")


if __name__ == "__main__":
    main()
