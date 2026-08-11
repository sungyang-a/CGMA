#!/usr/bin/env python3
"""Static pre-release checks; this script never trains or evaluates a model."""

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    "KNOWN_LIMITATIONS.md",
    "environment.yml",
    "src/cgma.py",
    "src/cgma_inference.py",
    "src/joint_grid.py",
    "scripts/aggregate_results.py",
    "scripts/export_inference.py",
    "results/expected/paper_tables.csv",
    "figures/source_data/fig4_grid2d_mean.csv",
    "figures/source_data/fig5a_gate_calibration.csv",
    "figures/source_data/fig5b_iemocap.csv",
]
FORBIDDEN_SUFFIXES = {".pyc", ".pt", ".pth", ".ckpt", ".npz", ".log"}
FORBIDDEN_NAMES = {".DS_Store", "__pycache__"}
PRIVATE_MARKERS = ("/Users/sungyang", "/home/wangbin", "~/depression", "~/lmvd_work")


def main():
    errors = []
    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            errors.append(f"missing required path: {relative}")

    for path in ROOT.rglob("*"):
        if path.name in FORBIDDEN_NAMES or (path.is_file() and path.suffix in FORBIDDEN_SUFFIXES):
            errors.append(f"generated/private artifact should not be released: {path.relative_to(ROOT)}")
        if (path.is_file()
                and path.resolve() != Path(__file__).resolve()):
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for marker in PRIVATE_MARKERS:
                if marker in text:
                    errors.append(f"private path marker {marker!r}: {path.relative_to(ROOT)}")

    if (ROOT / "results/raw_logs").exists():
        errors.append("results/raw_logs must remain outside the public release")

    expected_csv = ROOT / "results/expected/paper_tables.csv"
    if expected_csv.is_file():
        with expected_csv.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != ["table", "method", "condition", "mean", "std", "metric", "scale"]:
            errors.append("unexpected paper_tables.csv header")

    if errors:
        print("Release check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Static release checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
