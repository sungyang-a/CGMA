#!/usr/bin/env python3
"""Parse CGMA experiment logs into long-form and aggregated CSV files.

The parser only reads existing logs. It does not train or evaluate a model.
Reported standard deviations use the sample definition (ddof=1), matching the paper.
"""

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


HEADER_RE = re.compile(r">>>\s*\[([^\]]+)\]\s*(.*)")
SEED_RE = re.compile(r"seed(?:=)?(\d+)", re.IGNORECASE)
PAIR_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)=([+-]?\d+(?:\.\d+)?)")
MEAN_STD_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)=([+-]?\d+(?:\.\d+)?)±([+-]?\d+(?:\.\d+)?)")
AUDIT_RE = re.compile(
    r"\b(full|dropV|dropA|f50)(?:\s+F1/mac/acc)?="
    r"([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)"
)

TEXT_REPLACEMENTS = {
    "完整 wF1=": "intact_wf1=",
    "完整 ": "",
    "丢text=": "drop_text=",
    "丢audio=": "drop_audio=",
    "丢visual=": "drop_visual=",
    "话语级丢text ": "",
    "帧级 ": "",
    "整模态 ": "",
}


def parse_header(raw_header):
    parts = [part.strip() for part in raw_header.split("|")]
    seed = None
    for part in parts:
        match = SEED_RE.fullmatch(part)
        if match:
            seed = int(match.group(1))
            break

    if parts[0] == "audit" and len(parts) >= 3:
        experiment, variant = parts[1], parts[2]
        extras = parts[3:]
    else:
        experiment = parts[0]
        variant = parts[1] if len(parts) > 1 and not SEED_RE.fullmatch(parts[1]) else "default"
        extras = parts[2:]

    context_parts = [part for part in extras if part and not SEED_RE.fullmatch(part)]
    return experiment, variant, seed, ";".join(context_parts)


def normalize_body(body):
    for old, new in TEXT_REPLACEMENTS.items():
        body = body.replace(old, new)
    return body


def context_from_coordinates(body, base_context):
    coords = []
    for key in ("p", "r", "pv", "pa"):
        match = re.search(rf"\b{key}=([+-]?\d+(?:\.\d+)?)", body)
        if match:
            coords.append(f"{key}={match.group(1)}")
    pieces = [piece for piece in (base_context, ";".join(coords)) if piece]
    return ";".join(pieces) or "default"


def parse_log(path, root):
    records = []
    rel = path.relative_to(root)
    series = str(rel.parent)

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            match = HEADER_RE.search(line)
            if not match:
                continue
            experiment, variant, seed, base_context = parse_header(match.group(1))
            body = normalize_body(match.group(2))
            context = context_from_coordinates(body, base_context)

            audit_matches = list(AUDIT_RE.finditer(body))
            for audit in audit_matches:
                condition = audit.group(1)
                for metric, value in zip(("f1", "macro_f1", "accuracy"), audit.groups()[1:]):
                    records.append(
                        [series, str(rel), line_number, experiment, variant, seed, condition, metric, float(value)]
                    )
            if audit_matches:
                continue

            mean_std = {item.group(1): (float(item.group(2)), float(item.group(3))) for item in MEAN_STD_RE.finditer(body)}
            for key, (mean, std) in mean_std.items():
                records.append([series, str(rel), line_number, experiment, variant, seed, context, key, mean])
                records.append([series, str(rel), line_number, experiment, variant, seed, context, f"{key}_within_std", std])

            for item in PAIR_RE.finditer(body):
                key, value = item.group(1), float(item.group(2))
                if key in {"p", "r", "pv", "pa"} or key in mean_std:
                    continue
                records.append([series, str(rel), line_number, experiment, variant, seed, context, key, value])
    return records


def write_long(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "source_log", "line", "experiment", "variant", "seed", "context", "metric", "value"])
        writer.writerows(records)


def write_summary(records, path):
    grouped = defaultdict(list)
    for series, _, _, experiment, variant, seed, context, metric, value in records:
        if seed is None or metric.endswith("_within_std"):
            continue
        grouped[(series, experiment, variant, context, metric)].append((seed, value))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "experiment", "variant", "context", "metric", "n", "seeds", "mean", "sample_std"])
        for key in sorted(grouped):
            values_by_seed = {}
            for seed, value in grouped[key]:
                values_by_seed[seed] = value
            seeds = sorted(values_by_seed)
            values = [values_by_seed[seed] for seed in seeds]
            std = statistics.stdev(values) if len(values) > 1 else math.nan
            writer.writerow([*key, len(values), ";".join(map(str, seeds)), f"{statistics.mean(values):.6f}", f"{std:.6f}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=Path, default=Path("results/raw_logs"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/aggregated"))
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    records = []
    for path in sorted(log_dir.rglob("*.log")):
        if path.stat().st_size:
            records.extend(parse_log(path, log_dir))

    write_long(records, args.output_dir / "per_seed_metrics.csv")
    write_summary(records, args.output_dir / "summary_metrics.csv")
    print(f"Parsed {len(records)} metric records from {log_dir}")


if __name__ == "__main__":
    main()
