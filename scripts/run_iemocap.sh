#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <IEMOCAP_features.pkl> [OUTPUT_DIR]" >&2
  exit 2
fi

DATA_PATH="$1"
OUTPUT_DIR="${2:-outputs/iemocap}"
SEEDS=(42 1 2 3 123)
mkdir -p "$OUTPUT_DIR"

for method in naive ours; do
  for seed in "${SEEDS[@]}"; do
    python src/cgma_iemocap.py \
      --data "$DATA_PATH" --ablate "$method" --epochs 60 --patience 12 --seed "$seed" \
      2>&1 | tee "$OUTPUT_DIR/${method}_seed${seed}.log"
  done
done

python scripts/aggregate_results.py --log_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR/aggregated"
