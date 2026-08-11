#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <LMVD_DIR> [OUTPUT_DIR]" >&2
  exit 2
fi

DATA_DIR="$1"
OUTPUT_DIR="${2:-outputs/lmvd}"
SEEDS=(0 1 2 42 123)
mkdir -p "$OUTPUT_DIR"

for seed in "${SEEDS[@]}"; do
  python src/cgma.py \
    --data_dir "$DATA_DIR" --ablate no_proxy --epochs 45 --patience 12 \
    --seed "$seed" --output_dir "$OUTPUT_DIR" --save_checkpoint \
    2>&1 | tee "$OUTPUT_DIR/cgma_seed${seed}.log"

  python src/cgma.py \
    --data_dir "$DATA_DIR" --ablate no_gate --epochs 45 --patience 12 \
    --seed "$seed" --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/identity_gate_seed${seed}.log"

  python src/naive_fusion.py \
    --data_dir "$DATA_DIR" --drop_p 0 --epochs 45 --patience 12 --seed "$seed" \
    2>&1 | tee "$OUTPUT_DIR/naive_seed${seed}.log"

  python src/naive_aug.py \
    --data_dir "$DATA_DIR" --epochs 45 --patience 12 --seed "$seed" \
    2>&1 | tee "$OUTPUT_DIR/naive_aug_seed${seed}.log"
done

python scripts/aggregate_results.py --log_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR/aggregated"
