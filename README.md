# CGMA: Completeness Gating with Missing-Aware Augmentation

Reference implementation and result archive for *Lightweight Multimodal Depression
Detection under Real-World Frame-Level Missingness: Completeness-Aware Gating with
Full-Spectrum Augmentation*.

CGMA is a two-stream visual-acoustic model for video-level depression detection. It
combines modality-specific completeness gates with training-time perturbations that
cover intact inputs, whole-modality absence, and partial frame loss. The final
`no_proxy` prediction path performs no feature reconstruction. Training-only proxy
modules are retained in the research model for protocol compatibility and can be
removed from a checkpoint with the included inference exporter.

## Release contents

```text
configs/                 machine-readable experiment settings
figures/
  rendered/              manuscript figures in PNG and SVG
  scripts/               figure-generation scripts
  source_data/           aggregate values used by the manuscript figures
results/
  expected/              manuscript table values
scripts/
  aggregate_results.py   parse logs and compute mean +/- sample standard deviation
  export_inference.py    remove training-only proxy modules from a checkpoint
  run_lmvd.sh            explicit main LMVD run matrix
  run_iemocap.sh         explicit IEMOCAP transfer-probe run matrix
  validate_data.py       validate LMVD paths, folds, and feature dimensions
src/                     models, baselines, ablations, and robustness probes
```

No dataset, model weight, raw experiment log, per-sample probability, representation
dump, or personally identifying video is included.

## Environment

The original experiments used Python 3.8.13 and PyTorch 1.11.0. A compatible pinned
environment is supplied:

```bash
conda env create -f environment.yml
conda activate cgma
```

For pip-only installation:

```bash
python -m pip install -r requirements.txt
```

Install the CUDA-specific PyTorch 1.11.0 wheel appropriate for the target machine when
GPU acceleration is needed. `environment-historical.txt` records the versions observed
on the original server; see `KNOWN_LIMITATIONS.md` before using that combination.

## Data

### LMVD

Prepare LMVD under its original access terms. The code consumes pre-extracted 136-D
OpenFace visual features and 128-D VGGish acoustic features:

```text
<LMVD_DIR>/
  visual/<id>_visual.npy      # shape (T_v, 136)
  audio/<id>.npy              # shape (T_a, 128)
  lmvd_labels.csv             # id, label, fold
```

`fold` must be one of `train`, `valid`, or `test`. The depression label is mapped to
the positive class (`y=1`); the normal label is `y=0`. Invalid frames are represented
by all-zero feature vectors, and validity masks are derived from feature validity.

Validate a prepared directory before training:

```bash
python scripts/validate_data.py /path/to/LMVD
```

### IEMOCAP transfer probe

The transfer code expects a DialogueRNN-format `IEMOCAP_features.pkl` containing
100-D text, 1,582-D acoustic, and 342-D visual utterance features. The paper uses the
public session-level 120/31 dialogue split implemented in `src/cgma_iemocap.py`.

## Main LMVD runs

Main LMVD results use depressed-class F1 over seeds `{0, 1, 2, 42, 123}`. Means are
reported with the sample standard deviation (`ddof=1`). Main-table runs use 45 epochs
and early-stopping patience 12.

Run the four central configurations:

```bash
bash scripts/run_lmvd.sh /path/to/LMVD outputs/lmvd
```

The script executes final CGMA, the clean identity-gate ablation, naive fusion, and
Naive + Aug for all five seeds. Individual commands are shown below.

```bash
# Final CGMA and an inference-exportable checkpoint
python src/cgma.py --data_dir /path/to/LMVD --ablate no_proxy \
  --epochs 45 --patience 12 --seed 42 --save_checkpoint \
  --output_dir outputs/checkpoints

# Clean gate ablation: w_v = w_a = 1; no completeness loss
python src/cgma.py --data_dir /path/to/LMVD --ablate no_gate \
  --epochs 45 --patience 12 --seed 42 --output_dir outputs/no_gate

# Other Table 5 variants
python src/cgma.py --data_dir /path/to/LMVD --ablate full \
  --epochs 45 --patience 12 --seed 42
python src/cgma.py --data_dir /path/to/LMVD --ablate no_comp \
  --epochs 45 --patience 12 --seed 42
python src/ablation_noframeaug.py --data_dir /path/to/LMVD \
  --ablate no_frameaug --epochs 45 --patience 12 --seed 42
```

Mechanism baselines:

```bash
python src/naive_fusion.py --data_dir /path/to/LMVD --drop_p 0 \
  --epochs 45 --patience 12 --seed 42
python src/naive_aug.py --data_dir /path/to/LMVD \
  --epochs 45 --patience 12 --seed 42
python src/baselines.py --data_dir /path/to/LMVD --method zero \
  --epochs 45 --patience 12 --seed 42
python src/baselines.py --data_dir /path/to/LMVD --method token \
  --epochs 45 --patience 12 --seed 42
python src/baselines.py --data_dir /path/to/LMVD --method ae \
  --epochs 45 --patience 12 --seed 42
python src/mmin_core.py --data_dir /path/to/LMVD \
  --epochs 45 --patience 12 --seed 42
```

Robustness and gate probes:

```bash
python src/ratio_curve.py --data_dir /path/to/LMVD --method ours \
  --epochs 45 --patience 12 --seed 42
python src/ratio_curve.py --data_dir /path/to/LMVD --method adafuse \
  --epochs 45 --patience 12 --seed 42
python src/ratio_curve_mmin.py --data_dir /path/to/LMVD \
  --epochs 45 --patience 12 --seed 42
python src/ratio_curve_naive_aug.py --data_dir /path/to/LMVD \
  --epochs 45 --patience 12 --seed 42
python src/gate_endpoints.py --data_dir /path/to/LMVD \
  --ablate no_proxy --epochs 45 --patience 12 --seed 42
python src/gate_calibration.py --data_dir /path/to/LMVD \
  --ablate no_proxy --epochs 45 --patience 12 --seed 42
```

`src/joint_grid.py` is the historical Figure 4 reproduction script. Its protocol is
documented separately because the released naive-grid source log predates the unified
main-table training schedule.

## IEMOCAP transfer probe

IEMOCAP results use weighted F1 over seeds `{42, 1, 2, 3, 123}`:

```bash
bash scripts/run_iemocap.sh /path/to/IEMOCAP_features.pkl outputs/iemocap
```

## Aggregating generated logs

Raw experiment logs are not distributed. Logs generated by new runs can be converted
to tidy CSV and summarized with the sample standard deviation:

```bash
python scripts/aggregate_results.py \
  --log_dir outputs/lmvd \
  --output_dir outputs/lmvd/aggregated
```

The outputs are `per_seed_metrics.csv` and `summary_metrics.csv`. Manuscript-level
aggregate reference values are in `results/expected/paper_tables.csv`.

## Compact inference model

`src/cgma.py` retains proxy modules because several training ablations share the same
research class. They are detached from the final `no_proxy` prediction path but still
occupy parameters in a raw training checkpoint. Export the actual inference graph with:

```bash
python scripts/export_inference.py \
  --checkpoint outputs/checkpoints/cgma_no_proxy_seed42.pt \
  --output outputs/checkpoints/cgma_inference_seed42.pt
```

The exporter loads the training checkpoint, removes proxy weights, and checks logits on
a synthetic batch before saving. `src/cgma_inference.py` contains only the two BiLSTM
encoders, completeness gates, fusion gate, and classifier.

## Figures

Final figures are in `figures/rendered/`. Figure scripts read only files in
`figures/source_data/` and write publication formats through `pubstyle.py`.

| Figure | Script | Source data |
|---|---|---|
| Figure 2 | `fig2_missing_dist.py` | LMVD missingness distributions and barcode data |
| Figure 3 | `fig3_robustness.py` | one-dimensional robustness curves and main bars |
| Figure 4 | `fig4_missing_spectrum.py` | joint visual/audio grid means |
| Figure 5 | `fig5_gate_iemocap.py` | gate calibration and IEMOCAP probe values |

Figure 1 is supplied as final PNG/SVG artwork because it is a manually composed method
diagram rather than a data-driven chart.

## Reproducibility boundaries

Read `KNOWN_LIMITATIONS.md` for the historical Figure 4 protocol, CUDA determinism,
environment provenance, and data-release boundaries. The repository records these
limits explicitly so the main 45/12 results are not conflated with older diagnostic
runs.

## License and citation

Code is released under the MIT License. Please cite the associated paper when using
this implementation; author and publication information is provided by the paper.
