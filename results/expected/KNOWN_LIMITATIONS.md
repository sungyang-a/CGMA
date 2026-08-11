# Known limitations and provenance notes

1. **Data are not redistributed.** LMVD and IEMOCAP must be obtained under their
   original access terms. This repository expects the feature layouts described in
   `README.md`.
2. **Figure 4 uses a historical comparison protocol.** Its released aggregate grid
   values were produced with CGMA at 45 epochs / patience 12 and naive AdaFuse at 40
   epochs / patience 10. The main tables and one-dimensional robustness curves use the
   unified 45/12 protocol. `src/joint_grid.py` preserves the historical Figure 4
   protocol, while the public package contains only the plotted aggregate CSV.
3. **GPU runs may not be bitwise deterministic.** Seeds, splits, and aggregation rules
   are fixed, but CUDA/cuDNN kernels and hardware can still cause small numerical
   changes.
4. **Historical environment warning.** The original server combined SciPy 1.7.3 with
   NumPy 1.24.4, which emits a supported-version warning. `environment.yml` supplies a
   compatible reconstruction environment; it has not been used to rerun the paper's
   full experiment matrix.
5. **Inference export.** `scripts/export_inference.py` removes training-only proxy
   modules from a saved final-CGMA checkpoint and checks output equivalence before
   writing the compact checkpoint. The export utility is included but no public
   checkpoint or dataset is bundled here.
