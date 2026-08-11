# GitHub and Zenodo release checklist

- [ ] Confirm the MIT License is the intended code license.
- [ ] Confirm that no dataset, model checkpoint, raw experiment log, per-sample
      probability, or representation dump is staged for upload.
- [ ] Run `python scripts/verify_release.py`.
- [ ] Run `python scripts/aggregate_results.py` and inspect the two generated CSV files.
- [ ] Recreate the environment on a clean machine and run a one-seed smoke test.
- [ ] Reproduce the full five-seed matrix before claiming exact reproducibility in the
      public release notes.
- [ ] Create a versioned GitHub release, then archive the same commit on Zenodo.

