# Figure archive

`rendered/` contains the current manuscript PNG and SVG files. Data-driven figures can
be regenerated from `scripts/` and `source_data/`; Figure 1 is manually composed and is
therefore distributed only as final artwork.

| Figure | Script | Primary source files |
|---|---|---|
| 2 | `fig2_missing_dist.py` | `lmvd_missing_per_video.csv`, `lmvd_barcode.csv` |
| 3 | `fig3_robustness.py` | `fig3a_ratio_curves.csv`, `fig3b_frame_curve.csv`, `fig3c_main_bars.csv` |
| 4 | `fig4_missing_spectrum.py` | `fig4_grid2d_mean.csv` |
| 5(a) | `fig5_gate_iemocap.py` | `fig5a_gate_calibration.csv` |
| 5(b) | `fig5_gate_iemocap.py` | `fig5b_iemocap.csv` |

Raw grid logs, per-sample probabilities, and representation dumps are not included in
the public package. The plotted aggregate values are retained as CSV files.
