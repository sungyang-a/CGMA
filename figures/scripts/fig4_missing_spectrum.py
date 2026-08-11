#-*- coding: utf-8 -*-
"""Figure 4: joint visual/audio frame-missingness F1 surfaces."""
import csv
import os
import numpy as np
from pubstyle import plt, mm, save_pub, SRCDIR

GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def read_mean_grid(path):
    grids = {"ours": np.zeros((len(GRID), len(GRID))),
             "adafuse": np.zeros((len(GRID), len(GRID)))}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            pv, pa = float(row["pv"]), float(row["pa"])
            grids[method][GRID.index(pv), GRID.index(pa)] = float(row["f1_mean3seed"])
    return grids


MEAN_GRIDS = read_mean_grid(os.path.join(SRCDIR, "fig4_grid2d_mean.csv"))
H_OURS = MEAN_GRIDS["ours"]
H_ADA = MEAN_GRIDS["adafuse"]
vmin = min(H_OURS.min(), H_ADA.min()); vmax = max(H_OURS.max(), H_ADA.max())

fig = plt.figure(figsize=mm(122, 60))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.22)
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1]); cax = fig.add_subplot(gs[2])

def heat(ax, H, title):
    im = ax.imshow(H, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(GRID))); ax.set_xticklabels([f"{int(g*100)}" for g in GRID], fontsize=5.8)
    ax.set_yticks(range(len(GRID))); ax.set_yticklabels([f"{int(g*100)}" for g in GRID], fontsize=5.8)
    ax.set_xlabel("Audio frame-drop $p_a$ (%)", fontsize=6.4)
    ax.set_title(title, fontsize=7, pad=3)
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            v = H[i, j]
            lum = (v - vmin) / (vmax - vmin + 1e-9)
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.2,
                    color="black" if lum > 0.55 else "white")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=1.5)
    return im

im = heat(ax1, H_OURS, "CGMA (ours)")
heat(ax2, H_ADA, "Naive fusion")
ax1.set_ylabel("Visual frame-drop $p_v$ (%)", fontsize=6.4)
ax2.set_yticklabels([])
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=5.8)
cax.set_title("F1 (%)", fontsize=6.2, pad=4)

save_pub(fig, "fig4_missing_spectrum")
