#-*- coding: utf-8 -*-
"""Figure 5: gate calibration and the IEMOCAP transfer probe."""
import numpy as np
from pubstyle import (plt, mm, save_pub, panel_tag, dump_source,
                      C_CGMA, C_NAIVE, C_WA)

# ---- (a) 门控校准曲线 w_v vs r=1-p, 3种子 {0,1,42} ----
# CGMA(全谱系增强, no_proxy) vs 去帧级增强(no_frameaug); log/wr0712/wr_*.log (2026-07-13)
RVEC = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]          # 剩余有效视觉帧比例 (=1-p)
WR_CGMA = [[0.012, 0.226, 0.356, 0.597, 0.818, 0.993],   # seed0
           [0.179, 0.311, 0.340, 0.515, 0.855, 0.989],   # seed1
           [0.036, 0.159, 0.265, 0.502, 0.709, 0.981]]   # seed42
WR_NFA  = [[0.004, 0.955, 0.977, 0.996, 0.994, 0.997],   # seed0
           [0.056, 0.972, 0.989, 0.995, 0.997, 0.997],   # seed1
           [0.006, 0.970, 0.991, 0.992, 0.994, 0.997]]   # seed42

# ---- (b) IEMOCAP wF1(%), 5种子 {42,1,2,3,123} ----
ICOND = ["Intact", "Drop-Text", "Drop-Audio", "Drop-Visual"]
I_NAIVE = {"Intact": [61.53, 61.87, 61.92, 60.22, 59.08],
           "Drop-Text": [37.69, 45.71, 35.30, 35.53, 26.89],
           "Drop-Audio": [58.09, 57.99, 57.79, 58.01, 57.26],
           "Drop-Visual": [59.26, 58.98, 59.72, 58.95, 58.11]}
I_OURS = {"Intact": [60.53, 61.62, 61.45, 61.35, 60.19],
          "Drop-Text": [50.66, 48.66, 51.52, 52.88, 42.97],
          "Drop-Audio": [56.17, 57.92, 57.43, 57.84, 57.88],
          "Drop-Visual": [58.07, 59.29, 60.18, 59.52, 57.44]}

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=mm(183, 60),
                                 gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.35]))

# ---- (a) 门控 w_v 对真实剩余比例 r 的校准曲线 ----
R = np.array(RVEC)
ax_a.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=0.8, color="0.6", zorder=1,
          label=r"ideal $w=r$")                                       # 对角线参考
for rows, c, lab, ls in [(WR_CGMA, C_CGMA, "CGMA (full-spectrum)", "-"),
                         (WR_NFA, C_WA, "w/o frame-level aug.", "--")]:
    a = np.array(rows); m = a.mean(0); s = a.std(0, ddof=1)
    ax_a.plot(R, m, ls, color=c, lw=1.5, marker="o", ms=3, label=lab, zorder=3)
    ax_a.fill_between(R, m - s, m + s, color=c, alpha=0.15, lw=0)
ax_a.set_xlabel("Remaining valid-frame ratio $r=1-p$")
ax_a.set_ylabel("Visual gate $w^v$")
ax_a.set_xlim(-0.03, 1.03); ax_a.set_ylim(-0.03, 1.12)
ax_a.legend(loc="lower right", fontsize=6.0, handlelength=1.6, borderpad=0.3,
            labelspacing=0.3, frameon=False)
panel_tag(ax_a, "a", dy=1.18)

# ---- (b) ----
x = np.arange(len(ICOND)); w = 0.34
jit = np.linspace(-0.06, 0.06, 5)
for d, off, c, lab in [(I_NAIVE, -w / 2 - 0.01, C_NAIVE, "Naive fusion"),
                       (I_OURS, w / 2 + 0.01, C_CGMA, "CGMA (ours)")]:
    means = [np.mean(d[k]) for k in ICOND]
    stds = [np.std(d[k], ddof=1) for k in ICOND]
    ax_b.bar(x + off, means, width=w, color=c, alpha=0.85, label=lab,
             yerr=stds, error_kw=dict(lw=0.8, capsize=2, capthick=0.8, ecolor="0.25"))
    for xi, k in zip(x + off, ICOND):
        ax_b.scatter(np.full(5, xi) + jit, d[k], s=5, zorder=3,
                     facecolor="white", edgecolor="0.25", linewidth=0.45)
d = np.mean(I_OURS["Drop-Text"]) - np.mean(I_NAIVE["Drop-Text"])
ax_b.annotate(f"+{d:.1f}", xy=(1, 54), fontsize=7, color=C_CGMA,
              fontweight="bold", ha="center")
ax_b.set_xticks(x); ax_b.set_xticklabels(ICOND)
ax_b.set_ylabel("Weighted F1 (%)")
ax_b.set_ylim(0, 72)
ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2,
            columnspacing=1.2, handlelength=1.4)                      # 移到轴外, 不压柱
panel_tag(ax_b, "b", dx=-0.10, dy=1.18)

save_pub(fig, "fig5_gate_iemocap")

# ---- source data ----
rows = []
for label, mat in [("CGMA", WR_CGMA), ("no_frameaug", WR_NFA)]:
    for sd, row in zip([0, 1, 42], mat):
        for r, wv in zip(RVEC, row):
            rows.append([label, sd, r, wv])
dump_source("fig5a_gate_calibration.csv", ["variant", "seed", "r", "w_video"], rows)  # wr0712 实测
rows = []
for mname, d in [("naive", I_NAIVE), ("cgma", I_OURS)]:
    for k in ICOND:
        for sd, v in zip([42, 1, 2, 3, 123], d[k]):
            rows.append([mname, k, sd, v])
dump_source("fig5b_iemocap.csv", ["method", "condition", "seed", "wf1"], rows)
