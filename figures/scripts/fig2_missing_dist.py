#-*- coding: utf-8 -*-
"""
图2 · 动机: "真实缺失长什么样 + 有多普遍"(双层证据, schematic-led 布局)。
(a) 缺失条码图: 3 个真实视频(低/中/高缺失)的逐帧人脸在场时间轴 —— 展示真实缺失的
    间歇性/成段形态(原始数据可视化)。数据 source_data/lmvd_barcode.csv(2000 bins/视频)。
(b) 全库逐视频缺失率分布直方图(n=1823)。数据 source_data/lmvd_missing_per_video.csv。
对应正文: 5.2 节(图 2)。
"""
import csv, os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pubstyle import (plt, mm, save_pub, panel_tag, C_CGMA, C_DROP, SRCDIR)

# ---- (a) 条码数据 ----
bars = {}                                   # vid -> (miss_rate, frac array)
with open(os.path.join(SRCDIR, "lmvd_barcode.csv")) as f:
    for row in csv.DictReader(f):
        v = row["video"]
        bars.setdefault(v, [float(row["miss_rate"]), []])
        bars[v][1].append(float(row["miss_frac"]))
ORDER = ["1519", "0754", "1011"]            # 低/中/高
TAGS = ["low", "moderate", "severe"]

# ---- (b) 分布数据 ----
rates = []
with open(os.path.join(SRCDIR, "lmvd_missing_per_video.csv")) as f:
    for row in csv.DictReader(f):
        rates.append(float(row["miss_rate"]) * 100)
rates = np.asarray(rates)
med, mean = np.median(rates), rates.mean()
gt10 = (rates > 10).mean() * 100
gt50 = (rates > 50).mean() * 100

# ---- 画布: 上条码(hero) 下直方图 ----
fig = plt.figure(figsize=mm(89, 92))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.5)

# (a) 条码: 白=在场, 红系=缺失占比
ax_a = fig.add_subplot(gs[0])
cmap = LinearSegmentedColormap.from_list("miss", ["#FFFFFF", C_DROP])
M = np.array([bars[v][1] for v in ORDER])          # 3 × 2000
ax_a.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
for i, (v, tag) in enumerate(zip(ORDER, TAGS)):
    ax_a.text(1.01, i, f"{bars[v][0]*100:.0f}%", transform=ax_a.get_yaxis_transform(),
              va="center", fontsize=6.5, color=C_DROP if i == 2 else "0.25")
ax_a.set_yticks(range(len(ORDER)))
ax_a.set_yticklabels([f"video #{v}\n({t})" for v, t in zip(ORDER, TAGS)], fontsize=6)
ax_a.set_xticks([0, 500, 1000, 1500, 2000])
ax_a.set_xticklabels(["0", "25", "50", "75", "100"])
ax_a.set_xlabel("Video timeline (%)")
for s in ax_a.spines.values(): s.set_visible(True); s.set_linewidth(0.6)
ax_a.tick_params(length=2)
ax_a.set_title("Face-missing segments in real vlogs (red = face lost)",
               fontsize=6.8, loc="left", pad=3)
panel_tag(ax_a, "a", dx=-0.30, dy=1.30)

# (b) 直方图
ax_b = fig.add_subplot(gs[1])
bins = np.arange(0, 102.5, 2.5)
ax_b.hist(rates, bins=bins, color=C_CGMA, alpha=0.75, edgecolor="white", linewidth=0.3)
ax_b.axvline(med, color="0.25", lw=0.9, ls="--")
ax_b.axvline(mean, color=C_DROP, lw=0.9, ls="--")
ymax = ax_b.get_ylim()[1]
ax_b.text(med + 1.5, ymax * 0.92, f"median {med:.1f}%", fontsize=6.2, color="0.25")
ax_b.text(mean + 1.5, ymax * 0.74, f"mean {mean:.1f}%", fontsize=6.2, color=C_DROP)
ax_b.annotate(f"{gt10:.0f}% of videos > 10% missing\n({gt50:.0f}% exceed 50%)",
              xy=(34, ymax * 0.30), fontsize=6.2, color="0.2",
              bbox=dict(boxstyle="round,pad=0.3", fc="#F4F4F4", ec="0.8", lw=0.5))
ax_b.set_xlabel("Per-video face-missing rate (%)")
ax_b.set_ylabel("Number of videos")
ax_b.set_xlim(0, 100)
panel_tag(ax_b, "b", dx=-0.30)

save_pub(fig, "fig2_lmvd_missing_dist")
print(f"n={len(rates)}  median={med:.1f}%  mean={mean:.1f}%  >10%: {gt10:.0f}%  >50%: {gt50:.0f}%")
