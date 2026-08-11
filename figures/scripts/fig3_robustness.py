#-*- coding: utf-8 -*-
"""
图3 · 鲁棒性主图(hero 三面板)。
一句话结论: 缺失谱系上 CGMA 平缓、朴素断崖; 且 CGMA 把种子间失稳彻底消除。
(a) 整模态缺失比例曲线(5种子 mean±std 带)   ← log/curve/{ours,adafuse}_5seed.log
(b) 视觉帧级缺失曲线(5种子 mean±std 带)+消融参考点 ← log/abl/no_proxy_full_s*.log; log/frame/complete_s{0,1,42}
(c) 三档主结果 分组柱+逐种子散点               ← log/naive_baseline/5seed.log; log/naive_aug/5seed.log; log/abl/no_proxy_full_s*
对应正文: 5.3 表2 / 5.4 表3(原"图 2", 定稿编号图 3) / 5.5 表4。
所有数字 = 旧机(RTX 4090/csy) 5种子 {0,1,2,42,123}; (b) 消融参考点为 3 种子(正文已注)。
"""
import numpy as np
from pubstyle import (plt, mm, save_pub, panel_tag, mean_std, dump_source,
                      C_CGMA, C_NAIVE, C_NAUG, C_DROP)

# ================= 数据(带出处) =================
R = [0, 10, 25, 50, 75, 100]                      # 整模态丢弃比例 r(%)
CURVE_OURS = [                                    # log/rerun0707/curve45_ours_s*.log, seed 0/1/2/42/123 (45ep/pat12, 与表1同协议, 2026-07-07)
    [73.44, 72.89, 72.77, 69.30, 70.17, 69.66],
    [75.32, 74.19, 71.46, 71.57, 71.96, 68.68],
    [78.24, 76.46, 75.44, 73.40, 70.48, 70.37],
    [74.22, 75.18, 73.48, 68.63, 67.92, 63.39],
    [72.37, 72.33, 72.10, 69.87, 69.29, 68.80]]
CURVE_ADA = [                                     # log/rerun0707/curve45_adafuse_s*.log
    [75.60, 74.76, 70.77, 67.20, 64.35, 58.36],
    [75.32, 73.77, 71.96, 64.04, 61.10, 52.81],
    [77.44, 72.68, 74.65, 68.97, 66.48, 57.82],
    [76.76, 72.42, 68.41, 62.73, 53.07, 53.25],
    [75.00, 73.61, 69.80, 68.21, 62.69, 64.72]]
CURVE_MMIN = [                                    # log/rerun0708/mmin_ratio_s*.log, seed 0/1/2/42/123 (45ep/pat12, 整模态only训练+想象loss, 2026-07-08)
    [76.17, 75.97, 73.15, 73.13, 74.88, 70.64],
    [74.50, 73.89, 73.43, 71.53, 69.03, 69.93],
    [74.68, 73.74, 74.69, 75.43, 72.99, 69.57],
    [75.66, 75.92, 74.54, 71.82, 67.83, 67.45],
    [75.49, 72.98, 73.74, 69.27, 67.99, 60.06]]

P = [0, 25, 50, 75, 90, 100]                      # 视觉帧级丢弃比例 p(%)
FRAME_CGMA = [                                    # log/abl/no_proxy_full_s{0,1,2,42,123}
    [77.55, 77.08, 76.81, 73.46, 72.56, 68.79],
    [75.91, 76.80, 75.88, 65.64, 66.06, 67.52],
    [72.89, 72.44, 73.05, 73.32, 70.15, 67.85],
    [72.21, 72.41, 73.97, 71.98, 71.20, 68.16],
    [73.63, 74.75, 73.32, 69.45, 66.67, 65.68]]

P5 = [0, 25, 50, 75, 90]                          # (b) 多方法帧级曲线统一到 f90(f100=整模态, 归dropV)
FRAME_BASE = {                                    # f0=各自full; 全部45ep/pat12统一协议(2026-07-09重跑, log/rerun0709 + rerun0708 MMIN); seed顺序 0/1/2/42/123
    "Naive": [[78.05,49.22,26.05,10.20, 8.33],[74.45,49.03,25.35, 8.29, 4.26],
              [74.69,55.91,20.29, 7.33,13.20],[77.30,52.27,30.91, 7.33, 4.28],
              [78.93,50.77,24.53, 7.29,11.34]],
    "Zero-fill": [[76.96,30.36, 6.32, 4.28, 6.32],[76.21,52.36,40.49,30.97,24.66],
                  [75.62,35.09,15.92, 6.28, 8.21],[73.99,58.58,43.77,39.13,36.43],
                  [76.36,40.48,13.00, 5.29, 5.29]],
    "Missing-token": [[78.53,54.55,22.75, 9.23,11.22],[76.53,54.98,42.59,35.34,36.86],
                      [75.76,47.37,37.34,37.13,33.61],[75.45,58.94,42.91,34.71,33.74],
                      [75.98,54.11,39.52,25.89,23.15]],
    "AE": [[74.25,43.24,28.32,16.11,17.59],[73.35,60.59,45.69,17.65,11.22],
           [76.61,63.29,51.41,45.69,39.68],[76.09,70.34,43.62,15.92,16.67],
           [76.49,46.27,24.41,18.72,13.27]],
    "MMIN": [[76.17,61.73,54.90,51.68,54.55],[74.50,54.74,37.70,28.44,24.88],
             [74.68,38.49,31.86,49.06,48.78],[75.66,35.50,14.93,10.26,13.86],
             [75.49,58.36,46.22,34.63,23.64]],
    "Naive+Aug": [[77.31,75.18,74.76,74.94,74.94],[76.09,75.83,73.10,71.73,67.97],  # log/naive_aug/5seed.log 帧级逐档
                  [77.00,75.00,74.61,71.28,64.72],[77.61,76.10,75.97,74.88,70.74],  # 全谱系训练; seed 0/1/2/42/123
                  [77.18,74.79,71.98,60.33,47.97]],
}

COND = ["Intact", "Drop-Video", "Drop-Audio"]     # (c) 三条件 × 三方法, 每方法5种子
BARS = {                                          # 出处见 docstring; 朴素=45ep统一协议(rerun0709)
    "Naive fusion":   {"Intact": [78.05, 74.45, 74.69, 77.30, 78.93],
                       "Drop-Video": [32.13, 4.19, 65.81, 5.00, 42.62],
                       "Drop-Audio": [71.07, 71.96, 72.85, 73.43, 73.02]},
    "Naive + Aug":    {"Intact": [77.31, 76.09, 77.00, 77.61, 77.18],
                       "Drop-Video": [68.25, 57.65, 63.55, 64.76, 67.98],
                       "Drop-Audio": [73.81, 71.24, 72.73, 70.80, 40.34]},
    "CGMA (ours)":    {"Intact": [77.55, 75.91, 72.89, 72.21, 73.63],
                       "Drop-Video": [68.79, 67.52, 67.85, 68.16, 65.68],
                       "Drop-Audio": [72.40, 72.64, 73.68, 72.61, 72.98]}}
MCOLOR = {"Naive fusion": C_NAIVE, "Naive + Aug": C_NAUG, "CGMA (ours)": C_CGMA}

# ================= 画布 =================
fig = plt.figure(figsize=mm(183, 130))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])

# ---- (a) 整模态缺失比例曲线 (三线: CGMA / MMIN 强基线 / 朴素) ----
from pubstyle import C_WA as _C_WA
for rows, c, lab, ls, mk, z, al in [(CURVE_OURS, C_CGMA, "CGMA (ours)", "-", "o", 5, 0.15),
                                    (CURVE_MMIN, _C_WA, "MMIN (ACL'21)", "--", "s", 4, 0.12),
                                    (CURVE_ADA, C_NAIVE, "Naive fusion", ":", "x", 3, 0.13)]:
    m, s = mean_std(rows)
    ax_a.plot(R, m, ls, color=c, lw=1.4, marker=mk, ms=3, label=lab, zorder=z)
    ax_a.fill_between(R, m - s, m + s, color=c, alpha=al, lw=0)
m_o, _ = mean_std(CURVE_OURS); m_a, _ = mean_std(CURVE_ADA)
ax_a.annotate("", xy=(104, m_o[-1]), xytext=(104, m_a[-1]),        # r=100 差距括线
              arrowprops=dict(arrowstyle="<->", color=C_DROP, lw=0.8, shrinkA=0, shrinkB=0))
ax_a.text(106, (m_o[-1] + m_a[-1]) / 2, f"+{m_o[-1]-m_a[-1]:.1f}",
          fontsize=6.6, color=C_DROP, va="center")
ax_a.set_xlabel("Modality-drop ratio $r$ (%)")
ax_a.set_ylabel("F1 (%)")
ax_a.set_xlim(-4, 116)
ax_a.set_ylim(42, 82)
ax_a.legend(loc="lower left")
panel_tag(ax_a, "a")

# ---- (b) 帧级缺失曲线(多方法, 表4 f50 列的全曲线版) ----
from pubstyle import C_WA
m, s = mean_std([r[:5] for r in FRAME_CGMA])
ax_b.plot(P5, m, "-", color=C_CGMA, lw=1.6, marker="o", ms=3, label="CGMA (ours)", zorder=5)
ax_b.fill_between(P5, m - s, m + s, color=C_CGMA, alpha=0.15, lw=0)
ma, sa = mean_std(FRAME_BASE["Naive+Aug"])                                # 第二条全谱系曲线, 与 CGMA 近乎重合
ax_b.plot(P5, ma, "-", color=C_NAUG, lw=1.5, marker="D", ms=2.8, label="Naive + Aug", zorder=5)
ax_b.fill_between(P5, ma - sa, ma + sa, color=C_NAUG, alpha=0.12, lw=0)
mm, ss = mean_std(FRAME_BASE["MMIN"])
ax_b.plot(P5, mm, "--", color=C_WA, lw=1.3, marker="s", ms=2.6, label="MMIN (ACL'21)", zorder=4)
ax_b.fill_between(P5, mm - ss, mm + ss, color=C_WA, alpha=0.12, lw=0)    # 巨大方差带=不稳的证据
GRAY = "0.55"
for name, ls, mk in [("Missing-token", "-.", "^"), ("Zero-fill", "--", "v"),
                     ("AE", ":", "d"), ("Naive", ":", "x")]:
    gm, _ = mean_std(FRAME_BASE[name])
    ax_b.plot(P5, gm, ls, color=GRAY, lw=0.9, marker=mk, ms=2.6, label=name, zorder=3)
ax_b.set_xlabel("Visual frame-drop ratio $p$ (%)")
ax_b.set_ylabel("F1 (%)")
ax_b.set_ylim(0, 84)
ax_b.legend(loc="lower left", fontsize=5.6, ncols=2, columnspacing=0.9, handlelength=1.6)
panel_tag(ax_b, "b")

# ---- (c) 三档主结果: 分组柱 + 种子散点 ----
x = np.arange(len(COND))
w = 0.24
offs = {"Naive fusion": -w - 0.02, "Naive + Aug": 0, "CGMA (ours)": w + 0.02}
jit = np.linspace(-0.055, 0.055, 5)
for mname, per_cond in BARS.items():
    means = [np.mean(per_cond[c]) for c in COND]
    stds = [np.std(per_cond[c], ddof=1) for c in COND]
    xs = x + offs[mname]
    ax_c.bar(xs, means, width=w, color=MCOLOR[mname], alpha=0.85, label=mname,
             yerr=stds, error_kw=dict(lw=0.8, capsize=2, capthick=0.8, ecolor="0.25"))
    for xi, c in zip(xs, COND):                      # 逐种子散点(方差故事的主角)
        ax_c.scatter(np.full(5, xi) + jit, per_cond[c], s=6, zorder=3,
                     facecolor="white", edgecolor="0.25", linewidth=0.5)
ax_c.text(x[1] + offs["Naive fusion"], 72, "std 26.2\nfloor 4.2", fontsize=6.2,
          color=C_DROP, ha="center")
ax_c.text(x[1] + offs["CGMA (ours)"], 74, "std 1.2", fontsize=6.2,
          color=C_CGMA, ha="center")
ax_c.annotate("seed outlier 40.3", xy=(x[2] + offs["Naive + Aug"] + 0.05, 40.3),
              xytext=(x[2] + 0.42, 32), fontsize=6.2, color=C_DROP, ha="center",
              arrowprops=dict(arrowstyle="-", color=C_DROP, lw=0.6))
ax_c.set_xticks(x)
ax_c.set_xticklabels(COND)
ax_c.set_ylabel("F1 (%)")
ax_c.set_ylim(0, 92)
ax_c.legend(loc="upper right", ncols=3)
panel_tag(ax_c, "c", dx=-0.065)

save_pub(fig, "fig3_robustness_lmvd")

# ---- source data ----
rows = []
for tag, rows_ in [("curve_ours", CURVE_OURS), ("curve_adafuse", CURVE_ADA)]:
    for sd, vals in zip([0, 1, 2, 42, 123], rows_):
        rows.append([tag, sd] + vals)
dump_source("fig3a_ratio_curves.csv", ["series", "seed"] + [f"r{r}" for r in R], rows)
rows = [["frame_cgma", sd] + vals for sd, vals in zip([0, 1, 2, 42, 123], FRAME_CGMA)]
for name, rws in FRAME_BASE.items():
    for sd, vals in zip([0, 1, 2, 42, 123], rws):
        rows.append([f"frame_{name}", sd] + vals + ["", ""][:max(0, len(P) - len(vals))])
dump_source("fig3b_frame_curve.csv", ["series", "seed"] + [f"p{p}" for p in P], rows)
rows = []
for mname, per_cond in BARS.items():
    for c in COND:
        for sd, v in zip([0, 1, 2, 42, 123], per_cond[c]):
            rows.append([mname, c, sd, v])
dump_source("fig3c_main_bars.csv", ["method", "condition", "seed", "f1"], rows)
