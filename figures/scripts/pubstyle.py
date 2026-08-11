#-*- coding: utf-8 -*-
"""
投稿级图公共样式(Python/matplotlib 后端, 全组图统一)。
用法: from pubstyle import *; 出图统一走 save_pub(fig, name)。
配色策略: 方法族统一——CGMA=蓝(信号), 朴素基线=灰(中性), 朴素+增强=琥珀(中间档);
红色仅作"崩溃/下降"方向线索, 不作类别色。
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",          # SVG 文字可编辑
    "pdf.fonttype": 42,              # PDF TrueType 可编辑
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "axes.unicode_minus": False,
})

# ---- 方法族配色 ----
C_CGMA  = "#3D7EAA"   # CGMA(ours) 信号蓝
C_NAIVE = "#8C8C8C"   # 朴素基线 中性灰
C_NAUG  = "#E0A458"   # 朴素+增强 琥珀
C_DROP  = "#C0504D"   # 崩溃/下降 红(方向线索)
C_WA    = "#8E7CA6"   # 门控第二模态 淡紫

FIGROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "figures"))
FIGDIR = os.path.join(FIGROOT, "rendered")
SRCDIR = os.path.join(FIGROOT, "source_data")


def mm(w_mm, h_mm):
    """毫米 → 英寸 figsize。"""
    return (w_mm / 25.4, h_mm / 25.4)


def panel_tag(ax, letter, dx=-0.14, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def mean_std(rows):
    a = np.asarray(rows, dtype=float)
    return a.mean(axis=0), a.std(axis=0, ddof=1)


def save_pub(fig, name, dpi=600):
    os.makedirs(FIGDIR, exist_ok=True)
    base = os.path.join(FIGDIR, name)
    fig.savefig(base + ".svg", bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")   # 预览
    try:
        fig.savefig(base + ".tiff", dpi=dpi, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"})
    except Exception as e:                                     # 无 PIL 等
        print(f"[warn] TIFF 导出跳过: {e}")
    print(f">>> saved {base}.{{svg,pdf,png,tiff}}")


def dump_source(name, header, rows):
    """图旁落一份 source data CSV(可溯源)。"""
    import csv
    os.makedirs(SRCDIR, exist_ok=True)
    p = os.path.join(SRCDIR, name)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f">>> source data → {p}")
