"""Figure 4.2 — 湖内 CCM 因果筛查结果（RQ1）。

driver → WL 结果矩阵（热图）：10 个湖 × 6 个候选驱动变量。
    仅同时通过 BH-FDR（α = 0.05，检验族 = 420 条湖内候选边）与收敛诊断的
    24 条关系着色，未通过筛选的格子留白。
      格子颜色深浅 = cross-map skill ρ
      格内数字     = 最优时滞 d（月），带符号，如 +2 / 0 / −1
      格子描边     = 时序类别，紫 = d = 0（同月），灰 = d < 0（效应领先）；
                     d > 0 无描边，即正常的「原因领先」情形
该矩阵同时承载正文最重要的几个结果：RegFlow 最普遍、Runoff 与
Precipitation 次之、Temperature 无 positive-lag direct relationship、
Vaseux 关系最多而 Skaha 一条都没有，以及 4 例同月与 2 例逆序。

约定
----
· d 的符号约定与 ccm_full_pipeline 一致：d > 0 表示原因领先效应。
  （相对 Ye et al. 2015 的 ℓ，本文 d = −ℓ。）
· 数据源为 ch4_tables/T5_within_lake_edges.csv（420 条），出自
  2026-08-31 重跑结果。
· statistically_significant 字段本身已包含收敛诊断，见
  ccm_full_pipeline.apply_fdr_and_causal_evidence，因此不必再取一次交集。

输出
----
    results/figures/figure_4_2_ccm_driver_matrix.svg / .pdf / .png / .tiff
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
TAB_DIR = ROOT / "ch4_tables"
OUT_DIR = ROOT / "results" / "figures"

SYSTEMS = [
    ("Okanagan", ["Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake"]),
    ("Nelson–Winnipeg", ["Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
                         "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake"]),
]
LAKES = [lk for _, g in SYSTEMS for lk in g]
LABEL = {lk: lk.replace("_Lake", "").replace("_", " ") for lk in LAKES}
LABEL["Lake_of_the_Woods"] = "Lake of the Woods"

DRIVERS = ["RegFlow", "R", "P", "Evap", "SWE", "T"]
DRIVER_LABEL = {"RegFlow": "Regulated flow", "R": "Runoff", "P": "Precipitation",
                "Evap": "Evaporation", "SWE": "SWE", "T": "Temperature"}

C_ZERO = "#B9785A"       # d = 0  同月，低饱和度橙色
C_NEG = "#777777"        # d < 0  逆序
CMAP = LinearSegmentedColormap.from_list(
    "rho", ["#F4F8FB", "#C5DFED", "#84BBD8", "#347EAF", "#124B7A"])
NORM = Normalize(vmin=0.0, vmax=1.0)     # ρ 使用完整理论范围 0–1
TXT_FLIP = 0.62                          # 超过此 ρ 用白字

# 字号按最终印刷尺寸设定：图宽即正文栏宽 6.3 in（A4，2.5 cm 页边距），
# 因此这里写多少 pt，读者在纸上看到的就是多少 pt。插入文档时务必按 100%
# 置入，一旦被缩放，下面所有字号都会等比变小。
FS_TICK, FS_LAB, FS_TITLE, FS_IN, FS_LEG = 10.0, 11.0, 12.0, 9.5, 10.0
FS_HEAD = 10.0                  # 列首：旋转 30° 以便在等分列宽下保持字号
LEFT, RIGHT = 0.292, 0.900

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "axes.linewidth": 0.8, "axes.labelsize": FS_LAB,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "xtick.major.width": 0.7, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 0.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def lag_text(d):
    """+2 / 0 / −1，负号用 U+2212 以匹配正文排版。"""
    return "0" if d == 0 else (f"+{d}" if d > 0 else f"−{abs(d)}")


def draw_matrix(ax, sup):
    ny, nx = len(LAKES), len(DRIVERS)
    for i in range(ny):                                   # 先铺满空格
        for j in range(nx):
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="white",
                                   edgecolor="0.86", lw=0.55, zorder=1))

    for _, r in sup.iterrows():
        i, j, d = LAKES.index(r.lake), DRIVERS.index(r.cause), int(r.obs_lag)
        edge = C_ZERO if d == 0 else (C_NEG if d < 0 else "0.72")
        border_lw = 1.35 if d == 0 else (1.55 if d < 0 else 0.7)
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                               facecolor=CMAP(NORM(r.obs_rho)), edgecolor=edge,
                               lw=border_lw, zorder=2))
        ax.text(j, i, lag_text(d), ha="center", va="center", fontsize=FS_IN,
                fontweight="bold", zorder=3,
                color="white" if r.obs_rho > TXT_FLIP else "0.12")

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(ny - 0.5, -0.5)
    ax.set_xticks(range(nx))
    ax.set_xticklabels([DRIVER_LABEL[d] for d in DRIVERS], fontsize=FS_HEAD,
                       fontweight="bold", rotation=26, ha="left",
                       rotation_mode="anchor")
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks(range(ny))
    ax.set_yticklabels([LABEL[lk] for lk in LAKES], fontsize=FS_TICK)
    ax.tick_params(axis="y", pad=7)
    ax.axhline(len(SYSTEMS[0][1]) - 0.5, color="0.30", lw=0.55, zorder=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for name, grp in SYSTEMS:
        idx = [LAKES.index(lk) for lk in grp]
        ax.text(-0.315, 1 - (np.mean(idx) + 0.5) / ny, name, rotation=90,
                rotation_mode="anchor", transform=ax.transAxes,
                ha="center", va="center",
                fontsize=8.6, color="0.35")


def draw_key(axl):
    """时序类别描边说明和 ρ 色标。"""
    axl.set_axis_off()
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    # Two timing classes on one row, with the color scale tucked under them.
    for x, y, c, lab in ((0.015, 0.70, C_ZERO, "d = 0  contemporaneous"),
                         (0.515, 0.70, C_NEG, "d < 0  opposite temporal order")):
        axl.add_patch(Rectangle((x, y - 0.13), 0.026, 0.26,
                                facecolor="#E4EFF7", edgecolor=c, lw=1.25))
        axl.text(x + 0.036, y, lab, fontsize=FS_LEG, va="center", ha="left")

    axl.text(0.515, 0.30, "cross-map skill ρ", fontsize=FS_LEG,
             va="center", ha="left")
    cax = axl.inset_axes([0.515, 0.04, 0.350, 0.15])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=CMAP, norm=NORM,
                                   orientation="horizontal")
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(labelsize=8.5, length=2.2, width=0.6, pad=1.4)
    cb.outline.set_linewidth(0.55)


def save(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.svg", format="svg",
                bbox_inches=None)
    print(f"wrote {OUT_DIR / f'{stem}.svg'}")
    fig.savefig(OUT_DIR / f"{stem}.pdf", format="pdf",
                bbox_inches=None)
    print(f"wrote {OUT_DIR / f'{stem}.pdf'}")
    fig.savefig(OUT_DIR / f"{stem}.png", format="png", dpi=600,
                bbox_inches=None)
    print(f"wrote {OUT_DIR / f'{stem}.png'}")
    fig.savefig(OUT_DIR / f"{stem}.tiff", format="tiff", dpi=600,
                bbox_inches=None)
    print(f"wrote {OUT_DIR / f'{stem}.tiff'}")
    plt.close(fig)


def main():
    t5 = pd.read_csv(TAB_DIR / "T5_within_lake_edges.csv")
    sup = t5[(t5.effect == "WL") & t5.statistically_significant].copy()

    # ------------------------------------------------------ 独立矩阵图
    fig = plt.figure(figsize=(6.3, 5.30))
    gs = fig.add_gridspec(2, 1, height_ratios=[10.0, 2.15], hspace=0.0,
                          left=LEFT, right=RIGHT, top=0.874, bottom=0.058)
    draw_matrix(fig.add_subplot(gs[0]), sup)
    draw_key(fig.add_subplot(gs[1]))
    save(fig, "figure_4_2_ccm_driver_matrix")

    # ------------------------------------------------------------ 核对数字
    print(f"\nwithin-lake edges tested   : {len(t5)}")
    print(f"  supported                : {int(t5.statistically_significant.sum())}")
    print(f"driver -> WL tested        : {int((t5.effect == 'WL').sum())}")
    print(f"  supported                : {len(sup)}")
    cls = sup.obs_lag.map(lambda d: "pos" if d > 0 else ("zero" if d == 0 else "neg"))
    print("  temporal class           : "
          + ", ".join(f"{k}={v}" for k, v in cls.value_counts()
                      .reindex(["pos", "zero", "neg"]).items()))
    print(f"  rho range (coloured)     : {sup.obs_rho.min():.3f}-{sup.obs_rho.max():.3f}")
    ps = sup[sup.obs_lag > 0]
    print("\npositive-lag drivers:")
    for drv in DRIVERS:
        s = ps[ps.cause == drv]
        name = DRIVER_LABEL[drv].replace("\n", " ")
        if len(s):
            print(f"  {name:<16} n={s.lake.nunique()}  median rho="
                  f"{s.obs_rho.median():.3f}  lag {int(s.obs_lag.min())}-"
                  f"{int(s.obs_lag.max())}")
        else:
            print(f"  {name:<16} n=0")
    per_lake = sup[sup.obs_lag > 0].groupby("lake").size()
    print("\npositive-lag driver count per lake:")
    for lk in LAKES:
        print(f"  {LABEL[lk]:<18} {int(per_lake.get(lk, 0))}")


if __name__ == "__main__":
    main()


# 定稿 caption（图内不重复，正文排版时使用）
# -----------------------------------------------------------------------------
# Figure 4.2. CCM-supported driver-to-water-level relationships across the ten
# study lakes. The 24 driver → WL relationships that pass Benjamini–Hochberg
# FDR control at alpha = 0.05 within the 420-edge within-lake family together
# with the convergence diagnostic. Cell shading gives the cross-map skill rho
# and the cell label the optimal lag d in months, positive when the driver
# leads water level; cell outlines mark the temporal class. Relationships that
# do not pass are left blank, as is Temperature, for which no lake returned a
# supported relationship.
