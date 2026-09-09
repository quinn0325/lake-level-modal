"""附录图B1 — 420 条湖内候选关系的支持情况总览。

替代逐条列出 420 行的长表：行为 42 种有向变量对（7 × 6，按原因变量分块），
列为 10 个湖，格内着色表示该湖该关系通过了 BH-FDR 与收敛诊断，颜色深浅为
cross-map skill ρ，未通过者留白。一张图承载全部 420 个检验结果，同时保留
逐湖差异——这是聚合成 7×7 计数矩阵时丢失的信息。

左侧使用规范的两列表头：Cause 与 Effect。Cause 按 6 行一组居中显示，
Effect 逐行显示目标变量，不再用箭头表达关系。格内小字为最优时滞 d（月）：
正值表示原因领先，0 为同月，负值表示时序相反。为降低 42 × 10 矩阵的视觉
噪声，正滞后省略加号；d = 0 与 d < 0 不再使用整格粗描边，而是在右上角
加小角标。右侧窄栏给出每条变量关系在多少个湖中得到支持，帮助读者先看到
网络结构，再回看逐湖细节。

数据源 ch4_tables/T5_within_lake_edges.csv（420 行，2026-08-31 重跑）。

输出
----
    results/figures/figure_B1_within_lake_network.pdf / .png / .svg / .tiff
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "ch4_tables"
OUT = ROOT / "results" / "figures"

SYSTEMS = [
    ("Okanagan", ["Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake"]),
    ("Nelson–Winnipeg", ["Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
                         "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake"]),
]
LAKES = [lk for _, g in SYSTEMS for lk in g]
LABEL = {lk: lk.replace("_Lake", "").replace("_", " ") for lk in LAKES}
LABEL["Lake_of_the_Woods"] = "LotW"        # 列首空间有限，用简称

VARS = ["RegFlow", "R", "P", "Evap", "SWE", "T", "WL"]
CMAP = LinearSegmentedColormap.from_list(
    "rho", ["#F4F8FB", "#C5DFED", "#84BBD8", "#347EAF", "#124B7A"])
NORM = Normalize(vmin=0.0, vmax=1.0)
C_ZERO, C_NEG = "#B9785A", "#5F6468"
GRID = "#E8ECEF"
ROW_BAND = "#F7F9FA"
COUNT = "#8B98A4"

FS_ROW, FS_COL, FS_IN, FS_LEG, FS_BLOCK = 7.0, 8.0, 5.2, 8.2, 8.0

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def lag_text(d):
    """Show positive lags without '+', leaving 0 and negative lags explicit."""
    return "0" if d == 0 else (str(d) if d > 0 else f"−{abs(d)}")


def text_colour(rho):
    return "white" if rho > 0.62 else "0.16"


def main():
    t5 = pd.read_csv(TAB / "T5_within_lake_edges.csv")
    supported = t5[t5.statistically_significant].copy()
    sup = supported.set_index(["lake", "cause", "effect"])

    pairs = [(c, e) for c in VARS for e in VARS if c != e]      # 42 行
    ny, nx = len(pairs), len(LAKES)
    pair_counts = supported.groupby(["cause", "effect"]).lake.nunique()

    fig = plt.figure(figsize=(7.2, 8.25))
    axh = fig.add_axes([0.070, 0.052, 0.185, 0.838], sharey=None)
    ax = fig.add_axes([0.275, 0.052, 0.590, 0.838])
    axn = fig.add_axes([0.880, 0.052, 0.078, 0.838], sharey=ax)

    for b in range(len(VARS)):
        if b % 2:
            ax.add_patch(Rectangle((0, b * 6), nx, 6, facecolor=ROW_BAND,
                                   edgecolor="none", zorder=0))

    for j in range(nx + 1):
        ax.axvline(j, color=GRID, lw=0.42, zorder=1)
    for i in range(ny + 1):
        ax.axhline(i, color=GRID, lw=0.35, zorder=1)

    for i, (c, e) in enumerate(pairs):
        for j, lk in enumerate(LAKES):
            key = (lk, c, e)
            if key not in sup.index:
                continue
            r = sup.loc[key]
            d = int(r.obs_lag)
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=CMAP(NORM(r.obs_rho)),
                                   edgecolor="white", lw=0.22, zorder=2))
            ax.text(j + .5, i + .52, lag_text(d), ha="center",
                    va="center", fontsize=FS_IN, zorder=3,
                    color=text_colour(float(r.obs_rho)))
            if d <= 0:
                ax.scatter(j + 0.82, i + 0.20, s=9.5,
                           marker="s" if d == 0 else "v",
                           facecolor=C_ZERO if d == 0 else C_NEG,
                           edgecolor="white", linewidth=0.25, zorder=4)

    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.set_xticks([j + .5 for j in range(nx)])
    ax.set_xticklabels([LABEL[lk] for lk in LAKES], fontsize=FS_COL,
                       rotation=45, ha="left", rotation_mode="anchor")
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # 每 6 行一个原因变量分块
    for b in range(1, len(VARS)):
        ax.axhline(b * 6, color="0.40", lw=0.72, zorder=5)
    ax.axvline(len(SYSTEMS[0][1]), color="0.40", lw=0.72, zorder=5)

    # Dedicated row-header columns: Cause is merged by six-row blocks and
    # Effect is listed once per row, so the direction is explicit without arrows.
    axh.set_xlim(0, 2)
    axh.set_ylim(ny, 0)
    axh.set_xticks([0.5, 1.5])
    axh.set_xticklabels(["Cause", "Effect"], fontsize=FS_LEG)
    axh.xaxis.set_ticks_position("top")
    axh.tick_params(axis="x", length=0, pad=4)
    axh.set_yticks([])
    for x in (0, 1, 2):
        axh.axvline(x, color=GRID if x else "none", lw=0.42, zorder=1)
    for i in range(ny + 1):
        axh.axhline(i, color=GRID, lw=0.35, zorder=1)
    for b, cause in enumerate(VARS):
        axh.text(0.5, b * 6 + 3, cause, ha="center", va="center",
                 fontsize=FS_ROW, color="0.18")
        axh.axhline(b * 6, color="0.40", lw=0.72, zorder=5)
    for i, (_, effect) in enumerate(pairs):
        axh.text(1.5, i + 0.5, effect, ha="center", va="center",
                 fontsize=FS_ROW, color="0.18")
    for sp in axh.spines.values():
        sp.set_visible(False)

    axn.set_xlim(0, 11.1)
    for i, pair in enumerate(pairs):
        n = int(pair_counts.get(pair, 0))
        axn.barh(i + .5, n, height=0.54, color=COUNT, edgecolor="none",
                 alpha=0.82, zorder=2)
        if n:
            axn.text(n + 0.28, i + .5, str(n), ha="left", va="center",
                     fontsize=FS_IN, color="0.25")
    for b in range(1, len(VARS)):
        axn.axhline(b * 6, color="0.40", lw=0.72, zorder=5)
    axn.set_xticks([0, 5, 10])
    axn.xaxis.set_ticks_position("top")
    axn.tick_params(axis="x", labelsize=6.6, length=2.0, width=0.55, pad=1.5)
    axn.tick_params(axis="y", left=False, labelleft=False)
    axn.set_title("n lakes", fontsize=FS_LEG, pad=14, loc="left")
    for sp in axn.spines.values():
        sp.set_visible(False)

    axl = fig.add_axes([0.275, 0.954, 0.683, 0.040])
    axl.set_axis_off()
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    cax = axl.inset_axes([0.445, 0.30, 0.255, 0.38])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=CMAP, norm=NORM,
                                   orientation="horizontal")
    cb.set_ticks([0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=7.0, length=2.2, width=0.6, pad=1.4)
    cb.outline.set_linewidth(0.55)
    axl.text(0.428, 0.52, "ρ", fontsize=FS_LEG, ha="right", va="center")
    axl.text(0.00, 0.52, "Cell text = lag d (months; + omitted)",
             fontsize=FS_LEG, va="center", ha="left", color="0.22")
    for x, marker, c, lab in ((0.760, "s", C_ZERO, "$d=0$"),
                              (0.888, "v", C_NEG, "$d<0$")):
        axl.scatter(x, 0.52, s=18, marker=marker, facecolor=c,
                    edgecolor="white", linewidth=0.35)
        axl.text(x + 0.018, 0.52, lab, fontsize=FS_LEG, va="center",
                 ha="left", color="0.22")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", dict(dpi=600)), ("svg", {}),
                    ("tiff", dict(dpi=600))):
        fig.savefig(OUT / f"figure_B1_within_lake_network.{ext}", **kw)
        print(f"wrote {OUT / f'figure_B1_within_lake_network.{ext}'}")
    plt.close(fig)

    n = len(supported)
    print(f"\n着色格子 {n} / {len(pairs) * nx}；"
          f"正滞后 {int((supported.obs_lag > 0).sum())}、"
          f"同期 {int((supported.obs_lag == 0).sum())}、"
          f"负滞后 {int((supported.obs_lag < 0).sum())}")


if __name__ == "__main__":
    main()
