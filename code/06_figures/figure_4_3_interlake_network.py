"""Figure 4.3 — 湖间因果结构与水文连通性（RQ2）。

Local rendering from downloaded results; this script does not rerun CCM.
仅在本地读取已下载结果并绘图；本脚本不重新运行 CCM。

(a) 湖间关系网络：10 个湖按两个水系分区排布。
    浅灰粗线 = 7 条已知直接水道连接（先验的物理连通信息）；
    彩色箭头 = 22 条通过 BH-FDR（α = 0.05，检验族 = 90 条湖间候选边）
    与收敛诊断、且最优时滞 d > 0 的有向关系，箭头指向被影响的湖，
    线宽与颜色深浅同时编码 cross-map skill ρ；
    浅灰虚线 = 另外 4 条支持关系（3 条 d = 0、1 条 d < 0），
    时序方向未定或与假设方向相反，故不与主箭头同等呈现。

(b) 湖泊对强度 S_ij 按水文距离分组的分布（箱线图 + 抖动散点）。
    S_ij 为该湖泊对两个方向 |ρ| 的均值，见
    modal_inter_lake_ccm.py 中 pair_df 的聚合口径。
    正文的正式推断只有「direct vs all other pairs」一项 Mann–Whitney
    检验（两侧，与代码 alternative="two-sided" 一致），四分组仅为描述性
    深化，图上以括注标明哪三组被合并进 "other pairs"。

约定
----
· d > 0 表示原因领先效应，与 analysis_core 一致。
· 数据源 ch4_tables/T6_between_lake_edges.csv（90 条有向边）与
  T7_lake_pair_strength.csv（45 个湖泊对），均出自 2026-08-31 重跑。
· tier 字段为第四章写作阶段引入的事后水文距离分组，分析代码中不存在，
  因此只作描述性使用。

输出
----
    results/figures/figure_4_3_interlake_network.pdf / .png
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
TAB_DIR = ROOT / "ch4_tables"
OUT_DIR = ROOT / "results" / "figures"

LABEL = {"Kalamalka_Lake": "Kalamalka", "Okanagan_Lake": "Okanagan",
         "Skaha_Lake": "Skaha", "Vaseux_Lake": "Vaseux",
         "Rainy_Lake": "Rainy", "Lake_of_the_Woods": "Lake of\nthe Woods",
         "Playgreen_Lake": "Playgreen", "Kiskitto_Lake": "Kiskitto",
         "Sipiwesk_Lake": "Sipiwesk", "Split_Lake": "Split"}

POS = {"Kalamalka_Lake": (0.085, 0.86), "Okanagan_Lake": (0.255, 0.63),
       "Skaha_Lake": (0.085, 0.40), "Vaseux_Lake": (0.255, 0.16),
       "Rainy_Lake": (0.605, 0.878), "Lake_of_the_Woods": (0.895, 0.85),
       "Playgreen_Lake": (0.585, 0.58), "Kiskitto_Lake": (0.925, 0.55),
       "Sipiwesk_Lake": (0.665, 0.30), "Split_Lake": (0.915, 0.17)}

# Offset crowded labels from arrows. / 将拥挤标签移离箭头。
LABEL_OFF = {"Kalamalka_Lake": (0.0, 0.052, "center", "bottom"),
             "Okanagan_Lake": (0.0, -0.055, "center", "top"),
             "Skaha_Lake": (0.0, -0.055, "center", "top"),
             "Vaseux_Lake": (0.0, -0.055, "center", "top"),
             "Rainy_Lake": (0.0, 0.048, "center", "bottom"),
             "Lake_of_the_Woods": (0.030, -0.045, "center", "top"),
             "Playgreen_Lake": (-0.024, 0.0, "right", "center"),
             "Kiskitto_Lake": (0.0, -0.055, "center", "top"),
             "Sipiwesk_Lake": (-0.024, 0.0, "right", "center"),
             "Split_Lake": (0.0, -0.055, "center", "top")}

WATERWAYS = [("Kalamalka_Lake", "Okanagan_Lake"), ("Okanagan_Lake", "Skaha_Lake"),
             ("Skaha_Lake", "Vaseux_Lake"), ("Playgreen_Lake", "Sipiwesk_Lake"),
             ("Sipiwesk_Lake", "Split_Lake"), ("Rainy_Lake", "Lake_of_the_Woods"),
             ("Kiskitto_Lake", "Sipiwesk_Lake")]

TIERS = ["1_direct", "2_same_subsystem_indirect",
         "3_same_basin_diff_subsystem", "4_different_basin"]
TIER_LABEL = {"1_direct": "Direct connection",
              "2_same_subsystem_indirect": "Same subsystem, indirect",
              "3_same_basin_diff_subsystem": "Same basin, other subsystem",
              "4_different_basin": "Different basin"}

# Colour and width scales preserve weak retained edges in print. / 色彩与线宽范围确保较弱保留边在印刷中可见。
CMAP = LinearSegmentedColormap.from_list(
    "rho", ["#9CC3E0", "#6BAED6", "#3C8DC4", "#2171B5", "#08306B"])
NORM = Normalize(vmin=0.15, vmax=1.0)          # Matches Figure 4.2 / 与图 4.2 一致
RHO_LO, RHO_HI = 0.35, 0.95                    # Width mapping / 线宽映射范围

# Long-distance edges are lighter but remain visible. / 远距离边较浅，但仍保持可见。
TIER_EMPH = {"1_direct": (1.00, 1.00), "2_same_subsystem_indirect": (0.95, 0.92),
             "3_same_basin_diff_subsystem": (0.84, 0.84),
             "4_different_basin": (0.72, 0.76)}
WATERWAY_GREY = "0.76"      # Distinguish waterways from CCM edges. / 区分物理水道与 CCM 边。

# Typography assumes full-width placement in the dissertation. / 字号按论文正文全宽排版设置。
FS_TICK, FS_LAB, FS_TITLE, FS_NODE, FS_LEG = 10.0, 11.0, 12.0, 9.5, 10.0
FS_ANN = 9.0                    # Panel (b) test annotation / 面板 (b) 检验标注
LEFT, RIGHT = 0.235, 0.972

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.7, "axes.labelsize": FS_LAB,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def edge_lw(rho):
    """最细的边也要有 1.5 pt，否则印刷后与背景难以分辨。"""
    f = np.clip((rho - RHO_LO) / (RHO_HI - RHO_LO), 0, 1)
    return 1.5 + 3.0 * f


def draw_network(ax, sup):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.02, 1.02)
    ax.set_axis_off()

    for x0, x1, name in ((0.010, 0.345, "Okanagan system"),
                         (0.520, 1.010, "Nelson–Winnipeg system")):
        ax.add_patch(FancyBboxPatch((x0, 0.055), x1 - x0, 0.895,
                                    boxstyle="round,pad=0.004,rounding_size=0.02",
                                    facecolor="#F6F6F6", edgecolor="none", zorder=0))
        ax.text((x0 + x1) / 2, 0.985, name, ha="center", va="center",
                fontsize=FS_LEG + 0.2, color="0.35")

    for a, b in WATERWAYS:                       # Known waterways beneath CCM edges. / 已知水道置于 CCM 边下层。
        (xa, ya), (xb, yb) = POS[a], POS[b]
        ax.plot([xa, xb], [ya, yb], color=WATERWAY_GREY, lw=6.0, zorder=1,
                solid_capstyle="round")

    weak = sup[sup.obs_lag <= 0]
    for r in weak.itertuples():                  # d<=0 as dashed lines without arrows. / d<=0 使用无箭头虚线。
        (xa, ya), (xb, yb) = POS[r.cause_lake], POS[r.effect_lake]
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-",
                                     connectionstyle="arc3,rad=0.16",
                                     color="0.42", lw=1.3, ls=(0, (3.5, 2)),
                                     shrinkA=10, shrinkB=10, zorder=2))

    # Draw distant edges first. / 先绘制远距离边。
    pos = sup[sup.obs_lag > 0].copy()
    pos["ord_key"] = pos.tier.map({t: i for i, t in enumerate(TIERS[::-1])})
    for r in pos.sort_values(["ord_key", "obs_rho"]).itertuples():
        (xa, ya), (xb, yb) = POS[r.cause_lake], POS[r.effect_lake]
        rad = 0.16 if r.cause_lake < r.effect_lake else -0.16
        alpha, kw = TIER_EMPH[r.tier]
        lw = edge_lw(r.obs_rho) * kw
        ax.add_patch(FancyArrowPatch(
            (xa, ya), (xb, yb), arrowstyle="-|>",
            mutation_scale=11 + 4 * lw, alpha=alpha,
            connectionstyle=f"arc3,rad={rad}", color=CMAP(NORM(r.obs_rho)),
            lw=lw, shrinkA=9, shrinkB=9, zorder=3 + (r.ord_key >= 2)))

    for lake, (x, y) in POS.items():
        ax.scatter(x, y, s=150, facecolor="white", edgecolor="0.25", lw=1.1,
                   zorder=5)
        dx, dy, ha, va = LABEL_OFF[lake]
        ax.text(x + dx, y + dy, LABEL[lake], ha=ha, va=va,
                fontsize=FS_NODE, zorder=6, linespacing=1.05,
                path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])



def draw_key(axl):
    axl.set_axis_off()
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.plot([0.012, 0.058], [0.78, 0.78], color=WATERWAY_GREY, lw=6.0,
             solid_capstyle="round")
    axl.text(0.070, 0.78, "known direct waterway", fontsize=FS_LEG, va="center")
    axl.add_patch(FancyArrowPatch((0.345, 0.78), (0.397, 0.78),
                                  arrowstyle="-|>", mutation_scale=13,
                                  color="#2171B5", lw=2.6))
    axl.text(0.409, 0.78, "$d>0$ relationship", fontsize=FS_LEG, va="center")
    axl.plot([0.640, 0.692], [0.78, 0.78], color="0.42", lw=1.3, ls=(0, (3.5, 2)))
    axl.text(0.704, 0.78, "$d=0$ or $d<0$ relationship", fontsize=FS_LEG,
             va="center")

    axl.text(0.012, 0.26, "arrow width and shade give cross-map skill",
             fontsize=FS_LEG, va="center", color="0.35")
    axl.text(0.470, 0.26, r"$\rho$", fontsize=FS_LEG, va="center", color="0.35")
    cax = axl.inset_axes([0.510, 0.20, 0.210, 0.125])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=CMAP, norm=NORM,
                                   orientation="horizontal")
    cb.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(labelsize=7.8, length=2.5, width=0.6, pad=1.5)
    cb.outline.set_linewidth(0.6)


def draw_strength(axb, t7):
    rng = np.random.default_rng(0)
    data = [t7.loc[t7.tier == t, "S_ij"].values for t in TIERS]
    ypos = range(1, len(TIERS) + 1)

    bp = axb.boxplot(data, orientation="horizontal", widths=0.52, whis=1.5,
                     showfliers=False, patch_artist=True,
                     medianprops=dict(color="black", lw=1.3),
                     whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for patch in bp["boxes"]:
        patch.set(facecolor="#DCE9F7", edgecolor="#4E7EA8", lw=0.9)
    for y, v in zip(ypos, data):
        axb.scatter(v, y + rng.uniform(-0.15, 0.15, len(v)), s=11,
                    facecolor="0.25", edgecolor="none", alpha=0.75, zorder=4)

    axb.set_yticks(list(ypos))
    axb.set_yticklabels([f"{TIER_LABEL[t]}\n($n$ = {len(d)})"
                         for t, d in zip(TIERS, data)], fontsize=FS_TICK,
                        linespacing=1.2)
    axb.set_ylim(len(TIERS) + 0.60, -0.62)
    axb.set_xlim(0.0, 1.0)
    axb.set_xlabel(r"Pair strength $S_{ij}$  (mean $|\rho|$ over both directions)",
                   labelpad=4)
    axb.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        axb.spines[sp].set_visible(False)

    # Formal test compares direct pairs with all others. / 正式检验比较直接连接湖泊对与其他湖泊对。
    axb.text(0.0, -0.34, "Pre-specified test, direct vs all other pairs:",
             fontsize=FS_ANN, ha="left", va="center", color="0.20")
    axb.text(0.0, 0.02, "Mann–Whitney $U$ = 201, $p$ = 0.032, AUC = 0.756",
             fontsize=FS_ANN, ha="left", va="center", color="0.20")
    xb = 1.035
    axb.plot([xb, xb + 0.022, xb + 0.022, xb], [1.62, 1.62, 4.38, 4.38],
             color="0.45", lw=0.8, clip_on=False)
    axb.text(xb + 0.052, 3.00, "pooled as \"other pairs\" ($n$ = 38)",
             rotation=90, ha="center", va="center", fontsize=8.5, color="0.35",
             clip_on=False)


def main():
    t6 = pd.read_csv(TAB_DIR / "T6_between_lake_edges.csv")
    t7 = pd.read_csv(TAB_DIR / "T7_lake_pair_strength.csv")
    sup = t6[t6.statistically_significant].copy()

    def save(fig, stem):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for ext, kw in (("svg", {}), ("pdf", dict(dpi=300)),
                        ("png", dict(dpi=400)), ("tiff", dict(dpi=600))):
            path = OUT_DIR / f"{stem}.{ext}"
            fig.savefig(path, bbox_inches="tight", **kw)
            print(f"wrote {path}")
        plt.close(fig)

    # Render the network separately for clearer spacing. / 网络图单独输出以增加节点间距。
    fig = plt.figure(figsize=(6.3, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[10.0, 1.9], hspace=0.0,
                          left=0.020, right=0.992, top=0.988, bottom=0.030)
    draw_network(fig.add_subplot(gs[0]), sup)
    draw_key(fig.add_subplot(gs[1]))
    save(fig, "figure_4_3_interlake_network")

    fig = plt.figure(figsize=(6.3, 3.25))
    ax = fig.add_axes([0.335, 0.185, 0.525, 0.760])
    draw_strength(ax, t7)
    save(fig, "figure_4_3_pair_strength")

    # Numerical checks / 数值核对
    print(f"\nbetween-lake edges tested : {len(t6)}")
    print(f"  supported               : {len(sup)}")
    cls = sup.obs_lag.map(lambda d: "pos" if d > 0 else ("zero" if d == 0 else "neg"))
    print("  temporal class          : "
          + ", ".join(f"{k}={v}" for k, v in
                      cls.value_counts().reindex(["pos", "zero", "neg"]).items()))
    p = sup[sup.obs_lag > 0]
    print(f"  d>0 median rho          : {p.obs_rho.median():.3f}")
    print(f"  d>0 median lag          : {p.obs_lag.median()}  "
          f"(range {int(p.obs_lag.min())}-{int(p.obs_lag.max())}, "
          f"{int(p.obs_lag.between(1, 3).sum())}/{len(p)} in 1-3)")
    print(f"  d>0 on a direct waterway: {int(p.waterway_connected.sum())} of {len(p)}")
    print("\npair strength by tier:")
    print(t7.groupby("tier").agg(n=("S_ij", "size"), median_S=("S_ij", "median"),
                                 occurrence=("detected", "mean")).round(4).to_string())
    print("\ndirect vs other:")
    print(t7.groupby("directly_connected").agg(
        n=("S_ij", "size"), median_S=("S_ij", "median"),
        occurrence=("detected", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
