"""Figure 4.1 — 十个研究湖泊的水位变率。

(a) 逐湖时间序列（10 条堆叠条带，1994–2024）
(b) 逐湖横向箱线图

两个 panel 使用同一条 **centred water level**：

    WL_centered(i,t) = WL(i,t) − mean_t WL(i,t)

即每个湖减去自身长期均值，**不除以标准差、不去季节化**。术语上一律称
"centred water level"，不称 "anomaly"——后者在本文中专指去除月气候态之后的量。站点基准面差异由此消除、
各湖围绕 0 显示，而单位仍是米，因此 Sipiwesk 的大幅波动与 Skaha 的小幅波动之间
是真实差异。数据取自 lake_pkls 的原始多站水位，经与主流程完全相同的异常值筛查与
多站合成（clean_wide_wl → combine_station_water_levels），再对齐到完整月历，
真实缺测保留为 NaN。

制图规范
--------
· 按最终印刷尺寸作图（6.3 in 宽 = A4 双面 2.5 cm 页边距下的正文宽度），
  字号即为读者看到的字号，不依赖后期缩放。最小字号 8 pt。
· 配色取 Okabe–Ito 色盲安全调色板的蓝（#0072B2）与朱红（#D55E00）。
· 两个 panel 共用 y/x 量程以保证跨湖可比；Skaha 与 Vaseux 因此接近平线，
  这本身即是结果，其量级由 panel (b) 承担。
· 连续缺测 ≥2 个月的区间以浅灰竖带标出（BAND_GREY），孤立缺测仅表现为断线。
· 图内不设 legend：颜色与灰带的含义全部由 caption 承担，避免图例压占数据区。
· 字体嵌入为 TrueType（pdf.fonttype=42），满足投稿要求。

跑法
----
    python code/06_figures/figure_4_1_wl_variability.py

输出
----
    results/figures/figure_4_1_wl_variability.pdf   （矢量，投稿用）
    results/figures/figure_4_1_wl_variability.png   （400 dpi，预览用）

正式图注
--------
Figure 4.1. Temporal and distributional variability of observed monthly lake
water levels before deseasonalisation. (a) Cleaned monthly water levels
centred on each lake's long-term mean. All series share the same vertical
scale, allowing differences in amplitude to be compared directly. Breaks
indicate missing observations, with grey shading identifying contiguous gaps
of at least two months. (b) Distributions of the corresponding centred
monthly water levels. Blue and orange denote lakes in the Okanagan and
Nelson–Winnipeg systems, respectively.
"""
from pathlib import Path
import pickle
import sys
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR / "01_shared"))
import ccm_full_pipeline as p                                    # noqa: E402

p.log = lambda msg: None

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"
PKL_DIR = Path(p.PKL_DIR)

SYSTEMS = [
    ("Okanagan", ["Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake"]),
    ("Nelson–Winnipeg", ["Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
                         "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake"]),
]
LAKES = [lk for _, g in SYSTEMS for lk in g]
SYSTEM_OF = {lk: n for n, g in SYSTEMS for lk in g}
COLOUR = {"Okanagan": "#0072B2", "Nelson–Winnipeg": "#D55E00"}   # Okabe–Ito
LABEL = {lk: lk.replace("_Lake", "").replace("_", " ") for lk in LAKES}
LABEL["Lake_of_the_Woods"] = "Lake of the Woods"

# 字号按最终印刷尺寸设定：图宽 6.3 in 即 A4（2.5 cm 页边距）的正文栏宽，
# 所以这里写的 pt 就是纸面上的 pt；插入文档时必须按 100% 置入。
FS_TICK, FS_LABEL, FS_LAKE, FS_TITLE = 10.0, 11.0, 10.5, 12.0
BAND_GREY = "0.45"      # 缺测竖带颜色；透明度由 BAND_ALPHA 控制
BAND_ALPHA = 0.07
FS_CAPTION = 9.0

# 图内 caption。投稿时若期刊/handbook 要求 caption 写在正文，把 EMBED_CAPTION
# 置 False 即可，版面会自动收回底部留白，图形本身不变。
EMBED_CAPTION = False
CAPTION = (
    "Figure 4.1. Temporal and distributional variability of observed monthly "
    "lake water levels before deseasonalisation. (a) Cleaned monthly water "
    "levels centred on each lake's long-term mean. All series share the same "
    "vertical scale, allowing differences in amplitude to be compared "
    "directly. Breaks indicate missing observations, with grey shading "
    "identifying contiguous gaps of at least two months. (b) Distributions of "
    "the corresponding centred monthly water levels. Blue and orange denote "
    "lakes in the Okanagan and Nelson–Winnipeg systems, respectively."
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.7, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def centered_series(lake):
    with open(PKL_DIR / f"{lake}_result.pkl", "rb") as fh:
        cached = pickle.load(fh)
    wl = p.combine_station_water_levels(
        p.clean_wide_wl(cached["wide_wl"]), method="anomaly_mean")
    wl = wl.sort_index().asfreq("MS")
    return wl - wl.mean()


def gap_spans(s, min_len=2):
    """返回连续缺测 ≥min_len 个月的区间 [(start, end), ...]。"""
    na = s.isna().values
    out, i, n = [], 0, len(na)
    while i < n:
        if not na[i]:
            i += 1
            continue
        j = i
        while j < n and na[j]:
            j += 1
        if j - i >= min_len:
            out.append((s.index[i], s.index[min(j, n - 1)]))
        i = j
    return out


def main():
    data = {lk: centered_series(lk) for lk in LAKES}
    lim = np.ceil(max(np.nanmax(np.abs(s.values)) for s in data.values()) * 10) / 10
    x0, x1 = pd.Timestamp("1994-01-01"), pd.Timestamp("2025-01-01")

    cap_lines = (textwrap.wrap(CAPTION, width=104) if EMBED_CAPTION else [])
    cap_in = len(cap_lines) * FS_CAPTION * 1.42 / 72.0        # caption 占用高度
    fig_h = 9.3 + cap_in
    bottom = (0.50 + cap_in + 0.10) / fig_h if EMBED_CAPTION else 0.50 / fig_h

    fig = plt.figure(figsize=(6.3, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 3.6], hspace=0.30,
                          left=0.238, right=0.978, top=1 - 0.42 / fig_h,
                          bottom=bottom)
    gs_a = gs[0].subgridspec(10, 1, hspace=0.0)

    # ---------------------------------------------------------------- panel (a)
    for k, lake in enumerate(LAKES):
        ax = fig.add_subplot(gs_a[k])
        s, colour = data[lake], COLOUR[SYSTEM_OF[lake]]
        for g0, g1 in gap_spans(s):
            ax.axvspan(g0, g1, color=BAND_GREY, alpha=BAND_ALPHA,
                       lw=0, zorder=0)
        ax.axhline(0, lw=0.5, color="0.70", zorder=1)
        ax.plot(s.index, s.values, lw=0.8, color=colour, zorder=3,
                solid_capstyle="round")
        ax.set_xlim(x0, x1)
        ax.set_ylim(-lim, lim)
        ax.set_yticks([-2, 0, 2])
        ax.text(0.008, 0.90, LABEL[lake], transform=ax.transAxes,
                fontsize=FS_LAKE, fontweight="bold", color=colour,
                va="top", ha="left", zorder=5)
        ax.spines["top"].set_visible(k == 0)
        ax.spines["bottom"].set_visible(k == len(LAKES) - 1)
        if k < len(LAKES) - 1:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xlabel("Year", labelpad=3)
        if k == 0:
            ax.set_title("(a)  Monthly water levels centred on each lake's "
                         "long-term mean", fontsize=FS_TITLE, loc="left", pad=6,
                         x=-0.245)
        # 系统分隔：Okanagan 与 Nelson–Winnipeg 之间画一条细线
        if lake == "Rainy_Lake":
            ax.spines["top"].set_visible(True)
            ax.spines["top"].set_linewidth(1.1)
    pa_top, pa_bot = gs[0].get_position(fig).y1, gs[0].get_position(fig).y0
    fig.text(0.058, (pa_top + pa_bot) / 2, "Centred water level (m)",
             rotation=90, va="center", ha="center", fontsize=FS_LABEL)

    # ---------------------------------------------------------------- panel (b)
    axb = fig.add_subplot(gs[1])
    order = LAKES[::-1]
    _mpl_ver = tuple(int(v) for v in mpl.__version__.split(".")[:2])
    orient_kw = ({"orientation": "horizontal"} if _mpl_ver >= (3, 11)
                 else {"vert": False})
    bp = axb.boxplot([data[lk].dropna().values for lk in order], **orient_kw,
                     widths=0.62, whis=1.5, showfliers=True, patch_artist=True,
                     medianprops=dict(color="black", lw=1.2),
                     whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                     flierprops=dict(marker=".", ms=2.4, mfc="0.40",
                                     mec="none", alpha=0.55))
    for patch, lake in zip(bp["boxes"], order):
        c = COLOUR[SYSTEM_OF[lake]]
        patch.set(facecolor=c, alpha=0.32, edgecolor=c, lw=0.9)
    axb.axvline(0, lw=0.5, color="0.70", zorder=0)
    axb.set_yticks(range(1, len(order) + 1))
    axb.set_yticklabels([LABEL[lk] for lk in order], fontsize=FS_LAKE)
    for tick, lake in zip(axb.get_yticklabels(), order):
        tick.set_color(COLOUR[SYSTEM_OF[lake]])
    axb.set_xlim(-lim, lim)
    axb.set_xlabel("Centred monthly water level (m)", labelpad=3)
    axb.set_title("(b)  Distribution of centred monthly water levels",
                  fontsize=FS_TITLE, loc="left", pad=6, x=-0.245)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    if cap_lines:
        fig.text(0.045, 0.10 / fig_h, "\n".join(cap_lines), fontsize=FS_CAPTION,
                 va="bottom", ha="left", linespacing=1.42, color="0.15")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, dpi in ((OUT_DIR / "figure_4_1_wl_variability.pdf", 300),
                      (OUT_DIR / "figure_4_1_wl_variability.png", 400)):
        fig.savefig(path, dpi=dpi)
        print(f"wrote {path}")
    plt.close(fig)

    summary = pd.DataFrame({
        "lake": [LABEL[lk] for lk in LAKES],
        "system": [SYSTEM_OF[lk] for lk in LAKES],
        "n_obs": [int(data[lk].notna().sum()) for lk in LAKES],
        "n_missing": [int(data[lk].isna().sum()) for lk in LAKES],
        "sd_m": [round(float(data[lk].std()), 3) for lk in LAKES],
        "iqr_m": [round(float(data[lk].quantile(.75) - data[lk].quantile(.25)), 3)
                  for lk in LAKES],
        "range_m": [round(float(data[lk].max() - data[lk].min()), 3) for lk in LAKES],
    })
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()


# 定稿 caption（与正文一致，勿在图内重复）
# -----------------------------------------------------------------------------
# Figure 4.1. Temporal and distributional variability of observed monthly lake
# water levels before deseasonalisation. (a) Cleaned monthly water levels
# centred on each lake's long-term mean. All series share the same vertical
# scale, allowing differences in amplitude to be compared directly. Breaks
# indicate missing observations, with grey shading identifying contiguous gaps
# of at least two months. (b) Distributions of the corresponding centred
# monthly water levels. Blue and orange denote lakes in the Okanagan and
# Nelson–Winnipeg systems, respectively.
