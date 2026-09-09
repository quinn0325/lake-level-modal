"""Figure 4.4 — 逐湖预测表现小多图（RQ3）。

热图把 108 个数字铺开，逐格可查但看不出每个湖自己的走势；本版改用 small
multiples：每个湖一格，横轴为预见期，纵轴为该策略相对 AR-only（SARIMA(X) 面板
为相对 SARIMA）的 RMSE 比值。基线因此收缩成一条 y = 1 的参考线，四条策略线
直接读作"离基线多远、往哪个方向走"。

· 纵轴取对数并共用同一量程：0.5 与 2.0 到参考线的距离相等，即"误差减半"与
  "误差翻倍"视觉等重；共用量程使九个湖之间可以横向对读。
· 参考线以下浅灰底表示优于基线。
· XGBoost 的 9 个 matched 湖按水文系统顺序排列，前三行为 Okanagan 系统，
  后六行为 Nelson–Winnipeg 系统。
· 末行为 SARIMAX 的 3 个 matched 湖，顺序与 Okanagan 系统一致。

数据源 ch4_tables/T1_rolling_lake_horizon_method.csv（372 行，2026-08-31 重跑）。

输出
----
    results/figures/figure_4_4_per_lake_smallmultiples.pdf / .png / .svg / .tiff
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB_DIR = ROOT / "ch4_tables"
OUT_DIR = ROOT / "results" / "figures"

HORIZONS = [1, 3, 6, 12]
LABEL = {"Kalamalka_Lake": "Kalamalka", "Okanagan_Lake": "Okanagan",
         "Skaha_Lake": "Skaha", "Vaseux_Lake": "Vaseux", "Rainy_Lake": "Rainy",
         "Lake_of_the_Woods": "Lake of the Woods", "Playgreen_Lake": "Playgreen",
         "Kiskitto_Lake": "Kiskitto", "Sipiwesk_Lake": "Sipiwesk",
         "Split_Lake": "Split"}

PALETTE = {
    "all_vars": "#E68613",
    "stepwise": "#2CA02C",
    "ccm_direct": "#1F77B4",
    "ccm_ancestors": "#D62728",
    "baseline": "0.20",
    "baseline_fill": "0.982",
}

XGB = ("XGBoost_AR_only",
       [("XGBoost_all_vars", "All-vars", PALETTE["all_vars"], "s"),
        ("XGBoost_Stepwise", "Stepwise", PALETTE["stepwise"], "^"),
        ("XGBoost_CCM_direct", "CCM-direct", PALETTE["ccm_direct"], "D"),
        ("XGBoost_CCM_ancestors", "CCM-ancestors", PALETTE["ccm_ancestors"], "v")])
SAR = ("SARIMA",
       [("SARIMAX_all_vars", "All-vars", PALETTE["all_vars"], "s"),
        ("SARIMAX_Stepwise", "Stepwise", PALETTE["stepwise"], "^"),
        ("SARIMAX_CCM_direct", "CCM-direct", PALETTE["ccm_direct"], "D"),
        ("SARIMAX_CCM_ancestors", "CCM-ancestors", PALETTE["ccm_ancestors"], "v")])

FS_TICK, FS_LAB, FS_TITLE, FS_PANEL, FS_LEG = 8.0, 9.7, 10.1, 8.7, 8.8
YLIM = (0.28, 2.55)
# 四条策略线在多个湖上几乎重合（Sipiwesk、Split），沿横轴各让开一点，
# 使同一预见期上的四个点分开；线因此略带斜度，但可分辨性远优于完全叠合。
DODGE = 0.092

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "axes.linewidth": 0.75, "axes.labelsize": FS_LAB,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "xtick.major.width": 0.65, "ytick.major.width": 0.65,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.labelcolor": "0.15", "xtick.color": "0.20", "ytick.color": "0.20",
    "axes.titleweight": "semibold",
})


def matched(v, keys):
    sets = [set(v[(v.method == m) & (v.horizon_months == h)].lake)
            for m in keys for h in HORIZONS]
    return sorted(set.intersection(*sets))


XGB_ORDER = [
    "Kalamalka_Lake", "Okanagan_Lake", "Vaseux_Lake",
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
    "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake",
]
SAR_ORDER = ["Kalamalka_Lake", "Okanagan_Lake", "Vaseux_Lake"]


def ratios(v, base, strategies, lakes):
    cols = [base] + [c for c, *_ in strategies]
    piv = v[v.lake.isin(lakes) & v.method.isin(cols)].pivot_table(
        index=["lake", "horizon_months"], columns="method", values="rmse")
    return piv[[c for c, *_ in strategies]].div(piv[base], axis=0)


def panel(ax, r, lake, strategies, first_col, tag):
    ax.axhspan(YLIM[0], 1.0, color=PALETTE["baseline_fill"], zorder=0)
    ax.axhline(1.0, color=PALETTE["baseline"], lw=0.85, ls=(0, (4, 2.5)),
               zorder=1)
    ax.grid(axis="y", which="major", color="0.92", lw=0.55, zorder=0)
    for si, (col, _, colour, marker) in enumerate(strategies):
        dx = (si - (len(strategies) - 1) / 2) * DODGE
        x = [i + dx for i in range(len(HORIZONS))]
        y = [r.loc[(lake, h), col] for h in HORIZONS]
        ax.plot(x, y, color=colour, lw=1.05, marker=marker, ms=4.4,
                mfc=colour, mec="white", mew=0.55, zorder=3, clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_yticks([0.5, 1.0, 2.0])
    ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(
        lambda y, _: f"{y:g}"))
    ax.get_yaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlim(-0.34, len(HORIZONS) - 0.66)
    if not first_col:
        ax.set_yticklabels([])
    ax.tick_params(length=2.4, pad=1.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.03, 0.955, f"{LABEL[lake]}{tag}", transform=ax.transAxes,
            fontsize=FS_PANEL, va="top", ha="left", fontweight="semibold",
            color="0.10")


def main():
    t1 = pd.read_csv(TAB_DIR / "T1_rolling_lake_horizon_method.csv")
    v = t1[t1.n_origins > 0]
    lx = matched(v, [XGB[0]] + [c for c, *_ in XGB[1]])
    ls = matched(v, [SAR[0]] + [c for c, *_ in SAR[1]])
    rx, rs = ratios(v, *XGB, lx), ratios(v, *SAR, ls)
    order_x = [k for k in XGB_ORDER if k in lx]
    order_s = [k for k in SAR_ORDER if k in ls]

    # (a) 与 (b) 用两个 gridspec，中间留白供 (b) 的标题落位
    fig = plt.figure(figsize=(6.3, 8.4))
    gs_a = fig.add_gridspec(3, 3, hspace=0.30, wspace=0.09,
                            left=0.098, right=0.990, top=0.925, bottom=0.312)
    gs_b = fig.add_gridspec(1, 3, wspace=0.09,
                            left=0.098, right=0.990, top=0.238, bottom=0.118)
    for i, lake in enumerate(order_x):
        ax = fig.add_subplot(gs_a[i // 3, i % 3])
        panel(ax, rx, lake, XGB[1], i % 3 == 0, "")
        if i == 0:
            ax.set_title("(a)  XGBoost strategies relative to AR-only",
                         fontsize=FS_TITLE, loc="left", pad=6, x=-0.12)
    for i, lake in enumerate(order_s):
        ax = fig.add_subplot(gs_b[0, i])
        panel(ax, rs, lake, SAR[1], i == 0, "")
        if i == 0:
            ax.set_title("(b)  SARIMAX strategies relative to SARIMA",
                         fontsize=FS_TITLE, loc="left", pad=6, x=-0.12)

    fig.text(0.5, 0.056, "Forecast horizon (months)", ha="center",
             fontsize=FS_LAB)
    fig.text(0.016, 0.56, "RMSE ratio to no-exogenous baseline (log scale)",
             rotation=90, rotation_mode="anchor", va="center", ha="center",
             fontsize=FS_LAB)

    handles = [plt.Line2D([], [], color=c, lw=1.05, marker=m, ms=4.4,
                          mec="white", mew=0.55) for _, _, c, m in XGB[1]]
    handles.append(plt.Line2D([], [], color=PALETTE["baseline"], lw=0.85,
                              ls=(0, (4, 2.5))))
    fig.legend(handles, [n for _, n, _, _ in XGB[1]] + ["baseline"],
               fontsize=FS_LEG, frameon=False, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.002), handlelength=2.0,
               handletextpad=0.5, columnspacing=1.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", dict(dpi=600)), ("svg", {}), ("tiff", dict(dpi=600))):
        path = OUT_DIR / f"figure_4_4_per_lake_smallmultiples.{ext}"
        fig.savefig(path, **kw)
        print(f"wrote {path}")
    plt.close(fig)

    # ------------------------------------------------------------- 核对数字
    print("\npanel order (a):", ", ".join(LABEL[k] for k in order_x))
    print("panel order (b):", ", ".join(LABEL[k] for k in order_s))
    print("\nratio range (a): %.2f – %.2f" % (rx.values.min(), rx.values.max()))
    print("ratio range (b): %.2f – %.2f" % (rs.values.min(), rs.values.max()))
    print("\nlakes better than baseline, per horizon (a):")
    print((rx < 1).groupby(level=1).sum().astype(int).to_string())
    print("\n(b):")
    print((rs < 1).groupby(level=1).sum().astype(int).to_string())


if __name__ == "__main__":
    main()
