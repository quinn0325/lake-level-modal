"""Render Figure 4.1 from cleaned monthly water-level observations.

Each series is centred on its long-term mean without standardisation or
deseasonalisation. Gaps of at least two consecutive months are shaded.
"""
from pathlib import Path
import pickle
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR / "01_analysis_core"))
import analysis_core as p                                        # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"
PKL_DIR = Path(p.PKL_DIR)

SYSTEMS = [
    ("Okanagan", ["Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake"]),
    ("Nelson–Winnipeg", ["Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
                         "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake"]),
]
LAKES = [lk for _, g in SYSTEMS for lk in g]
SYSTEM_OF = {lk: n for n, g in SYSTEMS for lk in g}
COLOUR = {"Okanagan": "#0072B2", "Nelson–Winnipeg": "#D55E00"}
LABEL = {lk: lk.replace("_Lake", "").replace("_", " ") for lk in LAKES}
LABEL["Lake_of_the_Woods"] = "Lake of the Woods"

FS_TICK, FS_LABEL, FS_LAKE, FS_TITLE = 10.0, 11.0, 10.5, 12.0
BAND_GREY = "0.45"
BAND_ALPHA = 0.07

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.7, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "pdf.fonttype": 42,
})


def centered_series(lake):
    """Return one cleaned water-level series centred on its mean."""
    with open(PKL_DIR / f"{lake}_result.pkl", "rb") as fh:
        cached = pickle.load(fh)
    wl = p.combine_station_water_levels(p.clean_wide_wl(cached["wide_wl"]))
    wl = wl.sort_index().asfreq("MS")
    return wl - wl.mean()


def gap_spans(s, min_len=2):
    """Return spans containing at least min_len consecutive missing months."""
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

    fig_h = 9.3
    bottom = 0.50 / fig_h

    fig = plt.figure(figsize=(6.3, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 3.6], hspace=0.30,
                          left=0.238, right=0.978, top=1 - 0.42 / fig_h,
                          bottom=bottom)
    gs_a = gs[0].subgridspec(10, 1, hspace=0.0)

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
        if lake == "Rainy_Lake":
            ax.spines["top"].set_visible(True)
            ax.spines["top"].set_linewidth(1.1)
    pa_top, pa_bot = gs[0].get_position(fig).y1, gs[0].get_position(fig).y0
    fig.text(0.058, (pa_top + pa_bot) / 2, "Centred water level (m)",
             rotation=90, va="center", ha="center", fontsize=FS_LABEL)

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, dpi in ((OUT_DIR / "figure_4_1_wl_variability.pdf", 300),
                      (OUT_DIR / "figure_4_1_wl_variability.png", 400)):
        fig.savefig(path, dpi=dpi)
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
