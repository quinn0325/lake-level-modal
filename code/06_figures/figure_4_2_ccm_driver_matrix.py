"""Render Figure 4.2 from the complete within-lake CCM results.

Supported driver-to-water-level relationships are coloured by cross-map skill
and labelled with the selected signed lag.
"""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
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

C_ZERO = "#B9785A"
C_NEG = "#777777"
CMAP = LinearSegmentedColormap.from_list(
    "rho", ["#F4F8FB", "#C5DFED", "#84BBD8", "#347EAF", "#124B7A"])
NORM = Normalize(vmin=0.0, vmax=1.0)
TXT_FLIP = 0.62

FS_TICK, FS_LAB, FS_TITLE, FS_IN, FS_LEG = 10.0, 11.0, 12.0, 9.5, 10.0
FS_HEAD = 10.0
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


def load_supported_edges():
    """Validate the 420-edge input and return supported driver-to-WL edges."""
    edges = pd.read_csv(TAB_DIR / "T5_within_lake_edges.csv")
    required = {
        "lake", "cause", "effect", "status", "obs_lag", "obs_rho",
        "convergence_diagnostic_pass", "statistically_significant",
    }
    missing_columns = required - set(edges.columns)
    if missing_columns:
        raise ValueError(f"Missing input columns: {sorted(missing_columns)}")

    variables = ["WL", *DRIVERS]
    expected = {
        (lake, cause, effect)
        for lake in LAKES
        for cause in variables
        for effect in variables
        if cause != effect
    }
    actual = set(edges[["lake", "cause", "effect"]].itertuples(
        index=False, name=None))
    if len(edges) != 420 or actual != expected:
        raise ValueError(
            f"Expected exactly 420 within-lake edges; found {len(edges)} rows "
            f"and {len(actual)} unique edges")
    if not edges["status"].eq("OK").all():
        raise ValueError("Within-lake input contains non-OK results")

    supported = edges[
        (edges["effect"] == "WL") & edges["statistically_significant"]
    ].copy()
    if not supported["convergence_diagnostic_pass"].all():
        raise ValueError("A supported edge failed the convergence diagnostic")
    if not np.isfinite(supported[["obs_lag", "obs_rho"]].to_numpy()).all():
        raise ValueError("Supported edges contain invalid lag or skill values")
    return supported


def lag_text(d):
    """Format a signed lag using the typographic minus sign."""
    return "0" if d == 0 else (f"+{d}" if d > 0 else f"−{abs(d)}")


def draw_matrix(ax, sup):
    ny, nx = len(LAKES), len(DRIVERS)
    for i in range(ny):
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
    """Draw the temporal-order key and cross-map skill scale."""
    axl.set_axis_off()
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
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
    print(f"Wrote {OUT_DIR / f'{stem}.svg'}")
    fig.savefig(OUT_DIR / f"{stem}.pdf", format="pdf",
                bbox_inches=None)
    print(f"Wrote {OUT_DIR / f'{stem}.pdf'}")
    fig.savefig(OUT_DIR / f"{stem}.png", format="png", dpi=600,
                bbox_inches=None)
    print(f"Wrote {OUT_DIR / f'{stem}.png'}")
    fig.savefig(OUT_DIR / f"{stem}.tiff", format="tiff", dpi=600,
                bbox_inches=None)
    print(f"Wrote {OUT_DIR / f'{stem}.tiff'}")
    plt.close(fig)


def main():
    supported = load_supported_edges()

    fig = plt.figure(figsize=(6.3, 5.30))
    gs = fig.add_gridspec(2, 1, height_ratios=[10.0, 2.15], hspace=0.0,
                          left=LEFT, right=RIGHT, top=0.874, bottom=0.058)
    draw_matrix(fig.add_subplot(gs[0]), supported)
    draw_key(fig.add_subplot(gs[1]))
    save(fig, "figure_4_2_ccm_driver_matrix")


if __name__ == "__main__":
    main()
