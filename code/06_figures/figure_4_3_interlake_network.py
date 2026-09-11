"""Render Figures 4.3 and 4.4 from the complete between-lake CCM results.


"""
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from config import LAKES, WATERWAY_CONNECTED_PAIRS                 # noqa: E402

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

TIERS = ["1_direct", "2_same_subsystem_indirect",
         "3_same_basin_diff_subsystem", "4_different_basin"]
TIER_LABEL = {"1_direct": "Direct connection",
              "2_same_subsystem_indirect": "Same subsystem, indirect",
              "3_same_basin_diff_subsystem": "Same basin, other subsystem",
              "4_different_basin": "Different basin"}

CMAP = LinearSegmentedColormap.from_list(
    "rho", ["#9CC3E0", "#6BAED6", "#3C8DC4", "#2171B5", "#08306B"])
NORM = Normalize(vmin=0.15, vmax=1.0)
RHO_LO, RHO_HI = 0.35, 0.95

TIER_EMPH = {"1_direct": (1.00, 1.00), "2_same_subsystem_indirect": (0.95, 0.92),
             "3_same_basin_diff_subsystem": (0.84, 0.84),
             "4_different_basin": (0.72, 0.76)}
WATERWAY_GREY = "0.76"

FS_TICK, FS_LAB, FS_TITLE, FS_NODE, FS_LEG = 10.0, 11.0, 12.0, 9.5, 10.0
FS_ANN = 9.0
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


def load_inputs():
    """Validate and return the 90 directed edges and 45 lake-pair summaries."""
    edges = pd.read_csv(TAB_DIR / "T6_between_lake_edges.csv")
    edge_columns = {
        "cause_lake", "effect_lake", "status", "obs_lag", "obs_rho", "tier",
        "convergence_diagnostic_pass", "statistically_significant",
    }
    missing = edge_columns - set(edges.columns)
    if missing:
        raise ValueError(f"Missing between-lake columns: {sorted(missing)}")

    expected_edges = {
        (cause, effect)
        for cause in LAKES
        for effect in LAKES
        if cause != effect
    }
    actual_edges = set(edges[["cause_lake", "effect_lake"]].itertuples(
        index=False, name=None))
    if len(edges) != 90 or actual_edges != expected_edges:
        raise ValueError(
            f"Expected exactly 90 between-lake edges; found {len(edges)} rows "
            f"and {len(actual_edges)} unique edges")
    if not edges["status"].eq("OK").all():
        raise ValueError("Between-lake input contains non-OK results")
    if not set(edges["tier"]).issubset(TIERS):
        raise ValueError("Between-lake input contains an unknown tier")

    supported = edges[edges["statistically_significant"]].copy()
    if not supported["convergence_diagnostic_pass"].all():
        raise ValueError("A supported edge failed the convergence diagnostic")
    if not np.isfinite(supported[["obs_lag", "obs_rho"]].to_numpy()).all():
        raise ValueError("Supported edges contain invalid lag or skill values")

    pairs = pd.read_csv(TAB_DIR / "T7_lake_pair_strength.csv")
    pair_columns = {"pair", "S_ij", "tier", "directly_connected"}
    missing = pair_columns - set(pairs.columns)
    if missing:
        raise ValueError(f"Missing lake-pair columns: {sorted(missing)}")

    expected_pairs = {
        "|".join(sorted((first, second)))
        for index, first in enumerate(LAKES)
        for second in LAKES[index + 1:]
    }
    actual_pairs = set(pairs["pair"])
    if len(pairs) != 45 or actual_pairs != expected_pairs:
        raise ValueError(
            f"Expected exactly 45 lake pairs; found {len(pairs)} rows "
            f"and {len(actual_pairs)} unique pairs")
    if not set(pairs["tier"]).issubset(TIERS):
        raise ValueError("Lake-pair input contains an unknown tier")
    if not np.isfinite(pairs["S_ij"]).all() or not pairs["S_ij"].between(0, 1).all():
        raise ValueError("Lake-pair strengths must be finite values from zero to one")

    direct_pairs = {"|".join(sorted(pair)) for pair in WATERWAY_CONNECTED_PAIRS}
    reported_direct = set(pairs.loc[pairs["directly_connected"], "pair"])
    if reported_direct != direct_pairs:
        raise ValueError("Directly connected lake pairs do not match config.py")
    return supported, pairs


def edge_lw(rho):
    """Map cross-map skill to a visible printed line width."""
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

    for a, b in WATERWAY_CONNECTED_PAIRS:
        (xa, ya), (xb, yb) = POS[a], POS[b]
        ax.plot([xa, xb], [ya, yb], color=WATERWAY_GREY, lw=6.0, zorder=1,
                solid_capstyle="round")

    weak = sup[sup.obs_lag <= 0]
    for r in weak.itertuples():
        (xa, ya), (xb, yb) = POS[r.cause_lake], POS[r.effect_lake]
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-",
                                     connectionstyle="arc3,rad=0.16",
                                     color="0.42", lw=1.3, ls=(0, (3.5, 2)),
                                     shrinkA=10, shrinkB=10, zorder=2))

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

    mpl_version = tuple(int(value) for value in mpl.__version__.split(".")[:2])
    orientation = ({"orientation": "horizontal"} if mpl_version >= (3, 11)
                   else {"vert": False})
    bp = axb.boxplot(data, **orientation, widths=0.52, whis=1.5,
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

    direct = t7.loc[t7["directly_connected"], "S_ij"].to_numpy()
    other = t7.loc[~t7["directly_connected"], "S_ij"].to_numpy()
    test = mannwhitneyu(direct, other, alternative="two-sided", method="exact")
    auc = test.statistic / (len(direct) * len(other))
    axb.text(0.0, -0.34, "Pre-specified test, direct vs all other pairs:",
             fontsize=FS_ANN, ha="left", va="center", color="0.20")
    axb.text(0.0, 0.02,
             f"Mann–Whitney $U$ = {test.statistic:.0f}, "
             f"$p$ = {test.pvalue:.3f}, AUC = {auc:.3f}",
             fontsize=FS_ANN, ha="left", va="center", color="0.20")
    xb = 1.035
    axb.plot([xb, xb + 0.022, xb + 0.022, xb], [1.62, 1.62, 4.38, 4.38],
             color="0.45", lw=0.8, clip_on=False)
    axb.text(xb + 0.052, 3.00,
             f"pooled as \"other pairs\" ($n$ = {len(other)})",
             rotation=90, ha="center", va="center", fontsize=8.5, color="0.35",
             clip_on=False)


def save(fig, stem):
    """Save one figure in the publication and preview formats."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension, options in (
        ("svg", {}),
        ("pdf", {"dpi": 300}),
        ("png", {"dpi": 400}),
        ("tiff", {"dpi": 600}),
    ):
        path = OUT_DIR / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", **options)
        print(f"Wrote {path}")
    plt.close(fig)


def main():
    supported, pairs = load_inputs()

    fig = plt.figure(figsize=(6.3, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[10.0, 1.9], hspace=0.0,
                          left=0.020, right=0.992, top=0.988, bottom=0.030)
    draw_network(fig.add_subplot(gs[0]), supported)
    draw_key(fig.add_subplot(gs[1]))
    save(fig, "figure_4_3_interlake_network")

    fig = plt.figure(figsize=(6.3, 3.25))
    ax = fig.add_axes([0.335, 0.185, 0.525, 0.760])
    draw_strength(ax, pairs)
    save(fig, "figure_4_4_pair_strength")


if __name__ == "__main__":
    main()
