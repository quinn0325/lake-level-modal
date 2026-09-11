"""Derive the chapter-table inputs from downloaded Modal results.

This deterministic local step does not rerun CCM or forecasting.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import config

ROOT = CODE.parent
RESULTS = ROOT / "results"
OUT = ROOT / "ch4_tables"

SYSTEM_OF = {lk: ("Okanagan" if lk in config.LAKES[:4] else "Nelson-Winnipeg")
             for lk in config.LAKES}


def waterway_components():
    """Map each lake to its connected component in the waterway network."""
    comp, nxt = {}, 0
    adj = {lk: set() for lk in config.LAKES}
    for a, b in config.WATERWAY_CONNECTED_PAIRS:
        adj[a].add(b)
        adj[b].add(a)
    for lk in config.LAKES:
        if lk in comp:
            continue
        stack, nxt = [lk], nxt + 1
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp[cur] = nxt
            stack.extend(adj[cur] - set(comp))
    return comp


COMP = waterway_components()
DIRECT = {tuple(sorted(pair)) for pair in config.WATERWAY_CONNECTED_PAIRS}


def tier_of(a, b):
    """Return the descriptive hydrological-separation tier for a lake pair."""
    if tuple(sorted([a, b])) in DIRECT:
        return "1_direct"
    if COMP[a] == COMP[b]:
        return "2_same_subsystem_indirect"
    if SYSTEM_OF[a] == SYSTEM_OF[b]:
        return "3_same_basin_diff_subsystem"
    return "4_different_basin"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Reorder rolling-result columns and retain the complete full-result table.
    roll = pd.read_csv(RESULTS / "forecast_synchrony_filtered_rolling_results.csv")
    roll = roll[["lake", "horizon_months", "method", "rmse", "mae", "nse",
                 "n_origins", "min_exog_lag", "requires_foresight"]]
    roll.to_csv(OUT / "T1_rolling_lake_horizon_method.csv", index=False)

    full = pd.read_csv(RESULTS / "forecast_synchrony_filtered_full_results.csv")
    full.to_csv(OUT / "T4_single_split_full.csv", index=False)

    # A negative DM statistic favours method 1.
    dm = pd.read_csv(RESULTS / "forecast_synchrony_filtered_dm_results.csv")
    dm["sig_fdr"] = dm["p_fdr"] < config.FDR_ALPHA
    dm["winner"] = np.where(~dm["sig_fdr"], "",
                            np.where(dm["dm_stat"] < 0, dm["method_1"], dm["method_2"]))
    dm.to_csv(OUT / "T3_dm_bh.csv", index=False)

    # Add the effect variable's embedding parameters to each within-lake edge.
    emb = json.loads(
        (RESULTS / "embed_params_corrected.json").read_text(encoding="utf-8")
    )
    w = pd.read_csv(RESULTS / "ccm_all_edges_merged_fdr.csv")
    w["E_effect"] = [emb[r.lake][r.effect]["E"] for r in w.itertuples()]
    w["tau"] = [emb[r.lake][r.effect]["tau"] for r in w.itertuples()]
    w = w[["lake", "cause", "effect", "status", "obs_lag", "obs_rho", "obs_n",
           "E_effect", "tau", "p_value", "p_fdr", "kendall_tau", "kendall_p",
           "convergence_diagnostic_pass", "statistically_significant",
           "lag_resolution", "causal_evidence", "n_valid_surrogates"]]
    w.to_csv(OUT / "T5_within_lake_edges.csv", index=False)

    # Add a descriptive hydrological-separation tier to each between-lake edge.
    b = pd.read_csv(RESULTS / "connectivity_full_pairwise_ccm_results.csv")
    b["tier"] = [tier_of(r.cause_lake, r.effect_lake) for r in b.itertuples()]
    b.to_csv(OUT / "T6_between_lake_edges.csv", index=False)

    # Pair strength is mean directional |rho|; either causal direction marks detection.
    b["key"] = [tuple(sorted([r.cause_lake, r.effect_lake])) for r in b.itertuples()]
    b["ok"] = b["causal_evidence"]
    rows = []
    for (a, c), g in b.groupby("key"):
        rows.append({"pair": f"{a}|{c}",
                     "S_ij": g.obs_rho.abs().mean(),
                     "detected": bool(g.ok.any()),
                     "tier": tier_of(a, c),
                     "directly_connected": tuple(sorted([a, c])) in DIRECT})
    t7 = pd.DataFrame(rows).sort_values(["tier", "S_ij"], ascending=[True, False])
    t7.to_csv(OUT / "T7_lake_pair_strength.csv", index=False)

    for name, n in [("T1", len(roll)), ("T3", len(dm)), ("T4", len(full)),
                    ("T5", len(w)), ("T6", len(b)), ("T7", len(t7))]:
        print(f"  {name}: {n} rows")
    print(f"Wrote chapter-table inputs to {OUT}")


if __name__ == "__main__":
    main()
