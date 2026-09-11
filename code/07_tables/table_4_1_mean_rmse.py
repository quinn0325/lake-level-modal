"""Build Table 4.1 from matched forecast samples within each model family."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from config import LAKES, ROLLING_HORIZONS  # noqa: E402

TAB_DIR = ROOT / "ch4_tables"
HORIZONS = ROLLING_HORIZONS

BLOCKS = [
    ("SARIMA(X)", [("SARIMA", "No exogenous"),
                   ("SARIMAX_all_vars", "All vars"),
                   ("SARIMAX_Stepwise", "Stepwise"),
                   ("SARIMAX_CCM_direct", "CCM direct"),
                   ("SARIMAX_CCM_ancestors", "CCM ancestors")]),
    ("XGBoost", [("XGBoost_AR_only", "AR-only"),
                 ("XGBoost_all_vars", "All vars"),
                 ("XGBoost_Stepwise", "Stepwise"),
                 ("XGBoost_CCM_direct", "CCM direct"),
                 ("XGBoost_CCM_ancestors", "CCM ancestors")]),
]
EXPECTED_METHODS = {
    "Persistence",
    *(key for _, methods in BLOCKS for key, _ in methods),
    "SARIMAX_CCM_neighbor",
    "XGBoost_CCM_neighbor",
}
REQUIRED_COLUMNS = {"lake", "method", "horizon_months", "rmse", "n_origins"}


def load_valid_results():
    """Load and validate rolling-origin forecast results."""
    results = pd.read_csv(TAB_DIR / "T1_rolling_lake_horizon_method.csv")
    missing = REQUIRED_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(f"Missing rolling-result columns: {sorted(missing)}")
    if results.duplicated(["lake", "method", "horizon_months"]).any():
        raise ValueError("Rolling results contain duplicate lake-method-horizon rows")
    if set(results.lake) != set(LAKES):
        raise ValueError("Rolling results do not cover the configured ten lakes")
    if set(results.method) != EXPECTED_METHODS:
        raise ValueError("Rolling results do not contain the expected methods")
    if set(results.horizon_months) != set(HORIZONS):
        raise ValueError("Rolling results do not cover the configured horizons")

    valid = results[results.n_origins > 0].copy()
    if valid.rmse.isna().any() or not np.isfinite(valid.rmse).all():
        raise ValueError("Valid rolling results contain a non-finite RMSE")
    if (valid.rmse < 0).any():
        raise ValueError("Valid rolling results contain a negative RMSE")
    return valid


def matched_lakes(results, method_keys):
    """Return lakes available for every method and horizon in a block."""
    sets = [
        set(
            results[
                (results.method == method) & (results.horizon_months == horizon)
            ].lake
        )
        for method in method_keys
        for horizon in HORIZONS
    ]
    return sorted(set.intersection(*sets))


def main():
    valid = load_valid_results()

    rows, md_rows = [], []
    for block, methods in BLOCKS:
        lakes = matched_lakes(valid, [key for key, _ in methods])
        if not lakes:
            raise ValueError(f"No complete matched sample is available for {block}")
        md_rows.append(f"| **{block}** (matched on {len(lakes)} lakes) | | | | |")
        for key, name in methods + [("Persistence", "Persistence")]:
            subset = valid[(valid.method == key) & valid.lake.isin(lakes)]
            rec = {
                "model_family": block,
                "n_lakes": len(lakes),
                "configuration": name,
            }
            cells = []
            for h in HORIZONS:
                horizon = subset[subset.horizon_months == h]
                if set(horizon.lake) != set(lakes):
                    raise ValueError(f"Incomplete {block} data for {key} at horizon {h}")
                val = float(horizon.rmse.mean())
                rec[f"rmse_h{h}"] = round(val, 3)
                cells.append(f"{val:.3f}")
            rows.append(rec)
            md_rows.append(f"| {name} | " + " | ".join(cells) + " |")

    out = pd.DataFrame(rows)
    csv_path = TAB_DIR / "T4_1_mean_rmse.csv"
    out.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    header = ("| Configuration | " + " | ".join(f"h = {h}" for h in HORIZONS)
              + " |\n| --- | " + " | ".join(["---:"] * len(HORIZONS)) + " |")
    md = "\n".join([
        "**Table 4.1. Mean RMSE (m) by forecasting configuration.** Rows are "
        "comparable within a model-family block but not between blocks; "
        "persistence is recomputed on each block's matched lakes.",
        "",
        header,
        *md_rows,
    ])
    md_path = TAB_DIR / "T4_1_mean_rmse.md"
    md_path.write_text(md + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
