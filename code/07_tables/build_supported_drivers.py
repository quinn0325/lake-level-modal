"""Build the appendix input table of CCM-supported water-level drivers."""

import sys
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
TABLE_DIR = ROOT / "ch4_tables"
sys.path.insert(0, str(CODE_DIR))

from config import LAKES, VARIABLES  # noqa: E402


BASIN_OF = {
    lake: "Okanagan" if lake in LAKES[:4] else "Nelson–Winnipeg"
    for lake in LAKES
}
DISPLAY_NAME = {lake: lake.replace("_Lake", "").replace("_", " ") for lake in LAKES}
DISPLAY_NAME["Lake_of_the_Woods"] = "Lake of the Woods"

DRIVER_ORDER = ["RegFlow", "R", "P", "Evap", "SWE", "T"]
DRIVER_NAME = {
    "RegFlow": "Regulated flow",
    "R": "Runoff",
    "P": "Precipitation",
    "Evap": "Evaporation",
    "SWE": "SWE",
    "T": "Temperature",
}
TEMPORAL_CLASS = {
    "resolved": "Positive",
    "unresolved_contemporaneous": "Contemporaneous",
    "rejected_reverse": "Negative",
}


def as_bool(series: pd.Series) -> pd.Series:
    """Normalize a stored Boolean column."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    """Reject an input table that lacks required columns."""
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Within-lake table is missing columns: {', '.join(missing)}")


def main() -> None:
    source = TABLE_DIR / "T5_within_lake_edges.csv"
    edges = pd.read_csv(source)
    require_columns(
        edges,
        {
            "lake",
            "cause",
            "effect",
            "status",
            "obs_lag",
            "obs_rho",
            "obs_n",
            "E_effect",
            "p_fdr",
            "kendall_tau",
            "kendall_p",
            "statistically_significant",
            "lag_resolution",
            "causal_evidence",
        },
    )

    expected_edges = {
        (lake, cause, effect)
        for lake in LAKES
        for cause in VARIABLES
        for effect in VARIABLES
        if cause != effect
    }
    actual_edges = set(
        edges[["lake", "cause", "effect"]].itertuples(index=False, name=None)
    )
    if len(edges) != 420 or actual_edges != expected_edges:
        raise ValueError("Within-lake table must contain exactly the expected 420 edges")
    if not edges["status"].eq("OK").all():
        raise ValueError("Within-lake table contains one or more non-OK results")
    if set(DRIVER_ORDER) != set(VARIABLES) - {"WL"}:
        raise ValueError("Driver order does not match config.VARIABLES")

    supported = edges[
        edges["effect"].eq("WL") & as_bool(edges["statistically_significant"])
    ].copy()
    supported["_lake_order"] = supported["lake"].map(
        {lake: index for index, lake in enumerate(LAKES)}
    )
    supported["_driver_order"] = supported["cause"].map(
        {driver: index for index, driver in enumerate(DRIVER_ORDER)}
    )
    if supported[["_lake_order", "_driver_order"]].isna().any().any():
        raise ValueError("Supported-driver table contains an unknown lake or variable")
    supported = supported.sort_values(["_lake_order", "_driver_order"])

    output = pd.DataFrame({
        "basin": supported["lake"].map(BASIN_OF),
        "lake": supported["lake"].map(DISPLAY_NAME),
        "driver": supported["cause"].map(DRIVER_NAME),
        "rho": supported["obs_rho"].round(3),
        "lag_d_months": supported["obs_lag"].astype(int),
        "p_fdr": supported["p_fdr"].round(4),
        "kendall_tau_L": supported["kendall_tau"].round(2),
        "kendall_p": supported["kendall_p"].map(lambda value: f"{value:.1e}"),
        "n_months": supported["obs_n"].astype(int),
        "E": supported["E_effect"].astype(int),
        "temporal_class": supported["lag_resolution"].map(TEMPORAL_CLASS),
        "causal_evidence": as_bool(supported["causal_evidence"]).map(
            {True: "Yes", False: "No"}
        ),
    })
    if output.isna().any().any():
        raise ValueError("Supported-driver output contains an unmapped or missing value")

    output_path = TABLE_DIR / "supported_drivers.csv"
    output.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(output)} rows)")


if __name__ == "__main__":
    main()
