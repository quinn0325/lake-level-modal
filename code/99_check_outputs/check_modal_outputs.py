"""Check the completeness and structure of reproduced analysis outputs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from config import LAKES, VARIABLES


FORECAST_FILES = {
    "results/forecast_synchrony_filtered_full_results.csv": {
        "lake", "method", "rmse", "mae", "nse", "n_eval", "status",
    },
    "results/forecast_synchrony_filtered_rolling_results.csv": {
        "lake", "method", "horizon_months", "rmse", "mae", "nse",
        "n_origins",
    },
    "results/forecast_synchrony_filtered_dm_results.csv": {
        "lake", "method_1", "method_2", "horizon_months", "dm_stat",
        "p_value", "n", "p_fdr",
    },
    "results/forecast_synchrony_filtered_selected_lags.csv": {
        "lake", "method", "selected_vars", "selected_lags",
    },
}

LOCAL_OUTPUTS = {
    "chapter tables": [
        "ch4_tables/T1_rolling_lake_horizon_method.csv",
        "ch4_tables/T3_dm_bh.csv",
        "ch4_tables/T4_single_split_full.csv",
        "ch4_tables/T4_1_mean_rmse.csv",
        "ch4_tables/T4_1_mean_rmse.md",
        "ch4_tables/T5_within_lake_edges.csv",
        "ch4_tables/T6_between_lake_edges.csv",
        "ch4_tables/T7_lake_pair_strength.csv",
        "ch4_tables/supported_drivers.csv",
    ],
    "figures": [
        f"results/figures/{stem}.{suffix}"
        for stem in (
            "figure_3_1_study_area",
            "figure_4_1_wl_variability",
            "figure_4_2_ccm_driver_matrix",
            "figure_4_3_interlake_network",
            "figure_4_4_pair_strength",
            "figure_B1_within_lake_network",
        )
        for suffix in ("pdf", "png")
    ],
    "appendices": [
        "appendices/Appendices.md",
        *[
            f"appendices/{stem}.{suffix}"
            for stem in (
                "A1_lakes_and_stations",
                "A2_flagged_outliers",
                "B1_embedding_parameters",
                "B2_data_availability",
                "B3_supported_drivers",
                "B4_supported_between_lake_edges",
                "B5_lake_pair_strength",
                "C1_selected_predictors",
                "C2_model_configurations",
                "C3_xgboost_hyperparameters",
                "C4_dm_summary",
            )
            for suffix in ("csv", "md")
        ],
    ],
    "public dataset": [
        "dataset/raw_station_water_level.csv",
        "dataset/lake_monthly.csv",
        "dataset/lake_monthly_deseasonalised.csv",
        "dataset/data_dictionary.csv",
    ],
}


def record(ok: bool, message: str) -> bool:
    """Print and return one check result."""
    print(("[ok] " if ok else "[FAIL] ") + message)
    return ok


def read_csv(relative_path: str) -> tuple[list[dict[str, str]], set[str]] | None:
    """Read a CSV and report a clear failure when it is missing or invalid."""
    path = ROOT / relative_path
    if not path.is_file():
        record(False, f"missing {relative_path}")
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = set(reader.fieldnames or [])
    except Exception as exc:
        record(False, f"cannot read {relative_path}: {type(exc).__name__}: {exc}")
        return None
    return rows, columns


def check_local_files(label: str, relative_paths: list[str]) -> bool:
    """Check that a group of deterministic local products exists and is non-empty."""
    missing = [
        name
        for name in relative_paths
        if not (ROOT / name).is_file() or (ROOT / name).stat().st_size == 0
    ]
    if missing:
        return record(False, f"{label}: missing or empty {missing}")
    return record(True, f"{label}: all expected files are present")


def main() -> int:
    checks: list[bool] = []
    expected_lakes = set(LAKES)
    expected_variables = set(VARIABLES)

    pkl_dir = ROOT / "lake_pkls"
    pkl_names = {
        path.name for path in pkl_dir.glob("*_result.pkl")
    } if pkl_dir.is_dir() else set()
    expected_pkls = {f"{lake}_result.pkl" for lake in LAKES}
    checks.append(record(
        pkl_names == expected_pkls,
        "lake panels contain exactly the 10 study lakes",
    ))

    embed_path = ROOT / "results" / "embed_params_corrected.json"
    try:
        embed = json.loads(embed_path.read_text(encoding="utf-8"))
        correct_lakes = set(embed) == expected_lakes
        correct_variables = correct_lakes and all(
            set(embed[lake]) == expected_variables for lake in LAKES
        )
        valid_parameters = correct_variables and all(
            2 <= int(embed[lake][variable]["E"]) <= 10
            and int(embed[lake][variable]["tau"]) == 1
            for lake in LAKES
            for variable in VARIABLES
        )
        checks.append(record(
            bool(valid_parameters),
            "embedding JSON covers 10 lakes x 7 variables with valid E and tau",
        ))
    except Exception as exc:
        checks.append(record(
            False,
            f"cannot validate embedding JSON: {type(exc).__name__}: {exc}",
        ))

    within_rel = "results/ccm_all_edges_merged_fdr.csv"
    within_data = read_csv(within_rel)
    if within_data is None:
        checks.append(False)
    else:
        rows, columns = within_data
        required = {"lake", "cause", "effect", "status"}
        expected_edges = {
            (lake, cause, effect)
            for lake in LAKES
            for cause in VARIABLES
            for effect in VARIABLES
            if cause != effect
        }
        actual_edges = {
            (row.get("lake"), row.get("cause"), row.get("effect"))
            for row in rows
        }
        valid = (
            required <= columns
            and len(rows) == 420
            and actual_edges == expected_edges
            and all(row.get("status") == "OK" for row in rows)
        )
        checks.append(record(
            valid,
            f"within-lake CCM contains 420 unique expected edges with status=OK ({len(rows)} rows)",
        ))

    inter_rel = "results/connectivity_full_pairwise_ccm_results.csv"
    inter_data = read_csv(inter_rel)
    if inter_data is None:
        checks.append(False)
    else:
        rows, columns = inter_data
        required = {"cause_lake", "effect_lake", "status"}
        expected_edges = {
            (cause_lake, effect_lake)
            for cause_lake in LAKES
            for effect_lake in LAKES
            if cause_lake != effect_lake
        }
        actual_edges = {
            (row.get("cause_lake"), row.get("effect_lake"))
            for row in rows
        }
        valid = (
            required <= columns
            and len(rows) == 90
            and actual_edges == expected_edges
            and all(row.get("status") == "OK" for row in rows)
        )
        checks.append(record(
            valid,
            f"between-lake CCM contains 90 unique expected edges with status=OK ({len(rows)} rows)",
        ))

    summary_rel = "results/connectivity_connected_vs_unconnected_summary.csv"
    summary_data = read_csv(summary_rel)
    if summary_data is None:
        checks.append(False)
    else:
        rows, columns = summary_data
        required = {
            "unit", "n_connected", "n_unconnected", "mannwhitney_p",
            "effect_size_auc",
        }
        checks.append(record(
            len(rows) == 2 and required <= columns,
            f"connectivity summary has the two expected analysis units ({len(rows)} rows)",
        ))

    for relative_path, required_columns in FORECAST_FILES.items():
        data = read_csv(relative_path)
        if data is None:
            checks.append(False)
            continue
        rows, columns = data
        lakes = {row.get("lake") for row in rows}
        valid = bool(rows) and required_columns <= columns and lakes == expected_lakes
        if "horizon_months" in required_columns:
            try:
                horizons = {int(row["horizon_months"]) for row in rows}
            except (KeyError, TypeError, ValueError):
                horizons = set()
            valid = valid and horizons == {1, 3, 6, 12}
        checks.append(record(
            valid,
            f"{relative_path}: valid structure and all 10 lakes ({len(rows)} rows)",
        ))

    tuning_rel = "results/xgboost_tuning_results.csv"
    tuning_data = read_csv(tuning_rel)
    if tuning_data is None:
        checks.append(False)
    else:
        rows, columns = tuning_data
        required = {
            "max_depth", "learning_rate", "n_estimators", "cv_rmse_m",
            "selected", "n_lakes", "n_cv_folds",
        }
        selected = sum(
            row.get("selected", "").strip().lower() in {"true", "1", "yes"}
            for row in rows
        )
        provenance = {
            (row.get("n_lakes"), row.get("n_cv_folds")) for row in rows
        }
        valid = (
            len(rows) == 6
            and required <= columns
            and selected == 1
            and provenance == {("10", "30")}
        )
        checks.append(record(
            valid,
            "XGBoost tuning contains 6 candidates, one selection, 10 lakes and 30 folds",
        ))

    for label, relative_paths in LOCAL_OUTPUTS.items():
        checks.append(check_local_files(label, relative_paths))

    if all(checks):
        print("\nOutput checks passed.")
        return 0
    print("\nOutput checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
