"""Check downloaded Modal outputs and deterministic local products."""

from __future__ import annotations

import csv
import filecmp
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "reference"
sys.path.insert(0, str(ROOT / "code"))

from config import LAKES

EXPECTED_CSV_ROWS = {
    "results/ccm_all_edges_merged_fdr.csv": 420,
    "results/connectivity_full_pairwise_ccm_results.csv": 90,
    "results/connectivity_connected_vs_unconnected_summary.csv": 2,
    "results/forecast_synchrony_filtered_full_results.csv": 122,
    "results/forecast_synchrony_filtered_rolling_results.csv": 372,
    "results/forecast_synchrony_filtered_dm_results.csv": 341,
    "results/forecast_synchrony_filtered_selected_lags.csv": 38,
    "results/xgboost_tuning_results.csv": 6,
}


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def record(ok: bool, message: str) -> bool:
    print(("[ok] " if ok else "[FAIL] ") + message)
    return ok


def compare_dir(label: str, generated: Path, reference: Path, pattern: str = "*.csv") -> bool:
    if not reference.is_dir():
        return record(True, f"{label}: no reference directory present")
    if not generated.is_dir():
        return record(False, f"{label}: generated directory missing")
    good = True
    reference_files = sorted(reference.glob(pattern))
    expected_names = {path.name for path in reference_files}
    generated_names = {path.name for path in generated.glob(pattern)}
    unexpected = sorted(generated_names - expected_names)
    if unexpected:
        good = record(False, f"{label}: unexpected files {unexpected}") and good
    for ref_file in reference_files:
        out_file = generated / ref_file.name
        if not out_file.is_file():
            good = record(False, f"{label}: missing {out_file.relative_to(ROOT)}") and good
        elif not filecmp.cmp(ref_file, out_file, shallow=False):
            good = record(False, f"{label}: differs {out_file.relative_to(ROOT)}") and good
    if good:
        record(True, f"{label}: matches reference")
    return good


def main() -> int:
    checks = []

    pkl_dir = ROOT / "lake_pkls"
    pkl_names = {p.name for p in pkl_dir.glob("*_result.pkl")} if pkl_dir.is_dir() else set()
    expected_pkls = {f"{lake}_result.pkl" for lake in LAKES}
    checks.append(record(pkl_names == expected_pkls, "downloaded lake_pkls contain all 10 lakes"))

    embed_path = ROOT / "results" / "embed_params_corrected.json"
    try:
        embed = json.loads(embed_path.read_text(encoding="utf-8"))
        checks.append(record(set(embed) == set(LAKES), "embedding JSON contains all 10 lakes"))
    except Exception as exc:
        checks.append(record(False, f"cannot read embedding JSON: {type(exc).__name__}: {exc}"))

    for rel, expected_rows in EXPECTED_CSV_ROWS.items():
        path = ROOT / rel
        if not path.is_file():
            checks.append(record(False, f"missing {rel}"))
            continue
        try:
            rows = row_count(path)
        except Exception as exc:
            checks.append(record(False, f"cannot read {rel}: {type(exc).__name__}: {exc}"))
            continue
        checks.append(record(rows == expected_rows, f"{rel}: {rows} rows"))

    tuning_path = ROOT / "results" / "xgboost_tuning_results.csv"
    if tuning_path.is_file():
        try:
            tuning_rows = read_csv_rows(tuning_path)
            required = {
                "max_depth",
                "learning_rate",
                "n_estimators",
                "cv_rmse_m",
                "selected",
                "n_lakes",
                "n_cv_folds",
            }
            columns = set(tuning_rows[0]) if tuning_rows else set()
            selected = sum(
                row.get("selected", "").strip().lower() in {"true", "1", "yes"}
                for row in tuning_rows
            )
            provenance = {
                (row.get("n_lakes"), row.get("n_cv_folds"))
                for row in tuning_rows
            }
            checks.append(record(required <= columns, "XGBoost tuning columns are complete"))
            checks.append(record(selected == 1, "XGBoost tuning selects one configuration"))
            checks.append(record(provenance == {("10", "30")}, "XGBoost tuning used 10 lakes and 30 folds"))
        except Exception as exc:
            checks.append(
                record(
                    False,
                    f"cannot validate XGBoost tuning: {type(exc).__name__}: {exc}",
                )
            )

    checks.append(
        compare_dir(
            "chapter tables",
            ROOT / "ch4_tables",
            REF / "tables",
            "T4_1*.csv",
        )
    )
    checks.append(compare_dir("appendices", ROOT / "appendices", REF / "appendices"))
    checks.append(compare_dir("dataset", ROOT / "dataset", REF / "dataset"))

    expected_figures = {
        path.name for path in (REF / "figures").glob("figure_*.png")
    }
    generated_figures = {
        path.name for path in (ROOT / "results" / "figures").glob("figure_*.png")
    }
    checks.append(
        record(
            generated_figures == expected_figures,
            "rendered figure names match Figures 3.1, 4.1–4.4 and B1",
        )
    )

    if all(checks):
        print("\nModal output checks passed.")
        return 0
    print("\nModal output checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
