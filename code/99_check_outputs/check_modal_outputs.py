"""Check downloaded Modal outputs and deterministic local products.
检查已下载的 Modal 输出及本地确定性生成的成果。

This script does not rerun CCM or forecasting. It verifies that the outputs
downloaded from Modal have the expected shapes and, when reference files are
present, that deterministic post-processing products match the reference files.
"""

from __future__ import annotations

import csv
import filecmp
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "reference"

LAKES = [
    "Kalamalka_Lake",
    "Okanagan_Lake",
    "Skaha_Lake",
    "Vaseux_Lake",
    "Rainy_Lake",
    "Lake_of_the_Woods",
    "Playgreen_Lake",
    "Kiskitto_Lake",
    "Sipiwesk_Lake",
    "Split_Lake",
]

EXPECTED_CSV_ROWS = {
    "results/ccm_all_edges_merged_fdr.csv": 420,
    "results/connectivity_full_pairwise_ccm_results.csv": 90,
    "results/connectivity_connected_vs_unconnected_summary.csv": 2,
    "results/forecast_synchrony_filtered_full_results.csv": 122,
    "results/forecast_synchrony_filtered_rolling_results.csv": 372,
    "results/forecast_synchrony_filtered_dm_results.csv": 341,
    "results/forecast_synchrony_filtered_selected_lags.csv": 38,
}


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def record(ok: bool, message: str) -> bool:
    print(("[ok] " if ok else "[FAIL] ") + message)
    return ok


def compare_dir(label: str, generated: Path, reference: Path, pattern: str = "*.csv") -> bool:
    if not reference.is_dir():
        return record(True, f"{label}: no reference directory present")
    if not generated.is_dir():
        return record(False, f"{label}: generated directory missing")
    good = True
    for ref_file in sorted(reference.glob(pattern)):
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

    checks.append(compare_dir("appendices", ROOT / "appendices", REF / "appendices"))
    checks.append(compare_dir("dataset", ROOT / "dataset", REF / "dataset"))

    if all(checks):
        print("\nModal output checks passed.")
        return 0
    print("\nModal output checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
