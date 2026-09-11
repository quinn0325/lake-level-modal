"""Build tables, figures, appendices and datasets from downloaded Modal results."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from config import LAKES


REQUIRED_INPUTS = [
    ROOT / "results" / "embed_params_corrected.json",
    ROOT / "results" / "ccm_all_edges_merged_fdr.csv",
    ROOT / "results" / "connectivity_full_pairwise_ccm_results.csv",
    ROOT / "results" / "forecast_synchrony_filtered_full_results.csv",
    ROOT / "results" / "forecast_synchrony_filtered_rolling_results.csv",
    ROOT / "results" / "forecast_synchrony_filtered_dm_results.csv",
    ROOT / "results" / "forecast_synchrony_filtered_selected_lags.csv",
    ROOT / "results" / "xgboost_tuning_results.csv",
]

TABLE_SCRIPTS = [
    "build_ch4_tables.py",
    "build_supported_drivers.py",
    "table_4_1_mean_rmse.py",
]

FIGURE_SCRIPTS = [
    "figure_3_1_study_area.py",
    "figure_4_1_wl_variability.py",
    "figure_4_2_ccm_driver_matrix.py",
    "figure_4_3_interlake_network.py",
    "figure_B1_within_lake_network.py",
]

def require_inputs() -> None:
    """Stop before rendering if any required Modal output is missing."""
    missing = [
        str(path.relative_to(ROOT))
        for path in REQUIRED_INPUTS
        if not path.is_file()
    ]
    missing.extend(
        f"lake_pkls/{lake}_result.pkl"
        for lake in LAKES
        if not (ROOT / "lake_pkls" / f"{lake}_result.pkl").is_file()
    )
    if missing:
        raise SystemExit(
            "Missing Modal outputs. Download them first as described in README.md:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )


def run_script(path: Path) -> None:
    """Run one output script with an isolated Python process."""
    print(f"[postprocess] {path.relative_to(ROOT)}", flush=True)
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault(
        "MPLCONFIGDIR",
        str(ROOT / "results" / ".matplotlib_cache"),
    )
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        env=environment,
    )
    if completed.returncode:
        raise SystemExit(
            f"Post-processing failed: {path.relative_to(ROOT)} "
            f"(exit code {completed.returncode})"
        )


def main() -> int:
    require_inputs()

    for name in TABLE_SCRIPTS:
        run_script(CODE / "07_tables" / name)

    for name in FIGURE_SCRIPTS:
        run_script(CODE / "06_figures" / name)

    run_script(CODE / "07_tables" / "build_appendices.py")

    run_script(CODE / "08_dataset" / "build_public_dataset.py")
    print("[postprocess] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
