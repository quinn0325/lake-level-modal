"""Build local tables, figures and dataset files from downloaded Modal outputs.
根据已下载的 Modal 输出在本地生成表格、图形和数据集。

This is the only local post-processing entry point. It assumes that Modal has
already produced the analysis outputs and that they have been downloaded into
`results/` and `lake_pkls/`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"

for rel in ("", "01_analysis_core"):
    sys.path.insert(0, str(CODE / rel) if rel else str(CODE))


REQUIRED_INPUTS = [
    ROOT / "results" / "embed_params_corrected.json",
    ROOT / "results" / "ccm_all_edges_merged_fdr.csv",
    ROOT / "results" / "connectivity_full_pairwise_ccm_results.csv",
    ROOT / "results" / "connectivity_connected_vs_unconnected_summary.csv",
    ROOT / "results" / "forecast_synchrony_filtered_full_results.csv",
    ROOT / "results" / "forecast_synchrony_filtered_rolling_results.csv",
    ROOT / "results" / "forecast_synchrony_filtered_dm_results.csv",
    ROOT / "results" / "forecast_synchrony_filtered_selected_lags.csv",
]

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


def require_inputs() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_INPUTS if not path.is_file()]
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
    print(f"[postprocess] {path.relative_to(ROOT)}", flush=True)
    runpy.run_path(str(path), run_name="__main__")


def main() -> int:
    require_inputs()

    table_order = [
        "build_ch4_tables.py",
        "table_C1_supported_drivers.py",
        "table_4_2_dm_maintext.py",
        "table_4_4_forecast_summary.py",
        "table_4_4_matched_rmse.py",
    ]
    for name in table_order:
        run_script(CODE / "07_tables" / name)

    for script in sorted((CODE / "06_figures").glob("figure_*.py")):
        run_script(script)

    for name in ("build_appendices.py", "build_appendices_en.py"):
        run_script(CODE / "07_tables" / name)

    run_script(CODE / "08_dataset" / "build_public_dataset.py")
    print("[postprocess] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
