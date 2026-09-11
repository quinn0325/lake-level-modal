"""Modal execution stage for computing the authoritative embedding parameters.

"""

import json
import sys
from pathlib import Path

import modal

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR.parent))
sys.path.insert(0, str(CODE_DIR.parent / "01_analysis_core"))

app = modal.App("compute-embedding-params")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas==2.2.2", "numpy==1.26.4", "scipy", "networkx", "pyEDM==2.4.0")
    .add_local_python_source("analysis_core")
    .add_local_python_source("config")
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=False)
DATA_ROOT = "/data"
PKL_DIR_REMOTE = f"{DATA_ROOT}/lake_results"
OUT_REMOTE = "lake_results/full_pipeline_v2/embed_params_corrected.json"
MIN_EMBED_OBSERVATIONS = 30


@app.function(image=image, volumes={DATA_ROOT: volume}, cpu=2.0, memory=4096,
              timeout=60 * 30)
def compute_one_lake(lake: str) -> dict:
    """Compute one lake's embedding parameters from its training panel."""
    import pickle
    import numpy as np

    import config
    import analysis_core as p

    pkl_path = Path(PKL_DIR_REMOTE) / f"{lake}_result.pkl"
    with pkl_path.open("rb") as f:
        cached = pickle.load(f)
    if cached.get("status") != "OK":
        raise RuntimeError(f"Input checkpoint is not complete for {lake}: {cached.get('status')}")

    # Use the same training-panel workflow as the downstream analyses.
    clean_wide = p.clean_wide_wl(cached["wide_wl"], log_prefix=f"[{lake}] ")
    combined_wl = p.combine_station_water_levels(clean_wide)
    real_predictors = cached["real_predictors"]
    panel_deseason = p.build_variable_panel(
        combined_wl,
        {c: real_predictors[c] for c in real_predictors.columns},
        forecast_horizon=config.FORECAST_HORIZON,
    )
    panel_train = panel_deseason.iloc[:-config.FORECAST_HORIZON]

    out = {}
    for var in config.VARIABLES:
        if var not in panel_train.columns:
            continue
        values = panel_train[var].values  # Preserve calendar gaps.
        n_obs = int(np.sum(~np.isnan(values)))
        if n_obs < MIN_EMBED_OBSERVATIONS:
            out[var] = {"status": "insufficient_observations", "n_obs": n_obs}
            continue
        try:
            E, curve = p.select_E(
                values,
                tau=config.EMBED_TAU,
                candidate_E=config.EMBED_E_CANDIDATES,
            )
            if not curve["rho"].notna().any():
                out[var] = {"status": "embedding_selection_failed", "n_obs": n_obs}
                continue
            out[var] = {
                "E": int(E), "tau": int(config.EMBED_TAU), "n_obs": n_obs,
            }
        except Exception as exc:
            out[var] = {"status": f"ERROR: {type(exc).__name__}: {exc}", "n_obs": n_obs}

    return {"lake": lake, "params": out}


@app.local_entrypoint()
def main():
    """Compute all lakes and upload the complete merged JSON result."""
    import config

    results = list(compute_one_lake.map(config.LAKES))
    merged = {r["lake"]: r["params"] for r in results}

    problems = []
    for lake in config.LAKES:
        params = merged.get(lake)
        if params is None:
            problems.append(f"{lake}: missing lake result")
            continue
        for var in config.VARIABLES:
            if var not in params:
                problems.append(f"{lake}/{var}: missing variable")
            elif "E" not in params[var] or "tau" not in params[var]:
                problems.append(f"{lake}/{var}: {params[var].get('status', 'invalid result')}")
    if problems:
        raise RuntimeError("Embedding parameters are incomplete:\n" + "\n".join(problems))

    print(f"\n{'Lake':<20}{'Variable':<12}{'E':>5}{'tau':>5}  {'n_obs':>6}")
    print("-" * 48)
    n_total = 0
    for lake in config.LAKES:
        for var, d in merged.get(lake, {}).items():
            if "E" not in d:
                print(f"{lake:<20}{var:<9}  {d.get('status', '?')}")
                continue
            n_total += 1
            print(f"{lake:<20}{var:<9}{d['E']:>5}{d['tau']:>5}  {d['n_obs']:>6}")
    print("-" * 48)
    print(f"Computed embedding parameters for {n_total} lake-variable combinations")

    payload = json.dumps(merged, ensure_ascii=False, indent=2)

    with volume.batch_upload(force=True) as batch:
        import io
        batch.put_file(io.BytesIO(payload.encode("utf-8")), OUT_REMOTE)
    print(f"\nWrote complete embedding parameters to the Volume: {OUT_REMOTE}")
