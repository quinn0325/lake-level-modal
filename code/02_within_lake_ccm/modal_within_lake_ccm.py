"""Run the 420 directed within-lake CCM tests on Modal."""

import json
import os
import sys
from pathlib import Path

import modal

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR / "01_analysis_core"))
sys.path.insert(0, str(_CODE_DIR))

from config import LAKES, VARIABLES  # noqa: E402

app = modal.App("within-lake-ccm-v3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas==2.2.2", "numpy==1.26.4", "scipy", "statsmodels",
                 "networkx", "pyEDM==2.4.0")
    .add_local_python_source("analysis_core")
    .add_local_python_source("config")
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=False)
DATA_ROOT = "/data"
OUT_DIR = f"{DATA_ROOT}/lake_results/final_v3"
EDGE_DIR = f"{OUT_DIR}/edges"
PKL_DIR_REMOTE = f"{DATA_ROOT}/lake_results"
EMBED_PARAMS_ABS = f"{DATA_ROOT}/lake_results/full_pipeline_v2/embed_params_corrected.json"


def _edge_name(lake, cause, effect):
    """Return the filename for one edge shard."""
    return f"{lake}__{cause}__{effect}.json"


def _edge_path(lake, cause, effect):
    """Return the Volume path for one edge shard."""
    return os.path.join(EDGE_DIR, _edge_name(lake, cause, effect))


TASKS = tuple(
    (lake, cause, effect)
    for lake in LAKES
    for cause in VARIABLES
    for effect in VARIABLES
    if cause != effect
)
EXPECTED_SHARDS = frozenset(_edge_name(*task) for task in TASKS)


def _existing_shards():
    """Return existing JSON shard filenames from the Volume."""
    if not os.path.isdir(EDGE_DIR):
        return []
    return sorted(name for name in os.listdir(EDGE_DIR) if name.endswith(".json"))


# Retry an independent edge after a worker interruption.
@app.function(image=image, volumes={DATA_ROOT: volume}, cpu=1.0, memory=2048,
              timeout=3 * 3600, retries=3)
def run_one_edge(lake: str, cause: str, effect: str, n_surrogates: int) -> str:
    """Compute and persist one within-lake edge."""
    import analysis_core as p

    p.PKL_DIR = PKL_DIR_REMOTE
    p.EMBED_PARAMS_PATH = EMBED_PARAMS_ABS
    os.makedirs(EDGE_DIR, exist_ok=True)

    out_path = _edge_path(lake, cause, effect)
    if os.path.exists(out_path):
        return out_path

    panel, embed = p.load_lake_panel_for_ccm(lake)
    if effect not in embed:
        raise RuntimeError(f"Embedding parameters are missing for {lake}/{effect}")
    row = p.test_one_ccm_edge(
        panel,
        cause,
        effect,
        embed[effect]["E"],
        embed[effect]["tau"],
        n_surrogates=n_surrogates,
    )
    row["lake"] = lake
    if row.get("status") != "OK":
        raise RuntimeError(
            f"{lake} {cause}->{effect} did not complete successfully: "
            f"{row.get('status', 'missing status')}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, default=str)
    volume.commit()
    print(f"[{lake}] {cause}->{effect}: {row.get('status')} "
          f"lag={row.get('obs_lag')} rho={row.get('obs_rho')}", flush=True)
    return out_path


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1800)
def list_done() -> list:
    """List existing edge-shard files."""
    return _existing_shards()


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1800)
def merge_edges() -> dict:
    """Validate and merge all edge shards, then apply FDR."""
    import pandas as pd
    import analysis_core as p

    present = set(_existing_shards())
    missing = sorted(EXPECTED_SHARDS - present)
    unexpected = sorted(present - EXPECTED_SHARDS)
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                f"missing {len(missing)} shard(s), for example: {', '.join(missing[:5])}"
            )
        if unexpected:
            details.append(
                f"found {len(unexpected)} unexpected shard(s), for example: "
                f"{', '.join(unexpected[:5])}"
            )
        raise RuntimeError(
            f"Cannot merge the within-lake CCM family: expected exactly "
            f"{len(EXPECTED_SHARDS)} shards; " + "; ".join(details)
        )

    rows = []
    failed = []
    for name in sorted(EXPECTED_SHARDS):
        with open(os.path.join(EDGE_DIR, name), encoding="utf-8") as f:
            row = json.load(f)
        rows.append(row)
        if row.get("status") != "OK":
            failed.append(f"{name}: {row.get('status', 'missing status')}")
    if failed:
        raise RuntimeError(
            f"Cannot merge: {len(failed)} shard(s) do not have status=OK. "
            f"Examples: {'; '.join(failed[:5])}"
        )

    df = pd.DataFrame(rows)
    for col in ("obs_lag", "obs_rho", "obs_n", "p_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "convergence_diagnostic_pass" in df.columns:
        df["convergence_diagnostic_pass"] = df["convergence_diagnostic_pass"].astype(str).str.lower() == "true"

    df = p.apply_fdr_and_causal_evidence(df)
    out = f"{OUT_DIR}/ccm_all_edges_merged_fdr.csv"
    df.to_csv(out, index=False)
    volume.commit()

    return {
        "n_edges": len(df),
        "n_causal": int(df["causal_evidence"].sum()),
        "lag_resolution": df["lag_resolution"].value_counts().to_dict(),
        "path": out,
    }


@app.local_entrypoint()
def main(n_surrogates: int = 500, merge_only: bool = False, status: bool = False):
    """Submit missing edges, report status, or merge complete shards."""
    done = set(list_done.remote())
    todo = [task for task in TASKS if _edge_name(*task) not in done]

    print(
        f"Total edges: {len(TASKS)} | completed: {len(TASKS) - len(todo)} | "
        f"remaining: {len(todo)}"
    )
    if status:
        return
    if not merge_only and todo:
        print(f"Submitting {len(todo)} edges with n_surrogates={n_surrogates}")
        # Submitted edges continue independently of the local client.
        run_one_edge.spawn_map([t[0] for t in todo], [t[1] for t in todo],
                               [t[2] for t in todo], [n_surrogates] * len(todo))
        print("Submitted. The edge jobs are running on Modal.")
        return

    result = merge_edges.remote()
    print(
        f"Merged {result['n_edges']} edges; {result['n_causal']} passed the temporal rule"
    )
    print(f"Lag classifications: {result['lag_resolution']}")
    print(f"Wrote: {result['path']}")
