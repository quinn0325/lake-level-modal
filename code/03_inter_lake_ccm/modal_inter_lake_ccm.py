"""Run the 90 directed between-lake CCM tests on Modal."""

import itertools
import json
import os
import sys
from pathlib import Path

import modal

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR / "01_analysis_core"))
sys.path.insert(0, str(_CODE_DIR))

from config import LAKES, WATERWAY_CONNECTED_PAIRS  # noqa: E402

app = modal.App("inter-lake-ccm-v3")

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
EDGE_DIR = f"{OUT_DIR}/inter_edges"
PKL_DIR_REMOTE = f"{DATA_ROOT}/lake_results"
EMBED_PARAMS_ABS = f"{DATA_ROOT}/lake_results/full_pipeline_v2/embed_params_corrected.json"


def _edge_name(cause_lake, effect_lake):
    """Return the filename for one edge shard."""
    return f"{cause_lake}__{effect_lake}.json"


def _edge_path(cause_lake, effect_lake):
    """Return the Volume path for one edge shard."""
    return os.path.join(EDGE_DIR, _edge_name(cause_lake, effect_lake))


PAIRS = tuple(itertools.combinations(LAKES, 2))
TASKS = tuple(itertools.permutations(LAKES, 2))
EXPECTED_SHARDS = frozenset(_edge_name(*task) for task in TASKS)
CONNECTED_PAIRS = frozenset(frozenset(pair) for pair in WATERWAY_CONNECTED_PAIRS)


def _existing_shards():
    """Return existing JSON shard filenames from the Volume."""
    if not os.path.isdir(EDGE_DIR):
        return []
    return sorted(name for name in os.listdir(EDGE_DIR) if name.endswith(".json"))


# Retry an independent edge after a worker interruption.
@app.function(image=image, volumes={DATA_ROOT: volume}, cpu=1.0, memory=2048,
              timeout=3 * 3600, retries=3)
def run_one_edge(cause_lake: str, effect_lake: str, n_surrogates: int) -> str:
    """Compute and persist one between-lake edge."""
    import analysis_core as p

    p.PKL_DIR = PKL_DIR_REMOTE
    p.EMBED_PARAMS_PATH = EMBED_PARAMS_ABS
    os.makedirs(EDGE_DIR, exist_ok=True)

    out_path = _edge_path(cause_lake, effect_lake)
    if os.path.exists(out_path):
        return out_path

    panel, embed_effect = p.load_pair_panel_for_connectivity(
        cause_lake, effect_lake
    )
    row = p.test_one_ccm_edge(
        panel,
        cause_lake,
        effect_lake,
        embed_effect["E"],
        embed_effect["tau"],
        n_surrogates=n_surrogates,
    )
    row["cause_lake"] = cause_lake
    row["effect_lake"] = effect_lake
    if row.get("status") != "OK":
        raise RuntimeError(
            f"{cause_lake}->{effect_lake} did not complete successfully: "
            f"{row.get('status', 'missing status')}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, default=str)
    volume.commit()
    print(f"{cause_lake}->{effect_lake}: {row.get('status')} "
          f"lag={row.get('obs_lag')} rho={row.get('obs_rho')}", flush=True)
    return out_path


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1800)
def list_done() -> list:
    """List existing edge-shard files."""
    return _existing_shards()


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1800)
def merge_edges() -> dict:
    """Validate and merge all edge shards, then build connectivity summaries."""
    import pandas as pd
    from scipy import stats
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
            f"Cannot merge the between-lake CCM family: expected exactly "
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
        df["convergence_diagnostic_pass"] = (
            df["convergence_diagnostic_pass"].astype(str).str.lower() == "true")

    df = p.apply_fdr_and_causal_evidence(df)
    df["waterway_connected"] = [
        frozenset((r.cause_lake, r.effect_lake)) in CONNECTED_PAIRS
        for r in df.itertuples()]

    result_path = f"{OUT_DIR}/connectivity_full_pairwise_ccm_results.csv"
    df.to_csv(result_path, index=False)

    # Opposite directions are dependent, so lake pairs are the primary unit.
    df["abs_rho"] = df["obs_rho"].abs()
    df["pair"] = ["|".join(sorted([r.cause_lake, r.effect_lake])) for r in df.itertuples()]
    pair_df = df.groupby("pair").agg(
        abs_rho=("abs_rho", "mean"),
        waterway_connected=("waterway_connected", "first"),
        causal_evidence=("causal_evidence", "any")).reset_index()

    summaries = []
    for label, data in (
        ("lake pairs (main)", pair_df),
        ("directed edges (reference)", df),
    ):
        connected = data.loc[data.waterway_connected, "abs_rho"].dropna()
        unconnected = data.loc[~data.waterway_connected, "abs_rho"].dropna()
        if len(connected) and len(unconnected):
            statistic, p_value = stats.mannwhitneyu(
                connected, unconnected, alternative="two-sided"
            )
            effect_size = statistic / (len(connected) * len(unconnected))
        else:
            statistic = p_value = effect_size = float("nan")
        summaries.append({
            "unit": label,
            "n_connected": len(connected),
            "n_unconnected": len(unconnected),
            "median_connected": connected.median(),
            "median_unconnected": unconnected.median(),
            "frac_sig_connected": data.loc[
                data.waterway_connected, "causal_evidence"
            ].mean(),
            "frac_sig_unconnected": data.loc[
                ~data.waterway_connected, "causal_evidence"
            ].mean(),
            "mannwhitney_u": statistic,
            "mannwhitney_p": p_value,
            "effect_size_auc": effect_size,
        })

    summary_df = pd.DataFrame(summaries)
    summary_path = f"{OUT_DIR}/connectivity_connected_vs_unconnected_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    volume.commit()

    return {
        "n_edges": len(df),
        "n_causal": int(df["causal_evidence"].sum()),
        "lag_resolution": df["lag_resolution"].value_counts().to_dict(),
        "summary": summary_df.to_dict("records"),
        "result_path": result_path,
        "summary_path": summary_path,
    }


@app.local_entrypoint()
def main(n_surrogates: int = 500, merge_only: bool = False, status: bool = False):
    """Submit missing edges, report status, or merge complete shards."""
    done = set(list_done.remote())
    todo = [task for task in TASKS if _edge_name(*task) not in done]
    n_connected = sum(
        1 for pair in PAIRS if frozenset(pair) in CONNECTED_PAIRS
    )

    print(
        f"Lake pairs: {len(PAIRS)} ({n_connected} connected) | "
        f"directed edges: {len(TASKS)} | completed: {len(TASKS) - len(todo)} | "
        f"remaining: {len(todo)}"
    )
    if status:
        return
    if not merge_only and todo:
        print(f"Submitting {len(todo)} edges with n_surrogates={n_surrogates}")
        # Submitted edges continue independently of the local client.
        run_one_edge.spawn_map(
            [task[0] for task in todo],
            [task[1] for task in todo],
            [n_surrogates] * len(todo),
        )
        print("Submitted. The edge jobs are running on Modal.")
        return

    result = merge_edges.remote()
    print(
        f"Merged {result['n_edges']} edges; "
        f"{result['n_causal']} passed the temporal rule"
    )
    print(f"Lag classifications: {result['lag_resolution']}")
    for summary in result["summary"]:
        print(
            f"{summary['unit']}: connected={summary['n_connected']}, "
            f"unconnected={summary['n_unconnected']}, "
            f"p={summary['mannwhitney_p']:.4f}, "
            f"effect size={summary['effect_size_auc']:.3f}"
        )
    print(f"Wrote: {result['result_path']}")
    print(f"Wrote: {result['summary_path']}")
