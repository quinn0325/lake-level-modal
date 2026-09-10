"""Official Modal stage for 90 directed between-lake CCM edges.
正式的 Modal 湖间 CCM 阶段，共 90 条有向边。

The shared CCM algorithms come from ``01_analysis_core/analysis_core.py``.
This stage defines the experiment edges, stores one resumable shard per edge,
rejects incomplete shard sets and builds the connectivity summaries.
共享 CCM 算法来自 ``01_analysis_core/analysis_core.py``。本阶段定义实验边、
按边保存可续跑分片、拒绝不完整分片集合，并生成连通性汇总。

Commands and expected outputs are listed in the repository README.
运行命令与预期输出见仓库 README。
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import modal

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR / "01_analysis_core"))
sys.path.insert(0, str(_CODE_DIR))

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

LAKES = [
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
    "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake",
]

# Seven directly connected pairs within the study systems. / 研究水系内的 7 对直接连接湖泊。
WATERWAY_CONNECTED_PAIRS = {
    frozenset({"Kalamalka_Lake", "Okanagan_Lake"}),
    frozenset({"Okanagan_Lake", "Skaha_Lake"}),
    frozenset({"Skaha_Lake", "Vaseux_Lake"}),
    frozenset({"Playgreen_Lake", "Sipiwesk_Lake"}),
    frozenset({"Sipiwesk_Lake", "Split_Lake"}),
    frozenset({"Rainy_Lake", "Lake_of_the_Woods"}),
    frozenset({"Kiskitto_Lake", "Sipiwesk_Lake"}),
}


def _edge_path(cause, effect):
    """Return one edge-shard path. / 返回单条边的分片路径。"""
    return f"{EDGE_DIR}/{cause}__{effect}.json"


@app.function(image=image, volumes={DATA_ROOT: volume}, cpu=1.0, memory=2048,
              timeout=3 * 3600, retries=3)
def run_one_edge(cause_lake: str, effect_lake: str, n_surrogates: int) -> str:
    """Compute and persist one between-lake edge. / 计算并保存一条湖间边。"""
    import analysis_core as p

    p.PKL_DIR = PKL_DIR_REMOTE
    p.OUT_DIR = OUT_DIR
    p.EMBED_PARAMS_PATH = EMBED_PARAMS_ABS
    os.makedirs(EDGE_DIR, exist_ok=True)

    out_path = _edge_path(cause_lake, effect_lake)
    if os.path.exists(out_path):
        return out_path

    # Persist failures as shards so their causes survive merging. / 将失败写入分片，保留错误原因。
    try:
        panel, _embed_cause, embed_effect = p.load_pair_panel_for_connectivity(
            cause_lake, effect_lake)
        row = p.test_one_ccm_edge(panel, cause_lake, effect_lake,
                                  embed_effect["E"], embed_effect["tau"],
                                  n_surrogates=n_surrogates)
    except Exception as exc:
        row = {"status": f"ERROR: {type(exc).__name__}: {exc}"}
    # Preserve lake names beside generic cause/effect fields. / 在通用原因与结果字段外保留湖泊名。
    row["cause_lake"] = cause_lake
    row["effect_lake"] = effect_lake

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, default=str)
    volume.commit()
    print(f"{cause_lake}->{effect_lake}: {row.get('status')} "
          f"lag={row.get('obs_lag')} rho={row.get('obs_rho')}", flush=True)
    return out_path


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1800)
def list_done() -> list:
    """List completed shard files. / 列出已完成分片。"""
    if not os.path.isdir(EDGE_DIR):
        return []
    return sorted(os.listdir(EDGE_DIR))


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1800)
def merge_edges() -> dict:
    """Merge complete shards, apply FDR and build connectivity summaries.
    合并完整分片、应用 FDR 并生成连通性汇总。"""
    import numpy as np
    import pandas as pd
    from scipy import stats
    import analysis_core as p

    p.OUT_DIR = OUT_DIR

    # Reject incomplete families before BH-FDR. / BH-FDR 前拒绝不完整检验族。
    expected = {f"{a}__{b}.json" for a, b in itertools.permutations(LAKES, 2)}
    present = {n for n in os.listdir(EDGE_DIR) if n.endswith(".json")}
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(
            f"分片不齐，拒绝合并：期望 {len(expected)} 条，实到 {len(present)} 条，"
            f"缺 {len(missing)} 条。先跑完再合并。缺失示例："
            + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        )

    rows = []
    for name in sorted(present):
        with open(os.path.join(EDGE_DIR, name), encoding="utf-8") as f:
            rows.append(json.load(f))

    df = pd.DataFrame(rows)
    for col in ("obs_lag", "obs_rho", "obs_n", "p_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "convergence_diagnostic_pass" in df.columns:
        df["convergence_diagnostic_pass"] = (
            df["convergence_diagnostic_pass"].astype(str).str.lower() == "true")

    # Reuse the within-lake FDR and temporal rule. / 复用湖内 FDR 与时序规则。
    df = p.apply_fdr_and_causal_evidence(df)
    df["waterway_connected"] = [
        frozenset({r.cause_lake, r.effect_lake}) in WATERWAY_CONNECTED_PAIRS
        for r in df.itertuples()]

    out = f"{OUT_DIR}/connectivity_full_pairwise_ccm_results.csv"
    df.to_csv(out, index=False)

    # Use lake pairs as the primary unit because opposite directions are dependent. / 主比较使用湖泊对，避免将相反方向视为独立样本。
    df["abs_rho"] = df["obs_rho"].abs()
    df["pair"] = ["|".join(sorted([r.cause_lake, r.effect_lake])) for r in df.itertuples()]
    summ = []
    # Pair strength is the directional mean; support requires either direction. / 湖泊对强度取双向均值，任一方向成立即视为支持。
    pair_df = df.groupby("pair").agg(
        abs_rho=("abs_rho", "mean"),
        waterway_connected=("waterway_connected", "first"),
        causal_evidence=("causal_evidence", "any")).reset_index()
    for label, d in [("lake pairs (main)", pair_df), ("directed edges (reference)", df)]:
        a = d.loc[d.waterway_connected, "abs_rho"].dropna()
        b = d.loc[~d.waterway_connected, "abs_rho"].dropna()
        if len(a) and len(b):
            u, pv = stats.mannwhitneyu(a, b, alternative="two-sided")
        else:
            u, pv = float("nan"), float("nan")
        summ.append({"unit": label, "n_connected": len(a), "n_unconnected": len(b),
                     "median_connected": a.median(), "median_unconnected": b.median(),
                     "frac_sig_connected": d.loc[d.waterway_connected, "causal_evidence"].mean(),
                     "frac_sig_unconnected": d.loc[~d.waterway_connected, "causal_evidence"].mean(),
                     "mannwhitney_u": u, "mannwhitney_p": pv,
                     "effect_size_auc": u / (len(a) * len(b)) if len(a) and len(b) else np.nan})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(f"{OUT_DIR}/connectivity_connected_vs_unconnected_summary.csv", index=False)
    volume.commit()

    return {"n_edges": len(df),
            "n_causal": int(df["causal_evidence"].sum()),
            "lag_resolution": df["lag_resolution"].value_counts().to_dict(),
            "summary": sdf.to_dict("records")}


@app.local_entrypoint()
def main(n_surrogates: int = 500, merge_only: bool = False, status: bool = False):
    """Submit missing edges, report status or merge complete shards.
    提交缺失边、查看进度或合并完整分片。"""
    pairs = list(itertools.combinations(LAKES, 2))
    tasks = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]
    done = set(list_done.remote())
    todo = [t for t in tasks if f"{t[0]}__{t[1]}.json" not in done]

    n_conn = sum(1 for a, b in pairs if frozenset({a, b}) in WATERWAY_CONNECTED_PAIRS)
    print(f"湖泊对 {len(pairs)}（有水道连接 {n_conn}）｜有向边 {len(tasks)}"
          f"｜已完成 {len(tasks) - len(todo)}｜待跑 {len(todo)}")
    if status:
        return
    if not merge_only and todo:
        run_one_edge.spawn_map([t[0] for t in todo], [t[1] for t in todo],
                               [n_surrogates] * len(todo))
        print("已提交，任务在服务端独立运行。查看进度：")
        print("  modal volume ls ccm-data lake_results/final_v3/inter_edges | grep -c json")
        print("完成后合并：")
        print("  modal run code/03_inter_lake_ccm/modal_inter_lake_ccm.py --merge-only")
        return

    r = merge_edges.remote()
    print(f"\n合并完成：{r['n_edges']} 条边，{r['n_causal']} 条通过时序保留规则")
    print(f"滞后分类：{r['lag_resolution']}")
    for s in r["summary"]:
        print(f"  {s['unit']}: 连通 {s['n_connected']} / 非连通 {s['n_unconnected']}"
              f"｜p={s['mannwhitney_p']:.4f}｜效应量={s['effect_size_auc']:.3f}")
