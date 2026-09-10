"""Official Modal stage for 420 directed within-lake CCM edges.
正式的 Modal 湖内 CCM 阶段，共 420 条有向边。

The shared CCM algorithms come from ``01_analysis_core/analysis_core.py``.
This stage defines the experiment edges, stores one resumable shard per edge,
rejects incomplete shard sets and writes the FDR-corrected merged result.
共享 CCM 算法来自 ``01_analysis_core/analysis_core.py``。本阶段定义实验边、
按边保存可续跑分片、拒绝不完整分片集合，并写出经 FDR 校正的合并结果。

Commands and expected outputs are listed in the repository README.
运行命令与预期输出见仓库 README。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR / "01_analysis_core"))
sys.path.insert(0, str(_CODE_DIR))

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

VARIABLES = ["WL", "T", "P", "R", "SWE", "Evap", "RegFlow"]
LAKES = [
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
    "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake",
]


def _edge_path(lake, cause, effect):
    """Return one edge-shard path. / 返回单条边的分片路径。"""
    return f"{EDGE_DIR}/{lake}__{cause}__{effect}.json"


# Retry an independent edge after worker interruption. / 容器中断后仅重试对应的独立边。
@app.function(image=image, volumes={DATA_ROOT: volume}, cpu=1.0, memory=2048,
              timeout=3 * 3600, retries=3)
def run_one_edge(lake: str, cause: str, effect: str, n_surrogates: int) -> str:
    """Compute and persist one within-lake edge. / 计算并保存一条湖内边。"""
    import analysis_core as p

    p.PKL_DIR = PKL_DIR_REMOTE
    p.OUT_DIR = OUT_DIR
    p.EMBED_PARAMS_PATH = EMBED_PARAMS_ABS
    os.makedirs(EDGE_DIR, exist_ok=True)

    out_path = _edge_path(lake, cause, effect)
    if os.path.exists(out_path):          # Reuse a completed shard. / 复用已完成分片。
        return out_path

    panel, embed, _ = p.load_lake_panel_for_ccm(lake)
    if effect not in embed:
        row = {"lake": lake, "cause": cause, "effect": effect,
               "status": "effect_embed_params_missing"}
    else:
        row = p.test_one_ccm_edge(panel, cause, effect,
                                  embed[effect]["E"], embed[effect]["tau"],
                                  n_surrogates=n_surrogates)
        row["lake"] = lake

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, default=str)
    volume.commit()                        # Persist each shard immediately. / 每条分片立即持久化。
    print(f"[{lake}] {cause}->{effect}: {row.get('status')} "
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
    """Merge complete shards, apply FDR and write the main table.
    合并完整分片、应用 FDR 并写出主结果表。"""
    import pandas as pd
    import analysis_core as p

    p.OUT_DIR = OUT_DIR

    # Reject incomplete families before BH-FDR. / BH-FDR 前拒绝不完整检验族。
    expected = {f"{lk}__{c}__{e}.json" for lk in LAKES
                for c in VARIABLES for e in VARIABLES if c != e}
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
        df["convergence_diagnostic_pass"] = df["convergence_diagnostic_pass"].astype(str).str.lower() == "true"

    df = p.apply_fdr_and_causal_evidence(df)
    out = f"{OUT_DIR}/ccm_all_edges_merged_fdr.csv"
    df.to_csv(out, index=False)
    volume.commit()

    counts = df["lag_resolution"].value_counts().to_dict() if "lag_resolution" in df else {}
    return {"n_edges": len(df),
            "n_causal": int(df["causal_evidence"].sum()) if "causal_evidence" in df else 0,
            "lag_resolution": counts, "path": out}


@app.local_entrypoint()
def main(n_surrogates: int = 500, merge_only: bool = False, status: bool = False):
    """Submit missing edges, report status or merge complete shards.
    提交缺失边、查看进度或合并完整分片。"""
    tasks = [(lk, c, e) for lk in LAKES
             for c in VARIABLES for e in VARIABLES if c != e]
    done = set(list_done.remote())
    todo = [t for t in tasks
            if f"{t[0]}__{t[1]}__{t[2]}.json" not in done]

    print(f"总边数 {len(tasks)}｜已完成 {len(tasks) - len(todo)}｜待跑 {len(todo)}")
    if status:
        return
    if not merge_only and todo:
        print(f"提交 {len(todo)} 条边（n_surrogates={n_surrogates}，每边算完立即落盘）")
        # spawn_map keeps submitted work independent of the local client. / spawn_map 使已提交任务不依赖本地客户端。
        run_one_edge.spawn_map([t[0] for t in todo], [t[1] for t in todo],
                               [t[2] for t in todo], [n_surrogates] * len(todo))
        print("已提交，任务在服务端运行。查看进度：")
        print("  modal volume ls ccm-data lake_results/final_v3/edges | grep -c json")
        print("全部完成后合并：")
        print("  modal run code/02_within_lake_ccm/modal_within_lake_ccm.py --merge-only")
        return

    result = merge_edges.remote()
    print(f"\n合并完成：{result['n_edges']} 条边，{result['n_causal']} 条通过时序保留规则")
    print(f"滞后分类：{result['lag_resolution']}")
    print(f"写入：{result['path']}")
