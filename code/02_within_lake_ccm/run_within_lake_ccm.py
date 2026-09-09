"""湖内 CCM 420 条边——带断点续跑。

断点续跑的机制
--------------
单条边要 25–55 分钟，420 条边整批重来的代价无法承受，所以**每条边算完立刻把结果
写成 Volume 上的一个分片**，再由合并步骤读取全部分片统一做 FDR。任何中断只损失
在途的那几条边，重跑会自动跳过已完成的。

不要把 420 条边包在一个编排容器里用 `.map()` 跑：那个容器一旦被抢占，它内部 map
的全部输入会被批量取消，而 `--detach` 只保活最后触发的那个函数。

跑法
----
    modal run --detach code/02_within_lake_ccm/run_within_lake_ccm.py            # 跑（自动续）
    modal run code/02_within_lake_ccm/run_within_lake_ccm.py --merge-only        # 仅合并已有分片
    modal run code/02_within_lake_ccm/run_within_lake_ccm.py --status            # 只看进度

输出
----
    lake_results/final_v3/edges/<lake>__<cause>__<effect>.json   单边分片
    lake_results/final_v3/ccm_all_edges_merged_fdr.csv           合并 + BH-FDR + 时序保留规则
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR / "01_shared"))
sys.path.insert(0, str(_CODE_DIR))

app = modal.App("within-lake-ccm-v3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas==2.2.2", "numpy==1.26.4", "scipy", "statsmodels",
                 "networkx", "pyEDM==2.4.0")
    .add_local_python_source("ccm_full_pipeline")
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
    return f"{EDGE_DIR}/{lake}__{cause}__{effect}.json"


# retries=3 吸收单边容器被抢占；每条边独立、无共享状态，重试代价仅为该边本身。
@app.function(image=image, volumes={DATA_ROOT: volume}, cpu=1.0, memory=2048,
              timeout=3 * 3600, retries=3)
def run_one_edge(lake: str, cause: str, effect: str, n_surrogates: int) -> str:
    import ccm_full_pipeline as p

    p.PKL_DIR = PKL_DIR_REMOTE
    p.OUT_DIR = OUT_DIR
    p.EMBED_PARAMS_PATH = EMBED_PARAMS_ABS
    os.makedirs(EDGE_DIR, exist_ok=True)

    out_path = _edge_path(lake, cause, effect)
    if os.path.exists(out_path):          # 容器重试时不重复计算
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
    volume.commit()                        # 立刻落盘，中断只损失在途的边
    print(f"[{lake}] {cause}->{effect}: {row.get('status')} "
          f"lag={row.get('obs_lag')} rho={row.get('obs_rho')}", flush=True)
    return out_path


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1800)
def list_done() -> list:
    if not os.path.isdir(EDGE_DIR):
        return []
    return sorted(os.listdir(EDGE_DIR))


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1800)
def merge_edges() -> dict:
    """读取全部分片 → BH-FDR → 时序保留规则 → 写出主结果表。"""
    import pandas as pd
    import ccm_full_pipeline as p

    p.OUT_DIR = OUT_DIR

    # 理由同湖间脚本：BH-FDR 的检验族大小 = 参与校正的行数，
    # 分片缺失会把门槛算松并静默产出错误结果。检验族须预先划定。
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
        # 用 spawn_map 而非 map：
        #   .map() 会阻塞等待全部结果，且默认 return_exceptions=False——
        #   任何一个输入永久失败就抛异常、本地入口崩溃，Modal 随即取消其余
        #   全部在途任务（2026-08-27 05:08 实测：一次 "Worker disappeared"
        #   导致数百条 "Input was cancelled by user"，14 分钟计算全废）。
        #   spawn_map 提交后立即返回，任务在服务端独立执行，
        #   客户端断网/休眠/退出均不影响；进度由 Volume 上的分片文件反映。
        run_one_edge.spawn_map([t[0] for t in todo], [t[1] for t in todo],
                               [t[2] for t in todo], [n_surrogates] * len(todo))
        print("已提交，任务在服务端运行。查看进度：")
        print("  modal volume ls ccm-data lake_results/final_v3/edges | grep -c json")
        print("全部完成后合并：")
        print("  modal run code/02_within_lake_ccm/run_within_lake_ccm.py --merge-only")
        return

    result = merge_edges.remote()
    print(f"\n合并完成：{result['n_edges']} 条边，{result['n_causal']} 条通过时序保留规则")
    print(f"滞后分类：{result['lag_resolution']}")
    print(f"写入：{result['path']}")
