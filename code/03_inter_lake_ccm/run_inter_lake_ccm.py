"""湖间 CCM 90 条边——带断点续跑，与湖内同一套规范。

三个要点
--------
1. **显著性判定与湖内共用同一个函数**（`apply_fdr_and_causal_evidence`），因此
   时序保留规则一致：d>0 保留、d=0 标注 unresolved、d<0 不作因果证据。
2. **断点续跑**：每条边算完立刻写入 Volume 分片，中断只损失在途的边。
3. **用 spawn_map 而不是 `.map()`**：后者阻塞等待且默认 return_exceptions=False，
   任一输入永久失败就会抛异常、本地入口崩溃，Modal 随即取消其余全部在途任务。

湖间的混杂控制由「有水道连接 vs 无水道连接」的分组对照承担，不用 PCMCI。

跑法
----
    modal run --detach code/03_inter_lake_ccm/run_inter_lake_ccm.py     # 跑（自动续）
    modal run code/03_inter_lake_ccm/run_inter_lake_ccm.py --merge-only # 仅合并
    modal run code/03_inter_lake_ccm/run_inter_lake_ccm.py --status     # 只看进度

输出
----
    lake_results/final_v3/inter_edges/<cause>__<effect>.json
    lake_results/final_v3/connectivity_full_pairwise_ccm_results.csv
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import modal

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR / "01_shared"))
sys.path.insert(0, str(_CODE_DIR))

app = modal.App("inter-lake-ccm-v3")

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
EDGE_DIR = f"{OUT_DIR}/inter_edges"
PKL_DIR_REMOTE = f"{DATA_ROOT}/lake_results"
EMBED_PARAMS_ABS = f"{DATA_ROOT}/lake_results/full_pipeline_v2/embed_params_corrected.json"

LAKES = [
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
    "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake",
]

# 7 对具有直接水道连接（均在同一系统内）
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
    return f"{EDGE_DIR}/{cause}__{effect}.json"


@app.function(image=image, volumes={DATA_ROOT: volume}, cpu=1.0, memory=2048,
              timeout=3 * 3600, retries=3)
def run_one_edge(cause_lake: str, effect_lake: str, n_surrogates: int) -> str:
    import ccm_full_pipeline as p

    p.PKL_DIR = PKL_DIR_REMOTE
    p.OUT_DIR = OUT_DIR
    p.EMBED_PARAMS_PATH = EMBED_PARAMS_ABS
    os.makedirs(EDGE_DIR, exist_ok=True)

    out_path = _edge_path(cause_lake, effect_lake)
    if os.path.exists(out_path):
        return out_path

    # 面板构建也要包在 try 内：它若抛错，本函数会向上抛出 → retries 耗尽后
    # 该边**不产生任何分片**，而合并阶段的齐全性校验只能报"缺失"、看不到原因。
    # 统一捕获后写出带 status=ERROR 的分片，失败原因随分片留痕。
    try:
        panel, _embed_cause, embed_effect = p.load_pair_panel_for_connectivity(
            cause_lake, effect_lake)
        row = p.test_one_ccm_edge(panel, cause_lake, effect_lake,
                                  embed_effect["E"], embed_effect["tau"],
                                  n_surrogates=n_surrogates)
    except Exception as exc:
        row = {"status": f"ERROR: {type(exc).__name__}: {exc}"}
    # test_one_ccm_edge 用 cause/effect 记录列名，此处同时留下湖泊名字段
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
    if not os.path.isdir(EDGE_DIR):
        return []
    return sorted(os.listdir(EDGE_DIR))


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1800)
def merge_edges() -> dict:
    """合并分片 → BH-FDR → 时序保留规则 → 分组对照汇总。"""
    import numpy as np
    import pandas as pd
    from scipy import stats
    import ccm_full_pipeline as p

    p.OUT_DIR = OUT_DIR

    # 先校验分片齐全再合并。BH-FDR 的检验族大小 = 参与校正的行数：
    # 分片缺失时门槛会被算松（如 50 条边按 50 个检验校正而非 90），
    # 产出一份看似正常、实则 FDR 基数错误的结果表且无任何警告。
    # 检验族必须按研究问题预先划定，不能因运行未完成而缩小。
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

    # 与湖内使用同一函数：FDR 在预先设定的候选边集合内施加，随后叠加时序保留规则
    df = p.apply_fdr_and_causal_evidence(df)
    df["waterway_connected"] = [
        frozenset({r.cause_lake, r.effect_lake}) in WATERWAY_CONNECTED_PAIRS
        for r in df.itertuples()]

    out = f"{OUT_DIR}/connectivity_full_pairwise_ccm_results.csv"
    df.to_csv(out, index=False)

    # 分组对照：主口径为湖泊对（两方向取均值），有向边并列作参考。
    # 同一湖泊对的两个方向不独立，按有向边检验会高估有效样本量。
    df["abs_rho"] = df["obs_rho"].abs()
    df["pair"] = ["|".join(sorted([r.cause_lake, r.effect_lake])) for r in df.itertuples()]
    summ = []
    # 聚合口径说明：强度取两方向**均值**，显著性取**任一方向显著**（any）。
    # 两者不同调是有意的——强度检验关心的是该湖泊对整体的关联水平，
    # 而"这对湖泊之间是否检出因果联系"只要任一方向成立即可。
    # 代价：pair 层面的 frac_sig 会高于 edge 层面，报告时需注明口径。
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
        print("  modal run code/03_inter_lake_ccm/run_inter_lake_ccm.py --merge-only")
        return

    r = merge_edges.remote()
    print(f"\n合并完成：{r['n_edges']} 条边，{r['n_causal']} 条通过时序保留规则")
    print(f"滞后分类：{r['lag_resolution']}")
    for s in r["summary"]:
        print(f"  {s['unit']}: 连通 {s['n_connected']} / 非连通 {s['n_unconnected']}"
              f"｜p={s['mannwhitney_p']:.4f}｜效应量={s['effect_size_auc']:.3f}")
