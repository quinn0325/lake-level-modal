"""从 results/ 派生第四章的中间表，写入 ch4_tables/。

为什么需要这一步
----------------
`06_figures/` 与 `07_tables/` 下的脚本读的是 `ch4_tables/T*.csv`，而不是
`results/` 里的原始输出。原先这些中间表由分析当时的临时脚本产出，未纳入代码库，
链条因此在这里断了一节：拿到 `results/` 也画不出图。本脚本把这一节补上——
六张中间表全部是 `results/` 的确定性派生，不含任何新的建模。

派生关系
--------
    T1_rolling_lake_horizon_method  = forecast_..._rolling_results（列重排）
    T3_dm_bh                        = forecast_..._dm_results + sig_fdr + winner
    T4_single_split_full            = forecast_..._full_results（原样）
    T5_within_lake_edges            = ccm_all_edges_merged_fdr + 效应变量的 E/τ
    T6_between_lake_edges           = connectivity_..._results + tier
    T7_lake_pair_strength           = 由 T6 按湖泊对聚合

tier 是写作阶段引入的事后描述性分层，不参与任何统计判定：
    1  两湖之间有直接水道连接（config.WATERWAY_CONNECTED_PAIRS，7 对）
    2  同一水道连通分量内，但无直接连接
    3  同一水系，但分属不同连通分量
    4  分属不同水系
连通分量即把 WATERWAY_CONNECTED_PAIRS 视为无向图后的连通块：
Okanagan 干流、Winnipeg 河、Nelson 河。

跑法
----
    python code/07_tables/build_ch4_tables.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import config                                                    # noqa: E402

ROOT = CODE.parent
RESULTS = ROOT / "results"
OUT = ROOT / "ch4_tables"

SYSTEM_OF = {lk: ("Okanagan" if lk in config.LAKES[:4] else "Nelson-Winnipeg")
             for lk in config.LAKES}


def waterway_components():
    """把直接水道连接看作无向图，返回 lake -> 连通分量编号。"""
    comp, nxt = {}, 0
    adj = {lk: set() for lk in config.LAKES}
    for a, b in config.WATERWAY_CONNECTED_PAIRS:
        adj[a].add(b)
        adj[b].add(a)
    for lk in config.LAKES:
        if lk in comp:
            continue
        stack, nxt = [lk], nxt + 1
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp[cur] = nxt
            stack.extend(adj[cur] - set(comp))
    return comp


COMP = waterway_components()
DIRECT = {tuple(sorted(pair)) for pair in config.WATERWAY_CONNECTED_PAIRS}


def tier_of(a, b):
    if tuple(sorted([a, b])) in DIRECT:
        return "1_direct"
    if COMP[a] == COMP[b]:
        return "2_same_subsystem_indirect"
    if SYSTEM_OF[a] == SYSTEM_OF[b]:
        return "3_same_basin_diff_subsystem"
    return "4_different_basin"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- T1 / T4：预测结果，只做列序统一 ----------------------------------
    roll = pd.read_csv(RESULTS / "forecast_synchrony_filtered_rolling_results.csv")
    roll = roll[["lake", "horizon_months", "method", "rmse", "mae", "nse",
                 "n_origins", "min_exog_lag", "requires_foresight"]]
    roll.to_csv(OUT / "T1_rolling_lake_horizon_method.csv", index=False)

    full = pd.read_csv(RESULTS / "forecast_synchrony_filtered_full_results.csv")
    full.to_csv(OUT / "T4_single_split_full.csv", index=False)

    # ---- T3：DM 检验 + 显著标记 + 胜者 -------------------------------------
    # dm_stat = mean(loss_1 - loss_2)/se，因此 dm_stat < 0 表示 method_1 损失更小。
    dm = pd.read_csv(RESULTS / "forecast_synchrony_filtered_dm_results.csv")
    dm["sig_fdr"] = dm["p_fdr"] < config.FDR_ALPHA
    dm["winner"] = np.where(~dm["sig_fdr"], "",
                            np.where(dm["dm_stat"] < 0, dm["method_1"], dm["method_2"]))
    dm.to_csv(OUT / "T3_dm_bh.csv", index=False)

    # ---- T5：湖内边 + 效应变量的嵌入参数 -----------------------------------
    emb = json.loads((RESULTS / "embed_params_corrected.json").read_text())
    w = pd.read_csv(RESULTS / "ccm_all_edges_merged_fdr.csv")
    w["E_effect"] = [emb.get(r.lake, {}).get(r.effect, {}).get("E") for r in w.itertuples()]
    w["tau"] = [emb.get(r.lake, {}).get(r.effect, {}).get("tau") for r in w.itertuples()]
    w = w[["lake", "cause", "effect", "status", "obs_lag", "obs_rho", "obs_n",
           "E_effect", "tau", "p_value", "p_fdr", "kendall_tau", "kendall_p",
           "convergence_diagnostic_pass", "statistically_significant",
           "lag_resolution", "causal_evidence", "n_valid_surrogates"]]
    w.to_csv(OUT / "T5_within_lake_edges.csv", index=False)

    # ---- T6：湖间边 + tier -------------------------------------------------
    b = pd.read_csv(RESULTS / "connectivity_full_pairwise_ccm_results.csv")
    b["tier"] = [tier_of(r.cause_lake, r.effect_lake) for r in b.itertuples()]
    b.to_csv(OUT / "T6_between_lake_edges.csv", index=False)

    # ---- T7：按湖泊对聚合 --------------------------------------------------
    # S_ij 为一对湖两个方向 |rho| 的均值；只要有一个方向 FDR 显著、收敛
    # 且最优滞后 d >= 0，该对即记为 detected。
    b["key"] = [tuple(sorted([r.cause_lake, r.effect_lake])) for r in b.itertuples()]
    b["ok"] = (b.statistically_significant & b.convergence_diagnostic_pass
               & (b.obs_lag >= 0))
    rows = []
    for (a, c), g in b.groupby("key"):
        rows.append({"pair": f"{a}|{c}",          # 下游脚本自己做人读标签
                     "S_ij": g.obs_rho.abs().mean(),
                     "detected": bool(g.ok.any()),
                     "tier": tier_of(a, c),
                     "directly_connected": tuple(sorted([a, c])) in DIRECT})
    t7 = pd.DataFrame(rows).sort_values(["tier", "S_ij"], ascending=[True, False])
    t7.to_csv(OUT / "T7_lake_pair_strength.csv", index=False)

    for name, n in [("T1", len(roll)), ("T3", len(dm)), ("T4", len(full)),
                    ("T5", len(w)), ("T6", len(b)), ("T7", len(t7))]:
        print(f"  {name}: {n} 行")
    print(f"全部写入 {OUT}")


if __name__ == "__main__":
    main()
