"""生成全部附录表（附录A/B/C），输出 CSV（完整精度）与 Markdown（排版用）。

分类与正文的方法学结构对应：
    附录A  数据与数据可用性        ← §3.1–3.2、§4.1
    附录B  CCM 分析                ← §3.3、§4.2–4.3
    附录C  预测分析                ← §3.4、§4.4

数据源全部来自 2026-08-31 重跑：lake_pkls、results/、ch4_tables/。
本脚本不做任何再计算的建模，只从已有结果汇总；A2/A3 的统计量与
figure_4_1_wl_variability.py 使用同一条清洗后水位序列，两处必然一致。

跑法
----
    python code/07_tables/build_appendices.py

输出
----
    final/appendices/<表号>_<名称>.csv / .md
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "01_shared"))
import ccm_full_pipeline as p                                   # noqa: E402

p.log = lambda msg: None

ROOT = CODE.parent                       # 仓库根
TAB = ROOT / "ch4_tables"
OUT = ROOT / "appendices"
PKL = Path(p.PKL_DIR)

SYSTEMS = [
    ("Okanagan", ["Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake"]),
    ("Nelson–Winnipeg", ["Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
                         "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake"]),
]
LAKES = [lk for _, g in SYSTEMS for lk in g]
SYSTEM_OF = {lk: n for n, g in SYSTEMS for lk in g}
LABEL = {lk: lk.replace("_Lake", "").replace("_", " ") for lk in LAKES}
LABEL["Lake_of_the_Woods"] = "Lake of the Woods"

# 取自 ccm_modal_app.py 的 REGULATION_STATIONS / REGULATION_SUBPERIODS
REGFLOW_STATIONS = {
    "Kalamalka_Lake": ["08NM065"], "Okanagan_Lake": ["08NM050"],
    "Skaha_Lake": ["08NM002"], "Vaseux_Lake": ["08NM247"],
    "Rainy_Lake": ["05PC019"], "Lake_of_the_Woods": ["05PE011", "05PE006"],
    "Playgreen_Lake": ["05UB009"], "Kiskitto_Lake": ["05UB009"],
    "Sipiwesk_Lake": ["05UE005"], "Split_Lake": ["05UF006"],
}
REGFLOW_SUBPERIOD = {"Vaseux_Lake": "2012–2024"}
REGFLOW_NOTE = {
    "Lake_of_the_Woods": "两站流量相加（Norman Dam + Kenora Powerhouse）",
    "Playgreen_Lake": "与 Kiskitto 共用 Jenpeg 大坝出流",
    "Kiskitto_Lake": "与 Playgreen 共用 Jenpeg 大坝出流",
    "Sipiwesk_Lake": "无自有坝，以下游 Kelsey 发电站出流代理",
    "Vaseux_Lake": "测流站 2012 年始测，序列受限于该子区间",
}

VAR_ORDER = ["RegFlow", "R", "P", "Evap", "SWE", "T", "WL"]


# 少数列需要不同的小数位，按列名指定
COL_DECIMALS = {"coverage_pct": 1, "cv_rmse_m": 5, "learning_rate": 2, "cv_rmse_m": 5, "learning_rate": 2, "p_value": 4, "p_fdr": 4, "kendall_p": 4,
                "dm_stat": 3, "S_ij": 3}


def _fmt(v, nd):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}" if abs(v) >= 1e-4 or v == 0 else f"{v:.2e}"
    return str(v)


def to_md(df, nd=3):
    """自己渲染 Markdown 表，避免依赖 tabulate。数值列右对齐。"""
    cols = list(df.columns)
    num = [c for c in cols
           if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool]
    head = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join("---:" if c in num else "---" for c in cols) + " |"
    dec = [COL_DECIMALS.get(c, nd) for c in cols]
    body = ["| " + " | ".join(_fmt(v, d) for v, d in zip(row, dec)) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def write(df, stem, title, note="", float_fmt=None, max_md_rows=None):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{stem}.csv", index=False)
    nd = 0 if float_fmt == ".0f" else (2 if float_fmt == ".2f" else 3)
    body = df if max_md_rows is None else df.head(max_md_rows)
    md = to_md(body, nd)
    head = f"**{title}**"
    if note:
        head += f"\n\n{note}"
    if max_md_rows is not None and len(df) > max_md_rows:
        md += f"\n\n（共 {len(df)} 行，完整内容见 {stem}.csv）"
    (OUT / f"{stem}.md").write_text(head + "\n\n" + md + "\n")
    print(f"  {stem:<34} {len(df):>4} 行")


def cleaned_wl(lake):
    with open(PKL / f"{lake}_result.pkl", "rb") as fh:
        cached = pickle.load(fh)
    wl = p.combine_station_water_levels(
        p.clean_wide_wl(cached["wide_wl"]), method="anomaly_mean")
    return wl.sort_index().asfreq("MS"), cached


def longest_gap(s):
    mx = cur = 0
    for v in s.isna():
        cur = cur + 1 if v else 0
        mx = max(mx, cur)
    return mx


# ------------------------------------------------------------------ 附录A
def appendix_a():
    print("附录A 数据与数据可用性")
    rows_a1, rows_a2, rows_a3 = [], [], []
    for lk in LAKES:
        wl, cached = cleaned_wl(lk)
        raw = cached["wide_wl"]
        rows_a1.append({
            "lake": LABEL[lk], "system": SYSTEM_OF[lk],
            "hydrolakes_id": cached.get("hylak_id"),
            "n_wl_stations": raw.shape[1],
            "wl_stations": ", ".join(raw.columns),
            "regflow_stations": ", ".join(REGFLOW_STATIONS[lk]),
            "regflow_period": REGFLOW_SUBPERIOD.get(lk, "1994–2024"),
            "note": REGFLOW_NOTE.get(lk, ""),
        })
        wl_c = wl - wl.mean()
        rows_a2.append({
            "lake": LABEL[lk], "system": SYSTEM_OF[lk],
            "n_months": len(wl), "n_observed": int(wl.notna().sum()),
            "n_missing": int(wl.isna().sum()),
            "coverage_pct": round(100 * wl.notna().mean(), 1),
            "longest_gap_months": longest_gap(wl),
            "sd_m": round(float(wl_c.std()), 3),
            "iqr_m": round(float(wl_c.quantile(.75) - wl_c.quantile(.25)), 3),
            "range_m": round(float(wl_c.max() - wl_c.min()), 3),
        })
        for col in raw.columns:
            for d in p.wl_station_outliers(raw[col]):
                rows_a3.append({"lake": LABEL[lk], "station": col,
                                "month": str(d.date())[:7],
                                "value_m": round(float(raw.loc[d, col]), 3)})

    write(pd.DataFrame(rows_a1), "A1_lakes_and_stations",
          "表A1　研究湖泊、水位站与调控出流站",
          "水位站与调控站编号为 Water Survey of Canada（HYDAT）站号；"
          "hydrolakes_id 为 HydroLAKES 的 Hylak_id。")
    write(pd.DataFrame(rows_a2), "A2_data_availability",
          "表A2　逐湖水位数据可用性与变率",
          "n_months 为 1994-01 至 2024-12 的月份总数；coverage_pct 为非缺失"
          "月份占比；longest_gap_months 为最长连续缺失月数。sd_m、iqr_m 与 "
          "range_m 在各湖去均值（中心化）后计算，单位为米。")
    write(pd.DataFrame(rows_a3), "A3_flagged_outliers",
          "表A3　两级稳健筛查标记为缺失的水位观测",
          "判据：月度变化量稳健 z > 6，且该月水位数值自身稳健 z > 5。"
          "标记点置为缺失，原值不作修正。")


# ------------------------------------------------------------------ 附录B
def appendix_b():
    print("附录B CCM 分析")
    import json
    emb = json.load(open(ROOT / "results" / "embed_params_corrected.json"))
    # 矩阵版：τ 全湖全变量恒为 1，写进表注，表内只放 E
    m = pd.DataFrame({v: {LABEL[lk]: emb.get(lk, {}).get(v, {}).get("E")
                          for lk in LAKES} for v in VAR_ORDER})
    m = m.reindex([LABEL[lk] for lk in LAKES]).reset_index(names="lake")
    write(m, "B1_embedding_parameters",
          "表B1　逐湖逐变量的嵌入维数 E",
          "E 由训练期一步 simplex 自预测在 2–10 中选定；嵌入延迟 τ 对全部湖泊"
          "与变量固定为 1。", float_fmt=".0f")

    b3 = pd.read_csv(TAB / "TC1_supported_drivers.csv")
    write(b3, "B2_supported_drivers",
          "表B2　通过筛选的 24 条 driver → WL 关系",
          "同时满足 BH-FDR（α = 0.05，检验族为 420 条湖内边）与收敛诊断。")

    t6 = pd.read_csv(TAB / "T6_between_lake_edges.csv")
    s6 = t6[t6.statistically_significant]
    s6o = s6.assign(cause_lake=s6.cause_lake.map(LABEL),
                    effect_lake=s6.effect_lake.map(LABEL))[
        ["cause_lake", "effect_lake", "obs_rho", "obs_lag", "obs_n", "p_fdr",
         "kendall_tau", "lag_resolution", "causal_evidence",
         "waterway_connected", "tier"]].sort_values(
        ["tier", "obs_rho"], ascending=[True, False])
    write(s6o, "B3_supported_between_lake_edges",
          "表B3　通过筛选的 26 条湖间关系",
          "检验族为 45 个湖泊对的双向共 90 条候选边，其中 26 条同时满足 "
          "BH-FDR 与收敛诊断。"
          "tier 为写作阶段引入的事后描述性水文距离分组。", max_md_rows=26)

    t7 = pd.read_csv(TAB / "T7_lake_pair_strength.csv")
    t7o = t7.copy()
    t7o["pair"] = t7o.pair.map(lambda x: " – ".join(LABEL[y] for y in x.split("|")))
    write(t7o[["pair", "S_ij", "detected", "directly_connected", "tier"]],
          "B4_lake_pair_strength",
          "表B4　45 个湖泊对的 CCM 强度与水文连通性",
          "S_ij 为该对两个方向 |ρ| 的均值；detected 表示任一方向满足 FDR 显著、"
          "收敛且最优滞后 d ≥ 0。本表为 Mann–Whitney 检验的全部输入。",
          max_md_rows=45)



# ------------------------------------------------------------------ 附录C
def appendix_c():
    print("附录C 预测分析")
    s = pd.read_csv(ROOT / "results" / "forecast_synchrony_filtered_selected_lags.csv")
    s = s.assign(lake=s.lake.map(LABEL))[["lake", "method", "selected_vars",
                                          "selected_lags"]]
    write(s, "C1_selected_predictors",
          "表C1　各湖各策略实际进入模型的外生变量与预测域滞后",
          "滞后为预测域重新求得的值（下界 1 个月）；因果域最优滞后 d = 0 的边"
          "在此重新扫描。CCM_neighbor 行只列出相对 CCM_ancestors 新增的湖间"
          "水位项，该策略的完整变量集为同一湖的 CCM_ancestors 行加本行。"
          "All-vars 不单列：它包含全部通过数据可用性筛查的候选变量，滞后由"
          "滞后互相关（式 17）确定；本研究中唯一被长缺口规则剔除的是 Vaseux "
          "的 RegFlow（该站 2012 年始测），因此除 Vaseux 为 5 个变量外，"
          "其余各湖均为 6 个。")

    t4 = pd.read_csv(TAB / "T4_single_split_full.csv")
    NO_EXOG = {"Persistence", "SARIMA", "XGBoost_AR_only"}

    def cell(r):
        if pd.notna(r.status):
            return "失败"
        if isinstance(r.arima_order, str):
            return f"{r.arima_order}{r.arima_seasonal_order}".replace(" ", "")
        return "无外生" if r.method in NO_EXOG else f"{int(r.n_selected_vars)}"

    t4["cell"] = t4.apply(cell, axis=1)
    mt = t4.pivot_table(index="lake", columns="method", values="cell",
                        aggfunc="first").reindex([lk for lk in LAKES])
    mt = mt[[c for c in ["Persistence", "SARIMA", "SARIMAX_all_vars",
                         "SARIMAX_Stepwise", "SARIMAX_CCM_direct",
                         "SARIMAX_CCM_ancestors", "SARIMAX_CCM_neighbor",
                         "XGBoost_AR_only", "XGBoost_all_vars",
                         "XGBoost_Stepwise", "XGBoost_CCM_direct",
                         "XGBoost_CCM_ancestors", "XGBoost_CCM_neighbor"]
             if c in mt.columns]]
    mt = mt.fillna("不适用")
    mt.index = [LABEL[lk] for lk in mt.index]
    write(mt.reset_index(names="lake"), "C2_model_configurations",
          "表C2　各湖各方法的模型配置与拟合结果",
          "SARIMA(X) 单元格为 (p,d,q)(P,D,Q,12) 阶数，XGBoost 单元格为入模外生"
          "变量数。「不适用」指该湖没有对应的 CCM 关系可用（共 8 个组合，"
          "均出现在无支持关系的 Skaha 与无邻居关系的 Kiskitto）；「失败」指"
          "拟合时测试期外生变量缺口超过 6 个月（共 29 个组合）。"
          "13 种配置 × 10 湖中 122 个进入拟合、93 个产出有效滚动预测。")

    # 由 tune_xgboost_hyperparams() 复算（2026-09-05，pandas 2.2.2 / numpy
    # 1.26.4 / xgboost 3.4.1 的锁定环境，与 pandas 3.0.3 环境结果一致）。原始
    # 运行仅把该结果打印到 Modal 日志、未落盘；调优过程无随机性（random_state
    # = 0、折分固定），按同一代码与同一份 lake_pkls 重跑即可复现。
    grid = pd.DataFrame([
        {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 200,
         "cv_rmse_m": 0.15547, "selected": True},
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200,
         "cv_rmse_m": 0.15933, "selected": False},
        {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 100,
         "cv_rmse_m": 0.15871, "selected": False},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
         "cv_rmse_m": 0.16096, "selected": False},
        {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 400,
         "cv_rmse_m": 0.15958, "selected": False},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 200,
         "cv_rmse_m": 0.16270, "selected": False},
    ])
    write(grid, "C3_xgboost_hyperparameters",
          "表C3　XGBoost 超参数候选网格与交叉验证结果",
          "固定 subsample = 0.8、colsample_bytree = 0.8、random_state = 0；在 10 "
          "个湖泊的训练期内以 3 折扩展窗口交叉验证（共 30 折，仅使用 AR-only "
          "特征）比较平均 RMSE，选出一组供所有湖泊与全部策略共用。选中 "
          "max_depth = 2、learning_rate = 0.05、n_estimators = 200，"
          "平均 CV RMSE = 0.155 m。", float_fmt=".2f")


    t3 = pd.read_csv(TAB / "T3_dm_bh.csv")
    ok = t3[t3.p_value.notna()]
    rows = []
    for (m1, m2), g in ok.groupby(["method_1", "method_2"]):
        sig = g[g.sig_fdr]
        rows.append({"method_1": m1, "method_2": m2,
                     "n_lakes": g.lake.nunique(), "n_tested": len(g),
                     "n_significant": len(sig),
                     "n_favouring_method_1": int((sig.winner == m1).sum()),
                     "n_favouring_method_2": int((sig.winner == m2).sum())})
    write(pd.DataFrame(rows), "C4_dm_summary",
          "表C4　Diebold–Mariano 检验结果汇总",
          "14 组预先设定的方法对，逐湖逐时距共 341 项检验，其中 333 项返回可用"
          "统计量，BH 校正在这 333 项上一次性施加，42 项在校正后仍显著。"
          "", float_fmt=".0f")


# 合并成单一 Markdown，供直接贴入 Word
SECTIONS = [
    ("附录A　数据与数据可用性",
     ["A1_lakes_and_stations", "A2_data_availability",
      "A3_flagged_outliers"]),
    ("附录B　CCM 分析",
     ["B1_embedding_parameters", "B2_supported_drivers",
      "B3_supported_between_lake_edges", "B4_lake_pair_strength"]),
    ("附录C　预测分析",
     ["C1_selected_predictors", "C2_model_configurations",
      "C3_xgboost_hyperparameters", "C4_dm_summary"]),
]


def combine():
    parts = ["# 附录", "",
             "本附录内容来自 2026-08-31 重跑结果，由 "
             "`code/07_tables/build_appendices.py` 自动生成。分类与第三章方法学"
             "结构对应：附录A 对应 §3.1–3.2 数据，附录B 对应 §3.3 CCM 分析，"
             "附录C 对应 §3.4 预测分析。", ""]
    for title, stems in SECTIONS:
        parts += [f"## {title}", ""]
        if title.startswith("附录B"):
            parts += ["### 图B1　420 条湖内候选关系的支持情况总览", "",
                      "行为 42 种有向变量对（按原因变量分为七块），列为 10 个湖。"
                      "着色格子表示该关系同时通过 BH-FDR 与收敛诊断，颜色深浅为 "
                      "cross-map skill ρ，格内数字为最优时滞 d（月，正号省略）；"
                      "橙色方块标记 d = 0，灰色三角标记 d < 0；右侧条形为该关系"
                      "获得支持的湖泊数。420 个检验中 212 个获得支持"
                      "（正滞后 150、同期 23、负滞后 39）。",
                      "", "*（插入 figure_B1_within_lake_network.pdf）*", ""]
        for stem in stems:
            txt = (OUT / f"{stem}.md").read_text().strip().split("\n")
            parts += [f"### {txt[0].strip('*')}", "",
                      "\n".join(txt[1:]).strip(), ""]
    path = OUT / "全部附录.md"
    path.write_text("\n".join(parts) + "\n")
    n = sum(1 for l in path.read_text().split("\n") if l.startswith("|"))
    print(f"\n合并输出 {path}（表格行 {n}）")


if __name__ == "__main__":
    appendix_a()
    appendix_b()
    appendix_c()
    combine()
    print(f"全部输出至 {OUT}")
