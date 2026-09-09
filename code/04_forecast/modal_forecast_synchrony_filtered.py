"""条件预测评估：比较各变量选择策略在同一模型族下的预测表现。

每个湖泊运行 13 个方法 = 3 个无外生变量基线 + 5 个变量集 × 2 个模型族：

    无外生基线   SARIMA        Persistence      XGBoost_AR_only
    变量集(×2)   all_vars      Stepwise
                 CCM_direct    CCM_ancestors    CCM_neighbor

两域滞后
--------
因果域 d ∈ [-12, +12] 求 argmax ρ(d)，得 obs_lag；预测特征的滞后须落在预测域
d ∈ [1, 12]（d=0 需要预测起点当期的真实值）。因果最优落在 d=0 的边**不整条丢弃**，
而是在预测域内重新求最优（forecast_constrained_lag）——丢弃会损失该关系在预测
可用范围内的全部信息。direct、ancestors、neighbor 三个分支一律如此。

基线的独立性
------------
All_vars / Stepwise 的滞后由与目标水位的滞后 Pearson 相关确定
（select_all_var_lags_xcorr，1..12），**不使用任何 CCM 信息**。若让基线沿用 CCM
的滞后识别，它就无偿继承了 CCM 贡献的主要部分，RQ3 会被系统性低估。

长缺口筛选
----------
训练窗口内最长连续缺口 > MAX_FILLABLE_GAP_MONTHS 的变量整体不纳入候选，覆盖全部
策略（含 all_vars 与邻居湖水位），且必须在 stepwise 前向选择之前施加——见
_filter_long_gap_vars 与 _load_neighbor_series。

跑法
----
    modal run --detach code/04_forecast/modal_forecast_synchrony_filtered.py::detached

输入（由 02/03 阶段的 merge 步骤写到 Modal Volume）
    lake_results/final_v3/ccm_all_edges_merged_fdr.csv
    lake_results/final_v3/connectivity_full_pairwise_ccm_results.csv

输出（Modal Volume）
    forecast_synchrony_filtered_full_results.csv      单次切分，含 arima_order
    forecast_synchrony_filtered_rolling_results.csv   滚动起点，h = 1/3/6/12
    forecast_synchrony_filtered_dm_results.csv        DM 检验，含 BH 校正的 p_fdr
    forecast_synchrony_filtered_selected_lags.csv     各策略选中的变量与滞后
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# add_local_python_source 靠本地 import 解析；ccm_full_pipeline 位于 code/01_shared/
_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR / "01_shared"))
sys.path.insert(0, str(_CODE_DIR))


app = modal.App("forecast-synchrony-filtered")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scipy",
        "statsmodels",
        "pmdarima",
        "xgboost",
        "networkx",
        "pyEDM==2.4.0",
    )
    # 线程数钉死为 1（须排在 add_local_* 之前：Modal 不允许在其后再有构建步骤）。
    # 说明其作用范围：这**没有**消除 SARIMA/SARIMAX 的运行间数值差异——实测钉死
    # 前后差异幅度不变（最大约 1e-3 RMSE，来源是极大似然估计在不同容器宿主上
    # 收敛到相邻解，疑与 CPU 指令集差异有关）。保留它是为了消除线程调度这一个
    # 可控的变异来源，属复现性上的常规做法，不要误以为结果因此完全可复现。
    # XGBoost 与 Persistence 在两次运行间逐字节相同；阶数搜索经实测亦稳定(31/31)。
    .env({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    })
    .add_local_python_source("ccm_full_pipeline")
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=False)
DATA_ROOT = "/data"
# 与本批次 CCM 重跑一致
OUT_DIR = f"{DATA_ROOT}/lake_results/final_v3"
EMBED_PARAMS_ABS = f"{DATA_ROOT}/lake_results/full_pipeline_v2/embed_params_corrected.json"

WITHIN_ORIGINAL_INPUT = f"{OUT_DIR}/ccm_all_edges_merged_fdr.csv"
INTER_ORIGINAL_INPUT = f"{OUT_DIR}/connectivity_full_pairwise_ccm_results.csv"

FORECAST_LAKES = [
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake", "Kiskitto_Lake",
    "Sipiwesk_Lake", "Split_Lake",
]

OUTPUT_NAMES = {
    "full": "forecast_synchrony_filtered_full_results.csv",
    "rolling": "forecast_synchrony_filtered_rolling_results.csv",
    "dm": "forecast_synchrony_filtered_dm_results.csv",
    "selected": "forecast_synchrony_filtered_selected_lags.csv",
}


def _configure_module():
    import ccm_full_pipeline as p

    p.PKL_DIR = f"{DATA_ROOT}/lake_results"
    p.OUT_DIR = OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    # 不设则容器内找不到、静默回退到 PKL 中带测试期泄漏的旧 E/τ
    p.EMBED_PARAMS_PATH = EMBED_PARAMS_ABS
    return p


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _json_dict(values: dict) -> str:
    return json.dumps({str(k): int(v) for k, v in values.items()}, sort_keys=True)


def _select_stepwise_lags(panel_train, all_var_lags, wl_col="WL", criterion="aic"):
    """Select a subset from the same all-variable lag pool as the baseline."""
    import pandas as pd
    import statsmodels.api as sm

    candidates = list(all_var_lags)
    if not candidates:
        return {}

    features = pd.DataFrame(index=panel_train.index)
    for variable, lag in all_var_lags.items():
        features[variable] = panel_train[variable].shift(lag)
    data = pd.concat([panel_train[wl_col].rename(wl_col), features], axis=1).dropna()
    if len(data) < 20:
        return {}

    y = data[wl_col]

    def score(selected):
        if selected:
            design = sm.add_constant(data[selected])
        else:
            design = sm.add_constant(pd.Series(1.0, index=y.index, name="_const_only"))
        model = sm.OLS(y, design).fit()
        return model.aic if criterion == "aic" else model.bic

    selected = []
    remaining = list(candidates)
    current_score = score(selected)
    while remaining:
        best_variable = None
        best_score = current_score
        for variable in remaining:
            trial_score = score(selected + [variable])
            if trial_score < best_score:
                best_variable = variable
                best_score = trial_score
        if best_variable is None:
            break
        selected.append(best_variable)
        remaining.remove(best_variable)
        current_score = best_score

    return {variable: all_var_lags[variable] for variable in selected}


def _filtered_within_edges(pd, input_path=WITHIN_ORIGINAL_INPUT):
    """取湖内因果网络中的边。

    带符号扫描（CCM_LAGS = -12..+12）下，obs_lag 本身就是全局最优滞后，
    causal_evidence 也已内含时序保留规则（d >= 0），无需再读单独的诊断表。
    """
    df = pd.read_csv(input_path)
    sig = df[df["causal_evidence"].map(_as_bool)].copy()
    sig["obs_lag"] = pd.to_numeric(sig["obs_lag"], errors="coerce")
    sig["obs_rho"] = pd.to_numeric(sig["obs_rho"], errors="coerce")
    # 此处只要求 d ≥ 0（d < 0 已由时序保留规则排除）。
    # **不在这里丢弃 d = 0 的边**：那样下游的 forecast_constrained_lag 永远
    # 收不到它们。d = 0 交由下游在预测域 [1, 12] 内重新求最优滞后。
    sig = sig[sig["obs_lag"].notna() & (sig["obs_lag"] >= 0)].copy()
    return pd.DataFrame(
        {
            "lake": sig["lake"].astype(str),
            "cause": sig["cause"].astype(str),
            "effect": sig["effect"].astype(str),
            "obs_rho": sig["obs_rho"],
            "obs_lag": sig["obs_lag"].astype(int),
            "p_fdr": sig["p_fdr"],
            "causal_evidence": True,
        }
    )


def _filtered_interlake_edges(pd, input_path=INTER_ORIGINAL_INPUT):
    """取湖间因果网络中的边（理由同 _filtered_within_edges）。

    湖内湖间同一口径：由 causal_evidence 承担（d>=0 保留，d=0 标注 unresolved）。
    """
    df = pd.read_csv(input_path)
    sig = df[df["causal_evidence"].map(_as_bool)].copy()
    sig["obs_lag"] = pd.to_numeric(sig["obs_lag"], errors="coerce")
    sig["obs_rho"] = pd.to_numeric(sig["obs_rho"], errors="coerce")
    # 同上：此处只要求 d ≥ 0，d = 0 交由下游在预测域内重新求最优
    sig = sig[sig["obs_lag"].notna() & (sig["obs_lag"] >= 0)].copy()
    return pd.DataFrame(
        {
            "cause_lake": sig["cause_lake"].astype(str),
            "effect_lake": sig["effect_lake"].astype(str),
            "obs_rho": sig["obs_rho"],
            "obs_lag": sig["obs_lag"].astype(int),
            "p_fdr": sig["p_fdr"],
            "causal_evidence": True,
        }
    )


def _original_within_direct_lags(pd, lake_name):
    df = pd.read_csv(WITHIN_ORIGINAL_INPUT)
    df["causal_evidence"] = df["causal_evidence"].map(_as_bool)
    sub = df[(df["lake"] == lake_name) & (df["causal_evidence"]) & (df["effect"] == "WL")]
    return {str(row["cause"]): int(row["obs_lag"]) for _, row in sub.iterrows()}


def _original_neighbor_lags(pd, lake_name):
    df = pd.read_csv(INTER_ORIGINAL_INPUT)
    df["causal_evidence"] = df["causal_evidence"].map(_as_bool)
    sub = df[(df["effect_lake"] == lake_name) & (df["causal_evidence"])]
    return {str(row["cause_lake"]): int(row["obs_lag"]) for _, row in sub.iterrows()}


def _filter_long_gap_vars(p, panel_deseason, lag_sets, lake_name):
    """对本湖面板内的变量施加长缺口规则，返回 (筛后的 lag_sets, 被排除的变量集合)。

    **只适用于 panel_deseason 里已有的列。** find_long_gap_vars 对不在
    train.columns 中的变量会 `continue` 静默跳过，因此邻居湖水位（建模时才
    reindex 进来的 `_neighbor_*` 列）不能走这里——放进来会一条都筛不掉且不报错。
    邻居由 _load_neighbor_series() 在对齐后单独施加同一阈值。

    返回排除集合供调用方复用（例如需在其他候选选择上施加同一排除时）。
    """
    selected_cols = set()
    for values in lag_sets.values():
        selected_cols.update(values.keys())
    if not selected_cols:
        return lag_sets, set()

    p.check_train_gap_warning(
        panel_deseason,
        selected_cols,
        p.FORECAST_HORIZON,
        log_prefix=f"[{lake_name}] ",
    )
    long_gap_vars = p.find_long_gap_vars(
        panel_deseason,
        selected_cols,
        p.FORECAST_HORIZON,
        log_prefix=f"[{lake_name}] ",
    )
    if not long_gap_vars:
        return lag_sets, set()
    return {
        name: {key: value for key, value in values.items() if key not in long_gap_vars}
        for name, values in lag_sets.items()
    }, long_gap_vars


def _load_neighbor_series(p, panel_deseason, neighbor_lags, lake_name):
    """加载邻居湖水位、对齐到本湖面板索引，并施加与本湖变量相同的长缺口规则。

    邻居必须和本湖的外生变量适用同一条数据质量标准，否则方法间比较不公平。

    缺口在 **reindex 之后**测量：模型看到的是对齐到本湖日历的那条序列，
    邻居自身日历上的缺口长度未必相同。

    返回 (列名→序列, 列名→滞后, 保留下来的邻居名→滞后)。
    """
    cols, col_lags, kept = {}, {}, {}
    train_end = -p.FORECAST_HORIZON
    for neighbor_name, lag in neighbor_lags.items():
        neighbor_wl = p.load_neighbor_wl_series(neighbor_name)
        if neighbor_wl is None:
            continue
        aligned = neighbor_wl.reindex(panel_deseason.index)
        gap = int(p.longest_consecutive_gap_months(aligned.iloc[:train_end]))
        if gap > p.MAX_FILLABLE_GAP_MONTHS:
            print(f"[{lake_name}] 排除邻居{neighbor_name}: 训练窗口内最长连续缺口={gap}个月"
                  f"(超过{p.MAX_FILLABLE_GAP_MONTHS}个月阈值)，与本湖外生变量同一口径",
                  flush=True)
            continue
        column = f"_neighbor_{neighbor_name}"
        cols[column] = aligned
        col_lags[column] = lag
        kept[neighbor_name] = lag
    return cols, col_lags, kept


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=3600, cpu=1.0, memory=2048)
def run_lake_synchrony_filtered(
    lake_name: str,
    xgb_params: dict,
    within_edges_records: list[dict],
    inter_edges_records: list[dict],
) -> dict:
    import pickle
    import pandas as pd

    p = _configure_module()
    pkl_path = os.path.join(p.PKL_DIR, f"{lake_name}_result.pkl")
    if not os.path.exists(pkl_path):
        return {"lake": lake_name, "full_rows": [], "rolling_rows": [], "dm_rows": [], "selected_rows": []}

    within_edges = pd.DataFrame(within_edges_records)
    inter_edges = pd.DataFrame(inter_edges_records)

    with open(pkl_path, "rb") as handle:
        cached = pickle.load(handle)

    clean_wide = p.clean_wide_wl(cached["wide_wl"], log_prefix=f"[{lake_name}] ")
    combined_wl = p.combine_station_water_levels(clean_wide, method="anomaly_mean")
    real_predictors = cached["real_predictors"]
    _, panel_deseason = p.build_variable_panel(
        combined_wl,
        {column: real_predictors[column] for column in real_predictors.columns},
        forecast_horizon=p.FORECAST_HORIZON,
    )
    ccm_train_panel = panel_deseason.iloc[:-p.FORECAST_HORIZON]
    # 必须走 load_embed_params：PKL 里缓存的 E/τ 由数据生成阶段用**含测试期**的
    # 全序列去季节化算出，是已确认的泄漏点。直接读 cached["embed_params"] 会绕过修正。
    embed_params = p.load_embed_params(lake_name, cached)

    lake_edges = within_edges[within_edges["lake"] == lake_name].copy()
    G = p.build_new_causal_network(lake_edges)
    # 因果边的滞后来自因果域 [-12,12]；预测特征的滞后须在预测域 [1,12] 内重新求最优
    # （forecast-constrained optimal lag）。全局最优落在 d=0 的边不整条丢弃——
    # 丢弃会损失该关系在预测可用范围内的全部信息，而 CCM_ancestors 分支本就采用
    # 重新求最优的做法，此处统一口径。
    direct_lags = p.select_direct_predictor_lags(lake_edges)
    for _v, _l in list(direct_lags.items()):
        if _l < p.FORECAST_LAG_MIN:
            _nl, _nr = p.forecast_constrained_lag(ccm_train_panel, _v, "WL", embed_params)
            if _nl is None:
                direct_lags.pop(_v)
                print(f"[{lake_name}] {_v}→WL: 因果最优 d={_l}，预测域内无有效滞后，剔除", flush=True)
            else:
                direct_lags[_v] = _nl
                print(f"[{lake_name}] {_v}→WL: 因果最优 d={_l} → 预测域最优 d={_nl} (rho={_nr:.3f})", flush=True)
    # ancestors 分支同样受预测域约束：有直接边的祖先此前直接沿用因果域 obs_lag，
    # d=0 的边因此以当期外生变量入模（见 select_ancestor_lags 的说明）。
    # 现由该函数内部统一走 forecast_constrained_lag，三个分支口径一致。
    ancestor_lags = p.select_ancestor_lags(
        G, ccm_train_panel, embed_params, log_prefix=f"[{lake_name}] ")
    # 基线（All_vars / Stepwise）的滞后改用滞后 Pearson 相关，不再沿用 CCM 交叉映射。
    # 原实现使基线无偿继承 CCM 的滞后识别，令 RQ3 被系统性低估（实测 84% 的滞后会变）。
    # 与 CCM 保持相同的预处理、相同的 0–12 搜索窗口、相同的 conditional-information
    # 假设（两侧均不施加 d ≥ h 约束）。
    all_var_lags = p.select_all_var_lags_xcorr(
        ccm_train_panel, embed_params, min_lag=1, log_prefix=f"[{lake_name}] ")
    # stepwise 不在此处选：长缺口筛选必须先于前向选择，见下方 lag_sets 处的说明。

    neighbor_lags = p.load_neighbor_lags(inter_edges, lake_name)
    # 湖间边同样按预测域 [1,12] 重新求最优（理由同 direct 分支）。
    # 湖间面板由两湖 WL 对齐而成，效应湖的 WL 嵌入参数即扫描所需。
    for _nb, _l in list(neighbor_lags.items()):
        if _l < p.FORECAST_LAG_MIN:
            try:
                _pnl, _, _emb_eff = p.load_pair_panel_for_connectivity(_nb, lake_name)
                _nl, _nr = p.forecast_constrained_lag(
                    _pnl, _nb, lake_name, {lake_name: _emb_eff})
            except Exception as _exc:
                _nl, _nr = None, float("nan")
                print(f"[{lake_name}] 邻居 {_nb}: 预测域扫描失败 {type(_exc).__name__}", flush=True)
            if _nl is None:
                neighbor_lags.pop(_nb)
                print(f"[{lake_name}] 邻居 {_nb}: 因果最优 d={_l}，预测域内无有效滞后，剔除", flush=True)
            else:
                neighbor_lags[_nb] = _nl
                print(f"[{lake_name}] 邻居 {_nb}: 因果最优 d={_l} → 预测域最优 d={_nl} (rho={_nr:.3f})", flush=True)
    # 邻居须在此处（而非建模处）加载：只有对齐到本湖面板后才能量测缺口。
    neighbor_cols, neighbor_combined_lags, neighbor_lags = _load_neighbor_series(
        p, panel_deseason, neighbor_lags, lake_name)

    old_direct_lags = _original_within_direct_lags(pd, lake_name)
    old_neighbor_lags = _original_neighbor_lags(pd, lake_name)

    # 长缺口筛选覆盖**全部**以本湖面板为来源的策略（含 all_vars），且必须在
    # _select_stepwise_lags 之前完成：该函数对全部候选列一次性 dropna，池中
    # 留一个长缺口变量会把估计样本整体砍掉（实测 Vaseux 332→117），使前向
    # 选择的每一步 AIC 比较——包括不含该变量的模型——都在缩水样本上进行。
    # 事后再删该变量，留下的既不是含它的最优组合、也不是不含它的最优组合。
    lag_sets = {
        "direct": direct_lags,
        "ancestors": ancestor_lags,
        "all_vars": all_var_lags,
    }
    lag_sets, long_gap_vars = _filter_long_gap_vars(
        p, panel_deseason, lag_sets, lake_name)
    direct_lags = lag_sets["direct"]
    ancestor_lags = lag_sets["ancestors"]
    all_var_lags = lag_sets["all_vars"]
    stepwise_lags = _select_stepwise_lags(ccm_train_panel, all_var_lags)

    selected_rows = [{
        "lake": lake_name,
        "method": "CCM_direct",
        "selected_vars": ",".join(sorted(direct_lags)),
        "selected_lags": _json_dict(direct_lags),
        "old_direct_lags": _json_dict(old_direct_lags),
        "new_direct_lags": _json_dict(direct_lags),
        "old_neighbor_lags": _json_dict(old_neighbor_lags),
        "new_neighbor_lags": _json_dict(neighbor_lags),
    }, {
        "lake": lake_name,
        "method": "CCM_ancestors",
        "selected_vars": ",".join(sorted(ancestor_lags)),
        "selected_lags": _json_dict(ancestor_lags),
        "old_direct_lags": _json_dict(old_direct_lags),
        "new_direct_lags": _json_dict(direct_lags),
        "old_neighbor_lags": _json_dict(old_neighbor_lags),
        "new_neighbor_lags": _json_dict(neighbor_lags),
    }, {
        "lake": lake_name,
        "method": "Stepwise",
        "selected_vars": ",".join(sorted(stepwise_lags)),
        "selected_lags": _json_dict(stepwise_lags),
        "old_direct_lags": _json_dict(old_direct_lags),
        "new_direct_lags": _json_dict(direct_lags),
        "old_neighbor_lags": _json_dict(old_neighbor_lags),
        "new_neighbor_lags": _json_dict(neighbor_lags),
    }]

    wl_series = panel_deseason["WL"].sort_index().asfreq("MS")
    methods = {}
    method_exog_lags = {}

    # ---- 基线方法 ----
    # 基线必须和 CCM_* 在同一次运行里产出，否则胜负比较会跨两批数据、两套配置。
    # 全部方法共用同一份面板、同一套 XGB 超参、同一批 CCM 结果。
    # 无外生变量的三个基线不受"完美预知"影响，method_exog_lags 记为 None。
    methods["SARIMA"] = lambda: p.fit_auto_sarima(wl_series, test_size=p.FORECAST_HORIZON)
    method_exog_lags["SARIMA"] = None
    methods["Persistence"] = lambda: p.fit_persistence(wl_series, test_size=p.FORECAST_HORIZON)
    method_exog_lags["Persistence"] = None
    methods["XGBoost_AR_only"] = lambda: p.fit_xgboost_multi(
        wl_series, panel_deseason.iloc[:, :0], {}, xgb_params, test_size=p.FORECAST_HORIZON)
    method_exog_lags["XGBoost_AR_only"] = None

    if all_var_lags:
        methods["SARIMAX_all_vars"] = lambda: p.fit_auto_sarimax_multi(
            wl_series, panel_deseason[list(all_var_lags.keys())], all_var_lags,
            test_size=p.FORECAST_HORIZON)
        methods["XGBoost_all_vars"] = lambda: p.fit_xgboost_multi(
            wl_series, panel_deseason[list(all_var_lags.keys())], all_var_lags,
            xgb_params, test_size=p.FORECAST_HORIZON)
        method_exog_lags["SARIMAX_all_vars"] = all_var_lags
        method_exog_lags["XGBoost_all_vars"] = all_var_lags
    if direct_lags:
        methods["SARIMAX_CCM_direct"] = lambda: p.fit_auto_sarimax_multi(
            wl_series, panel_deseason[list(direct_lags)], direct_lags,
            test_size=p.FORECAST_HORIZON
        )
        methods["XGBoost_CCM_direct"] = lambda: p.fit_xgboost_multi(
            wl_series, panel_deseason[list(direct_lags)], direct_lags, xgb_params, test_size=p.FORECAST_HORIZON
        )
        method_exog_lags["SARIMAX_CCM_direct"] = direct_lags
        method_exog_lags["XGBoost_CCM_direct"] = direct_lags

    if ancestor_lags:
        methods["SARIMAX_CCM_ancestors"] = lambda: p.fit_auto_sarimax_multi(
            wl_series, panel_deseason[list(ancestor_lags)], ancestor_lags,
            test_size=p.FORECAST_HORIZON
        )
        methods["XGBoost_CCM_ancestors"] = lambda: p.fit_xgboost_multi(
            wl_series, panel_deseason[list(ancestor_lags)], ancestor_lags, xgb_params, test_size=p.FORECAST_HORIZON
        )
        method_exog_lags["SARIMAX_CCM_ancestors"] = ancestor_lags
        method_exog_lags["XGBoost_CCM_ancestors"] = ancestor_lags

    if stepwise_lags:
        methods["SARIMAX_Stepwise"] = lambda: p.fit_auto_sarimax_multi(
            wl_series, panel_deseason[list(stepwise_lags)], stepwise_lags,
            test_size=p.FORECAST_HORIZON
        )
        methods["XGBoost_Stepwise"] = lambda: p.fit_xgboost_multi(
            wl_series, panel_deseason[list(stepwise_lags)], stepwise_lags, xgb_params, test_size=p.FORECAST_HORIZON
        )
        method_exog_lags["SARIMAX_Stepwise"] = stepwise_lags
        method_exog_lags["XGBoost_Stepwise"] = stepwise_lags

    if neighbor_lags and ancestor_lags:
        # neighbor_cols / neighbor_combined_lags 已由 _load_neighbor_series 在
        # 上游备好（含长缺口筛选），此处不再重复加载。
        if neighbor_cols:
            neighbor_panel = pd.DataFrame(neighbor_cols)
            combined_exog = pd.concat([panel_deseason[list(ancestor_lags)], neighbor_panel], axis=1)
            combined_lags = {**ancestor_lags, **neighbor_combined_lags}
            methods["SARIMAX_CCM_neighbor"] = lambda: p.fit_auto_sarimax_multi(
                wl_series, combined_exog, combined_lags,
                test_size=p.FORECAST_HORIZON
            )
            methods["XGBoost_CCM_neighbor"] = lambda: p.fit_xgboost_multi(
                wl_series, combined_exog, combined_lags, xgb_params, test_size=p.FORECAST_HORIZON
            )
            method_exog_lags["SARIMAX_CCM_neighbor"] = combined_lags
            method_exog_lags["XGBoost_CCM_neighbor"] = combined_lags
            selected_rows.append({
                "lake": lake_name,
                "method": "CCM_neighbor",
                "selected_vars": ",".join(sorted(neighbor_lags)),
                "selected_lags": _json_dict(neighbor_combined_lags),
                "old_direct_lags": _json_dict(old_direct_lags),
                "new_direct_lags": _json_dict(direct_lags),
                "old_neighbor_lags": _json_dict(old_neighbor_lags),
                "new_neighbor_lags": _json_dict(neighbor_lags),
            })

    full_rows, rolling_rows, dm_rows = [], [], []
    rolling_results = {}
    for name, fit_fn in methods.items():
        try:
            result = fit_fn()
            used_lags = method_exog_lags[name]
            min_exog_lag = min(used_lags.values()) if used_lags else None
            free_months = p.FORECAST_HORIZON if min_exog_lag is None else min(min_exog_lag, p.FORECAST_HORIZON)
            full_rows.append({
                "lake": lake_name,
                "method": name,
                "rmse": result["rmse"],
                "mae": result["mae"],
                "nse": result["nse"],
                "pbias": result.get("pbias"),
                "n_eval": result["n_eval"],
                # 无外生基线的 used_lags 为 None，直接 len() 会抛 TypeError。
                "n_selected_vars": len(used_lags) if used_lags else 0,
                # 记录实际用的 ARIMA 阶数：阶数搜索在不同容器宿主上不完全
                # 可复现，写进结果表便于核对。XGBoost/Persistence 无此字段。
                "arima_order": str(result.get("order")) if "order" in result else None,
                "arima_seasonal_order": (str(result.get("seasonal_order"))
                                         if "seasonal_order" in result else None),
                "min_exog_lag": min_exog_lag,
                "n_foresight_free_months": free_months,
                "frac_foresight_free": free_months / p.FORECAST_HORIZON,
            })
            # Persistence 不拟合模型、结果里没有 "model" 键，送进
            # rolling_origin_sarimax 会抛 KeyError；共享库为它单独提供了
            # rolling_origin_persistence。
            if name.startswith("XGBoost"):
                rolling = p.rolling_origin_xgb(result)
            elif name == "Persistence":
                rolling = p.rolling_origin_persistence(wl_series, test_size=p.FORECAST_HORIZON)
            else:                                     # SARIMA / SARIMAX_*
                rolling = p.rolling_origin_sarimax(result, wl_series, test_size=p.FORECAST_HORIZON)
            rolling_results[name] = rolling
            for horizon, metrics in rolling.items():
                rolling_rows.append({
                    "lake": lake_name,
                    "method": name,
                    "horizon_months": horizon,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "nse": metrics["nse"],
                    "n_origins": metrics["n_origins"],
                    "min_exog_lag": min_exog_lag,
                    "requires_foresight": min_exog_lag is not None and horizon > min_exog_lag,
                })
        except Exception as exc:
            full_rows.append({
                "lake": lake_name,
                "method": name,
                "status": f"ERROR: {type(exc).__name__}: {exc}",
            })
            print(f"[{lake_name}] {name} failed: {type(exc).__name__}: {exc}", flush=True)

    # 14 对预先设定的比较：CCM 策略彼此之间，以及各自对无外生/不筛选基线——
    # RQ3 问的正是"因果筛选相对基线是否更准"。某个湖缺某个方法时下方 continue
    # 跳过该对，不报错。
    comparisons = [
        # CCM 策略之间（逐步扩大所用的因果信息）
        ("XGBoost_CCM_ancestors", "XGBoost_CCM_direct"),
        ("SARIMAX_CCM_ancestors", "SARIMAX_CCM_direct"),
        ("XGBoost_CCM_neighbor", "XGBoost_CCM_ancestors"),
        ("SARIMAX_CCM_neighbor", "SARIMAX_CCM_ancestors"),
        # 对传统统计变量选择
        ("XGBoost_CCM_ancestors", "XGBoost_Stepwise"),
        ("SARIMAX_CCM_ancestors", "SARIMAX_Stepwise"),
        # 对基线（无外生变量 / 不做因果筛选）
        ("XGBoost_CCM_ancestors", "XGBoost_AR_only"),
        ("XGBoost_CCM_ancestors", "Persistence"),
        ("XGBoost_CCM_ancestors", "XGBoost_all_vars"),
        ("XGBoost_CCM_direct", "XGBoost_AR_only"),
        ("SARIMAX_CCM_ancestors", "SARIMA"),
        ("SARIMAX_CCM_ancestors", "Persistence"),
        # 跨模型族
        ("XGBoost_CCM_ancestors", "SARIMAX_CCM_ancestors"),
        ("XGBoost_all_vars", "SARIMAX_all_vars"),
    ]
    for method_1, method_2 in comparisons:
        if method_1 not in rolling_results or method_2 not in rolling_results:
            continue
        for horizon in p.ROLLING_HORIZONS:
            actual, pred1, pred2 = p.align_rolling_by_origin(
                rolling_results[method_1], rolling_results[method_2], horizon
            )
            if actual is None:
                continue
            dm = p.diebold_mariano_test(actual, pred1, pred2, h=horizon)
            dm_rows.append({
                "lake": lake_name,
                "method_1": method_1,
                "method_2": method_2,
                "horizon_months": horizon,
                "dm_stat": dm["dm_stat"],
                "p_value": dm["p_value"],
                "n": dm["n"],
            })

    print(
        f"[{lake_name}] filtered forecast complete: methods={list(methods)}, "
        f"direct={direct_lags}, ancestors={ancestor_lags}, stepwise={stepwise_lags}, "
        f"neighbors={neighbor_lags}",
        flush=True,
    )
    return {
        "lake": lake_name,
        "full_rows": full_rows,
        "rolling_rows": rolling_rows,
        "dm_rows": dm_rows,
        "selected_rows": selected_rows,
    }


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1800, cpu=1.0, memory=2048)
def tune_hyperparams() -> dict:
    p = _configure_module()
    return p.tune_xgboost_hyperparams(p.LAKES)


def _apply_dm_fdr(dm_df):
    """对全部 Diebold–Mariano 检验施加 Benjamini–Hochberg FDR 校正。

    检验族的划定
    ------------
    **全部 DM 检验作为单一检验族**（湖泊 × 方法对 × 预测时距）。RQ3 由这整组
    对比共同回答，因此检验族按研究问题预先划定、不因结果而切分——与 CCM 边检验
    采用同一条原则（RQ1 的 420 条、RQ2 的 90 条各自成族）。文献中也有把 DM 与
    BH-FDR 合用的先例，但那里的族是「空间单元」（见 Environmental Research
    Letters 的数据驱动天气模式评估）；本研究改按研究问题划族，是更保守的划法
    （实测同一族下显著数 50，按方法对×时距分族则为 62）。

    依赖性
    ------
    BH 在独立或正回归依赖(PRDS)下控制 FDR。此处各检验共用预测误差序列
    （同一方法出现在多个对比中），属正依赖，BH 适用；若要在任意依赖下严格控制，
    应改用 Benjamini–Yekutieli，代价是显著数进一步下降。

    定位
    ----
    逐对 DM 不处理跨全部竞争设定的 data snooping——那是 White (2000) Reality
    Check、Hansen 的 SPA 与 Hansen–Lunde–Nason 的 Model Confidence Set 要解决的
    问题。BH 在此仅作为更简单的错误发现率控制，不冒充上述程序。

    n < 10 的对比返回 NaN 的 p 值（diebold_mariano_test 的样本量下限），不参与
    校正，其 p_fdr 保持 NaN。
    """
    from statsmodels.stats.multitest import multipletests

    if dm_df.empty or "p_value" not in dm_df.columns:
        return dm_df
    dm_df = dm_df.copy()
    dm_df["p_fdr"] = float("nan")
    valid = dm_df["p_value"].notna()
    if valid.any():
        dm_df.loc[valid, "p_fdr"] = multipletests(
            dm_df.loc[valid, "p_value"], alpha=0.05, method="fdr_bh")[1]
        print(f"DM 检验 BH-FDR：{int(valid.sum())} 个检验作一族，"
              f"未校正 p<0.05 有 {int((dm_df.loc[valid, 'p_value'] < 0.05).sum())} 个，"
              f"校正后 {int((dm_df.loc[valid, 'p_fdr'] < 0.05).sum())} 个",
              flush=True)
    return dm_df


# ---------------------------------------------------------------------------
# 服务端编排入口：客户端可断开，本机关机不影响运行
# ---------------------------------------------------------------------------
# 编排容器本身也可能被抢占，所以每个湖算完立刻写成 Volume 上的 JSON 分片并
# commit；retries 重跑时跳过已完成的湖，只补在途的。
#
#     modal run --detach code/04_forecast/modal_forecast_synchrony_filtered.py::detached
#
# 取回结果：
#     modal volume get ccm-data lake_results/final_v3/forecast_synchrony_filtered_*.csv results/

def _jsonable(obj):
    """numpy 标量 → Python 标量。分片必须存成真正的数字，否则合并后
    整列变成字符串，to_csv 出来的表与主流程不一致。"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


@app.function(image=image, volumes={DATA_ROOT: volume},
              timeout=10 * 3600, cpu=1.0, memory=4096, retries=2)
def orchestrate() -> dict:
    import pandas as pd

    p = _configure_module()
    volume.reload()                      # 看见上一次（被抢占前）提交的分片

    within_edges = _filtered_within_edges(pd, WITHIN_ORIGINAL_INPUT)
    inter_edges = _filtered_interlake_edges(pd, INTER_ORIGINAL_INPUT)
    print(f"within edges={len(within_edges)}  inter edges={len(inter_edges)}", flush=True)

    print("Tuning XGBoost hyperparameters once globally...", flush=True)
    xgb_params = p.tune_xgboost_hyperparams(FORECAST_LAKES)
    print(f"Selected XGBoost parameters: {xgb_params}", flush=True)

    out_dir_remote = OUT_DIR
    shard_dir = f"{OUT_DIR}/forecast_shards"
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(out_dir_remote, exist_ok=True)

    done = {n[:-5] for n in os.listdir(shard_dir) if n.endswith(".json")}
    todo = [lk for lk in FORECAST_LAKES if lk not in done]
    print(f"\n=== 已完成 {len(done)}｜待跑 {len(todo)} ===", flush=True)

    if todo:
        results = list(run_lake_synchrony_filtered.map(
            todo,
            [xgb_params] * len(todo),
            [within_edges.to_dict("records")] * len(todo),
            [inter_edges.to_dict("records")] * len(todo),
            return_exceptions=True,
        ))
        for lake, result in zip(todo, results):
            if isinstance(result, Exception):
                print(f"  [FAIL] {lake}: {type(result).__name__}: {result}", flush=True)
                continue
            with open(f"{shard_dir}/{lake}.json", "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, default=_jsonable)
            volume.commit()          # 立刻落盘：被抢占只损失在途的湖
            print(f"  [OK] {lake}", flush=True)

    rows = {"full_rows": [], "rolling_rows": [], "dm_rows": [], "selected_rows": []}
    present = sorted(n for n in os.listdir(shard_dir) if n.endswith(".json"))
    for name in present:
        with open(f"{shard_dir}/{name}", encoding="utf-8") as fh:
            shard = json.load(fh)
        for key in rows:
            rows[key].extend(shard.get(key, []))

    outputs = {
        "full": pd.DataFrame(rows["full_rows"]),
        "rolling": pd.DataFrame(rows["rolling_rows"]),
        "dm": _apply_dm_fdr(pd.DataFrame(rows["dm_rows"])),
        "selected": pd.DataFrame(rows["selected_rows"]),
    }
    for key, frame in outputs.items():
        path = f"{out_dir_remote}/{OUTPUT_NAMES[key]}"
        frame.to_csv(path, index=False)
        print(f"  wrote {path} ({len(frame)} rows)", flush=True)
    volume.commit()

    summary = {"lakes_done": len(present),
               **{k: len(v) for k, v in outputs.items()}}
    print(f"=== 完成：{summary} ===", flush=True)
    return summary


@app.local_entrypoint()
def detached():
    """spawn 服务端编排后立即返回。配合 `modal run --detach` 使用。"""
    call = orchestrate.spawn()
    print("已提交服务端编排")
    print(f"call id: {call.object_id}")
    print("本机现在可以关机。查看进度：modal app logs（或 Modal 网页控制台）")
    print("跑完取回结果：")
    print("  modal volume get ccm-data lake_results/final_v3/forecast_synchrony_filtered_full_results.csv results/")
