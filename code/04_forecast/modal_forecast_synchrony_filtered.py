"""Official Modal stage for the conditional forecasting experiment.
正式的 Modal 条件预测实验阶段。

Shared preprocessing and model functions come from
``01_analysis_core/analysis_core.py``. This stage defines the experiment-specific
predictor strategies, evaluates all methods, stores one resumable shard per lake
and merges only the complete ten-lake result set.
共享预处理与模型函数来自 ``01_analysis_core/analysis_core.py``。本阶段定义实验专用的
预测变量策略、评估全部方法、按湖保存可续跑分片，并仅合并完整的十湖结果。

Commands, inputs and expected outputs are listed in the repository README.
运行命令、输入与预期输出见仓库 README。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# Modal packages the shared scientific module from code/01_analysis_core/. / Modal 打包分析核心模块。
_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR / "01_analysis_core"))
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
    # Fix thread counts to reduce controllable run-to-run variation. / 固定线程数，减少可控的运行间差异。
    .env({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    })
    .add_local_python_source("analysis_core")
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=False)
DATA_ROOT = "/data"
# Output directory shared with the current CCM stages. / 与当前 CCM 阶段共用输出目录。
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
    """Configure shared analysis paths inside the Modal container.
    在 Modal 容器中配置共享分析路径。"""
    import analysis_core as p

    p.PKL_DIR = f"{DATA_ROOT}/lake_results"
    p.OUT_DIR = OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    # Use the authoritative Modal JSON. / 使用 Modal 中的正式参数 JSON。
    p.EMBED_PARAMS_PATH = EMBED_PARAMS_ABS
    return p


def _as_bool(value) -> bool:
    """Normalize stored values to Boolean form.
    将存储值规范为布尔值。"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _json_dict(values: dict) -> str:
    """Serialize a lag mapping with stable key order.
    按稳定键顺序序列化滞后映射。"""
    return json.dumps({str(k): int(v) for k, v in values.items()}, sort_keys=True)


def _select_stepwise_lags(panel_train, all_var_lags, wl_col="WL", criterion="aic"):
    """Select a forward-stepwise subset from the baseline lag pool.
    从基线滞后池中以前向逐步法选择变量子集。"""
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
    """Return retained within-lake driver edges for one target lake.
    返回单个目标湖泊保留的湖内驱动边。"""
    df = pd.read_csv(input_path)
    sig = df[df["causal_evidence"].map(_as_bool)].copy()
    sig["obs_lag"] = pd.to_numeric(sig["obs_lag"], errors="coerce")
    sig["obs_rho"] = pd.to_numeric(sig["obs_rho"], errors="coerce")
    # Keep d=0 for later re-selection in the forecast lag range. / 保留 d=0，随后在预测滞后范围内重新选择。
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
    """Return retained incoming between-lake edges for one target lake.
    返回指向单个目标湖泊的保留湖间边。"""
    df = pd.read_csv(input_path)
    sig = df[df["causal_evidence"].map(_as_bool)].copy()
    sig["obs_lag"] = pd.to_numeric(sig["obs_lag"], errors="coerce")
    sig["obs_rho"] = pd.to_numeric(sig["obs_rho"], errors="coerce")
    # Keep d=0 for forecast-constrained lag selection. / 保留 d=0 以便按预测范围重新选择滞后。
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
    """Read direct driver lags from retained within-lake edges.
    从保留的湖内边读取直接驱动滞后。"""
    df = pd.read_csv(WITHIN_ORIGINAL_INPUT)
    df["causal_evidence"] = df["causal_evidence"].map(_as_bool)
    sub = df[(df["lake"] == lake_name) & (df["causal_evidence"]) & (df["effect"] == "WL")]
    return {str(row["cause"]): int(row["obs_lag"]) for _, row in sub.iterrows()}


def _original_neighbor_lags(pd, lake_name):
    """Read neighbouring-lake lags from retained between-lake edges.
    从保留的湖间边读取邻湖滞后。"""
    df = pd.read_csv(INTER_ORIGINAL_INPUT)
    df["causal_evidence"] = df["causal_evidence"].map(_as_bool)
    sub = df[(df["effect_lake"] == lake_name) & (df["causal_evidence"])]
    return {str(row["cause_lake"]): int(row["obs_lag"]) for _, row in sub.iterrows()}


def _filter_long_gap_vars(p, panel_deseason, lag_sets, lake_name):
    """Remove predictors whose training gaps exceed the allowed limit.
    移除训练期缺口超过上限的预测变量。"""
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
    """Load, align and gap-filter neighbouring water-level series.
    读取、对齐并筛除长缺口邻湖水位序列。"""
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
    """Run every forecasting method for one lake and save its shard.
    运行单湖全部预测方法并保存分片。"""
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
    # Load the authoritative parameters produced by the embedding stage. / 读取嵌入阶段生成的正式参数。
    embed_params = p.load_embed_params(lake_name)

    lake_edges = within_edges[within_edges["lake"] == lake_name].copy()
    G = p.build_new_causal_network(lake_edges)
    # Re-select retained CCM edges within forecast-usable lags 1–12. / 在可用于预测的 1–12 月滞后内重新选择保留的 CCM 边。
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
    # Apply the same forecast-lag constraint to ancestor predictors. / 对祖先预测变量应用相同的预测滞后约束。
    ancestor_lags = p.select_ancestor_lags(
        G, ccm_train_panel, embed_params, log_prefix=f"[{lake_name}] ")
    # Baseline lags use training-only Pearson correlation, not CCM. / 基线滞后仅用训练期 Pearson 相关，不使用 CCM。
    all_var_lags = p.select_all_var_lags_xcorr(
        ccm_train_panel, embed_params, min_lag=1, log_prefix=f"[{lake_name}] ")
    # Apply the long-gap rule before stepwise selection. / 逐步选择前先应用长缺口规则。

    neighbor_lags = p.load_neighbor_lags(inter_edges, lake_name)
    # Re-select between-lake lags in the same 1–12 month forecast range. / 湖间边也在 1–12 月预测范围内重新选择滞后。
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
    # Align neighbour series before measuring their gaps. / 对齐邻湖序列后再计算缺口。
    neighbor_cols, neighbor_combined_lags, neighbor_lags = _load_neighbor_series(
        p, panel_deseason, neighbor_lags, lake_name)

    old_direct_lags = _original_within_direct_lags(pd, lake_name)
    old_neighbor_lags = _original_neighbor_lags(pd, lake_name)

    # Apply one long-gap rule to every strategy before selection. / 所有策略在变量选择前统一应用长缺口规则。
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

    # Baselines and CCM methods share one panel and one XGBoost configuration. / 基线与 CCM 方法共用同一面板和 XGBoost 参数。
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
        # Neighbour inputs are already aligned and gap-filtered. / 邻湖输入已完成对齐与缺口筛选。
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
                # Baselines without exogenous variables have no lag map. / 无外生变量基线没有滞后映射。
                "n_selected_vars": len(used_lags) if used_lags else 0,
                # Record fitted ARIMA orders for reproducibility checks. / 记录实际 ARIMA 阶数以便复现核对。
                "arima_order": str(result.get("order")) if "order" in result else None,
                "arima_seasonal_order": (str(result.get("seasonal_order"))
                                         if "seasonal_order" in result else None),
                "min_exog_lag": min_exog_lag,
                "n_foresight_free_months": free_months,
                "frac_foresight_free": free_months / p.FORECAST_HORIZON,
            })
            # Persistence uses its dedicated rolling evaluator. / 持续性方法使用专用滚动评估函数。
            if name.startswith("XGBoost"):
                rolling = p.rolling_origin_xgb(result)
            elif name == "Persistence":
                rolling = p.rolling_origin_persistence(wl_series, test_size=p.FORECAST_HORIZON)
            else:                                     # SARIMA or SARIMAX / SARIMA 或 SARIMAX
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

    # Predefined comparisons cover CCM strategies and their baselines. / 预设比较覆盖 CCM 策略及其基线。
    comparisons = [
        # Between CCM strategies / CCM 策略之间
        ("XGBoost_CCM_ancestors", "XGBoost_CCM_direct"),
        ("SARIMAX_CCM_ancestors", "SARIMAX_CCM_direct"),
        ("XGBoost_CCM_neighbor", "XGBoost_CCM_ancestors"),
        ("SARIMAX_CCM_neighbor", "SARIMAX_CCM_ancestors"),
        # Against statistical selection / 与统计变量选择比较
        ("XGBoost_CCM_ancestors", "XGBoost_Stepwise"),
        ("SARIMAX_CCM_ancestors", "SARIMAX_Stepwise"),
        # Against unfiltered or no-exogenous baselines / 与不筛选或无外生变量基线比较
        ("XGBoost_CCM_ancestors", "XGBoost_AR_only"),
        ("XGBoost_CCM_ancestors", "Persistence"),
        ("XGBoost_CCM_ancestors", "XGBoost_all_vars"),
        ("XGBoost_CCM_direct", "XGBoost_AR_only"),
        ("SARIMAX_CCM_ancestors", "SARIMA"),
        ("SARIMAX_CCM_ancestors", "Persistence"),
        # Across model families / 跨模型族
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
    """Return the fixed global XGBoost parameters used by this experiment.
    返回本实验使用的固定全局 XGBoost 参数。"""
    p = _configure_module()
    return p.tune_xgboost_hyperparams(p.LAKES)


def _apply_dm_fdr(dm_df):
    """Apply BH-FDR to valid Diebold-Mariano comparisons.
    对有效 Diebold–Mariano 比较应用 BH-FDR。"""
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


# Server-side orchestration persists one shard per lake for resume support. / 服务端编排按湖保存分片以支持续跑。

def _jsonable(obj):
    """Convert NumPy and missing values to JSON-safe Python values.
    将 NumPy 与缺测值转换为可写入 JSON 的 Python 值。"""
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
    """Run or resume all lake shards and merge complete outputs.
    运行或续跑全部湖泊分片，并合并完整结果。"""
    import pandas as pd

    p = _configure_module()
    volume.reload()                      # Load previously committed shards. / 读取此前已提交分片。

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
            volume.commit()          # Persist after each lake. / 每个湖完成后立即持久化。
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
    """Start server-side orchestration and return immediately.
    启动服务端编排并立即返回。"""
    call = orchestrate.spawn()
    print("已提交服务端编排")
    print(f"call id: {call.object_id}")
    print("本机现在可以关机。查看进度：modal app logs（或 Modal 网页控制台）")
    print("跑完取回结果：")
    print("  modal volume get ccm-data lake_results/final_v3/forecast_synchrony_filtered_full_results.csv results/")
