"""Shared scientific algorithms for preprocessing, CCM and forecasting.
数据预处理、CCM 与预测分析的共享科学算法。

This library is imported by the official ``modal_*.py`` stages and is not run
directly. It provides training-only preprocessing, missing-data rules, CCM with
IAAFT significance tests, forecast models and evaluation utilities.
本库由正式的 ``modal_*.py`` 阶段导入，不单独运行。它提供仅基于训练期的预处理、
缺测规则、带 IAAFT 显著性检验的 CCM、预测模型与评估函数。

Data retrieval and spatial masking remain in ``data_acquisition_lib.py``.
数据获取与空间掩膜位于 ``data_acquisition_lib.py``。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Configuration / 配置

START_YEAR, END_YEAR = 1994, 2024
WSC_BASE_URL = "https://wateroffice.ec.gc.ca/services/monthly_data/csv/inline"

# Repository-relative defaults support local rendering scripts; Modal stages replace them with container paths.
# 仓库相对路径供本地图表与数据集脚本读取；Modal 阶段会将其替换为容器路径。
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Repository root / 仓库根目录
_DEFAULT_RESULTS = os.path.join(_PKG_DIR, "results")

PKL_DIR = os.environ.get("CCM_PKL_DIR", os.path.join(_PKG_DIR, "lake_pkls"))
OUT_DIR = os.environ.get("CCM_OUT_DIR", _DEFAULT_RESULTS)
os.makedirs(OUT_DIR, exist_ok=True)
LOG_TXT = os.path.join(OUT_DIR, "run.log")


FORECAST_HORIZON = 37           # Fixed holdout near a 9:1 split. / 接近 9:1 切分的固定测试期。
N_SURROGATES = 500              # Gives minimum surrogate p = 1/501. / 替代检验最小 p 值为 1/501。
# Observed and surrogate series use the same signed lag family. / 观测与替代序列使用同一带符号滞后集合。
CCM_LAGS = list(range(-12, 13))
ROLLING_HORIZONS = [1, 3, 6, 12]
WL_AR_LAGS = (1, 2, 3, 6, 12)
EXOG_ANTECEDENT_WINDOWS = (3, 6)  # Antecedent-window lengths in months. / 前期累积窗口（月）。

MAX_FILLABLE_GAP_MONTHS = 6      # Fill short gaps; exclude variables with longer gaps. / 填补短缺口，长缺口变量整体排除。
OUTLIER_Z_THRESH = 6             # Robust z threshold for monthly changes. / 月度变化的稳健 z 阈值。
OUTLIER_LEVEL_Z_THRESH = 5       # Confirmation threshold for levels. / 水位值二次确认阈值。
TRAIN_GAP_WARN_FRACTION = 0.15   # Warn on missing fraction; exclusion uses gap length. / 缺测比例仅警告，排除依据为缺口长度。

LAKES = [
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
    "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake",
]


XGB_PARAM_GRID = [
    {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 200},
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
    {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 100},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 400},
    {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 200},
]

def log(msg):
    """Write a timestamped message to stdout and the run log.
    将带时间戳的消息写入终端和运行日志。"""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_TXT, "a", encoding="utf-8") as f:
        f.write(line + "\n")
# 2. Preprocessing: gauges, outliers, seasonality, panels and gaps / 预处理：测站、异常值、季节性、面板与缺测

def wl_station_outliers(series, z_thresh=OUTLIER_Z_THRESH, level_z_thresh=OUTLIER_LEVEL_Z_THRESH):
    """Return the two-stage outlier mask for one water-level gauge.
    返回单个水位站的两阶段异常值掩膜。"""
    s = series.dropna()
    diffs = s.diff().dropna()
    if len(diffs) < 10:
        return pd.DatetimeIndex([])
    med = diffs.median()
    mad = (diffs - med).abs().median()
    if mad == 0:
        return pd.DatetimeIndex([])
    robust_z = (diffs - med).abs() / (1.4826 * mad)
    flagged_diff_dates = set(diffs[robust_z > z_thresh].index)
    flagged_months = set()
    for d in flagged_diff_dates:
        flagged_months.add(d)
        flagged_months.add(d - pd.DateOffset(months=1))
    flagged_months = sorted(m for m in flagged_months if m in s.index)
    candidates = s.loc[flagged_months]
    if candidates.empty:
        return pd.DatetimeIndex([])

    level_med = s.median()
    level_mad = (s - level_med).abs().median()
    if level_mad == 0:
        return pd.DatetimeIndex([])
    level_robust_z = (candidates - level_med).abs() / (1.4826 * level_mad)
    return candidates[level_robust_z > level_z_thresh].index


def clean_wide_wl(wide, log_prefix=""):
    """Replace confirmed station outliers with NaN without dropping other gauges.
    将确认的测站异常值设为 NaN，同时保留其他测站。"""
    wide = wide.copy()
    for col in wide.columns:
        bad_idx = wl_station_outliers(wide[col])
        if len(bad_idx):
            log(f"{log_prefix}WL站点{col}: 排除{len(bad_idx)}个异常点(设为NaN): "
                f"{[str(d.date()) for d in bad_idx]}")
            wide.loc[bad_idx, col] = np.nan
    return wide


def combine_station_water_levels(wide, method="anomaly_mean"):
    """Combine centred gauge anomalies and restore the mean lake level.
    合成中心化后的测站距平，并恢复湖泊平均水位。"""
    wide = wide.sort_index()
    if method == "raw_mean":
        return wide.mean(axis=1, skipna=True)
    station_means = wide.mean(axis=0, skipna=True)
    global_mean = station_means.mean(skipna=True)
    return (wide - station_means).mean(axis=1, skipna=True) + global_mean


def deseasonalize(series, train_end=None):
    """Subtract monthly climatology estimated from the training period only.
    减去仅由训练期估计的逐月气候态。"""
    train_series = series if train_end is None else series.iloc[:train_end]
    monthly_clim = train_series.groupby(train_series.index.month).mean()
    month_of_each_point = pd.Series(series.index.month, index=series.index)
    return series - month_of_each_point.map(monthly_clim)


def build_variable_panel(wl_series, era5_vars, forecast_horizon=None):
    """Align variables to a complete calendar and return raw and deseasonalised panels.
    将变量对齐到完整月历，并返回原始及去季节化面板。"""
    panel = pd.DataFrame({"WL": wl_series})
    for name, s in era5_vars.items():
        panel[name] = s
    panel.index = pd.DatetimeIndex(panel.index)
    panel = panel.sort_index()
    full_calendar = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
    panel = panel.reindex(full_calendar)
    train_end = len(panel) - forecast_horizon if forecast_horizon else None
    panel_deseason = panel.apply(lambda col: deseasonalize(col, train_end=train_end))
    return panel, panel_deseason


def calendar_month_offsets(index, start=None, one_based=True):
    """Convert dated observations to integer month offsets.
    将日期观测转换为整数月份偏移。"""
    idx = pd.DatetimeIndex(index)
    start = pd.Timestamp(idx.min() if start is None else start)
    offsets = (idx.year * 12 + idx.month) - (start.year * 12 + start.month)
    return offsets + 1 if one_based else offsets


def longest_consecutive_gap_months(series):
    """Return the longest consecutive missing interval in calendar months.
    返回按日历月份计算的最长连续缺测期。"""
    is_na = series.isna().values
    max_run = 0
    cur_run = 0
    for v in is_na:
        if v:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def find_long_gap_vars(panel, candidate_vars, forecast_horizon, log_prefix=""):
    """Identify training variables whose longest gap exceeds the allowed limit.
    识别训练期最长缺口超过上限的变量。"""
    train = panel.iloc[:-forecast_horizon]
    bad = set()
    for var in candidate_vars:
        if var not in train.columns:
            continue
        gap = longest_consecutive_gap_months(train[var])
        if gap > MAX_FILLABLE_GAP_MONTHS:
            bad.add(var)
            log(f"{log_prefix}排除变量{var}: 训练窗口内最长连续缺口={gap}个月"
                f"(超过{MAX_FILLABLE_GAP_MONTHS}个月阈值)，按'长缺失应整段排除'原则不纳入候选"
                f"(只看训练期，不看测试期——测试期缺口在预测阶段单独处理，不用来决定候选资格)")
    return bad


def check_train_gap_warning(panel, cols, forecast_horizon, log_prefix=""):
    """Log variables whose training-period missing fraction exceeds the warning level.
    记录训练期缺测比例超过警告阈值的变量。"""
    train = panel.iloc[:-forecast_horizon]
    for col in cols:
        if col not in train.columns:
            continue
        frac_na = train[col].isna().mean()
        if frac_na > TRAIN_GAP_WARN_FRACTION:
            log(f"{log_prefix}警告: 变量{col}训练窗口缺测比例={frac_na:.1%}，结果需谨慎解读")


def _fill_series_block(s, limit):
    """Fill only missing runs no longer than the configured limit.
    仅填补不超过设定上限的连续缺测段。"""
    s = s.copy()
    is_na = s.isna().values
    n = len(s)
    i = 0
    while i < n:
        if not is_na[i]:
            i += 1
            continue
        j = i
        while j < n and is_na[j]:
            j += 1
        gap_len = j - i
        if gap_len <= limit:
            left_val = s.iloc[i - 1] if i > 0 else None
            right_val = s.iloc[j] if j < n else None
            if left_val is not None and right_val is not None:
                s.iloc[i:j] = np.linspace(left_val, right_val, gap_len + 2)[1:-1]
            elif left_val is not None:
                s.iloc[i:j] = left_val
            elif right_val is not None:
                s.iloc[i:j] = right_val
        i = j
    return s


def _fill_training_block(obj, limit=MAX_FILLABLE_GAP_MONTHS):
    """Apply short-gap filling independently to each training variable.
    分别对各训练变量进行短缺口填补。"""
    if isinstance(obj, pd.Series):
        return _fill_series_block(obj, limit)
    return obj.apply(lambda col: _fill_series_block(col, limit))


# 3. CCM: explicit embedding, lag maximisation and IAAFT tests / CCM：显式嵌入、滞后最大化与 IAAFT 检验

def iaaft_surrogate(x, n_iter=100, rng=None):
    """Generate an IAAFT surrogate preserving the value distribution and power spectrum.
    生成保留数值分布与功率谱的 IAAFT 替代序列。"""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    n = len(x)
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))
    surrogate = rng.permutation(x)
    for _ in range(n_iter):
        s_fft = np.fft.rfft(surrogate)
        new_fft = target_amp * np.exp(1j * np.angle(s_fft))
        s = np.fft.irfft(new_fft, n=n)
        surrogate = sorted_x[np.argsort(np.argsort(s))]
    return surrogate


def make_effect_embedding_dataframe(panel, cause_col, effect_col, lag, E_eff, tau_eff):
    """Build an explicit effect embedding for one directed CCM lag.
    为单个有向 CCM 滞后构建显式效应嵌入。"""
    if cause_col not in panel.columns or effect_col not in panel.columns:
        return None, None
    if E_eff is None or tau_eff is None:
        return None, None
    E_eff, tau_eff = int(E_eff), int(tau_eff)

    work = pd.DataFrame(index=panel.index)
    work[cause_col] = panel[cause_col]
    embed_cols = []
    for k in range(E_eff):
        col = f"{effect_col}__lag{lag}__emb{k}"
        work[col] = panel[effect_col].shift(-(lag - k * tau_eff))
        embed_cols.append(col)

    work = work.dropna(subset=[cause_col] + embed_cols)
    if len(work) < max(20, E_eff + 5):
        return None, None

    df_time = work.reset_index(drop=True)
    df_time.insert(0, "time", calendar_month_offsets(work.index))
    return df_time, embed_cols


def _extract_ccm_rho(result_df, source_col):
    """Extract the final valid CCM skill value from a pyEDM result.
    从 pyEDM 结果中提取最后一个有效 CCM 技巧值。"""
    if result_df is None or len(result_df) == 0:
        return np.nan
    non_lib_cols = [c for c in result_df.columns if c != "LibSize"]
    preferred = [c for c in non_lib_cols if str(c).endswith(f":{source_col}")]
    col = preferred[0] if preferred else (non_lib_cols[0] if non_lib_cols else None)
    if col is None:
        return np.nan
    return float(result_df[col].iloc[-1])


def ccm_rho_embedded(df_time, embed_cols, source_col, sample=1, seed=1):
    """Compute CCM skill from an explicit embedding at one library size.
    在单个库长度下根据显式嵌入计算 CCM 技巧值。"""
    import pyEDM
    n = len(df_time)
    if n < 20:
        return np.nan
    lib_max = n - 1
    if lib_max < max(10, len(embed_cols) + 2):
        return np.nan
    lib_sizes = f"{lib_max} {lib_max} 1"
    try:
        result = pyEDM.CCM(dataFrame=df_time, columns=" ".join(embed_cols), target=source_col,
                            E=len(embed_cols), embedded=True, libSizes=lib_sizes,
                            sample=sample, seed=seed)
        return _extract_ccm_rho(result, source_col)
    except Exception:
        return np.nan


def ccm_curve_embedded(df_time, embed_cols, source_col, sample=50, seed=1):
    """Compute the CCM convergence curve for an explicit embedding.
    计算显式嵌入的 CCM 收敛曲线。"""
    import pyEDM
    n = len(df_time)
    if n < 30:
        return pd.DataFrame({"LibSize": [], "rho": []})
    lib_max = n - 1
    lib_start = max(10, len(embed_cols) + 2, lib_max // 3)
    if lib_start >= lib_max:
        return pd.DataFrame({"LibSize": [], "rho": []})
    step = max(5, (lib_max - lib_start) // 8)
    lib_sizes = f"{lib_start} {lib_max} {step}"
    try:
        result = pyEDM.CCM(dataFrame=df_time, columns=" ".join(embed_cols), target=source_col,
                            E=len(embed_cols), embedded=True, libSizes=lib_sizes,
                            sample=sample, seed=seed)
        non_lib_cols = [c for c in result.columns if c != "LibSize"]
        preferred = [c for c in non_lib_cols if str(c).endswith(f":{source_col}")]
        rho_col = preferred[0] if preferred else (non_lib_cols[0] if non_lib_cols else None)
        if rho_col is None:
            return pd.DataFrame({"LibSize": [], "rho": []})
        return result[["LibSize", rho_col]].rename(columns={rho_col: "rho"})
    except Exception:
        return pd.DataFrame({"LibSize": [], "rho": []})


def max_over_lags(panel, cause_col, effect_col, E_eff, tau_eff, lags=CCM_LAGS):
    """Scan the signed lag family and return its maximum CCM skill.
    扫描带符号滞后集合并返回最大 CCM 技巧值。"""
    best_lag, best_rho, best_n = None, -np.inf, 0
    for lag in lags:
        df_time, embed_cols = make_effect_embedding_dataframe(panel, cause_col, effect_col, lag, E_eff, tau_eff)
        if df_time is None:
            continue
        rho = ccm_rho_embedded(df_time, embed_cols, cause_col, sample=1, seed=1)
        if rho is not None and not np.isnan(rho) and rho > best_rho:
            best_lag, best_rho, best_n = lag, rho, len(df_time)
    if best_lag is None:
        return None, np.nan, 0
    return best_lag, best_rho, best_n


def test_one_ccm_edge(panel, cause_col, effect_col, E_eff, tau_eff, n_surrogates=N_SURROGATES):
    """Test one directed edge with lag maximisation and IAAFT surrogates.
    使用滞后最大化与 IAAFT 替代序列检验单条有向边。"""
    obs_lag, obs_rho, obs_n = max_over_lags(panel, cause_col, effect_col, E_eff, tau_eff)
    if obs_lag is None or np.isnan(obs_rho):
        return {"cause": cause_col, "effect": effect_col, "status": "insufficient_data"}

    cause_mask = panel[cause_col].notna()
    cause_observed = panel.loc[cause_mask, cause_col].values
    if len(cause_observed) < 30:
        return {"cause": cause_col, "effect": effect_col, "status": "insufficient_cause_observations",
                "obs_lag": obs_lag, "obs_rho": obs_rho, "obs_n": obs_n}

    seed_text = f"{cause_col}|{effect_col}".encode("utf-8")
    seed_base = int(hashlib.md5(seed_text).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_base)

    null_max_rhos = np.empty(n_surrogates, dtype=float)
    for i in range(n_surrogates):
        surrogate_vals = iaaft_surrogate(cause_observed, n_iter=100, rng=rng)
        panel_s = panel.copy()
        panel_s.loc[cause_mask, cause_col] = surrogate_vals
        _, s_rho, _ = max_over_lags(panel_s, cause_col, effect_col, E_eff, tau_eff)
        null_max_rhos[i] = s_rho if s_rho is not None else np.nan

    valid_null = null_max_rhos[~np.isnan(null_max_rhos)]
    if len(valid_null) < n_surrogates * 0.5:
        return {"cause": cause_col, "effect": effect_col, "status": "insufficient_surrogates",
                "obs_lag": obs_lag, "obs_rho": obs_rho, "obs_n": obs_n,
                "n_valid_surrogates": int(len(valid_null))}
    p_value = (np.sum(valid_null >= obs_rho) + 1) / (len(valid_null) + 1)

    df_time, embed_cols = make_effect_embedding_dataframe(panel, cause_col, effect_col, obs_lag, E_eff, tau_eff)
    curve = ccm_curve_embedded(df_time, embed_cols, cause_col, sample=50, seed=1) if df_time is not None else pd.DataFrame()
    if len(curve):
        curve = curve.dropna(subset=["LibSize", "rho"]).sort_values("LibSize")

    convergence_diagnostic_pass = False
    kendall_tau = kendall_p = np.nan
    if len(curve) >= 4:
        kendall_tau, kendall_p = sp_stats.kendalltau(curve["LibSize"], curve["rho"])
        convergence_diagnostic_pass = bool((curve["rho"].iloc[-1] > 0) and (kendall_tau > 0) and (kendall_p < 0.05))

    return {
        "cause": cause_col, "effect": effect_col, "status": "OK",
        "obs_lag": int(obs_lag), "obs_rho": float(obs_rho), "obs_n": int(obs_n),
        "p_value": float(p_value), "null_mean": float(np.nanmean(null_max_rhos)),
        "null_std": float(np.nanstd(null_max_rhos)), "n_valid_surrogates": int(len(valid_null)),
        "convergence_diagnostic_pass": convergence_diagnostic_pass,
        "kendall_tau": float(kendall_tau) if not pd.isna(kendall_tau) else np.nan,
        "kendall_p": float(kendall_p) if not pd.isna(kendall_p) else np.nan,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def lag_scan(panel_df, cause, effect, embed_params, lags=None, seed=0):
    """Scan lags for predictor selection without retesting edge significance.
    扫描滞后用于预测变量选择，不重新判定边的显著性。"""
    if lags is None:
        lags = CCM_LAGS if "CCM_LAGS" in globals() else range(-12, 13)
    import pyEDM
    results = []
    for l in lags:
        col_name = f"{effect}__lag{l}"
        pair = panel_df[[cause, effect]].copy()
        pair[col_name] = panel_df[effect].shift(-l)
        pair = pair[[cause, col_name]]
        df_time = pair.reset_index(drop=True)
        df_time.insert(0, "time", calendar_month_offsets(pair.index))
        n_span = len(df_time)
        E_eff, tau_eff = embed_params[effect]["E"], embed_params[effect]["tau"]
        try:
            res = pyEDM.CCM(dataFrame=df_time, columns=col_name, target=cause, E=E_eff, tau=-tau_eff,
                             libSizes=f"{n_span} {n_span} 1", sample=1, seed=seed)
            rho = float(res[f"{col_name}:{cause}"].iloc[-1])
        except Exception:
            rho = np.nan
        results.append({"lag": l, "rho": rho})
    return pd.DataFrame(results)


# 4. Within-lake panel loading for 420 directed edges / 湖内 420 条有向边的面板读取

def load_lake_panel_for_ccm(lake_name):
    """Load one cleaned training panel and its authoritative embedding parameters.
    读取单湖清理后的训练面板及正式嵌入参数。"""
    with open(os.path.join(PKL_DIR, f"{lake_name}_result.pkl"), "rb") as f:
        cached = pickle.load(f)
    clean_wide = clean_wide_wl(cached["wide_wl"], log_prefix=f"[{lake_name}] ")
    combined_wl = combine_station_water_levels(clean_wide, method="anomaly_mean")
    real_predictors = cached["real_predictors"]
    _, panel_deseason = build_variable_panel(combined_wl, {c: real_predictors[c] for c in real_predictors.columns},
                                              forecast_horizon=FORECAST_HORIZON)
    panel_train = panel_deseason.iloc[:-FORECAST_HORIZON]
    return panel_train, load_embed_params(lake_name), cached


# Authoritative embedding parameters generated on Modal / Modal 生成的正式嵌入参数

EMBED_PARAMS = None
EMBED_PARAMS_PATH = os.environ.get(
    "CCM_EMBED_PARAMS", os.path.join(_DEFAULT_RESULTS, "embed_params_corrected.json")
)


def load_embed_params(lake_name):
    """Load one lake's authoritative embedding parameters from JSON.
    从 JSON 读取单湖正式嵌入参数。"""
    global EMBED_PARAMS
    if EMBED_PARAMS is None:
        try:
            with open(EMBED_PARAMS_PATH, encoding="utf-8") as f:
                EMBED_PARAMS = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load embedding parameters from {EMBED_PARAMS_PATH}. "
                "Run code/00_data_generation/modal_compute_embedding_params.py first."
            ) from exc
    if lake_name not in EMBED_PARAMS:
        raise KeyError(f"Embedding parameters missing for {lake_name}: {EMBED_PARAMS_PATH}")
    return EMBED_PARAMS[lake_name]




def apply_fdr_and_causal_evidence(df):
    """Apply BH-FDR, convergence checks and the temporal retention rule.
    依次应用 BH-FDR、收敛诊断与时序保留规则。"""
    from statsmodels.stats.multitest import multipletests
    df = df.copy()
    valid = (df["status"] == "OK") & df["p_value"].notna()
    df["p_fdr"] = np.nan
    if valid.any():
        _, fdr_p, _, _ = multipletests(df.loc[valid, "p_value"], alpha=0.05, method="fdr_bh")
        df.loc[valid, "p_fdr"] = fdr_p

    # Statistical layer: BH-FDR plus convergence. / 统计层：BH-FDR 与收敛诊断。
    df["statistically_significant"] = (
        (df["p_fdr"] < 0.05) & df["convergence_diagnostic_pass"].fillna(False) & df["obs_lag"].notna()
    )

    # Temporal layer after FDR: d>0 resolved, d=0 unresolved, d<0 rejected. / FDR 后按时序分类：正滞后保留、零滞后未决、负滞后拒绝。
    lag = pd.to_numeric(df["obs_lag"], errors="coerce")
    df["lag_resolution"] = np.where(
        ~df["statistically_significant"], "not_significant",
        np.where(lag > 0, "resolved",
                 np.where(lag == 0, "unresolved_contemporaneous", "rejected_reverse")),
    )
    df["causal_evidence"] = df["statistically_significant"] & (lag >= 0)

    return df


# 5. Between-lake water-level connectivity / 湖间水位连通性

def load_pair_panel_for_connectivity(lake_a, lake_b, forecast_horizon=FORECAST_HORIZON):
    """Build an aligned training panel for a directed lake-to-lake CCM test.
    构建用于湖间有向 CCM 检验的对齐训练面板。"""
    with open(os.path.join(PKL_DIR, f"{lake_a}_result.pkl"), "rb") as f:
        cached_a = pickle.load(f)
    with open(os.path.join(PKL_DIR, f"{lake_b}_result.pkl"), "rb") as f:
        cached_b = pickle.load(f)

    wl_a = combine_station_water_levels(clean_wide_wl(cached_a["wide_wl"], log_prefix=f"[{lake_a}] "), method="anomaly_mean")
    wl_b = combine_station_water_levels(clean_wide_wl(cached_b["wide_wl"], log_prefix=f"[{lake_b}] "), method="anomaly_mean")

    panel = pd.DataFrame({lake_a: wl_a, lake_b: wl_b})
    panel.index = pd.DatetimeIndex(panel.index)
    panel = panel.sort_index()
    full_calendar = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
    panel = panel.reindex(full_calendar)
    train_end = len(panel) - forecast_horizon if forecast_horizon else None
    panel_deseason = panel.apply(lambda col: deseasonalize(col, train_end=train_end))
    if forecast_horizon and forecast_horizon > 0:
        panel_deseason = panel_deseason.iloc[:-forecast_horizon]

    embed_a = load_embed_params(lake_a)["WL"]
    embed_b = load_embed_params(lake_b)["WL"]
    return panel_deseason, embed_a, embed_b
# 6. Forecast metrics and Diebold–Mariano tests / 预测指标与 Diebold–Mariano 检验
# KGE is omitted because its mean ratio is unstable for near-zero anomalies. / 距平均值接近零，故不使用均值比不稳定的 KGE。

def _forecast_metrics(actual, predicted):
    """Calculate forecast metrics after removing paired missing values.
    去除成对缺测值后计算预测指标。"""
    actual = pd.Series(np.asarray(actual, dtype=float))
    predicted = pd.Series(np.asarray(predicted, dtype=float))
    mask = actual.notna() & predicted.notna()
    if not mask.any():
        return {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "pbias": np.nan, "n_eval": 0}
    a, p = actual[mask].values, predicted[mask].values
    rmse = np.sqrt(np.mean((p - a) ** 2))
    mae = np.mean(np.abs(p - a))
    denom = np.sum((a - a.mean()) ** 2)
    nse = np.nan if denom == 0 else 1 - np.sum((a - p) ** 2) / denom
    sum_a = np.sum(a)
    pbias_unstable = abs(sum_a) < 1e-6 * max(1.0, np.sum(np.abs(a)))
    pbias = np.nan if pbias_unstable else 100.0 * np.sum(a - p) / sum_a
    return {"rmse": rmse, "mae": mae, "nse": nse, "pbias": pbias, "n_eval": int(mask.sum())}


def diebold_mariano_test(actual, pred1, pred2, h=1, loss="squared"):
    """Compare paired forecast errors using a Newey-West Diebold-Mariano test.
    使用 Newey-West Diebold–Mariano 检验比较配对预测误差。"""
    a = np.asarray(actual, dtype=float)
    p1 = np.asarray(pred1, dtype=float)
    p2 = np.asarray(pred2, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p1) | np.isnan(p2))
    a, p1, p2 = a[mask], p1[mask], p2[mask]
    n = len(a)
    if n < 10:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": n}
    e1, e2 = a - p1, a - p2
    d = (e1 ** 2 - e2 ** 2) if loss == "squared" else (np.abs(e1) - np.abs(e2))
    d_mean = d.mean()
    max_lag = max(0, h - 1)
    var_d = np.var(d, ddof=0)
    for lag in range(1, max_lag + 1):
        if n - lag < 2:
            break
        cov = np.cov(d[:-lag], d[lag:], ddof=0)[0, 1]
        var_d += 2 * (1 - lag / (max_lag + 1)) * cov
    var_d = max(var_d, 1e-12)
    dm_stat = d_mean / np.sqrt(var_d / n)
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(dm_stat)))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "n": int(n)}


def _arima_fit_is_degenerate(model):
    """Detect non-finite or implausibly large ARIMA fit values.
    识别非有限或异常巨大的 ARIMA 拟合值。"""
    try:
        bse = np.asarray(model.arima_res_.bse, dtype=float)
    except Exception:
        return False
    return len(bse) == 0 or not np.all(np.isfinite(bse))


def compute_vif(exog_df):
    """Compute variance-inflation factors for numeric predictors.
    计算数值预测变量的方差膨胀因子。"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = exog_df.dropna()
    if X.shape[1] < 2 or len(X) <= X.shape[1]:
        return pd.Series([np.nan] * X.shape[1], index=X.columns, dtype=float)
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)


# 7. Training-only global XGBoost tuning / 仅用训练期的全局 XGBoost 调参

def expanding_window_splits(n, n_splits=3, min_train_frac=0.5):
    """Create expanding-window training and validation splits.
    创建扩展窗口训练与验证切分。"""
    min_train = int(n * min_train_frac)
    remaining = n - min_train
    fold_size = remaining // (n_splits + 1)
    if fold_size < 5:
        return []
    splits = []
    for i in range(n_splits):
        train_end = min_train + i * fold_size
        val_end = train_end + fold_size
        if val_end > n:
            break
        splits.append((train_end, val_end))
    return splits


# XGBoost feature engineering / XGBoost 特征工程
# Every method uses the same water-level state features; only its exogenous set changes. / 各方法使用相同水位状态特征，仅外生变量集合不同。
# Selected exogenous lags also receive 3- and 6-month antecedent means. / 入选外生变量同时构造 3 与 6 个月前期均值。

def build_wl_state_features(wl_series, wl_ar_lags=WL_AR_LAGS):
    """Build lag, change and rolling-summary features for water level.
    构建水位滞后、变化率与滚动汇总特征。"""
    lag1, lag2, lag4 = wl_series.shift(1), wl_series.shift(2), wl_series.shift(4)
    feats = {f"WL_lag{k}": wl_series.shift(k) for k in wl_ar_lags}
    feats["WL_diff1"] = lag1 - lag2
    feats["WL_diff3"] = lag1 - lag4
    feats["WL_mean3"] = lag1.rolling(3).mean()
    feats["WL_mean6"] = lag1.rolling(6).mean()
    feats["WL_std3"] = lag1.rolling(3).std()
    return pd.DataFrame(feats, index=wl_series.index)


def wl_state_features_from_getter(get_wl, t, wl_ar_lags=WL_AR_LAGS):
    """Build one water-level feature row from a history accessor.
    根据历史值访问器构建一行水位特征。"""
    def get(offset):
        return get_wl(t - offset)
    row = {f"WL_lag{k}": get(k) for k in wl_ar_lags}
    v1, v2, v4 = get(1), get(2), get(4)
    row["WL_diff1"] = v1 - v2 if pd.notna(v1) and pd.notna(v2) else np.nan
    row["WL_diff3"] = v1 - v4 if pd.notna(v1) and pd.notna(v4) else np.nan
    window3 = [get(1), get(2), get(3)]
    window6 = window3 + [get(4), get(5), get(6)]
    row["WL_mean3"] = float(np.mean(window3)) if all(pd.notna(v) for v in window3) else np.nan
    row["WL_mean6"] = float(np.mean(window6)) if all(pd.notna(v) for v in window6) else np.nan
    row["WL_std3"] = float(np.std(window3, ddof=1)) if all(pd.notna(v) for v in window3) else np.nan
    return row


def wl_state_features_at(wl_values, t, wl_ar_lags=WL_AR_LAGS):
    """Build one water-level feature row at the requested position.
    在指定位置构建一行水位特征。"""
    def get_wl(idx):
        return wl_values.iloc[idx] if idx >= 0 else np.nan
    return wl_state_features_from_getter(get_wl, t, wl_ar_lags=wl_ar_lags)


def build_exog_antecedent_features(exog_df, exog_lags, windows=EXOG_ANTECEDENT_WINDOWS):
    """Build lagged and antecedent-window features for exogenous variables.
    构建外生变量的滞后与前期累积窗口特征。"""
    feats = {}
    for col, lag in exog_lags.items():
        feats[col] = exog_df[col].shift(lag)
        for w in windows:
            feats[f"{col}_mean{w}m"] = exog_df[col].rolling(w).mean().shift(lag)
    return pd.DataFrame(feats, index=exog_df.index)


def tune_xgboost_hyperparams(lake_names=LAKES, forecast_horizon=FORECAST_HORIZON):
    """Tune one global XGBoost configuration using training-only time-series CV.
    仅用训练期时间序列交叉验证选择一组全局 XGBoost 参数。"""
    import xgboost as xgb

    all_scores = {i: [] for i in range(len(XGB_PARAM_GRID))}
    n_folds_used = 0

    for lake_name in lake_names:
        pkl_path = os.path.join(PKL_DIR, f"{lake_name}_result.pkl")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            cached = pickle.load(f)
        clean_wide = clean_wide_wl(cached["wide_wl"], log_prefix=f"[{lake_name}/调优] ")
        combined_wl = combine_station_water_levels(clean_wide, method="anomaly_mean")
        _, panel_deseason = build_variable_panel(combined_wl, {}, forecast_horizon=forecast_horizon)
        wl_series = panel_deseason["WL"].sort_index().asfreq("MS")
        train_wl_full = wl_series.iloc[:-forecast_horizon]

        wl_state = build_wl_state_features(train_wl_full)
        combined = pd.concat([train_wl_full.rename("WL"), wl_state], axis=1).dropna()
        n = len(combined)
        if n < 60:
            continue

        for start, end in expanding_window_splits(n, n_splits=3):
            fold_train = combined.iloc[:start]
            fold_val = combined.iloc[start:end]
            if len(fold_train) < 30 or len(fold_val) < 5:
                continue
            X_train, y_train = fold_train.drop(columns="WL"), fold_train["WL"]
            X_val, y_val = fold_val.drop(columns="WL"), fold_val["WL"]
            n_folds_used += 1
            for i, params in enumerate(XGB_PARAM_GRID):
                model = xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=0, **params)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                rmse = float(np.sqrt(np.mean((pred - y_val.values) ** 2)))
                all_scores[i].append(rmse)

    avg_scores = {i: float(np.mean(scores)) for i, scores in all_scores.items() if scores}
    if not avg_scores:
        log("超参数调优：没有足够数据做CV，回退到经验性默认配置")
        return {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200}

    best_i = min(avg_scores, key=avg_scores.get)
    best_params = XGB_PARAM_GRID[best_i]
    log(f"超参数调优：{len(lake_names)}个湖泊、共{n_folds_used}折CV：")
    for i, params in enumerate(XGB_PARAM_GRID):
        marker = " <- 选中" if i == best_i else ""
        log(f"    {params}: 平均CV RMSE={avg_scores.get(i, float('nan')):.5f}{marker}")
    return best_params


# 8. SARIMA, SARIMAX, persistence and XGBoost fitting / 模型拟合

def fit_auto_sarima(wl_series, test_size=12, m=12):
    """Fit and forecast an automatically selected seasonal ARIMA model.
    拟合并预测自动选择的季节 ARIMA 模型。"""
    import pmdarima as pm
    wl_series = wl_series.sort_index().asfreq("MS")
    train_raw, test = wl_series[:-test_size], wl_series[-test_size:]
    train = _fill_training_block(train_raw).dropna()
    model = pm.auto_arima(train, seasonal=True, m=m, stepwise=True, suppress_warnings=True, error_action="ignore")
    if _arima_fit_is_degenerate(model):
        raise ValueError(f"auto_arima选中的模型{model.order}x{model.seasonal_order}协方差矩阵奇异/近奇异")
    preds, conf_int = model.predict(n_periods=test_size, return_conf_int=True)
    fc = pd.Series(np.asarray(preds), index=test.index)
    conf_df = pd.DataFrame(np.asarray(conf_int), index=test.index, columns=["lower", "upper"])
    return {**_forecast_metrics(test.values, fc.values), "order": model.order, "seasonal_order": model.seasonal_order,
            "predicted": fc, "actual": test, "conf_int": conf_df, "model": model, "has_exog": False}


def fit_auto_sarimax_multi(wl_series, exog_df, exog_lags, test_size=12, m=12):
    """Fit and forecast seasonal ARIMAX with selected exogenous variables.
    使用选定外生变量拟合并预测季节 ARIMAX。"""
    import pmdarima as pm
    wl_series = wl_series.sort_index().asfreq("MS")
    exog_df = exog_df.sort_index().asfreq("MS")
    exog_shifted = pd.DataFrame({col: exog_df[col].shift(lag) for col, lag in exog_lags.items()}, index=exog_df.index)
    # Fill the full series with the same gap limit before slicing the test period. / 切分测试期前按同一缺口上限填补完整序列。
    exog_shifted_ffilled = exog_shifted.ffill(limit=MAX_FILLABLE_GAP_MONTHS)
    combined = pd.concat([wl_series.rename("WL"), exog_shifted], axis=1).reindex(wl_series.index)
    train_raw, test_raw = combined.iloc[:-test_size], combined.iloc[-test_size:]
    test_wl = test_raw.iloc[:, 0]
    train = _fill_training_block(train_raw).dropna()
    train_wl, train_exog = train.iloc[:, 0], train.iloc[:, 1:]
    test_exog = exog_shifted_ffilled.iloc[-test_size:]
    if test_exog.isna().any().any():
        raise ValueError("测试期外生变量缺口超过MAX_FILLABLE_GAP_MONTHS上限，SARIMAX无法完成整块预测")
    # Return the fitted auto_arima model because refitting fixed orders can change its intercept choice. / 直接返回已拟合模型，避免固定阶数重拟合改变截距选择。
    model = pm.auto_arima(train_wl, X=train_exog.values, seasonal=True, m=m,
                          stepwise=True, suppress_warnings=True, error_action="ignore")
    if _arima_fit_is_degenerate(model):
        raise ValueError(f"auto_arima选中的模型{model.order}x{model.seasonal_order}协方差矩阵奇异/近奇异")
    preds, conf_int = model.predict(n_periods=test_size, X=test_exog.values, return_conf_int=True)
    fc = pd.Series(np.asarray(preds), index=test_raw.index)
    conf_df = pd.DataFrame(np.asarray(conf_int), index=test_raw.index, columns=["lower", "upper"])
    metrics = _forecast_metrics(test_wl.values, fc.values)
    return {**metrics, "order": model.order, "seasonal_order": model.seasonal_order, "vif": compute_vif(train_exog),
            "predicted": fc, "actual": test_wl, "conf_int": conf_df,
            "model": model, "has_exog": True, "exog_shifted": exog_shifted}


def fit_persistence(wl_series, test_size=12):
    """Forecast every test month with the final observed training value.
    用训练期最后一个观测值预测全部测试月份。"""
    wl_series = wl_series.sort_index().asfreq("MS")
    train, test = wl_series.iloc[:-test_size], wl_series.iloc[-test_size:]
    last_known = train.iloc[-1]
    fc = pd.Series([last_known] * test_size, index=test.index)
    return {**_forecast_metrics(test.values, fc.values), "predicted": fc, "actual": test}


def fit_xgboost_multi(wl_series, exog_df, exog_lags, xgb_params, test_size=12, wl_ar_lags=WL_AR_LAGS):
    """Fit XGBoost and recursively forecast the full test horizon.
    拟合 XGBoost 并递归预测完整测试期。"""
    import xgboost as xgb

    wl_series = wl_series.sort_index().asfreq("MS")
    exog_df = exog_df.sort_index().asfreq("MS") if len(exog_df.columns) else exog_df
    exog_shifted = build_exog_antecedent_features(exog_df, exog_lags)

    n = len(wl_series)
    train_end = n - test_size

    wl_state_train = build_wl_state_features(wl_series, wl_ar_lags=wl_ar_lags)
    features_train_full = pd.concat([wl_state_train, exog_shifted], axis=1)
    combined_train = pd.concat([wl_series.rename("WL"), features_train_full], axis=1).iloc[:train_end]
    train = _fill_training_block(combined_train).dropna()
    train_wl, train_X = train["WL"], train.drop(columns="WL")

    model = xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=0, **xgb_params)
    model.fit(train_X, train_wl)

    # Recursive lag features use the same short-gap-filled history as training. / 递归滞后特征使用与训练一致的短缺口填补历史。
    wl_values = wl_series.copy()
    wl_values.iloc[:train_end] = _fill_training_block(wl_series.iloc[:train_end])
    # Limit forward filling across train and test to short gaps only. / 训练与测试期的前向填补仅覆盖短缺口。
    exog_shifted_test = exog_shifted.ffill(limit=MAX_FILLABLE_GAP_MONTHS).iloc[train_end:]

    preds = []
    n_skipped = 0
    pred_index = wl_series.index[train_end:]
    for i in range(test_size):
        t = train_end + i
        row = wl_state_features_at(wl_values, t, wl_ar_lags=wl_ar_lags)
        for col in exog_shifted_test.columns:
            row[col] = exog_shifted_test.iloc[i][col]
        x_row = pd.DataFrame([row])[train_X.columns]
        if x_row.isna().any(axis=1).iloc[0]:
            # Keep an unforecastable month as NaN instead of discarding the full horizon. / 无法预测的月份保留 NaN，不废弃整个预测期。
            preds.append(np.nan)
            wl_values.iloc[t] = np.nan  # Prevent later lags from using the true test value. / 防止后续滞后特征使用测试期真值。
            n_skipped += 1
            continue
        pred_t = float(model.predict(x_row)[0])
        preds.append(pred_t)
        wl_values.iloc[t] = pred_t
    if n_skipped:
        log(f"    XGBoost递归预测: {n_skipped}/{test_size}个月因WL_lag回看到超过"
            f"{MAX_FILLABLE_GAP_MONTHS}个月的未填补缺口而跳过(留NaN，不计入评估)")

    fc = pd.Series(preds, index=pred_index)
    test_wl = wl_series.iloc[train_end:]
    metrics = _forecast_metrics(test_wl.values, fc.values)
    importance = pd.Series(model.feature_importances_, index=train_X.columns).sort_values(ascending=False)
    return {**metrics, "predicted": fc, "actual": test_wl, "feature_importance": importance,
            "model": model, "wl_series": wl_series, "exog_shifted": exog_shifted,
            "wl_ar_lags": wl_ar_lags, "train_X_columns": list(train_X.columns), "train_end": train_end}


# 9. Rolling-origin evaluation / 滚动起点评估

def rolling_origin_xgb(fit_result, horizons=ROLLING_HORIZONS):
    """Evaluate recursive XGBoost forecasts at rolling origins.
    在滚动起点上评估递归 XGBoost 预测。"""
    model = fit_result["model"]
    wl_series = fit_result["wl_series"]
    exog_shifted = fit_result["exog_shifted"]
    wl_ar_lags = fit_result["wl_ar_lags"]
    train_X_columns = fit_result["train_X_columns"]
    train_end = fit_result["train_end"]

    # Limit filling so origins with long exogenous gaps are skipped. / 限制填补长度，跳过外生变量长缺口对应的起点。
    exog_ffilled = exog_shifted.ffill(limit=MAX_FILLABLE_GAP_MONTHS)
    n = len(wl_series)
    max_h = max(horizons)
    # Retain origin indices so DM comparisons use matched origins. / 保留起点索引，使 DM 检验使用相同起点。
    pooled = {h: {"t0": [], "actual": [], "pred": []} for h in horizons}

    for t0 in range(train_end - 1, n - 1):
        steps_available = min(max_h, n - 1 - t0)
        if steps_available < 1:
            continue
        # Use the training-consistent short-gap-filled history. / 使用与训练一致的短缺口填补历史。
        wl_known = _fill_training_block(wl_series.iloc[: t0 + 1])
        mini_preds = {}

        def get_wl(idx):
            # Use recursive predictions after the origin and known history before it. / 起点后使用递归预测，起点前使用已知历史。
            if idx > t0:
                return mini_preds.get(idx, np.nan)
            elif idx >= 0:
                return wl_known.iloc[idx]
            return np.nan

        ok = True
        for step in range(1, steps_available + 1):
            t = t0 + step
            row = wl_state_features_from_getter(get_wl, t, wl_ar_lags=wl_ar_lags)
            for col in exog_ffilled.columns:
                row[col] = exog_ffilled.iloc[t][col] if t < len(exog_ffilled) else np.nan
            x_row = pd.DataFrame([row])[train_X_columns]
            if x_row.isna().any(axis=1).iloc[0]:
                ok = False
                break
            pred_t = float(model.predict(x_row)[0])
            mini_preds[t] = pred_t
            if step in horizons:
                pooled[step]["t0"].append(t0)
                pooled[step]["actual"].append(wl_series.iloc[t])
                pooled[step]["pred"].append(pred_t)
        if not ok:
            continue

    out = {}
    for h in horizons:
        a, p = pooled[h]["actual"], pooled[h]["pred"]
        out[h] = {**_forecast_metrics(a, p), "n_origins": len(a),
                   "t0": pooled[h]["t0"], "actual": a, "predicted": p}
    return out


def rolling_origin_sarimax(fit_result, wl_series, test_size=FORECAST_HORIZON, horizons=ROLLING_HORIZONS):
    """Evaluate SARIMA or SARIMAX forecasts at rolling origins.
    在滚动起点上评估 SARIMA 或 SARIMAX 预测。"""
    model = fit_result["model"]
    has_exog = fit_result.get("has_exog", False)
    exog_shifted = fit_result.get("exog_shifted")
    # Apply the same short-gap limit as rolling_origin_xgb. / 使用与 rolling_origin_xgb 相同的短缺口上限。
    exog_ffilled = exog_shifted.ffill(limit=MAX_FILLABLE_GAP_MONTHS) if has_exog else None

    wl_series = wl_series.sort_index().asfreq("MS")
    n = len(wl_series)
    train_end = n - test_size
    max_h = max(horizons)

    # Retain origin indices for matched DM comparisons. / 保留起点索引用于配对 DM 检验。
    pooled = {h: {"t0": [], "actual": [], "pred": []} for h in horizons}
    state_end = train_end - 1

    for t0 in range(train_end - 1, n - 1):
        if t0 > state_end:
            new_y = wl_series.iloc[state_end + 1: t0 + 1]
            try:
                if has_exog:
                    new_X = exog_ffilled.iloc[state_end + 1: t0 + 1]
                    if new_X.isna().any().any():
                        continue
                    model.update(new_y.values, X=new_X.values)
                else:
                    model.update(new_y.values)
                state_end = t0
            except Exception as e:
                log(f"    滚动更新在起点{t0}失败，提前结束: {type(e).__name__}: {e}")
                break

        steps_available = min(max_h, n - 1 - t0)
        if steps_available < 1:
            continue
        try:
            if has_exog:
                future_exog = exog_ffilled.iloc[t0 + 1: t0 + 1 + steps_available]
                if future_exog.isna().any().any() or len(future_exog) < steps_available:
                    continue
                preds = model.predict(n_periods=steps_available, X=future_exog.values)
            else:
                preds = model.predict(n_periods=steps_available)
        except Exception:
            continue

        preds = np.asarray(preds)
        for h in horizons:
            if h <= steps_available:
                pooled[h]["t0"].append(t0)
                pooled[h]["actual"].append(wl_series.iloc[t0 + h])
                pooled[h]["pred"].append(preds[h - 1])

    out = {}
    for h in horizons:
        a, p = pooled[h]["actual"], pooled[h]["pred"]
        out[h] = {**_forecast_metrics(a, p), "n_origins": len(a),
                   "t0": pooled[h]["t0"], "actual": a, "predicted": p}
    return out


def rolling_origin_persistence(wl_series, horizons=ROLLING_HORIZONS, test_size=FORECAST_HORIZON):
    """Evaluate persistence forecasts at rolling origins.
    在滚动起点上评估持续性预测。"""
    wl_series = wl_series.sort_index().asfreq("MS")
    n = len(wl_series)
    train_end = n - test_size
    max_h = max(horizons)
    pooled = {h: {"t0": [], "actual": [], "pred": []} for h in horizons}
    for t0 in range(train_end - 1, n - 1):
        last_known = wl_series.iloc[t0]
        if pd.isna(last_known):
            continue
        steps_available = min(max_h, n - 1 - t0)
        for h in horizons:
            if h <= steps_available:
                pooled[h]["t0"].append(t0)
                pooled[h]["actual"].append(wl_series.iloc[t0 + h])
                pooled[h]["pred"].append(last_known)
    out = {}
    for h in horizons:
        a, p = pooled[h]["actual"], pooled[h]["pred"]
        out[h] = {**_forecast_metrics(a, p), "n_origins": len(a),
                   "t0": pooled[h]["t0"], "actual": a, "predicted": p}
    return out


def align_rolling_by_origin(roll1, roll2, h):
    """Align two rolling forecasts on their shared origin indices.
    按共同起点索引对齐两组滚动预测。"""
    if h not in roll1 or h not in roll2:
        return None, None, None
    r1, r2 = roll1[h], roll2[h]
    common_t0 = sorted(set(r1["t0"]) & set(r2["t0"]))
    if len(common_t0) < 10:
        return None, None, None
    idx1 = {t0: i for i, t0 in enumerate(r1["t0"])}
    idx2 = {t0: i for i, t0 in enumerate(r2["t0"])}
    actual = [r1["actual"][idx1[t0]] for t0 in common_t0]
    pred1 = [r1["predicted"][idx1[t0]] for t0 in common_t0]
    pred2 = [r2["predicted"][idx2[t0]] for t0 in common_t0]
    return actual, pred1, pred2


# 10. Predictor selection from causal_evidence / 根据 causal_evidence 选择预测变量



def build_new_causal_network(lake_edges_df):
    """Build a directed graph from retained causal-evidence edges.
    根据保留的因果证据边构建有向图。"""
    G = nx.DiGraph()
    for _, row in lake_edges_df.iterrows():
        G.add_edge(row["cause"], row["effect"], rho=row["obs_rho"], lag=row["obs_lag"], p_fdr=row["p_fdr"])
    return G






def select_direct_predictor_lags(lake_edges_df):
    """Return direct driver lags for one target variable.
    返回单个目标变量的直接驱动滞后。"""
    wl_edges = lake_edges_df[lake_edges_df["effect"] == "WL"]
    return {row["cause"]: int(row["obs_lag"]) for _, row in wl_edges.iterrows()}


def best_positive_lag(scan):
    """Return the strongest forecast-usable positive lag.
    返回技巧值最高且可用于预测的正滞后。"""
    valid = scan[(scan["lag"] >= 1) & scan["rho"].notna()]
    if valid.empty:
        return None, None
    best_row = valid.loc[valid["rho"].idxmax()]
    return int(best_row["lag"]), float(best_row["rho"])


def select_ancestor_lags(G, ccm_train_panel, embed_params, log_prefix=""):
    """Select forecast lags for all retained ancestors of the target.
    为目标变量的全部保留祖先选择预测滞后。"""
    if "WL" not in G.nodes():
        return {}
    ancestor_vars = sorted(nx.ancestors(G, "WL"))
    ancestor_vars = [v for v in ancestor_vars if v in embed_params]
    lags = {}
    for var in ancestor_vars:
        if G.has_edge(var, "WL"):
            edge_lag = int(G[var]["WL"]["lag"])
            if edge_lag >= FORECAST_LAG_MIN:
                lags[var] = edge_lag
                continue
            # Negative lags were already rejected; only d=0 can remain here. / 负滞后已被拒绝，此处仅可能剩 d=0。
            new_lag, new_rho = forecast_constrained_lag(
                ccm_train_panel, var, "WL", embed_params)
            if new_lag is None:
                log(f"{log_prefix}祖先{var}: 因果最优 d={edge_lag}，预测域内无有效滞后，剔除")
            else:
                lags[var] = new_lag
                log(f"{log_prefix}祖先{var}: 因果最优 d={edge_lag} → "
                    f"预测域最优 d={new_lag} (rho={new_rho:.3f})")
        else:
            scan = lag_scan(ccm_train_panel, var, "WL", embed_params, lags=range(-12, 13))
            if scan.empty:
                continue
            l_star, rho_star = best_positive_lag(scan)
            if l_star is not None:
                lags[var] = l_star
            else:
                log(f"{log_prefix}间接祖先{var}: lag>=0的范围内没有任何有效rho，跳过不纳入")
    return lags


def select_all_var_lags_xcorr(ccm_train_panel, embed_params, wl_col="WL",
                              min_lag=1, max_lag=12, min_obs=20, log_prefix=""):
    """Select baseline predictor lags by training-only cross-correlation.
    仅根据训练期互相关选择基线预测变量滞后。"""
    lags = {}
    for var in embed_params:
        if var == wl_col or var not in ccm_train_panel.columns:
            continue
        best_lag, best_abs_r = None, -1.0
        for lag in range(min_lag, max_lag + 1):    # d>=1 excludes contemporaneous inputs. / d>=1 排除同期输入。
            pair = pd.concat([ccm_train_panel[var].shift(lag),
                              ccm_train_panel[wl_col]], axis=1).dropna()
            if len(pair) < min_obs:
                continue
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if np.isfinite(r) and abs(r) > best_abs_r:     # Strict comparison keeps the shorter tied lag. / 严格比较使并列时保留较短滞后。
                best_abs_r, best_lag = abs(r), lag
        if best_lag is not None:
            lags[var] = best_lag
            log(f"{log_prefix}{var}: 互相关最优滞后={best_lag} (|r|={best_abs_r:.3f})")
    return lags


FORECAST_LAG_MIN = 1        # d=0 would require the current true value. / d=0 需要当期真实值，故不用于预测。
FORECAST_LAG_MAX = 12


def forecast_constrained_lag(panel_df, cause, effect, embed_params,
                             lag_min=FORECAST_LAG_MIN, lag_max=FORECAST_LAG_MAX):
    """Re-select the best CCM lag within the forecast-usable lag range.
    在可用于预测的滞后范围内重新选择最佳 CCM 滞后。"""
    scan = lag_scan(panel_df, cause, effect, embed_params,
                    lags=range(lag_min, lag_max + 1))
    if scan.empty:
        return None, np.nan
    valid = scan[scan["rho"].notna()]
    if valid.empty:
        return None, np.nan
    best = valid.loc[valid["rho"].idxmax()]
    return int(best["lag"]), float(best["rho"])


def load_neighbor_lags(connectivity_df, lake_name):
    """Return retained incoming lake-neighbour lags for one target lake.
    返回指向目标湖泊的保留邻湖滞后。"""
    sub = connectivity_df[(connectivity_df["effect_lake"] == lake_name) & (connectivity_df["causal_evidence"] == True)]
    return {row["cause_lake"]: int(row["obs_lag"]) for _, row in sub.iterrows()}


def load_neighbor_wl_series(neighbor_lake_name):
    """Load and deseasonalise one neighbouring lake's water-level series.
    读取并去季节化单个邻湖水位序列。"""
    pkl_path = os.path.join(PKL_DIR, f"{neighbor_lake_name}_result.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        cached = pickle.load(f)
    clean_wide = clean_wide_wl(cached["wide_wl"], log_prefix=f"[{neighbor_lake_name}/邻居] ")
    combined_wl = combine_station_water_levels(clean_wide, method="anomaly_mean").sort_index().asfreq("MS")
    train_end = len(combined_wl) - FORECAST_HORIZON
    return deseasonalize(combined_wl, train_end=train_end)
if __name__ == "__main__":
    raise SystemExit(
        "analysis_core.py 是核心函数库，不应直接运行。论文结果的 Modal 执行入口：\n"
        "  湖内 CCM   modal run --detach code/02_within_lake_ccm/modal_within_lake_ccm.py\n"
        "  湖间 CCM   modal run --detach code/03_inter_lake_ccm/modal_inter_lake_ccm.py\n"
        "  条件预测   modal run --detach code/04_forecast/modal_forecast_synchrony_filtered.py::detached"
    )
