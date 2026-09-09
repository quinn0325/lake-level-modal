"""Shared library: preprocessing, CCM, surrogates, FDR, forecasting.

Everything downstream of data acquisition lives here. The stage scripts in
02_/03_/04_ are thin wrappers that parallelise calls into this module, so if two
stages ever disagree about how a panel is built, the bug is in here.

Entry point is a `{lake}_result.pkl` written by ccm_modal_app.process_lake_modal,
holding wide_wl and real_predictors. ERA5 download and catchment masking are not
duplicated here -- they live in ccm_lib.py and need CDS credentials plus the
HydroLAKES/HydroBASINS shapefiles.

Design decisions that change how results should be read
-------------------------------------------------------
Deseasonalisation uses training-period monthly climatology only. Computing it
over the full record and splitting afterwards leaks: a test-period value would
be inside the mean that is then subtracted from it. build_variable_panel() and
deseasonalize() therefore require an explicit train_end.

Long gaps drop a variable rather than being filled. find_long_gap_vars() checks
the training window only. Checking the test period too would catch more gaps,
but deciding whether a variable is eligible using test-period data is test-aware
model selection -- a subtler leak than the one it fixes. Test-period gaps are
instead handled at prediction time: the XGBoost recursion and both
rolling_origin_* functions skip the affected origins, and SARIMAX, which cannot
skip inside a single predict(), fails honestly.

Forecast features must lag by at least one month, and only non-negative lags are
eligible. best_positive_lag() takes the argmax over lag >= 0 rather than filtering
a global argmax afterwards: a variable whose global best is at lag -2 may still
have a perfectly usable peak at +3, and discarding it would throw away real
information. A negative lag means the effect moved before the cause, which is not
something you can forecast with.

Evaluation is in anomaly space throughout. This is deliberate -- scoring against
raw water level lets a model look good by predicting the seasonal cycle. It also
means the Climatology baseline degenerates to predicting zero, which is why it
is not among the baselines.

Perfect foresight is disclosed, not hidden. Methods with exogenous variables use
observed driver values over the test period, not forecasts of them. The question
being asked is how much predictive information the causal relationships carry,
not whether this could be deployed as a live warning system. `min_exog_lag`,
`n_foresight_free_months` and `requires_foresight` mark which results need
foresight: when h <= lag the driver values were already history at the forecast
origin and nothing is assumed.

Diebold-Mariano runs on the rolling-origin results, pooled by horizon, with the
Newey-West lag set to that horizon. Running it on a single 37-month trajectory
with h=1 would assume errors at horizons 30 and 31 are independent, which they
are not. Methods are aligned on the intersection of their origins before
comparison, since they skip different ones.

Known limitations, not fixed
----------------------------
Playgreen, Kiskitto and Sipiwesk have no water level at all for 2024, which sits
inside the 37-month test window. The NaN mask in _forecast_metrics() keeps those
months out of the scores, but n_eval for those three lakes is correspondingly
lower and should be quoted alongside their errors.

Pettitt tests find real non-seasonal level shifts in several lakes, probably
genuine changes in regulation. Deseasonalisation still uses one climatology per
series. Test periods usually fall entirely on one side of a break, so the effect
on forecasting is small; the effect on CCM significance is not characterised.

Evaporation shows a systematic anomaly across seven lakes in 2022-2023, most
likely an ERA5/ERA5T transition artefact. Left uncorrected.

Playgreen and Kiskitto share one regulated-outflow gauge (05UB009). That is
physical, not a bug -- both sit behind the Jenpeg dam -- and their water levels
are almost uncorrelated (r = -0.20), so the two RegFlow -> WL findings are not
redundant.
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

# ============================================================
# 1. 配置
# ============================================================

START_YEAR, END_YEAR = 1994, 2024
WSC_BASE_URL = "https://wateroffice.ec.gc.ca/services/monthly_data/csv/inline"

# 路径默认值取包内相对位置，不硬编码到某台机器上。Modal 上运行时这几个变量会被
# 容器路径覆盖（见 04_forecast 里的 _configure_module），所以默认值主要服务本地运行。
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
_DEFAULT_RESULTS = os.path.join(_PKG_DIR, "results")

PKL_DIR = os.environ.get("CCM_PKL_DIR", os.path.join(_PKG_DIR, "lake_pkls"))
OUT_DIR = os.environ.get("CCM_OUT_DIR", _DEFAULT_RESULTS)
os.makedirs(OUT_DIR, exist_ok=True)
LOG_TXT = os.path.join(OUT_DIR, "run.log")


FORECAST_HORIZON = 37          # 9:1切分附近，固定月份数；也是滚动起点评估能取到的样本量的来源
N_SURROGATES = 500              # IAAFT 替代序列个数。1/(N+1) 是 p 值的分辨率下限，必须
                                 # 细于 BH-FDR 的拒绝边界，否则边界附近的显著性判定会被
                                 # 单个替代序列的随机结果翻转。500 给出 1/501 ≈ 0.002。
CCM_LAGS = list(range(-12, 13))  # CCM候选滞后扫描范围(月)：统一的带符号扫描
# 每条边在 -12..+12 上做一次完整的带符号扫描，而不是先扫 0..12 做显著性、
# 再单独扫负滞后做诊断。这样每条边只有一个 lag profile 和一个全局最优 d，
# 也才能知道全局最优是不是落在非负窗口之外。
# 观测数据与每一个 IAAFT 替代序列都经由 max_over_lags 走同一个滞后族（本常量是
# 其默认值的唯一来源），p 值定义因此自洽，不存在滞后多重选择偏差。
# 代价是取最大值的候选数由 13 增至 25，零分布右移、门槛提高——保守方向。
ROLLING_HORIZONS = [1, 3, 6, 12]
WL_AR_LAGS = (1, 2, 3, 6, 12)
EXOG_ANTECEDENT_WINDOWS = (3, 6)  # 外生变量antecedent window特征(该滞后往前
                                   # 3/6个月的滚动均值)的窗口长度，见build_exog_
                                   # antecedent_features()的说明

MAX_FILLABLE_GAP_MONTHS = 6      # 半年。<= 6 个月的连续缺口允许插值/前后填充，更长的把
                                  # 整个变量排除。选半年这个整数而非贴着某个具体缺口的长度；
                                  # 已核实这个取值不改变任何外生候选变量的排除判定。
OUTLIER_Z_THRESH = 6            # WL异常检测：月度变化量的稳健z分数阈值
OUTLIER_LEVEL_Z_THRESH = 5      # WL异常检测：数值本身的稳健z分数阈值(二次确认)
TRAIN_GAP_WARN_FRACTION = 0.15  # 总缺测比例超过这个只警告，不排除(排除用的是长缺失判据)

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
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_TXT, "a", encoding="utf-8") as f:
        f.write(line + "\n")
# ============================================================
# 2. 预处理：多站合成 / 异常值剔除 / 去季节化 / 面板构建 / 长短缺失分级
# ============================================================

def wl_station_outliers(series, z_thresh=OUTLIER_Z_THRESH, level_z_thresh=OUTLIER_LEVEL_Z_THRESH):
    """单个WL站点异常值检测：先找月度变化量(一阶差分)的稳健异常，抓"骤变"；再用
    "数值本身相对该站点整体历史分布是不是也异常"做二次确认。只标"骤变+数值本身也是
    历史级异常"的点——正常但幅度较大的季节性变化(比如融雪期水位快速上涨)不会被
    误判，只有类似Kiskitto_Lake 2011年5-8月那种"数值在站点历史上从未出现过"的点
    才会被标记。返回应该设成NaN的时间点。"""
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
    """对每个WL站点分别做异常值检测，把确认的异常点设成NaN(不删除整行，其他站点
    该月份如果没问题继续保留)——不自动修正数值，只是不再让这几个点污染多站合成
    和去季节化。每一处改动都打印出来，这是"异常处理要留痕"的最低要求。"""
    wide = wide.copy()
    for col in wide.columns:
        bad_idx = wl_station_outliers(wide[col])
        if len(bad_idx):
            log(f"{log_prefix}WL站点{col}: 排除{len(bad_idx)}个异常点(设为NaN): "
                f"{[str(d.date()) for d in bad_idx]}")
            wide.loc[bad_idx, col] = np.nan
    return wide


def combine_station_water_levels(wide, method="anomaly_mean"):
    """多站水位合成：每站减自己长期均值(anomaly)->跨站平均->加回全部站均值的均值。
    支撑文献：Risien & Strub (2016, Scientific Data)同样用"减去长期均值再合并"的
    anomaly-referencing方法。"""
    wide = wide.sort_index()
    if method == "raw_mean":
        return wide.mean(axis=1, skipna=True)
    station_means = wide.mean(axis=0, skipna=True)
    global_mean = station_means.mean(skipna=True)
    return (wide - station_means).mean(axis=1, skipna=True) + global_mean


def deseasonalize(series, train_end=None):
    """去季节化：每个值减去该变量所在月份的多年平均值。月度气候态均值只用
    train_end之前(训练窗口)的数据算——同一套均值同时套用到训练段和测试段。
    不能用全部数据(含测试期)一起算月度均值再切分train/test：那样测试期某个月
    的真实值，在被算进"该月气候态均值"的那一刻，就已经把自己的信息用于标准化
    自己了，这是发生在train/test切分之前的信息泄漏，会让测试期指标虚高。
    train_end=None时退化为用全部数据算(仅用于探索性/诊断场景，正式训练/评估
    流程必须传入train_end)。"""
    train_series = series if train_end is None else series.iloc[:train_end]
    monthly_clim = train_series.groupby(train_series.index.month).mean()
    month_of_each_point = pd.Series(series.index.month, index=series.index)
    return series - month_of_each_point.map(monthly_clim)


def build_variable_panel(wl_series, era5_vars, forecast_horizon=None):
    """重建完整日历面板：从 wl_series + era5_vars 原始序列出发，reindex 到完整月历，
    真实缺测保留 NaN。
    去季节化时只用训练窗口(排除最后forecast_horizon个月测试期)算月度气候态均值，
    详见deseasonalize()的说明——这是修复train/test切分前信息泄漏的关键一步，
    forecast_horizon必须由调用方显式传入，不能省略。"""
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
    idx = pd.DatetimeIndex(index)
    start = pd.Timestamp(idx.min() if start is None else start)
    offsets = (idx.year * 12 + idx.month) - (start.year * 12 + start.month)
    return offsets + 1 if one_based else offsets


def longest_consecutive_gap_months(series):
    """训练窗口里最长的连续缺测段有多少个月。比"总缺测比例"更准确地判断一个变量
    是"短缺失可以插值"还是"长缺失应该整个排除"。"""
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
    """找出"长缺失"变量(最长连续缺口>MAX_FILLABLE_GAP_MONTHS)，这些变量应该在候选池
    阶段就整个排除，不再指望插值/前后填充硬填(比如Vaseux_Lake的RegFlow训练窗口里
    有连续18年缺测)。

    **只扫训练窗口，不扫全序列。** 用测试期的缺测情况决定"这个变量要不要进候选池"
    等于模型选择阶段偷看了未来——某个变量训练期完整、只是测试期恰好缺 8 个月，
    训练时根本不可能预先知道。测试期真的出现长缺口，由预测阶段各自处理：
    fit_xgboost_multi 的递归循环与两个 rolling_origin_* 只跳过受影响的月份/起点，
    SARIMAX 的单次 predict() 做不到部分跳过，缺口超限时整个方法诚实报错。
    返回该排除的变量名集合。"""
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
    """总缺测比例超过阈值只警告，不排除(排除用find_long_gap_vars的连续缺口判据)。"""
    train = panel.iloc[:-forecast_horizon]
    for col in cols:
        if col not in train.columns:
            continue
        frac_na = train[col].isna().mean()
        if frac_na > TRAIN_GAP_WARN_FRACTION:
            log(f"{log_prefix}警告: 变量{col}训练窗口缺测比例={frac_na:.1%}，结果需谨慎解读")


def _fill_series_block(s, limit):
    """对一列做"只填补长度<=limit的连续缺口"的插值：逐段扫描原始连续NaN游程，
    只有游程长度<=limit才填(两端都有锚点用线性插值，只有一端有锚点——缺口在序列
    开头或结尾——用那一端的值常数延展)；长度>limit的游程整段保持NaN，不填。

    不能写成 interpolate(limit=).ffill(limit=).bfill(limit=) 链式调用：三次限流
    互不知道对方填过什么，一个远超 limit 的缺口会被三段各填一点、合起来填满。"""
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
    """只填补长度<=limit个月的连续缺口，不会跨越长缺口硬填(见_fill_series_block)。"""
    if isinstance(obj, pd.Series):
        return _fill_series_block(obj, limit)
    return obj.apply(lambda col: _fill_series_block(col, limit))


# ============================================================
# 3. CCM核心：显式嵌入 + max-over-lags + IAAFT替代数据显著性检验
# ============================================================

def iaaft_surrogate(x, n_iter=100, rng=None):
    """Schreiber & Schmitz (1996) IAAFT替代数据：保留原序列振幅分布+功率谱，
    打乱与effect的时序耦合。"""
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
    """显式构造cause->effect在指定lag下的CCM输入表。effect(t+lag), effect(t+lag-tau),
    effect(t+lag-2*tau), ...(往过去看，不能写成+k*tau，那样会偷看未来——这是之前
    notebook审查发现并修复过的符号bug，这里直接用已验证正确的实现)。"""
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
    if result_df is None or len(result_df) == 0:
        return np.nan
    non_lib_cols = [c for c in result_df.columns if c != "LibSize"]
    preferred = [c for c in non_lib_cols if str(c).endswith(f":{source_col}")]
    col = preferred[0] if preferred else (non_lib_cols[0] if non_lib_cols else None)
    if col is None:
        return np.nan
    return float(result_df[col].iloc[-1])


def ccm_rho_embedded(df_time, embed_cols, source_col, sample=1, seed=1):
    """不传 sequential：pyEDM 2.5.0 起移除了该参数，不传则 2.4.0 与 2.5.x 都能跑。"""
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
    """对真实数据(或替代数据)扫描全部候选滞后，取最大rho。真实数据和每一个IAAFT
    替代数据都要走这同一个函数，才能构成公平的null分布(不能只对真实数据扫多个
    滞后取最大值，却让替代数据只测单个滞后)。"""
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
    """单条边(cause->effect)完整的滞后感知max-over-lags + IAAFT显著性检验，
    附带收敛诊断。返回一个dict，跟420条边/连通性分析用的是同一套逻辑。"""
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
    """只用来给"间接祖先但没有直接显著边"的变量挑一个结构上最优的滞后，不是显著性
    检验。跟ccm_lib.py::lag_scan逐字一致(非embedded写法，pyEDM自己内部做嵌入)。"""
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


# ============================================================
# 4. within-lake 全部边 CCM 检验(7变量x6方向=42条边/湖，10湖共420条边)
# ============================================================

def load_lake_panel_for_ccm(lake_name):
    """读取pkl，清洗WL异常值，重建完整日历面板，切掉最后
    FORECAST_HORIZON个月留给测试。返回(panel_train, embed_params, cached)。"""
    with open(os.path.join(PKL_DIR, f"{lake_name}_result.pkl"), "rb") as f:
        cached = pickle.load(f)
    clean_wide = clean_wide_wl(cached["wide_wl"], log_prefix=f"[{lake_name}] ")
    combined_wl = combine_station_water_levels(clean_wide, method="anomaly_mean")
    real_predictors = cached["real_predictors"]
    _, panel_deseason = build_variable_panel(combined_wl, {c: real_predictors[c] for c in real_predictors.columns},
                                              forecast_horizon=FORECAST_HORIZON)
    panel_train = panel_deseason.iloc[:-FORECAST_HORIZON]
    return panel_train, load_embed_params(lake_name, cached), cached


# ============================================================
# 嵌入参数来源：优先用修正版，PKL 缓存值仅作回退
# ============================================================

EMBED_PARAMS_CORRECTED = None       # 由 load_embed_params 惰性载入并缓存
EMBED_PARAMS_PATH = os.environ.get(
    "CCM_EMBED_PARAMS", os.path.join(_DEFAULT_RESULTS, "embed_params_corrected.json")
)


def load_embed_params(lake_name, cached=None):
    """返回该湖的 {变量: {"E":…, "tau":…}}。

    PKL 里缓存的 embed_params 由数据生成阶段算出，当时去季节化用了含测试期的
    全序列气候态，因此 E/τ 携带测试期信息——这是修复后**唯一残留**的泄漏点
    （两个面板构建函数本身早已改为只用训练期去季节化）。

    recompute_embed_params.py 用正确的训练期面板重算并写出
    embed_params_corrected.json；此处优先读它，读不到才回退 PKL 并告警，
    以免在缺文件时静默地用回带泄漏的值。
    """
    global EMBED_PARAMS_CORRECTED
    if EMBED_PARAMS_CORRECTED is None:
        try:
            with open(EMBED_PARAMS_PATH, encoding="utf-8") as f:
                EMBED_PARAMS_CORRECTED = json.load(f)
        except Exception:
            EMBED_PARAMS_CORRECTED = {}
    if lake_name in EMBED_PARAMS_CORRECTED:
        return EMBED_PARAMS_CORRECTED[lake_name]
    log(f"[{lake_name}] 警告：未找到修正版嵌入参数({EMBED_PARAMS_PATH})，"
        f"回退使用 PKL 缓存值——该值含测试期信息泄漏，仅供调试，不可用于正式结果")
    if cached is None:
        with open(os.path.join(PKL_DIR, f"{lake_name}_result.pkl"), "rb") as fh:
            cached = pickle.load(fh)
    return cached["embed_params"]




def apply_fdr_and_causal_evidence(df):
    """跨全部行统一做一次 BH-FDR 多重比较校正，随后叠加时序保留规则。

    产出三个字段：

      statistically_significant = FDR显著 且 收敛诊断通过 且 obs_lag非空
      lag_resolution            = not_significant / resolved(d>0) /
                                  unresolved_contemporaneous(d=0) / rejected_reverse(d<0)
      causal_evidence           = statistically_significant 且 d >= 0

    施加顺序为「先 FDR 后过滤」：BH 的检验族由**预先设定的候选边集合**划定，
    不因时序过滤而改变；过滤属结果解释层，作用于已校正的显著集合之上。

    检验族要预先划定好范围，不能中途扩张——这批 420 条边是一个完整的检验族，
    不要跟别的检验(比如湖泊间连通性)混在一起校正。
    """
    from statsmodels.stats.multitest import multipletests
    df = df.copy()
    valid = (df["status"] == "OK") & df["p_value"].notna()
    df["p_fdr"] = np.nan
    if valid.any():
        _, fdr_p, _, _ = multipletests(df.loc[valid, "p_value"], alpha=0.05, method="fdr_bh")
        df.loc[valid, "p_fdr"] = fdr_p

    # 第一层：统计显著性（不含任何时序判据）
    df["statistically_significant"] = (
        (df["p_fdr"] < 0.05) & df["convergence_diagnostic_pass"].fillna(False) & df["obs_lag"].notna()
    )

    # 第二层：时序保留规则（temporal retention rule，2026-08-26 锁定，看结果前定死）
    #   d > 0  → resolved：原因领先效应，符合因果时序，作为因果证据保留
    #   d = 0  → unresolved_contemporaneous：同月响应，保留但标注方向未定
    #   d < 0  → rejected_reverse：效应领先原因，不作为因果证据
    # 施加顺序为「先 FDR 后过滤」：BH 的检验族由预先设定的候选边集合划定，
    # 不因时序过滤而改变；过滤属于结果解释层，作用于已校正的显著集合之上。
    lag = pd.to_numeric(df["obs_lag"], errors="coerce")
    df["lag_resolution"] = np.where(
        ~df["statistically_significant"], "not_significant",
        np.where(lag > 0, "resolved",
                 np.where(lag == 0, "unresolved_contemporaneous", "rejected_reverse")),
    )
    df["causal_evidence"] = df["statistically_significant"] & (lag >= 0)

    return df


# ============================================================
# 5. 湖泊间(inter-lake) WL 连通性检验
# ============================================================

def load_pair_panel_for_connectivity(lake_a, lake_b, forecast_horizon=FORECAST_HORIZON):
    """两湖各自WL(清洗异常值+anomaly_mean合成)对齐到两湖日历并集，各自去季节化，
    不跨湖dropna。返回(panel, embed_a, embed_b)。"""
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

    embed_a = load_embed_params(lake_a, cached_a)["WL"]
    embed_b = load_embed_params(lake_b, cached_b)["WL"]
    return panel_deseason, embed_a, embed_b
# ============================================================
# 6. 评估指标：RMSE / MAE / NSE / PBIAS + Diebold-Mariano 检验
# ============================================================
# 不用 KGE：它的 β 项是 mean(predicted)/mean(actual)，在去季节化后的零均值距平
# 序列上分母近零、极不稳定。RMSE/MAE/NSE/PBIAS 都不依赖均值比值。

def _forecast_metrics(actual, predicted):
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
    """Diebold & Mariano (1995)检验：两个预测序列的精度差异是否统计显著。
    Newey-West长期方差估计(滞后阶数按原论文取h-1)处理预测误差的序列相关。"""
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
    try:
        bse = np.asarray(model.arima_res_.bse, dtype=float)
    except Exception:
        return False
    return len(bse) == 0 or not np.all(np.isfinite(bse))


def compute_vif(exog_df):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = exog_df.dropna()
    if X.shape[1] < 2 or len(X) <= X.shape[1]:
        return pd.Series([np.nan] * X.shape[1], index=X.columns, dtype=float)
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)


# ============================================================
# 7. XGBoost 超参数：全局一次性时间序列CV调优(只用训练期数据)
# ============================================================

def expanding_window_splits(n, n_splits=3, min_train_frac=0.5):
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


# ============================================================
# XGBoost特征工程：WL自身状态特征 + 外生变量antecedent window特征
# ============================================================
# 之前只有"WL_lag{1,2,3,6,12}"+"每个外生变量CCM选定滞后处的单点值"，基本是把
# XGBoost当线性自回归模型在用，没发挥树模型的特征交互能力，而且水位这类有明显
# persistence/惯性的变量，变化率(在涨还是在跌)、前期累积条件往往比单点滞后值
# 更有信息量。这里补充：
#   WL状态特征：滞后项 + 一阶/三阶变化率(WL_diff1/WL_diff3) + 3/6个月滚动均值
#   (WL_mean3/WL_mean6) + 3个月滚动标准差(WL_std3)。
#   外生变量：除了CCM选定滞后处的单点值，再加该滞后往前3/6个月的滚动均值
#   (antecedent window，水文系统的前期累积水量条件通常比单月值更有解释力)。
# 关键原则：全部XGBoost方法(AR_only/CCM_top/CCM_direct/CCM_ancestors/all_vars/
# CCM_neighbor)必须用同一套构造函数，只有"喂哪些外生变量"因方法而异——不能
# 只给CCM方法加这些特征、AR_only或all_vars不加，否则最终"CCM_ancestors比
# AR_only准"这类结论，分不清是"CCM筛选真的有效"还是"CCM方法额外被喂了更好的
# 人工特征"这两种解释，实验就不公平了。
# 也刻意没有做的事(避免特征爆炸)：没有加日历/季节特征、没有加变量间交互项、
# 没有给每个变量加更多滚动窗口——训练样本量只有~300个月量级，"all_vars"方法
# 如果塞进6个气候变量，特征数已经逼近10(WL状态)+6*3(每个外生变量单点+两个
# antecedent window)=28个，特征数:样本数已经到1:10左右，不宜再加。

def build_wl_state_features(wl_series, wl_ar_lags=WL_AR_LAGS):
    """WL自身状态特征，向量化版本(训练用)：对整段Series直接shift()/rolling()。"""
    lag1, lag2, lag4 = wl_series.shift(1), wl_series.shift(2), wl_series.shift(4)
    feats = {f"WL_lag{k}": wl_series.shift(k) for k in wl_ar_lags}
    feats["WL_diff1"] = lag1 - lag2
    feats["WL_diff3"] = lag1 - lag4
    feats["WL_mean3"] = lag1.rolling(3).mean()
    feats["WL_mean6"] = lag1.rolling(6).mean()
    feats["WL_std3"] = lag1.rolling(3).std()
    return pd.DataFrame(feats, index=wl_series.index)


def wl_state_features_from_getter(get_wl, t, wl_ar_lags=WL_AR_LAGS):
    """跟build_wl_state_features()同一套构造规则，但通过get_wl(idx)这个可调用
    对象逐点取值，不假设有一个现成的、可以直接shift()/rolling()的Series——
    fit_xgboost_multi的整段递归预测用一个wl_values Series查表即可(见
    wl_state_features_at)，rolling_origin_xgb每个起点的mini-walk则是"起点前
    用固定的已知历史，起点后用这次mini-walk自己预测出的值"两段拼接，取值
    逻辑不同，但特征定义公式必须完全一致，所以抽成这样一个通用版本，两处各自
    传一个合适的get_wl。返回值必须跟build_wl_state_features()在同样输入下
    逐位对齐，否则训练/预测两边特征定义不一致，模型学到的规律用错了地方。"""
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
    """wl_state_features_from_getter()的简单包装：整段历史都在一个Series里
    (wl_values)时用这个，见fit_xgboost_multi。"""
    def get_wl(idx):
        return wl_values.iloc[idx] if idx >= 0 else np.nan
    return wl_state_features_from_getter(get_wl, t, wl_ar_lags=wl_ar_lags)


def build_exog_antecedent_features(exog_df, exog_lags, windows=EXOG_ANTECEDENT_WINDOWS):
    """每个被选中的外生变量：CCM选定滞后处的单点值 + 该滞后往前3/6个月的滚动
    均值(antecedent window)。外生变量从不递归预测(始终用真实观测值/有限ffill，
    见fit_xgboost_multi的"完美强迫"说明)，所以不需要像WL状态特征那样另外写
    一个按位置查表的版本，训练和预测都可以直接向量化shift()/rolling()。"""
    feats = {}
    for col, lag in exog_lags.items():
        feats[col] = exog_df[col].shift(lag)
        for w in windows:
            feats[f"{col}_mean{w}m"] = exog_df[col].rolling(w).mean().shift(lag)
    return pd.DataFrame(feats, index=exog_df.index)


def tune_xgboost_hyperparams(lake_names=LAKES, forecast_horizon=FORECAST_HORIZON):
    """只用WL自身滞后(AR-only)特征，在全部指定湖泊的训练期内部做扩展窗口CV，挑一组
    在多个湖上平均表现最好的超参数，全部湖泊统一使用(不按湖单独调，避免小样本
    调参过拟合调参本身)。"""
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


# ============================================================
# 8. 拟合函数：SARIMA/SARIMAX/Persistence/XGBoost
# ============================================================

def fit_auto_sarima(wl_series, test_size=12, m=12):
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
    import pmdarima as pm
    wl_series = wl_series.sort_index().asfreq("MS")
    exog_df = exog_df.sort_index().asfreq("MS")
    exog_shifted = pd.DataFrame({col: exog_df[col].shift(lag) for col, lag in exog_lags.items()}, index=exog_df.index)
    # 测试期外生变量的ffill要在切片前、对完整(训练+测试)序列做，并且限流到
    # MAX_FILLABLE_GAP_MONTHS：(1)只对窄切片ffill的话，如果测试期起始几个月本身
    # 就是NaN，切片内部找不到锚点可垫，即使训练期紧邻着有合法观测值也补不到；
    # (2)不限流的话，测试期哪怕连续缺测远超"长缺失"阈值，也会被一路拿训练期最后
    # 一个值垫到底，跟训练期"长缺失应整段排除"的处理原则不一致。
    exog_shifted_ffilled = exog_shifted.ffill(limit=MAX_FILLABLE_GAP_MONTHS)
    combined = pd.concat([wl_series.rename("WL"), exog_shifted], axis=1).reindex(wl_series.index)
    train_raw, test_raw = combined.iloc[:-test_size], combined.iloc[-test_size:]
    test_wl = test_raw.iloc[:, 0]
    train = _fill_training_block(train_raw).dropna()
    train_wl, train_exog = train.iloc[:, 0], train.iloc[:, 1:]
    test_exog = exog_shifted_ffilled.iloc[-test_size:]
    if test_exog.isna().any().any():
        raise ValueError("测试期外生变量缺口超过MAX_FILLABLE_GAP_MONTHS上限，SARIMAX无法完成整块预测")
    # 不要改成「先取阶数、再按阶数重新拟合」：auto_arima 在搜索中一并决定截距项
    # (d=1 时关闭截距)，只钉 (p,d,q)(P,D,Q,m) 重拟合会得到另一个模型——实测同一组
    # 阶数下 27/31 个方法的 RMSE 改变，Split_Lake 的 SARIMA 从 1.2426 变成 1.2883。
    # 阶数搜索本身经实测是稳定的(两次独立运行 31/31 一致)，无需固化。
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
    """naive基线：预测=最后一个已知观测值(重复到底)。"""
    wl_series = wl_series.sort_index().asfreq("MS")
    train, test = wl_series.iloc[:-test_size], wl_series.iloc[-test_size:]
    last_known = train.iloc[-1]
    fc = pd.Series([last_known] * test_size, index=test.index)
    return {**_forecast_metrics(test.values, fc.values), "predicted": fc, "actual": test}


def fit_xgboost_multi(wl_series, exog_df, exog_lags, xgb_params, test_size=12, wl_ar_lags=WL_AR_LAGS):
    """WL自身滞后特征在测试期内做真正的递归多步预测：预测第t个月时，如果"t减k个月"
    这个位置本身也落在测试期里，用模型自己之前预测出来的值，不用真实观测值。外生
    变量依然用真实观测值("完美强迫"假设)。

    特征工程(WL状态特征+外生变量antecedent window特征)对全部方法用同一套
    build_wl_state_features/build_exog_antecedent_features构造规则，只有
    "喂哪些外生变量"因方法而异，详见这两个函数前的说明。"""
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

    # 递归预测时WL_lag特征往训练期回看，要用跟训练时同一套"短缺失已填补"的历史，
    # 不能用原始未填补的wl_series——否则训练期里任何一个短缺口(哪怕只有1个月、
    # 明明在MAX_FILLABLE_GAP_MONTHS允许填补的范围内)一旦被某个WL_lag回看到，
    # 递归预测第一步就会直接崩(实测验证过：Playgreen_Lake训练期最后一个月
    # 2021-11本身缺测，WL_lag1在测试期第一步(2021-12)回看到它，如果不修，
    # 整个方法直接报错退出，而不是"填补后正常预测")。测试期部分保留原始值即可，
    # 反正下面的循环会按顺序把每个测试期位置都覆盖成模型自己的预测值。
    wl_values = wl_series.copy()
    wl_values.iloc[:train_end] = _fill_training_block(wl_series.iloc[:train_end])
    # ffill整段(训练+测试)后再切出测试段，并且限流到MAX_FILLABLE_GAP_MONTHS：
    # 只ffill测试期窄切片的话，测试期开头连续几个月本身缺测就补不到训练期的值；
    # 不限流的话，测试期缺口再长也会被一路垫到底，跟"长缺失应整段排除"矛盾。
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
            # 某个WL_lag回看到了一个真正超过MAX_FILLABLE_GAP_MONTHS、填不了的长缺口
            # (比如Playgreen_Lake训练期末尾2020-11~2021-03连续5个月缺测)。不应该
            # 因为个别月份预测不了就让整个37个月的结果全部作废——只跳过这一步，
            # 留NaN(_forecast_metrics的NaN掩码会正确把这个月排除在评估外)，
            # wl_values这个位置也保持NaN，后面步骤如果WL_lag回看到它会一样正确地
            # 传播成NaN(不会拿一个不存在的预测值当真)。等回看窗口滑过这段缺口，
            # 后面的月份该能正常预测就会恢复正常。
            preds.append(np.nan)
            wl_values.iloc[t] = np.nan  # 显式清空，不能让这个位置保留wl_series.copy()
                                        # 带过来的原始真实观测值，否则后续步骤的WL_lag
                                        # 回看到这里会悄悄用上测试期真值，重新打开
                                        # "递归预测不该偷看测试期真实观测值"这个已经
                                        # 修过的泄漏口子。
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


# ============================================================
# 9. 滚动起点(rolling-origin)评估
# ============================================================

def rolling_origin_xgb(fit_result, horizons=ROLLING_HORIZONS):
    """XGBoost已经是递归实现，逐起点滚动几乎不加成本。每个起点t0独立mini-walk，
    互不共享predicted history。"""
    model = fit_result["model"]
    wl_series = fit_result["wl_series"]
    exog_shifted = fit_result["exog_shifted"]
    wl_ar_lags = fit_result["wl_ar_lags"]
    train_X_columns = fit_result["train_X_columns"]
    train_end = fit_result["train_end"]

    # 限流到MAX_FILLABLE_GAP_MONTHS：测试期任意起点的外生变量缺口如果超过这个
    # 上限，不应该被一路垫到底，让下面每个起点的构造特征在遇到超限缺口时正常
    # 触发NaN检查、跳过该起点，而不是悄悄拿很久以前的值填。
    exog_ffilled = exog_shifted.ffill(limit=MAX_FILLABLE_GAP_MONTHS)
    n = len(wl_series)
    max_h = max(horizons)
    # 记录每条(actual,predicted)对应的起点位置t0(不只是位置对齐的list)：
    # DM检验要比较两个方法在同一个horizon上的预测精度，前提是两个方法用的是
    # 同一批起点——不同方法因为各自的NaN/长缺口跳过情况不同，成功产出预测的
    # 起点集合可能不完全一样，光看list位置对不上号，必须靠t0对齐取交集。
    pooled = {h: {"t0": [], "actual": [], "pred": []} for h in horizons}

    for t0 in range(train_end - 1, n - 1):
        steps_available = min(max_h, n - 1 - t0)
        if steps_available < 1:
            continue
        # 跟fit_xgboost_multi里的道理一样：回看用的历史要跟训练时同一套"短缺失已
        # 填补"处理，不能用原始未填补的wl_series——否则任何一个短缺口(哪怕在
        # MAX_FILLABLE_GAP_MONTHS允许范围内)一旦被某个WL_lag回看到，这个起点就会
        # 被跳过(ok=False)，白白损失掉本可以正常预测的样本。长于limit的缺口
        # 依然保持NaN，该起点该跳过还是会跳过，不受影响。
        wl_known = _fill_training_block(wl_series.iloc[: t0 + 1])
        mini_preds = {}

        def get_wl(idx):
            # 闭包捕获当前t0/wl_known/mini_preds(mini_preds是原地更新，不是
            # 重新赋值，闭包读到的始终是最新内容)：idx落在起点t0之后就查这次
            # mini-walk自己预测出的值，落在t0及之前就查固定的已知历史。
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
    """SARIMAX/SARIMA滚动起点评估：用pmdarima的update()方法做轻量状态更新，不是
    每个起点都重新完整网格搜索定阶——这是计算成本约束下的工程简化，不是教科书
    验证过的、跟完整滚动重新拟合统计上等价的方法，写方法部分要如实说明。"""
    model = fit_result["model"]
    has_exog = fit_result.get("has_exog", False)
    exog_shifted = fit_result.get("exog_shifted")
    # 限流到MAX_FILLABLE_GAP_MONTHS，理由同rolling_origin_xgb。
    exog_ffilled = exog_shifted.ffill(limit=MAX_FILLABLE_GAP_MONTHS) if has_exog else None

    wl_series = wl_series.sort_index().asfreq("MS")
    n = len(wl_series)
    train_end = n - test_size
    max_h = max(horizons)

    # 同样记录t0，理由同rolling_origin_xgb：两个方法在同一horizon上做DM比较前，
    # 必须先按起点对齐取交集，不能假设两个方法产出的起点list天然一一对应。
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
    """Persistence基线没有一个可以调用rolling_origin_xgb/sarimax的"model"对象——
    它本身就是"重复起点当月的已知值"，不需要拟合。单独补一个滚动起点版本，
    只是为了让DM检验里涉及Persistence的两个比较也能用跟其余方法同样的"同一
    horizon、同一批起点"口径，而不是退回到单起点37个月轨迹那种错误结构。"""
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
    """两个方法在同一horizon h上的滚动起点结果按t0取交集对齐，返回
    (actual, pred1, pred2)——DM检验要求比较的是"同一批预测目标"，不能假设两个
    方法的起点list天然一一对应(不同方法因为各自的NaN/缺口跳过情况不同，成功
    产出预测的起点集合可能不完全一样)。"""
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


# ============================================================
# 10. 候选变量选择：从420条边merged_fdr.csv的causal_evidence读
# ============================================================



def build_new_causal_network(lake_edges_df):
    G = nx.DiGraph()
    for _, row in lake_edges_df.iterrows():
        G.add_edge(row["cause"], row["effect"], rho=row["obs_rho"], lag=row["obs_lag"], p_fdr=row["p_fdr"])
    return G






def select_direct_predictor_lags(lake_edges_df):
    wl_edges = lake_edges_df[lake_edges_df["effect"] == "WL"]
    return {row["cause"]: int(row["obs_lag"]) for _, row in wl_edges.iterrows()}


def best_positive_lag(scan):
    """从 lag_scan() 的结果里挑"预测可用"的最优滞后。

    **实际下界是 lag >= 1，不是函数名暗示的 >= 0**：d=0 的外生变量在任何 h>=1 的
    预测中都需要预测起点当期的真实值，不构成可用的预测信息（函数名沿用历史命名
    以免影响调用点）。

    只在可用范围内取 rho 最大值，
    不能对全部lag(含负数)取全局argmax再判断是否>=0——那样如果全局最优恰好是
    负的，就会把整个变量直接丢掉，即使旁边有一个rho几乎一样高的正滞后完全可用
    (比如lag=-2 rho=0.61、lag=+3 rho=0.60，两者只差0.01，全局argmax会选中
    lag=-2导致变量被排除，但lag=+3明明是可以直接拿来做预测特征的)。负滞后
    本身也不是"稍差一点"的选项——lag_scan()里effect往未来位移l步再跟cause比较，
    l<0时相当于用cause[t]去预测effect[t-2]这种"结果已经在原因之前发生"的关系，
    对预测毫无用处，不是"次优"而是"根本不能用"，所以要先筛掉负滞后再取最优，
    不是取全局最优再筛。返回(l_star, rho_star)，如果lag>=0里没有任何有效rho
    则返回(None, None)。"""
    valid = scan[(scan["lag"] >= 1) & scan["rho"].notna()]
    if valid.empty:
        return None, None
    best_row = valid.loc[valid["rho"].idxmax()]
    return int(best_row["lag"]), float(best_row["rho"])


def select_ancestor_lags(G, ccm_train_panel, embed_params, log_prefix=""):
    """WL 的全部祖先变量 → 预测特征滞后。

    有直接边的祖先原先直接沿用因果域的 obs_lag，**包括 d = 0**——而 direct 与
    neighbor 两个分支都在上游把 d < 1 的边送去 forecast_constrained_lag 重新求
    最优。结果是同一个湖、同一个变量在 CCM_direct 里滞后被修正为 >= 1，在
    CCM_ancestors 里仍以 d = 0 入模（实测 4 例：Kalamalka/Okanagan 的 R、
    Lake_of_the_Woods/Playgreen 的 RegFlow），使 CCM_ancestors 及继承其变量集的
    CCM_neighbor 用到目标月份当期的外生观测，违反方法论声明的预测域
    ℓ ∈ [1, 12]，且 min_exog_lag = 0 让这些方法在任何预见期下都需要起点之后的信息。
    此处补上与另两个分支相同的处理：不整条丢弃，在预测域内重新求最优滞后。

    """
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
            # 时序保留规则已排除 d < 0，此处只可能是 d = 0。
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
    """关联式基线（All_vars / Stepwise）的滞后选择：用与目标水位的滞后 Pearson 相关，
    不使用任何 CCM 信息。

    为什么需要它
    ------------
    早期实现用 `lag_scan()`（CCM 交叉映射）为基线挑滞后，导致所谓"非因果基线"
    实际继承了 CCM 的滞后识别结果——而识别正确滞后正是
    CCM 贡献的主要组成部分。这使 RQ3（因果信息是否带来额外预测价值）被系统性低估。
    实测：改用互相关后，45 个变量中 38 个（84%）的滞后发生改变。

    与 CCM 保持一致的三点（保证可比）
    ----------------------------------
    1. 相同的数据预处理（同一 ccm_train_panel）；
    2. 相同的 0–12 个月搜索窗口；
    3. 相同的 conditional-information 假设——**不对基线单独施加 d ≥ h 的约束**。
       CCM 侧同样没有该约束；若只给基线加，比较反而不公平
       （CCM 取 d=5、基线取 d=0，在 12 个月预见期下同样需要预测起点之后的外生值）。

    设计选择
    --------
    · 取 |r| 而非 r：正负关联都可能含预测信息（如蒸发增加对应水位下降）；
    · 并列时取较短滞后（严格大于比较自然实现 parsimony）；
    · 仅使用训练期数据。

    预期表现（如实报告，不作为"基线选错"的证据）
    ----------------------------------------------
    RegFlow 在多数湖泊会被选到 d=0 且 |r| 极高（Split 达 0.987），因为当月放水量
    与当月水位近乎完全线性相关。Pearson 回答的是"哪个滞后下线性关联最强"，
    并不判断因果方向、操作反馈或未来可得性——锁定 d=0 是关联式基准应当允许
    表现出的局限，而非实现缺陷。
    """
    lags = {}
    for var in embed_params:
        if var == wl_col or var not in ccm_train_panel.columns:
            continue
        best_lag, best_abs_r = None, -1.0
        for lag in range(min_lag, max_lag + 1):    # d>=1：排除同期，与 CCM 侧对称
            pair = pd.concat([ccm_train_panel[var].shift(lag),
                              ccm_train_panel[wl_col]], axis=1).dropna()
            if len(pair) < min_obs:
                continue
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if np.isfinite(r) and abs(r) > best_abs_r:     # 严格大于 → 并列取较短滞后
                best_abs_r, best_lag = abs(r), lag
        if best_lag is not None:
            lags[var] = best_lag
            log(f"{log_prefix}{var}: 互相关最优滞后={best_lag} (|r|={best_abs_r:.3f})")
    return lags


FORECAST_LAG_MIN = 1        # 预测特征的滞后下界：d=0 需要预测起点当期的真实值
FORECAST_LAG_MAX = 12


def forecast_constrained_lag(panel_df, cause, effect, embed_params,
                             lag_min=FORECAST_LAG_MIN, lag_max=FORECAST_LAG_MAX):
    """在预测可用的滞后窗口内重新求最优滞后（forecast-constrained optimal lag）。

    因果分析与预测特征构建是**两个预先定义的优化问题**，各自的候选滞后域不同：

        d*_causal   = argmax_{d ∈ [-12, 12]} rho(d)      ← 因果解释，符号即时序判据
        d*_forecast = argmax_{d ∈ [1, 12]}  rho(d)       ← 预测特征，d=0 不可用

    因此当某条边的 d*_causal = 0 时，正确做法**不是整条丢弃**，而是在预测域内
    重新求解。这与 CCM 的标准滞后选择原则一致——在预先设定的滞后域内比较
    cross-map skill 并取最优（Ye et al., 2015；Martin et al., 2024, Chaos）。
    水文 CCM 应用亦采用同一原则（"the optimal time lag is decided when the
    cross map skill is largest"）。

    两点必须在方法论中写明：
    1. 预测域 [1, 12] 是**事先声明**的，不是看到结果后选定，否则构成事后挑选；
    2. 边的**显著性**来自因果域上的 IAAFT 检验（max over 25 lags），
       此处仅重新选择特征滞后，不对该滞后另作显著性声明。

    实现上复用 `lag_scan()`——它只逐滞后计算 rho，**不涉及替代序列**，
    因此代价是秒级的 pyEDM 调用，无需重跑显著性检验。
    """
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
    """从重新验证过的湖泊间连通性结果里，找出对本湖泊WL有显著直接因果关系的邻居
    湖泊，返回{邻居湖泊名: 滞后月数}。"""
    sub = connectivity_df[(connectivity_df["effect_lake"] == lake_name) & (connectivity_df["causal_evidence"] == True)]
    return {row["cause_lake"]: int(row["obs_lag"]) for _, row in sub.iterrows()}


def load_neighbor_wl_series(neighbor_lake_name):
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
        "ccm_full_pipeline.py 是共享库，不应直接运行。论文结果的产出入口：\n"
        "  湖内 CCM   modal run code/02_within_lake_ccm/run_within_lake_ccm.py\n"
        "  湖间 CCM   modal run code/03_inter_lake_ccm/run_inter_lake_ccm.py\n"
        "  条件预测   modal run --detach code/04_forecast/modal_forecast_synchrony_filtered.py::detached"
    )
