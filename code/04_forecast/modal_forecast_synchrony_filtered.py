"""Run the conditional forecasting experiment on Modal."""

import itertools
import json
import os
import sys
from pathlib import Path

import modal

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR / "01_analysis_core"))
sys.path.insert(0, str(_CODE_DIR))

from config import LAKES, VARIABLES  # noqa: E402

app = modal.App("forecast-synchrony-filtered")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scipy",
        "statsmodels",
        "pmdarima",
        "xgboost==3.4.1",
        "networkx",
        "pyEDM==2.4.0",
    )
    # Fixed thread counts reduce controllable run-to-run variation.
    .env({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    })
    .add_local_python_source("analysis_core")
    .add_local_python_source("config")
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=False)
DATA_ROOT = "/data"
OUT_DIR = f"{DATA_ROOT}/lake_results/final_v3"
EMBED_PARAMS_ABS = f"{DATA_ROOT}/lake_results/full_pipeline_v2/embed_params_corrected.json"

WITHIN_ORIGINAL_INPUT = f"{OUT_DIR}/ccm_all_edges_merged_fdr.csv"
INTER_ORIGINAL_INPUT = f"{OUT_DIR}/connectivity_full_pairwise_ccm_results.csv"

OUTPUT_NAMES = {
    "full": "forecast_synchrony_filtered_full_results.csv",
    "rolling": "forecast_synchrony_filtered_rolling_results.csv",
    "dm": "forecast_synchrony_filtered_dm_results.csv",
    "selected": "forecast_synchrony_filtered_selected_lags.csv",
}
TUNING_OUTPUT_NAME = "xgboost_tuning_results.csv"
SHARD_DIR = f"{OUT_DIR}/forecast_shards"
SHARD_ROW_KEYS = ("full_rows", "rolling_rows", "dm_rows", "selected_rows")
EXPECTED_SHARDS = frozenset(f"{lake}.json" for lake in LAKES)
EXPECTED_WITHIN_EDGES = frozenset(
    (lake, cause, effect)
    for lake in LAKES
    for cause in VARIABLES
    for effect in VARIABLES
    if cause != effect
)
EXPECTED_INTER_EDGES = frozenset(itertools.permutations(LAKES, 2))


def _configure_module():
    """Configure shared analysis paths inside the Modal container."""
    import analysis_core as p

    p.PKL_DIR = f"{DATA_ROOT}/lake_results"
    os.makedirs(OUT_DIR, exist_ok=True)
    p.EMBED_PARAMS_PATH = EMBED_PARAMS_ABS
    return p


def _as_bool(value) -> bool:
    """Normalize stored values to Boolean form."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _json_dict(values: dict) -> str:
    """Serialize a lag mapping with stable key order."""
    return json.dumps({str(k): int(v) for k, v in values.items()}, sort_keys=True)


def _require_columns(frame, required, label):
    """Reject an input table that lacks required columns."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {', '.join(missing)}")


def _load_forecast_inputs(pd):
    """Validate and load the fixed ten-lake forecasting inputs."""
    panel_dir = f"{DATA_ROOT}/lake_results"
    required_files = [EMBED_PARAMS_ABS, WITHIN_ORIGINAL_INPUT, INTER_ORIGINAL_INPUT]
    required_files.extend(f"{panel_dir}/{lake}_result.pkl" for lake in LAKES)
    missing_files = [path for path in required_files if not os.path.isfile(path)]
    if missing_files:
        raise FileNotFoundError(
            "Forecasting requires all ten lake panels and the completed CCM outputs. "
            f"Missing {len(missing_files)} file(s): {', '.join(missing_files)}"
        )

    with open(EMBED_PARAMS_ABS, encoding="utf-8") as handle:
        embedding_params = json.load(handle)
    missing_lakes = sorted(set(LAKES) - set(embedding_params))
    unexpected_lakes = sorted(set(embedding_params) - set(LAKES))
    if missing_lakes or unexpected_lakes:
        raise RuntimeError(
            "The embedding-parameter JSON must contain exactly the ten study lakes; "
            f"missing={missing_lakes}, unexpected={unexpected_lakes}"
        )

    within = pd.read_csv(WITHIN_ORIGINAL_INPUT)
    _require_columns(
        within,
        {
            "lake", "cause", "effect", "obs_rho", "obs_lag", "p_fdr",
            "causal_evidence", "status",
        },
        "Within-lake CCM input",
    )
    within_edges = set(zip(
        within["lake"].astype(str),
        within["cause"].astype(str),
        within["effect"].astype(str),
    ))
    if len(within) != len(EXPECTED_WITHIN_EDGES) or within_edges != EXPECTED_WITHIN_EDGES:
        raise RuntimeError(
            "Within-lake CCM input must contain exactly the 420 expected directed edges"
        )
    if not within["status"].astype(str).eq("OK").all():
        raise RuntimeError("Within-lake CCM input contains a non-OK row")

    inter = pd.read_csv(INTER_ORIGINAL_INPUT)
    _require_columns(
        inter,
        {
            "cause_lake", "effect_lake", "obs_rho", "obs_lag", "p_fdr",
            "causal_evidence", "status",
        },
        "Between-lake CCM input",
    )
    inter_edges = set(zip(
        inter["cause_lake"].astype(str),
        inter["effect_lake"].astype(str),
    ))
    if len(inter) != len(EXPECTED_INTER_EDGES) or inter_edges != EXPECTED_INTER_EDGES:
        raise RuntimeError(
            "Between-lake CCM input must contain exactly the 90 expected directed edges"
        )
    if not inter["status"].astype(str).eq("OK").all():
        raise RuntimeError("Between-lake CCM input contains a non-OK row")

    return within, inter


def _shard_path(lake_name):
    """Return the Volume path for one lake shard."""
    return os.path.join(SHARD_DIR, f"{lake_name}.json")


def _existing_shards():
    """Return existing forecasting shard filenames from the Volume."""
    if not os.path.isdir(SHARD_DIR):
        return []
    return sorted(name for name in os.listdir(SHARD_DIR) if name.endswith(".json"))


def _validate_lake_shard(lake_name, shard):
    """Validate the identity and row structure of one lake shard."""
    if not isinstance(shard, dict) or shard.get("lake") != lake_name:
        raise RuntimeError(f"Invalid forecasting shard identity for {lake_name}")
    for key in SHARD_ROW_KEYS:
        rows = shard.get(key)
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"Forecasting shard {lake_name} has no valid {key}")
        for row in rows:
            if not isinstance(row, dict) or row.get("lake") != lake_name:
                raise RuntimeError(
                    f"Forecasting shard {lake_name} contains an invalid row in {key}"
                )

    full_methods = [row.get("method") for row in shard["full_rows"]]
    if None in full_methods or len(full_methods) != len(set(full_methods)):
        raise RuntimeError(f"Forecasting shard {lake_name} has duplicate full-result methods")


def _load_lake_shard(lake_name):
    """Load and validate one existing lake shard."""
    with open(_shard_path(lake_name), encoding="utf-8") as handle:
        shard = json.load(handle)
    _validate_lake_shard(lake_name, shard)
    return shard


def _select_stepwise_lags(panel_train, all_var_lags, wl_col="WL", criterion="aic"):
    """Select a forward-stepwise subset from the baseline lag pool."""
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


def _filtered_within_edges(pd, frame):
    """Return all retained within-lake edges used by forecasting."""
    sig = frame[frame["causal_evidence"].map(_as_bool)].copy()
    sig["obs_lag"] = pd.to_numeric(sig["obs_lag"], errors="coerce")
    sig["obs_rho"] = pd.to_numeric(sig["obs_rho"], errors="coerce")
    # Retain d=0 for re-selection within the forecast-usable lag range.
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


def _filtered_interlake_edges(pd, frame):
    """Return all retained between-lake edges used by forecasting."""
    sig = frame[frame["causal_evidence"].map(_as_bool)].copy()
    sig["obs_lag"] = pd.to_numeric(sig["obs_lag"], errors="coerce")
    sig["obs_rho"] = pd.to_numeric(sig["obs_rho"], errors="coerce")
    # Retain d=0 for re-selection within the forecast-usable lag range.
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


def _filter_long_gap_vars(p, panel_deseason, lag_sets, lake_name):
    """Remove predictors whose training gaps exceed the allowed limit."""
    selected_cols = set()
    for values in lag_sets.values():
        selected_cols.update(values.keys())
    if not selected_cols:
        return lag_sets

    long_gap_vars = p.find_long_gap_vars(
        panel_deseason,
        selected_cols,
        p.FORECAST_HORIZON,
        log_prefix=f"[{lake_name}] ",
    )
    if not long_gap_vars:
        return lag_sets
    return {
        name: {key: value for key, value in values.items() if key not in long_gap_vars}
        for name, values in lag_sets.items()
    }


def _load_neighbor_series(p, panel_deseason, neighbor_lags, lake_name):
    """Load, align, and gap-filter neighbouring water-level series."""
    cols, col_lags, kept = {}, {}, {}
    train_end = -p.FORECAST_HORIZON
    for neighbor_name, lag in neighbor_lags.items():
        neighbor_wl = p.load_neighbor_wl_series(neighbor_name)
        if neighbor_wl is None:
            continue
        aligned = neighbor_wl.reindex(panel_deseason.index)
        gap = int(p.longest_consecutive_gap_months(aligned.iloc[:train_end]))
        if gap > p.MAX_FILLABLE_GAP_MONTHS:
            print(
                f"[{lake_name}] Excluding neighbour {neighbor_name}: "
                f"longest training gap is {gap} months "
                f"(limit={p.MAX_FILLABLE_GAP_MONTHS})",
                flush=True,
            )
            continue
        column = f"_neighbor_{neighbor_name}"
        cols[column] = aligned
        col_lags[column] = lag
        kept[neighbor_name] = lag
    return cols, col_lags, kept


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=3600, cpu=1.0,
              memory=2048, retries=2)
def run_lake_synchrony_filtered(
    lake_name: str,
    xgb_params: dict,
    within_edges_records: list[dict],
    inter_edges_records: list[dict],
) -> dict:
    """Run every forecasting method for one lake and return its result rows."""
    import pickle
    import pandas as pd

    p = _configure_module()
    pkl_path = os.path.join(p.PKL_DIR, f"{lake_name}_result.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing lake panel: {pkl_path}")

    within_edges = pd.DataFrame(within_edges_records)
    inter_edges = pd.DataFrame(inter_edges_records)

    with open(pkl_path, "rb") as handle:
        cached = pickle.load(handle)

    clean_wide = p.clean_wide_wl(cached["wide_wl"], log_prefix=f"[{lake_name}] ")
    combined_wl = p.combine_station_water_levels(clean_wide)
    real_predictors = cached["real_predictors"]
    panel_deseason = p.build_variable_panel(
        combined_wl,
        {column: real_predictors[column] for column in real_predictors.columns},
        forecast_horizon=p.FORECAST_HORIZON,
    )
    ccm_train_panel = panel_deseason.iloc[:-p.FORECAST_HORIZON]
    embed_params = p.load_embed_params(lake_name)

    lake_edges = within_edges[within_edges["lake"] == lake_name].copy()
    G = p.build_new_causal_network(lake_edges)
    direct_lags = p.select_direct_predictor_lags(lake_edges)
    old_direct_lags = dict(direct_lags)
    for variable, original_lag in list(direct_lags.items()):
        if original_lag < p.FORECAST_LAG_MIN:
            new_lag, new_rho = p.forecast_constrained_lag(
                ccm_train_panel, variable, "WL", embed_params
            )
            if new_lag is None:
                direct_lags.pop(variable)
                print(
                    f"[{lake_name}] {variable}->WL: causal optimum d={original_lag}; "
                    "no valid forecast lag, excluded",
                    flush=True,
                )
            else:
                direct_lags[variable] = new_lag
                print(
                    f"[{lake_name}] {variable}->WL: causal optimum d={original_lag}; "
                    f"forecast optimum d={new_lag} (rho={new_rho:.3f})",
                    flush=True,
                )

    ancestor_lags = p.select_ancestor_lags(
        G, ccm_train_panel, embed_params, log_prefix=f"[{lake_name}] ")
    all_var_lags = p.select_all_var_lags_xcorr(
        ccm_train_panel, embed_params, min_lag=1, log_prefix=f"[{lake_name}] ")

    neighbor_lags = p.load_neighbor_lags(inter_edges, lake_name)
    old_neighbor_lags = dict(neighbor_lags)
    for neighbor, original_lag in list(neighbor_lags.items()):
        if original_lag < p.FORECAST_LAG_MIN:
            try:
                pair_panel, effect_embedding = p.load_pair_panel_for_connectivity(
                    neighbor, lake_name
                )
                new_lag, new_rho = p.forecast_constrained_lag(
                    pair_panel,
                    neighbor,
                    lake_name,
                    {lake_name: effect_embedding},
                )
            except Exception as exc:
                new_lag, new_rho = None, float("nan")
                print(
                    f"[{lake_name}] Neighbour {neighbor}: forecast-lag scan failed "
                    f"with {type(exc).__name__}",
                    flush=True,
                )
            if new_lag is None:
                neighbor_lags.pop(neighbor)
                print(
                    f"[{lake_name}] Neighbour {neighbor}: causal optimum "
                    f"d={original_lag}; no valid forecast lag, excluded",
                    flush=True,
                )
            else:
                neighbor_lags[neighbor] = new_lag
                print(
                    f"[{lake_name}] Neighbour {neighbor}: causal optimum "
                    f"d={original_lag}; forecast optimum d={new_lag} "
                    f"(rho={new_rho:.3f})",
                    flush=True,
                )

    neighbor_cols, neighbor_combined_lags, neighbor_lags = _load_neighbor_series(
        p, panel_deseason, neighbor_lags, lake_name)

    lag_sets = {
        "direct": direct_lags,
        "ancestors": ancestor_lags,
        "all_vars": all_var_lags,
    }
    lag_sets = _filter_long_gap_vars(
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

    # All methods use the same panel and global XGBoost configuration.
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
            wl_series,
            panel_deseason[list(direct_lags)],
            direct_lags,
            xgb_params,
            test_size=p.FORECAST_HORIZON,
        )
        method_exog_lags["SARIMAX_CCM_direct"] = direct_lags
        method_exog_lags["XGBoost_CCM_direct"] = direct_lags

    if ancestor_lags:
        methods["SARIMAX_CCM_ancestors"] = lambda: p.fit_auto_sarimax_multi(
            wl_series, panel_deseason[list(ancestor_lags)], ancestor_lags,
            test_size=p.FORECAST_HORIZON
        )
        methods["XGBoost_CCM_ancestors"] = lambda: p.fit_xgboost_multi(
            wl_series,
            panel_deseason[list(ancestor_lags)],
            ancestor_lags,
            xgb_params,
            test_size=p.FORECAST_HORIZON,
        )
        method_exog_lags["SARIMAX_CCM_ancestors"] = ancestor_lags
        method_exog_lags["XGBoost_CCM_ancestors"] = ancestor_lags

    if stepwise_lags:
        methods["SARIMAX_Stepwise"] = lambda: p.fit_auto_sarimax_multi(
            wl_series, panel_deseason[list(stepwise_lags)], stepwise_lags,
            test_size=p.FORECAST_HORIZON
        )
        methods["XGBoost_Stepwise"] = lambda: p.fit_xgboost_multi(
            wl_series,
            panel_deseason[list(stepwise_lags)],
            stepwise_lags,
            xgb_params,
            test_size=p.FORECAST_HORIZON,
        )
        method_exog_lags["SARIMAX_Stepwise"] = stepwise_lags
        method_exog_lags["XGBoost_Stepwise"] = stepwise_lags

    if neighbor_lags and ancestor_lags:
        neighbor_panel = pd.DataFrame(neighbor_cols)
        combined_exog = pd.concat(
            [panel_deseason[list(ancestor_lags)], neighbor_panel], axis=1
        )
        combined_lags = {**ancestor_lags, **neighbor_combined_lags}
        methods["SARIMAX_CCM_neighbor"] = lambda: p.fit_auto_sarimax_multi(
            wl_series,
            combined_exog,
            combined_lags,
            test_size=p.FORECAST_HORIZON,
        )
        methods["XGBoost_CCM_neighbor"] = lambda: p.fit_xgboost_multi(
            wl_series,
            combined_exog,
            combined_lags,
            xgb_params,
            test_size=p.FORECAST_HORIZON,
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
        except Exception as exc:
            full_rows.append({
                "lake": lake_name,
                "method": name,
                "status": f"ERROR: {type(exc).__name__}: {exc}",
            })
            print(f"[{lake_name}] {name} failed: {type(exc).__name__}: {exc}", flush=True)
            continue

        used_lags = method_exog_lags[name]
        min_exog_lag = min(used_lags.values()) if used_lags else None
        free_months = (
            p.FORECAST_HORIZON
            if min_exog_lag is None
            else min(min_exog_lag, p.FORECAST_HORIZON)
        )
        full_rows.append({
            "lake": lake_name,
            "method": name,
            "rmse": result["rmse"],
            "mae": result["mae"],
            "nse": result["nse"],
            "pbias": result.get("pbias"),
            "n_eval": result["n_eval"],
            "n_selected_vars": len(used_lags) if used_lags else 0,
            "arima_order": str(result.get("order")) if "order" in result else None,
            "arima_seasonal_order": (
                str(result.get("seasonal_order"))
                if "seasonal_order" in result else None
            ),
            "min_exog_lag": min_exog_lag,
            "n_foresight_free_months": free_months,
            "frac_foresight_free": free_months / p.FORECAST_HORIZON,
        })

        try:
            if name.startswith("XGBoost"):
                rolling = p.rolling_origin_xgb(result)
            elif name == "Persistence":
                rolling = p.rolling_origin_persistence(
                    wl_series, test_size=p.FORECAST_HORIZON
                )
            else:
                rolling = p.rolling_origin_sarimax(
                    result, wl_series, test_size=p.FORECAST_HORIZON
                )
        except Exception as exc:
            raise RuntimeError(
                f"Rolling evaluation failed for {lake_name}/{name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

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
                "requires_foresight": (
                    min_exog_lag is not None and horizon > min_exog_lag
                ),
            })

    # Predefined comparisons cover CCM strategies and their baselines.
    comparisons = [
        # Between CCM strategies.
        ("XGBoost_CCM_ancestors", "XGBoost_CCM_direct"),
        ("SARIMAX_CCM_ancestors", "SARIMAX_CCM_direct"),
        ("XGBoost_CCM_neighbor", "XGBoost_CCM_ancestors"),
        ("SARIMAX_CCM_neighbor", "SARIMAX_CCM_ancestors"),
        # Against statistical selection.
        ("XGBoost_CCM_ancestors", "XGBoost_Stepwise"),
        ("SARIMAX_CCM_ancestors", "SARIMAX_Stepwise"),
        # Against unfiltered or no-exogenous baselines.
        ("XGBoost_CCM_ancestors", "XGBoost_AR_only"),
        ("XGBoost_CCM_ancestors", "Persistence"),
        ("XGBoost_CCM_ancestors", "XGBoost_all_vars"),
        ("XGBoost_CCM_direct", "XGBoost_AR_only"),
        ("SARIMAX_CCM_ancestors", "SARIMA"),
        ("SARIMAX_CCM_ancestors", "Persistence"),
        # Across model families.
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


def _apply_dm_fdr(dm_df):
    """Apply BH-FDR to valid Diebold-Mariano comparisons."""
    from statsmodels.stats.multitest import multipletests

    if dm_df.empty or "p_value" not in dm_df.columns:
        return dm_df
    dm_df = dm_df.copy()
    dm_df["p_fdr"] = float("nan")
    valid = dm_df["p_value"].notna()
    if valid.any():
        dm_df.loc[valid, "p_fdr"] = multipletests(
            dm_df.loc[valid, "p_value"], alpha=0.05, method="fdr_bh")[1]
        print(
            f"DM BH-FDR family: {int(valid.sum())} tests; "
            f"uncorrected p<0.05: "
            f"{int((dm_df.loc[valid, 'p_value'] < 0.05).sum())}; "
            f"adjusted p<0.05: "
            f"{int((dm_df.loc[valid, 'p_fdr'] < 0.05).sum())}",
            flush=True,
        )
    return dm_df


def _jsonable(obj):
    """Convert NumPy and missing values to JSON-safe Python values."""
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
    """Run or resume all lake shards and merge only the complete result set."""
    import pandas as pd

    p = _configure_module()
    volume.reload()
    within_input, inter_input = _load_forecast_inputs(pd)
    within_edges = _filtered_within_edges(pd, within_input)
    inter_edges = _filtered_interlake_edges(pd, inter_input)
    print(
        f"Retained within-lake edges: {len(within_edges)} | "
        f"retained between-lake edges: {len(inter_edges)}",
        flush=True,
    )

    os.makedirs(SHARD_DIR, exist_ok=True)
    present = set(_existing_shards())
    unexpected = sorted(present - EXPECTED_SHARDS)
    if unexpected:
        raise RuntimeError(
            f"Forecast shard directory contains {len(unexpected)} unexpected file(s): "
            f"{', '.join(unexpected[:5])}"
        )

    done = set()
    for lake_name in LAKES:
        if f"{lake_name}.json" not in present:
            continue
        try:
            _load_lake_shard(lake_name)
        except Exception as exc:
            print(
                f"Ignoring invalid shard for {lake_name}; it will be recomputed: {exc}",
                flush=True,
            )
        else:
            done.add(lake_name)

    todo = [lake_name for lake_name in LAKES if lake_name not in done]
    print(
        f"Forecast lakes: {len(LAKES)} | completed: {len(done)} | "
        f"remaining: {len(todo)}",
        flush=True,
    )

    print("Tuning XGBoost hyperparameters once globally...", flush=True)
    xgb_params, tuning_rows = p.tune_xgboost_hyperparams(
        LAKES,
        return_report=True,
    )
    tuning = pd.DataFrame(tuning_rows)
    if len(tuning) != len(p.XGB_PARAM_GRID) or int(tuning["selected"].sum()) != 1:
        raise RuntimeError("XGBoost tuning did not return one row per candidate")
    tuning_path = f"{OUT_DIR}/{TUNING_OUTPUT_NAME}"
    tuning.to_csv(tuning_path, index=False)
    volume.commit()
    print(f"Selected XGBoost parameters: {xgb_params}", flush=True)
    print(f"Wrote {tuning_path} ({len(tuning)} rows)", flush=True)

    if todo:
        results = run_lake_synchrony_filtered.map(
            todo,
            [xgb_params] * len(todo),
            [within_edges.to_dict("records")] * len(todo),
            [inter_edges.to_dict("records")] * len(todo),
            return_exceptions=True,
        )
        for lake, result in zip(todo, results):
            if isinstance(result, Exception):
                print(f"  [FAIL] {lake}: {type(result).__name__}: {result}", flush=True)
                continue
            _validate_lake_shard(lake, result)
            with open(_shard_path(lake), "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, default=_jsonable)
            volume.commit()
            print(f"  [OK] {lake}", flush=True)

    present = set(_existing_shards())
    missing = sorted(EXPECTED_SHARDS - present)
    unexpected = sorted(present - EXPECTED_SHARDS)
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                f"missing {len(missing)} shard(s), for example: {', '.join(missing[:5])}"
            )
        if unexpected:
            details.append(
                f"found {len(unexpected)} unexpected shard(s), for example: "
                f"{', '.join(unexpected[:5])}"
            )
        raise RuntimeError(
            "Cannot merge forecasting outputs: expected exactly ten lake shards; "
            + "; ".join(details)
        )

    rows = {key: [] for key in SHARD_ROW_KEYS}
    for filename in sorted(EXPECTED_SHARDS):
        lake_name = filename[:-5]
        shard = _load_lake_shard(lake_name)
        for key in rows:
            rows[key].extend(shard[key])

    outputs = {
        "full": pd.DataFrame(rows["full_rows"]),
        "rolling": pd.DataFrame(rows["rolling_rows"]),
        "dm": _apply_dm_fdr(pd.DataFrame(rows["dm_rows"])),
        "selected": pd.DataFrame(rows["selected_rows"]),
    }
    for key, frame in outputs.items():
        path = f"{OUT_DIR}/{OUTPUT_NAMES[key]}"
        frame.to_csv(path, index=False)
        print(f"Wrote {path} ({len(frame)} rows)", flush=True)
    volume.commit()

    summary = {
        "lakes_done": len(LAKES),
        "tuning": len(tuning),
        **{key: len(frame) for key, frame in outputs.items()},
    }
    print(f"Forecasting completed: {summary}", flush=True)
    return summary


@app.local_entrypoint()
def detached():
    """Start server-side orchestration and return immediately."""
    call = orchestrate.spawn()
    print(f"Submitted forecasting orchestration: {call.object_id}")
