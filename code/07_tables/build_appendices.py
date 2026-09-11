"""Build appendix CSV and Markdown files from downloaded Modal results."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "01_analysis_core"))

from config import LAKES, VARIABLES  # noqa: E402
import analysis_core as analysis  # noqa: E402


TABLE_DIR = ROOT / "ch4_tables"
OUTPUT_DIR = ROOT / "appendices"
PANEL_DIR = Path(analysis.PKL_DIR)

SYSTEMS = [
    ("Okanagan", LAKES[:4]),
    ("Nelson–Winnipeg", LAKES[4:]),
]
SYSTEM_OF = {lake: system for system, lakes in SYSTEMS for lake in lakes}
DISPLAY_NAME = {lake: lake.replace("_Lake", "").replace("_", " ") for lake in LAKES}
DISPLAY_NAME["Lake_of_the_Woods"] = "Lake of the Woods"

REGULATION_STATIONS = {
    "Kalamalka_Lake": ["08NM065"],
    "Okanagan_Lake": ["08NM050"],
    "Skaha_Lake": ["08NM002"],
    "Vaseux_Lake": ["08NM247"],
    "Rainy_Lake": ["05PC019"],
    "Lake_of_the_Woods": ["05PE011", "05PE006"],
    "Playgreen_Lake": ["05UB009"],
    "Kiskitto_Lake": ["05UB009"],
    "Sipiwesk_Lake": ["05UE005"],
    "Split_Lake": ["05UF006"],
}
REGULATION_PERIOD = {"Vaseux_Lake": "2012–2024"}
VARIABLE_ORDER = ["RegFlow", "R", "P", "Evap", "SWE", "T", "WL"]
NO_EXOGENOUS = {"Persistence", "SARIMA", "XGBoost_AR_only"}
METHOD_ORDER = [
    "Persistence",
    "SARIMA",
    "SARIMAX_all_vars",
    "SARIMAX_Stepwise",
    "SARIMAX_CCM_direct",
    "SARIMAX_CCM_ancestors",
    "SARIMAX_CCM_neighbor",
    "XGBoost_AR_only",
    "XGBoost_all_vars",
    "XGBoost_Stepwise",
    "XGBoost_CCM_direct",
    "XGBoost_CCM_ancestors",
    "XGBoost_CCM_neighbor",
]

COLUMN_DECIMALS = {
    "coverage_pct": 1,
    "cv_rmse_m": 5,
    "learning_rate": 2,
    "p_value": 4,
    "p_fdr": 4,
    "kendall_p": 4,
    "dm_stat": 3,
    "S_ij": 3,
}


def require_columns(frame, required, label):
    """Reject an input table that lacks required columns."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def as_bool(series):
    """Normalize a stored Boolean column."""
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def format_cell(value, decimals):
    """Format one value for a Markdown table."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if abs(value) >= 1e-4 or value == 0:
            return f"{value:.{decimals}f}"
        return f"{value:.2e}"
    return str(value)


def to_markdown(frame, default_decimals=3):
    """Render a Markdown table without requiring tabulate."""
    columns = list(frame.columns)
    numeric = {
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].dtype != bool
    }
    decimals = [COLUMN_DECIMALS.get(column, default_decimals) for column in columns]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(
        "---:" if column in numeric else "---" for column in columns
    ) + " |"
    rows = [
        "| "
        + " | ".join(
            format_cell(value, nd) for value, nd in zip(row, decimals)
        )
        + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows])


def write_table(frame, stem, title, note="", default_decimals=3):
    """Write one appendix table as CSV and Markdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    markdown = to_markdown(frame, default_decimals)
    text = f"**{title}**"
    if note:
        text += f"\n\n{note}"
    (OUTPUT_DIR / f"{stem}.md").write_text(
        text + "\n\n" + markdown + "\n",
        encoding="utf-8",
    )
    print(f"  {stem:<34} {len(frame):>4} rows")


def cleaned_water_level(lake):
    """Return the cleaned monthly lake-level series and its stored inputs."""
    with (PANEL_DIR / f"{lake}_result.pkl").open("rb") as handle:
        cached = pickle.load(handle)
    cleaned = analysis.clean_wide_wl(cached["wide_wl"])
    level = analysis.combine_station_water_levels(cleaned)
    return level.sort_index().asfreq("MS"), cached


def longest_gap(series):
    """Return the longest run of missing values."""
    longest = current = 0
    for missing in series.isna():
        current = current + 1 if missing else 0
        longest = max(longest, current)
    return longest


def build_appendix_a():
    """Build study-sample and water-level data tables."""
    if set(REGULATION_STATIONS) != set(LAKES):
        raise ValueError("Regulated-flow station metadata does not match config.LAKES")
    print("Appendix A: data and data availability")
    sample_rows = []
    availability_rows = []
    outlier_rows = []

    for lake in LAKES:
        level, cached = cleaned_water_level(lake)
        raw = cached["wide_wl"]
        sample_rows.append({
            "lake": DISPLAY_NAME[lake],
            "basin": SYSTEM_OF[lake],
            "hydrolakes_id": cached.get("hylak_id"),
            "n_wl_stations": raw.shape[1],
            "wl_stations": ", ".join(raw.columns),
            "regflow_stations": ", ".join(REGULATION_STATIONS[lake]),
            "regflow_period": REGULATION_PERIOD.get(lake, "1994–2024"),
        })

        centred = level - level.mean()
        availability_rows.append({
            "lake": DISPLAY_NAME[lake],
            "basin": SYSTEM_OF[lake],
            "n_months": len(level),
            "n_observed": int(level.notna().sum()),
            "n_missing": int(level.isna().sum()),
            "coverage_pct": round(100 * level.notna().mean(), 1),
            "longest_gap_months": longest_gap(level),
            "sd_m": round(float(centred.std()), 3),
            "iqr_m": round(float(centred.quantile(0.75) - centred.quantile(0.25)), 3),
            "range_m": round(float(centred.max() - centred.min()), 3),
        })

        for station in raw.columns:
            for month in analysis.wl_station_outliers(raw[station]):
                outlier_rows.append({
                    "lake": DISPLAY_NAME[lake],
                    "station": station,
                    "month": str(month.date())[:7],
                    "value_m": round(float(raw.loc[month, station]), 3),
                })

    write_table(
        pd.DataFrame(sample_rows),
        "A1_lakes_and_stations",
        "Table A1. Study lakes, water-level gauges and regulated-outflow gauges",
        "Gauge identifiers are Water Survey of Canada station numbers; "
        "hydrolakes_id is the HydroLAKES Hylak_id.",
    )
    write_table(
        pd.DataFrame(outlier_rows),
        "A2_flagged_outliers",
        "Table A2. Water level observations set to missing by the two-stage "
        "robust screen",
        "A point is removed when the robust z-score exceeds 6 for its monthly "
        "change and 5 for its level. Recorded values are not adjusted.",
    )
    return pd.DataFrame(availability_rows)


def build_appendix_b(availability):
    """Build embedding and CCM appendix tables."""
    print("Appendix B: CCM analysis")
    embedding = json.loads(
        (ROOT / "results" / "embed_params_corrected.json").read_text(encoding="utf-8")
    )
    if set(embedding) != set(LAKES):
        raise ValueError("Embedding parameters must contain exactly the ten study lakes")
    if set(VARIABLE_ORDER) != set(VARIABLES):
        raise ValueError("Appendix variable order does not match config.VARIABLES")
    missing_embeddings = [
        f"{lake}/{variable}"
        for lake in LAKES
        for variable in VARIABLES
        if embedding.get(lake, {}).get(variable, {}).get("E") is None
    ]
    if missing_embeddings:
        raise ValueError(
            "Missing embedding dimensions: " + ", ".join(missing_embeddings)
        )
    matrix = pd.DataFrame({
        variable: {
            DISPLAY_NAME[lake]: embedding.get(lake, {}).get(variable, {}).get("E")
            for lake in LAKES
        }
        for variable in VARIABLE_ORDER
    })
    matrix = matrix.reindex([DISPLAY_NAME[lake] for lake in LAKES])
    matrix = matrix.reset_index(names="lake")
    write_table(
        matrix,
        "B1_embedding_parameters",
        "Table B1. Embedding dimension E by lake and variable",
        "E is selected from 2–10 by one-step Simplex self-prediction on the "
        "training period; embedding delay tau is fixed at 1.",
        default_decimals=0,
    )

    drivers = pd.read_csv(TABLE_DIR / "supported_drivers.csv")
    write_table(
        availability,
        "B2_data_availability",
        "Table B2. Water level data availability and variability by lake",
        "coverage_pct is the percentage of months with an observation; "
        "longest_gap_months is the longest consecutive gap. Variability is "
        "calculated after centring each lake on its mean, in metres.",
    )

    write_table(
        drivers,
        "B3_supported_drivers",
        f"Table B3. The {len(drivers)} supported driver-to-water-level relationships",
        "Relationships pass both Benjamini–Hochberg FDR control at alpha = "
        "0.05 within the 420-edge family and the convergence diagnostic.",
    )

    between = pd.read_csv(TABLE_DIR / "T6_between_lake_edges.csv")
    require_columns(
        between,
        {
            "cause_lake",
            "effect_lake",
            "statistically_significant",
            "obs_rho",
            "obs_lag",
        },
        "Between-lake table",
    )
    identities = set(
        between[["cause_lake", "effect_lake"]].itertuples(index=False, name=None)
    )
    expected_identities = {
        (cause, effect)
        for cause in LAKES
        for effect in LAKES
        if cause != effect
    }
    if len(between) != 90 or identities != expected_identities:
        raise ValueError("Between-lake table must contain exactly 90 directed edges")
    supported = between[as_bool(between.statistically_significant)].copy()
    supported["cause_lake"] = supported.cause_lake.map(DISPLAY_NAME)
    supported["effect_lake"] = supported.effect_lake.map(DISPLAY_NAME)
    supported = supported[
        [
            "cause_lake",
            "effect_lake",
            "obs_rho",
            "obs_lag",
            "obs_n",
            "p_fdr",
            "kendall_tau",
            "lag_resolution",
            "causal_evidence",
            "waterway_connected",
            "tier",
        ]
    ].sort_values(["tier", "obs_rho"], ascending=[True, False])
    write_table(
        supported,
        "B4_supported_between_lake_edges",
        f"Table B4. The {len(supported)} supported between-lake relationships",
        "The family contains 90 directed edges from 45 lake pairs. Supported "
        "edges pass both FDR control and the convergence diagnostic. Tier is "
        "a post-hoc descriptive grouping by hydrological separation.",
    )

    pair_strength = pd.read_csv(TABLE_DIR / "T7_lake_pair_strength.csv")
    if len(pair_strength) != 45 or pair_strength.pair.nunique() != 45:
        raise ValueError("Lake-pair strength table must contain exactly 45 pairs")
    pair_strength["pair"] = pair_strength.pair.map(
        lambda value: " – ".join(DISPLAY_NAME[lake] for lake in value.split("|"))
    )
    pair_strength = pair_strength[
        ["pair", "S_ij", "detected", "directly_connected", "tier"]
    ]
    write_table(
        pair_strength,
        "B5_lake_pair_strength",
        f"Table B5. CCM strength and hydrological connectivity for all "
        f"{len(pair_strength)} lake pairs",
        "S_ij is the mean absolute cross-map skill over both directions. A pair "
        "is detected when either direction is FDR-significant, convergent and "
        "has an optimal lag of at least zero months.",
    )


def model_configuration_cell(row):
    """Format one fitted-model configuration for Table C2."""
    if pd.notna(row.status):
        return "failed"
    if isinstance(row.arima_order, str):
        return f"{row.arima_order}{row.arima_seasonal_order}".replace(" ", "")
    if row.method in NO_EXOGENOUS:
        return "no exogenous"
    return str(int(row.n_selected_vars))


def build_appendix_c():
    """Build forecasting appendix tables."""
    print("Appendix C: forecasting analysis")
    selected = pd.read_csv(
        ROOT / "results" / "forecast_synchrony_filtered_selected_lags.csv"
    )
    require_columns(
        selected,
        {"lake", "method", "selected_vars", "selected_lags"},
        "Selected-predictor table",
    )
    selected["lake"] = selected.lake.map(DISPLAY_NAME)
    selected = selected[["lake", "method", "selected_vars", "selected_lags"]]
    write_table(
        selected,
        "C1_selected_predictors",
        "Table C1. Exogenous predictors and forecast-domain lags by model",
        "Lags are re-optimised in the forecast domain with a lower bound of one "
        "month. CCM-neighbour rows contain only the between-lake terms added to "
        "the corresponding CCM-ancestors predictor set.",
    )

    full = pd.read_csv(TABLE_DIR / "T4_single_split_full.csv")
    require_columns(
        full,
        {
            "lake",
            "method",
            "status",
            "arima_order",
            "arima_seasonal_order",
            "n_selected_vars",
        },
        "Forecast full-results table",
    )
    if set(full.lake) - set(LAKES) or set(full.method) - set(METHOD_ORDER):
        raise ValueError("Forecast full-results table contains unexpected identities")
    full["cell"] = full.apply(model_configuration_cell, axis=1)
    configurations = full.pivot_table(
        index="lake",
        columns="method",
        values="cell",
        aggfunc="first",
    ).reindex(LAKES)
    configurations = configurations[
        [method for method in METHOD_ORDER if method in configurations.columns]
    ]
    configurations = configurations.fillna("not applicable")
    configurations.index = [DISPLAY_NAME[lake] for lake in configurations.index]
    n_expected = len(LAKES) * len(METHOD_ORDER)
    n_fitted = len(full)
    n_failed = int(full.status.notna().sum())
    write_table(
        configurations.reset_index(names="lake"),
        "C2_model_configurations",
        "Table C2. Model configuration and fitting outcome by lake and method",
        "SARIMA(X) cells give model orders and XGBoost cells give the number of "
        f"exogenous predictors. Of {n_expected} possible lake-method "
        f"combinations, {n_fitted} entered fitting, {n_failed} failed and "
        f"{n_fitted - n_failed} produced valid forecasts.",
    )

    tuning = pd.read_csv(ROOT / "results" / "xgboost_tuning_results.csv")
    require_columns(
        tuning,
        {
            "max_depth",
            "learning_rate",
            "n_estimators",
            "cv_rmse_m",
            "selected",
            "n_lakes",
            "n_cv_folds",
        },
        "XGBoost tuning table",
    )
    tuning["selected"] = as_bool(tuning.selected)
    expected_grid = {
        (row["max_depth"], row["learning_rate"], row["n_estimators"])
        for row in analysis.XGB_PARAM_GRID
    }
    actual_grid = set(
        tuning[["max_depth", "learning_rate", "n_estimators"]].itertuples(
            index=False,
            name=None,
        )
    )
    if len(tuning) != len(expected_grid) or actual_grid != expected_grid:
        raise ValueError("XGBoost tuning table does not match the configured grid")
    if int(tuning.selected.sum()) != 1:
        raise ValueError("XGBoost tuning table must select exactly one configuration")
    tuning["cv_rmse_m"] = tuning.cv_rmse_m.round(5)
    write_table(
        tuning,
        "C3_xgboost_hyperparameters",
        "Table C3. XGBoost hyperparameter grid and cross-validation results",
        "Subsample = 0.8, colsample_bytree = 0.8 and random_state = 0 were fixed. "
        "The table reports the training-only expanding-window cross-validation "
        "results saved by the Modal forecasting stage.",
    )

    dm = pd.read_csv(TABLE_DIR / "T3_dm_bh.csv")
    require_columns(
        dm,
        {"lake", "method_1", "method_2", "p_value", "sig_fdr", "winner"},
        "Diebold–Mariano table",
    )
    usable = dm[dm.p_value.notna()]
    usable = usable.copy()
    usable["sig_fdr"] = as_bool(usable.sig_fdr)
    rows = []
    for (method_1, method_2), group in usable.groupby(["method_1", "method_2"]):
        significant = group[group.sig_fdr]
        rows.append({
            "method_1": method_1,
            "method_2": method_2,
            "n_lakes": group.lake.nunique(),
            "n_tested": len(group),
            "n_significant": len(significant),
            "n_favouring_method_1": int((significant.winner == method_1).sum()),
            "n_favouring_method_2": int((significant.winner == method_2).sum()),
        })
    summary = pd.DataFrame(rows)
    write_table(
        summary,
        "C4_dm_summary",
        "Table C4. Summary of Diebold–Mariano tests",
        f"The input contains {len(dm)} comparisons, of which {len(usable)} "
        f"returned a usable statistic and {int(usable.sig_fdr.sum())} remained "
        "significant after Benjamini–Hochberg correction.",
        default_decimals=0,
    )


SECTIONS = [
    (
        "Appendix A. Data and data availability",
        ["A1_lakes_and_stations", "A2_flagged_outliers"],
    ),
    (
        "Appendix B. CCM analysis",
        [
            "B1_embedding_parameters",
            "B2_data_availability",
            "B3_supported_drivers",
            "B4_supported_between_lake_edges",
            "B5_lake_pair_strength",
        ],
    ),
    (
        "Appendix C. Forecasting analysis",
        [
            "C1_selected_predictors",
            "C2_model_configurations",
            "C3_xgboost_hyperparameters",
            "C4_dm_summary",
        ],
    ),
]


def combine_markdown():
    """Combine the individual Markdown tables into one appendix document."""
    edges = pd.read_csv(TABLE_DIR / "T5_within_lake_edges.csv")
    require_columns(
        edges,
        {"lake", "cause", "effect", "statistically_significant", "obs_lag"},
        "Within-lake table",
    )
    if len(edges) != 420:
        raise ValueError("Within-lake table must contain exactly 420 directed edges")
    supported = edges[as_bool(edges.statistically_significant)]
    n_positive = int((supported.obs_lag > 0).sum())
    n_zero = int((supported.obs_lag == 0).sum())
    n_negative = int((supported.obs_lag < 0).sum())

    parts = [
        "# Appendices",
        "",
        "These appendix files are generated from the downloaded Modal results.",
        "",
    ]
    for title, stems in SECTIONS:
        parts.extend([f"## {title}", ""])
        if title.startswith("Appendix B"):
            parts.extend([
                "### Figure B1. Support across all within-lake candidate relationships",
                "",
                f"Of the {len(edges)} candidate edges, {len(supported)} were "
                f"supported ({n_positive} with a positive lag, {n_zero} "
                f"contemporaneous and {n_negative} with a negative lag).",
                "",
                "See `results/figures/figure_B1_within_lake_network.pdf`.",
                "",
            ])
        for stem in stems:
            lines = (OUTPUT_DIR / f"{stem}.md").read_text(
                encoding="utf-8"
            ).strip().split("\n")
            parts.extend([
                f"### {lines[0].strip('*')}",
                "",
                "\n".join(lines[1:]).strip(),
                "",
            ])

    path = OUTPUT_DIR / "Appendices.md"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")


def main():
    availability = build_appendix_a()
    build_appendix_b(availability)
    build_appendix_c()
    combine_markdown()
    print(f"All appendix files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
