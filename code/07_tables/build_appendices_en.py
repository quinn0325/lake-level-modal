"""English edition of the appendix tables.

Reads the CSVs written by build_appendices.py and re-renders them with English
titles and notes, then assembles a single Markdown file for conversion to Word.
The Chinese edition is left untouched; only the surrounding prose differs, the
numbers are the same files.

Run
---
    python code/07_tables/build_appendices.py       # writes the CSVs
    python code/07_tables/build_appendices_en.py    # English titles and notes
    pandoc appendices/Appendices_EN.md -o appendices/Appendices_EN.docx \
        --from=markdown+pipe_tables

Output
------
    final/appendices/en/<stem>.md
    final/appendices/Appendices_EN.md
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "appendices"
EN = OUT / "en"

COL_DECIMALS = {"coverage_pct": 1, "p_value": 4, "p_fdr": 4, "kendall_p": 4,
                "dm_stat": 3, "S_ij": 3, "cv_rmse_m": 5, "learning_rate": 2}

# stem -> (title, note, decimals)
META = {
    "A1_lakes_and_stations": (
        "Table A1. Study lakes, water-level gauges and regulated-outflow gauges",
        "Gauge identifiers are Water Survey of Canada (HYDAT) station numbers; "
        "hydrolakes_id is the HydroLAKES Hylak_id. Regflow_period gives the "
        "years over which regulated outflow is available.", 3),
    "A2_data_availability": (
        "Table A2. Water-level data availability and variability by lake",
        "n_months is the number of months from 1994-01 to 2024-12; "
        "coverage_pct is the percentage of those months with an observation; "
        "longest_gap_months is the longest run of consecutive missing months. "
        "sd_m, iqr_m and range_m are computed after centring each lake on its "
        "own long-term mean, in metres.", 3),
    "A3_flagged_outliers": (
        "Table A3. Water-level observations set to missing by the two-stage "
        "robust screen",
        "Criterion: a month-to-month change with a robust z above 6, confirmed "
        "by a robust z above 5 on the level itself. Flagged points were set to "
        "missing; the recorded values were not adjusted.", 3),
    "B1_embedding_parameters": (
        "Table B1. Embedding dimension E by lake and variable",
        "E was chosen from 2–10 by one-step simplex self-prediction on the "
        "training period; the embedding delay tau was fixed at 1 for every lake "
        "and variable.", 0),
    "B2_supported_drivers": (
        "Table B2. The 24 supported driver-to-water-level relationships",
        "Relationships passing both Benjamini–Hochberg FDR control at alpha = "
        "0.05 within the 420-edge within-lake family and the convergence "
        "diagnostic. d is the optimal lag in months, positive when the driver "
        "leads water level; tau_L is the Kendall rank correlation between "
        "cross-map skill and library size.", 3),
    "B3_supported_between_lake_edges": (
        "Table B3. The 26 supported between-lake relationships",
        "The testing family comprises 90 directed candidate edges from 45 lake "
        "pairs; 26 passed both FDR control and the convergence diagnostic. "
        "Tier is a "
        "post-hoc descriptive grouping by hydrological separation, introduced "
        "at the writing stage.", 3),
    "B4_lake_pair_strength": (
        "Table B4. CCM strength and hydrological connectivity for all 45 lake "
        "pairs",
        "S_ij is the mean |rho| over the two directions of a pair. A pair is "
        "recorded as detected when at least one direction is FDR-significant, "
        "convergent and has an optimal lag d >= 0. This table is the complete "
        "input to the Mann–Whitney test reported in Section 4.3.", 3),
    "C1_selected_predictors": (
        "Table C1. Exogenous predictors and forecast-domain lags entering each "
        "model",
        "Lags are re-optimised within the forecast domain with a lower bound of "
        "one month. CCM_neighbor rows list only the between-lake terms added on "
        "top of CCM_ancestors; the full predictor set for that strategy is the "
        "CCM_ancestors row for the same lake plus this row. All-vars is not "
        "listed separately: it comprises every variable passing the "
        "data-availability screen, with lags from lagged cross-correlation. The "
        "only variable excluded by the long-gap rule in this study was RegFlow "
        "at Vaseux, whose gauge began recording in 2012.", 3),
    "C2_model_configurations": (
        "Table C2. Model configuration and fitting outcome by lake and method",
        "SARIMA(X) cells give the (p,d,q)(P,D,Q,12) orders; XGBoost cells give "
        "the number of exogenous predictors. \"Not applicable\" marks the eight "
        "combinations for which no CCM relationship was available, all at Skaha "
        "and Kiskitto; \"failed\" marks the 29 combinations in which an "
        "exogenous gap longer than six months in the test period prevented "
        "fitting. Of 13 configurations across 10 lakes, 122 were fitted and 93 "
        "produced valid rolling forecasts.", 3),
    "C3_xgboost_hyperparameters": (
        "Table C3. XGBoost hyperparameter grid and cross-validation results",
        "Subsample = 0.8, colsample_bytree = 0.8 and random_state = 0 were "
        "fixed. Mean RMSE was compared over three expanding-window folds in "
        "each of the ten lakes (30 folds in total) using autoregressive "
        "features only, and one configuration was then shared by every lake and "
        "strategy.", 5),
    "C4_dm_summary": (
        "Table C4. Summary of Diebold–Mariano tests",
        "Fourteen pre-specified method pairs were tested lake by lake and "
        "horizon by horizon, giving 341 comparisons of which 333 returned a "
        "usable statistic. Benjamini–Hochberg correction was applied once "
        "across those 333 tests, leaving 42 significant.", 3),
}

SECTIONS = [
    ("Appendix A. Data and data availability",
     ["A1_lakes_and_stations", "A2_data_availability",
      "A3_flagged_outliers"]),
    ("Appendix B. CCM analysis",
     ["B1_embedding_parameters", "B2_supported_drivers",
      "B3_supported_between_lake_edges", "B4_lake_pair_strength",]),
    ("Appendix C. Forecasting analysis",
     ["C1_selected_predictors", "C2_model_configurations",
      "C3_xgboost_hyperparameters", "C4_dm_summary"]),
]

FIG_B1 = (
    "Figure B1. Support for all 420 within-lake candidate relationships",
    "Rows are the 42 directed variable pairs, grouped into seven blocks by "
    "cause variable; columns are the ten lakes. A shaded cell marks a "
    "relationship passing both Benjamini–Hochberg FDR control and the "
    "convergence diagnostic, with shading giving cross-map skill rho and the "
    "cell label the optimal lag d in months (the plus sign is omitted). Orange "
    "squares mark d = 0 and grey triangles d < 0; the bars at the right give "
    "the number of lakes supporting each relationship. Of the 420 tests, 212 "
    "were supported (150 with a positive lag, 23 contemporaneous and 39 with a "
    "negative lag).")


# 表格内仍为中文的单元格：A1 的备注列与 C2 的状态标记
CELL_EN = {
    "无外生": "no exogenous",
    "不适用": "not applicable",
    "失败": "failed",
    "测流站 2012 年始测，序列受限于该子区间":
        "gauge began recording in 2012; series limited to that sub-period",
    "两站流量相加（Norman Dam + Kenora Powerhouse）":
        "sum of two gauges (Norman Dam + Kenora Powerhouse)",
    "与 Kiskitto 共用 Jenpeg 大坝出流":
        "shares the Jenpeg dam outflow with Kiskitto",
    "与 Playgreen 共用 Jenpeg 大坝出流":
        "shares the Jenpeg dam outflow with Playgreen",
    "无自有坝，以下游 Kelsey 发电站出流代理":
        "no dam of its own; proxied by outflow at the downstream Kelsey "
        "generating station",
}


def fmt(v, nd):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}" if abs(v) >= 1e-4 or v == 0 else f"{v:.2e}"
    return CELL_EN.get(str(v), str(v))


def to_md(df, nd):
    cols = list(df.columns)
    num = [c for c in cols
           if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool]
    dec = [COL_DECIMALS.get(c, nd) for c in cols]
    head = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join("---:" if c in num else "---" for c in cols) + " |"
    body = ["| " + " | ".join(fmt(v, d) for v, d in zip(row, dec)) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def main():
    EN.mkdir(parents=True, exist_ok=True)
    parts = ["# Appendices", "",
             "All content is derived from the 2026-08-31 re-run and generated "
             "by `code/07_tables/build_appendices.py`. The three appendices "
             "mirror the structure of Chapter 3: Appendix A corresponds to "
             "Sections 3.1–3.2 (data), Appendix B to Section 3.3 (CCM "
             "analysis) and Appendix C to Section 3.4 (forecasting).", ""]
    for section, stems in SECTIONS:
        parts += [f"## {section}", ""]
        if section.startswith("Appendix B"):
            parts += [f"### {FIG_B1[0]}", "", FIG_B1[1], "",
                      "*(insert figure_B1_within_lake_network.pdf here)*", ""]
        for stem in stems:
            title, note, nd = META[stem]
            df = pd.read_csv(OUT / f"{stem}.csv")
            md = to_md(df, nd)
            (EN / f"{stem}.md").write_text(f"**{title}**\n\n{note}\n\n{md}\n")
            parts += [f"### {title}", "", note, "", md, ""]
            print(f"  {stem:<34} {len(df):>4} rows")
    path = OUT / "Appendices_EN.md"
    path.write_text("\n".join(parts) + "\n")
    n = sum(1 for l in path.read_text().split("\n") if l.startswith("|"))
    print(f"\nwrote {path}  ({n} table rows)")


if __name__ == "__main__":
    main()
