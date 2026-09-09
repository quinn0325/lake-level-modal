"""Table 4.X — 13 种预测配置的滚动起点 RMSE 摘要（RQ3）。

正文 Figure 4.4 只画完全 matched 的 9 湖 XGBoost；本表把 13 种配置全部列出，
并对每一格标明有效湖数 n，因为各配置的有效样本并不相同：

  · 13 种配置 × 10 个湖 = 130 个理论组合；
  · 8 个组合因数据要求不适用，122 个进入拟合；
  · 29 个在拟合阶段报错（测试期外生变量缺口超过 MAX_FILLABLE_GAP_MONTHS，
    SARIMAX 无法完成整块预测），最终 93 个组合产出有效滚动预测。

n 不同意味着不同配置的 pooled mean 不能直接互相排名——这正是本表必须给出 n、
而 Figure 4.4 只取共同湖的原因。

数据源 ch4_tables/T1_rolling_lake_horizon_method.csv（372 行）与
T4_single_split_full.csv（122 行），均出自 2026-08-31 重跑结果。

输出
----
    ch4_tables/T4X_forecast_rmse_summary.csv   完整精度，供复算
    ch4_tables/T4X_forecast_rmse_summary.md    排版用，直接贴进正文
"""
from pathlib import Path

import pandas as pd

TAB_DIR = Path(__file__).resolve().parents[2] / "ch4_tables"
HORIZONS = [1, 3, 6, 12]

# 展示顺序：基线 → SARIMA 族 → XGBoost 族；族内 常规策略 → CCM 策略
METHODS = [
    ("Persistence",           "Persistence"),
    ("SARIMA",                "SARIMA (no exogenous)"),
    ("SARIMAX_all_vars",      "SARIMAX + all vars"),
    ("SARIMAX_Stepwise",      "SARIMAX + stepwise"),
    ("SARIMAX_CCM_direct",    "SARIMAX + CCM direct"),
    ("SARIMAX_CCM_ancestors", "SARIMAX + CCM ancestors"),
    ("SARIMAX_CCM_neighbor",  "SARIMAX + CCM neighbour"),
    ("XGBoost_AR_only",       "XGBoost, AR-only"),
    ("XGBoost_all_vars",      "XGBoost + all vars"),
    ("XGBoost_Stepwise",      "XGBoost + stepwise"),
    ("XGBoost_CCM_direct",    "XGBoost + CCM direct"),
    ("XGBoost_CCM_ancestors", "XGBoost + CCM ancestors"),
    ("XGBoost_CCM_neighbor",  "XGBoost + CCM neighbour"),
]


def main():
    t1 = pd.read_csv(TAB_DIR / "T1_rolling_lake_horizon_method.csv")
    t4 = pd.read_csv(TAB_DIR / "T4_single_split_full.csv")
    v = t1[t1.n_origins > 0]

    rows = []
    for key, name in METHODS:
        rec = {"method": name, "method_key": key,
               "n_lakes_fitted": int((t4.method == key).sum())}
        for h in HORIZONS:
            s = v[(v.method == key) & (v.horizon_months == h)]
            rec[f"rmse_h{h}"] = round(float(s.rmse.mean()), 3) if len(s) else None
            rec[f"n_h{h}"] = len(s)
        rows.append(rec)
    out = pd.DataFrame(rows)
    csv = TAB_DIR / "T4X_forecast_rmse_summary.csv"
    out.to_csv(csv, index=False)
    print(f"wrote {csv}")

    # ------------------------------------------------------------- markdown
    head = "| Configuration | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |"
    rule = "| --- | " + " | ".join(["---:"] * len(HORIZONS)) + " |"
    body = []
    for r in out.itertuples():
        cells = []
        for h in HORIZONS:
            val, n = getattr(r, f"rmse_h{h}"), getattr(r, f"n_h{h}")
            cells.append("—" if val is None else f"{val:.3f} ({n})")
        body.append(f"| {r.method} | " + " | ".join(cells) + " |")
    md = "\n".join([
        "**Table 4.X.** Mean RMSE (m) of rolling-origin forecasts by configuration "
        "and forecast horizon, with the number of lakes contributing to each mean "
        "in parentheses. Means are taken over the lakes for which that "
        "configuration produced valid rolling forecasts; because this number "
        "differs between configurations, pooled means are not directly comparable "
        "across rows and no ranking is implied. Of the 130 lake-by-configuration "
        "combinations, 8 were not applicable, 122 were fitted and 93 produced "
        "valid rolling forecasts; the shortfall is concentrated in the SARIMAX "
        "configurations, for which exogenous predictors were unavailable over the "
        "full evaluation period in some lakes. Figure 4.4 shows the fully matched "
        "subset.", "", head, rule, *body])
    md_path = TAB_DIR / "T4X_forecast_rmse_summary.md"
    md_path.write_text(md + "\n")
    print(f"wrote {md_path}")

    # ------------------------------------------------------------- 核对数字
    print("\n" + out[["method", "n_lakes_fitted"] +
                     [f"{p}_h{h}" for h in HORIZONS for p in ("rmse", "n")]]
          .to_string(index=False))
    print(f"\nlake x config fitted (T4 rows) : {len(t4)}  of {13 * 10}")
    print(f"  of which errored              : {int(t4.status.notna().sum())}")
    print(f"  valid rolling combinations    : {v.groupby(['lake', 'method']).ngroups}")


if __name__ == "__main__":
    main()
