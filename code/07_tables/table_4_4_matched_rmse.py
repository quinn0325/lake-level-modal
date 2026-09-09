"""Table 4.X — matched-sample 平均 RMSE（RQ3）。

按模型族分成两个 block，每个 block 内部所有策略使用**完全相同的湖泊集合**，
因此块内各行可以直接比较；两个 block 之间不可比，也不应比较。

  Block A  SARIMA(X)：五种策略（SARIMA / all vars / stepwise / CCM direct /
           CCM ancestors）× 四个预见期全部有效的共同湖只有 3 个
           （Kalamalka、Okanagan、Vaseux）。注意不是 4 个——各外生策略
           自己的 4 湖集合并不相同（all vars 与 stepwise 含 Skaha，
           两个 CCM 策略含 Kiskitto），取交集后只剩 3 个。
  Block B  XGBoost：五种策略 × 四个预见期全部有效的共同湖为 9 个（除 Skaha）。

Persistence 在两个 block 内各按该 block 的湖泊集合另算一次，作为同尺度参照。
CCM neighbour 的有效湖数随预见期变化，无法进入任何 matched block，单列在表下。

数据源 ch4_tables/T1_rolling_lake_horizon_method.csv（372 行，2026-08-31 重跑）。

输出
----
    ch4_tables/T4X_matched_rmse.csv   完整精度，供复算
    ch4_tables/T4X_matched_rmse.md    排版用，直接贴进正文
"""
from pathlib import Path

import pandas as pd

TAB_DIR = Path(__file__).resolve().parents[2] / "ch4_tables"
HORIZONS = [1, 3, 6, 12]

BLOCKS = [
    ("SARIMA(X)", [("SARIMA", "No exogenous"),
                   ("SARIMAX_all_vars", "All vars"),
                   ("SARIMAX_Stepwise", "Stepwise"),
                   ("SARIMAX_CCM_direct", "CCM direct"),
                   ("SARIMAX_CCM_ancestors", "CCM ancestors")]),
    ("XGBoost", [("XGBoost_AR_only", "AR-only"),
                 ("XGBoost_all_vars", "All vars"),
                 ("XGBoost_Stepwise", "Stepwise"),
                 ("XGBoost_CCM_direct", "CCM direct"),
                 ("XGBoost_CCM_ancestors", "CCM ancestors")]),
]
NEIGHBOUR = [("SARIMAX_CCM_neighbor", "SARIMAX"),
             ("XGBoost_CCM_neighbor", "XGBoost")]


def matched_lakes(v, keys):
    sets = [set(v[(v.method == m) & (v.horizon_months == h)].lake)
            for m in keys for h in HORIZONS]
    return sorted(set.intersection(*sets))


def main():
    t1 = pd.read_csv(TAB_DIR / "T1_rolling_lake_horizon_method.csv")
    v = t1[t1.n_origins > 0]

    rows, md_rows = [], []
    for block, methods in BLOCKS:
        lakes = matched_lakes(v, [k for k, _ in methods])
        md_rows.append(f"| **{block}** (matched on {len(lakes)} lakes) | | | | |")
        for key, name in methods + [("Persistence", "Persistence")]:
            s = v[(v.method == key) & v.lake.isin(lakes)]
            rec = {"block": block, "n_lakes": len(lakes), "method": name,
                   "method_key": key}
            cells = []
            for h in HORIZONS:
                val = float(s[s.horizon_months == h].rmse.mean())
                rec[f"rmse_h{h}"] = round(val, 3)
                cells.append(f"{val:.3f}")
            rows.append(rec)
            md_rows.append(f"| {name} | " + " | ".join(cells) + " |")

    # CCM neighbour：样本随预见期变化，只做脚注
    foot = []
    for key, name in NEIGHBOUR:
        s = v[v.method == key]
        cells = [f"{s[s.horizon_months == h].rmse.mean():.3f} "
                 f"({len(s[s.horizon_months == h])})" for h in HORIZONS]
        foot.append(f"{name} " + ", ".join(cells))
        rows.append({"block": "unmatched", "n_lakes": None, "method": name,
                     "method_key": key,
                     **{f"rmse_h{h}": round(float(
                         s[s.horizon_months == h].rmse.mean()), 3)
                        for h in HORIZONS},
                     **{f"n_h{h}": len(s[s.horizon_months == h])
                        for h in HORIZONS}})

    out = pd.DataFrame(rows)
    csv = TAB_DIR / "T4X_matched_rmse.csv"
    out.to_csv(csv, index=False)
    print(f"wrote {csv}")

    header = ("| Configuration | " + " | ".join(f"h = {h}" for h in HORIZONS)
              + " |\n| --- | " + " | ".join(["---:"] * len(HORIZONS)) + " |")
    md = "\n".join([
        "**Table 4.X.** Mean RMSE (m) of rolling-origin forecasts. Rows are "
        "comparable within a block but not between blocks; persistence is "
        "recomputed on each block's lakes.", "", header,
        *md_rows, "",
        "CCM neighbour is unmatched (valid lakes change with horizon): "
        + "; ".join(foot) + ".",
    ])
    md_path = TAB_DIR / "T4X_matched_rmse.md"
    md_path.write_text(md + "\n")
    print(f"wrote {md_path}")

    # ------------------------------------------------------------- 核对数字
    for block, methods in BLOCKS:
        lakes = matched_lakes(v, [k for k, _ in methods])
        print(f"\n{block}: matched on {len(lakes)} lakes — "
              + ", ".join(lk.replace('_Lake', '') for lk in lakes))
    print()
    print(out[out.block != "unmatched"][
        ["block", "n_lakes", "method"] + [f"rmse_h{h}" for h in HORIZONS]]
        .to_string(index=False))
    print("\nunmatched (footnote):")
    print(out[out.block == "unmatched"][
        ["method"] + [f"{p}_h{h}" for h in HORIZONS for p in ("rmse", "n")]]
        .to_string(index=False))


if __name__ == "__main__":
    main()
