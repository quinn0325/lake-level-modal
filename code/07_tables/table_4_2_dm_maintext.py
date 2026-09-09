"""Table 4.2 — 正文用的 Diebold–Mariano 结果摘要。

只列正文实际引用的方法对，每行给出：可比湖数、检验项数、BH 校正后显著项数、
显著项在两个方向上的分配，以及显著项按预见期的分布。逐项统计量与 p 值见表C5
与补充材料 S4。

DM 检验族为 14 组预先设定的方法对 × 湖泊 × 预见期，共 341 项，其中 333 项返回
可用统计量；BH 校正在这 333 项上一次性施加，42 项在校正后仍显著。本表所列
10 组覆盖其中 231 项、34 项显著。

输出
----
    ch4_tables/T4_2_dm_maintext.csv / .md
"""
from pathlib import Path

import pandas as pd

TAB = Path(__file__).resolve().parents[2] / "ch4_tables"

PAIRS = [("XGBoost", "XGBoost_CCM_direct", "XGBoost_AR_only"),
         ("XGBoost", "XGBoost_CCM_ancestors", "XGBoost_AR_only"),
         ("XGBoost", "XGBoost_CCM_ancestors", "Persistence"),
         ("XGBoost", "XGBoost_CCM_ancestors", "XGBoost_all_vars"),
         ("XGBoost", "XGBoost_CCM_ancestors", "XGBoost_Stepwise"),
         ("XGBoost", "XGBoost_CCM_neighbor", "XGBoost_CCM_ancestors"),
         ("SARIMA(X)", "SARIMAX_CCM_ancestors", "SARIMA"),
         ("SARIMA(X)", "SARIMAX_CCM_ancestors", "SARIMAX_Stepwise"),
         ("SARIMA(X)", "SARIMAX_CCM_ancestors", "SARIMAX_CCM_direct"),
         ("SARIMA(X)", "SARIMAX_CCM_neighbor", "SARIMAX_CCM_ancestors")]
HORIZONS = [1, 3, 6, 12]


def short(m):
    return (m.replace("XGBoost_", "").replace("SARIMAX_", "")
             .replace("CCM_", "CCM-").replace("_", " ")
             .replace("AR only", "AR-only").replace("all vars", "All-vars")
             .replace("neighbor", "neighbour"))


def main():
    t3 = pd.read_csv(TAB / "T3_dm_bh.csv")
    ok = t3[t3.p_value.notna()]
    rows = []
    for fam, a, b in PAIRS:
        g = ok[(ok.method_1 == a) & (ok.method_2 == b)]
        if not len(g):
            continue
        sig = g[g.sig_fdr]
        rows.append({
            "family": fam,
            "strategy": short(a),
            "reference": short(b),
            "lakes": g.lake.nunique(),
            "tests": len(g),
            "significant": len(sig),
            "favouring_strategy": int((sig.winner == a).sum()),
            "favouring_reference": int((sig.winner == b).sum()),
            "significant_by_horizon": " / ".join(
                str(int((sig.horizon_months == h).sum())) for h in HORIZONS),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "T4_2_dm_maintext.csv", index=False)

    def md(df):
        cols = list(df.columns)
        num = {"lakes", "tests", "significant", "favouring_strategy",
               "favouring_reference"}
        head = "| " + " | ".join(cols) + " |"
        rule = "| " + " | ".join("---:" if c in num else "---" for c in cols) + " |"
        body = ["| " + " | ".join(str(v) for v in r) + " |"
                for r in df.itertuples(index=False)]
        return "\n".join([head, rule] + body)

    note = (
        "**表4.2　Diebold–Mariano 检验结果摘要。** 每行为一组预先设定的方法对，"
        "逐湖、逐预见期比较滚动起点的平方误差损失差；「significant」为 "
        "Benjamini–Hochberg 校正后仍显著的项数，校正在全部 333 项可用检验上"
        "一次性施加。「favouring」两列给出显著项中各方向的占比，"
        "「significant by horizon」为显著项在 h = 1 / 3 / 6 / 12 上的分布。"
        "全部 14 组方法对共 341 项检验（333 项可用、42 项显著），"
        "逐项结果见表C5 与补充材料 S4。")
    (TAB / "T4_2_dm_maintext.md").write_text(note + "\n\n" + md(out) + "\n")
    print(out.to_string(index=False))
    print(f"\n本表覆盖 {out.tests.sum()} 项检验、{out.significant.sum()} 项显著"
          f"（全部 333 项中 42 项显著）")


if __name__ == "__main__":
    main()
