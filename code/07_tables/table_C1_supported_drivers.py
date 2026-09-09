"""Appendix Table C1 — 通过 CCM 检验的 driver → WL 关系（正文 §4.2 的完整数值）。

正文只给汇总句子与 Figure 4.2，逐条数值放附录。本脚本从
ch4_tables/T5_within_lake_edges.csv（420 条湖内候选边，2026-08-31 重跑）
里筛出 effect == "WL" 且 statistically_significant 为真的 24 条。

注意 statistically_significant 字段本身已经包含收敛诊断
（见 ccm_full_pipeline.apply_fdr_and_causal_evidence:813），
因此这 24 条即"BH-FDR 显著 且 收敛诊断通过"的全集，
不需要再额外与 convergence_diagnostic_pass 取交集。

输出
----
    ch4_tables/TC1_supported_drivers.csv   完整精度，供复算
    ch4_tables/TC1_supported_drivers.md    排版用，直接贴进附录
"""
from pathlib import Path

import pandas as pd

TAB_DIR = Path(__file__).resolve().parents[2] / "ch4_tables"

SYSTEMS = [
    ("Okanagan", ["Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake"]),
    ("Nelson–Winnipeg", ["Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
                         "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake"]),
]
LAKES = [lk for _, g in SYSTEMS for lk in g]
SYSTEM_OF = {lk: n for n, g in SYSTEMS for lk in g}
LABEL = {lk: lk.replace("_Lake", "").replace("_", " ") for lk in LAKES}
LABEL["Lake_of_the_Woods"] = "Lake of the Woods"

DRIVERS = ["RegFlow", "R", "P", "Evap", "SWE", "T"]
DRIVER_LABEL = {"RegFlow": "Regulated flow", "R": "Runoff", "P": "Precipitation",
                "Evap": "Evaporation", "SWE": "SWE", "T": "Temperature"}
CLASS_LABEL = {"resolved": "Positive", "unresolved_contemporaneous": "Contemporaneous",
               "rejected_reverse": "Negative"}


def main():
    t5 = pd.read_csv(TAB_DIR / "T5_within_lake_edges.csv")
    sup = t5[(t5.effect == "WL") & t5.statistically_significant].copy()
    sup["_lk"] = sup.lake.map(LAKES.index)
    sup["_dr"] = sup.cause.map(DRIVERS.index)
    sup = sup.sort_values(["_lk", "_dr"])

    out = pd.DataFrame({
        "lake": sup.lake.map(LABEL),
        "system": sup.lake.map(SYSTEM_OF),
        "driver": sup.cause.map(DRIVER_LABEL),
        "rho": sup.obs_rho.round(3),
        "lag_d_months": sup.obs_lag.astype(int),
        "p_fdr": sup.p_fdr.round(4),
        "kendall_tau_L": sup.kendall_tau.round(2),
        "kendall_p": sup.kendall_p.map(lambda v: f"{v:.1e}"),
        "n_months": sup.obs_n.astype(int),
        "E": sup.E_effect.astype(int),
        "temporal_class": sup.lag_resolution.map(CLASS_LABEL),
        "causal_evidence": sup.causal_evidence.map({True: "Yes", False: "No"}),
    })
    csv_path = TAB_DIR / "TC1_supported_drivers.csv"
    out.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}  ({len(out)} rows)")

    # ------------------------------------------------------------- markdown
    head = ("| Lake | Driver | ρ | d (months) | FDR p | Kendall τ_L | n | E "
            "| Temporal class |")
    rule = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    rows = [f"| {r.lake} | {r.driver} | {r.rho:.3f} "
            f"| {r.lag_d_months:+d}".replace("+0", "0") + " "
            f"| {r.p_fdr:.4f} | {r.kendall_tau_L:.2f} | {r.n_months} | {r.E} "
            f"| {r.temporal_class} |" for r in out.itertuples()]
    md = "\n".join([
        "**Table C1.** CCM-supported candidate drivers of lake water level "
        "(all 24 driver → WL edges passing BH-FDR at α = 0.05 within the "
        "420-edge within-lake family and the convergence diagnostic). "
        "*d* is the optimal lag, positive when the driver leads water level; "
        "τ_L is the Kendall rank correlation between cross-map skill and "
        "library size. Lakes are ordered by system, drivers by overall "
        "prevalence. Skaha Lake returned no supported driver and does not "
        "appear.", "", head, rule, *rows])
    md_path = TAB_DIR / "TC1_supported_drivers.md"
    md_path.write_text(md + "\n")
    print(f"wrote {md_path}")

    print("\n" + out.to_string(index=False))
    print("\ntemporal class:", out.temporal_class.value_counts().to_dict())
    print("lakes represented:", out.lake.nunique(), "of", len(LAKES))


if __name__ == "__main__":
    main()
