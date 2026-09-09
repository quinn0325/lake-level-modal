"""构建拟公开的月度数据集：十个受调控加拿大湖泊，1994-01 至 2024-12。

产出三个层次，对应三种使用需求：

  raw_station     逐站原始水位（未做异常筛查、未合成），保留观测溯源
  lake_monthly    每湖每月一行的分析级数据集：合成后的 WL 与六个驱动变量，
                  真实缺测保留为空，未去季节化
  lake_monthly_deseasonalised
                  同上，但各变量已减去训练期的逐月气候态（式 3），
                  即 CCM 与预测分析实际使用的序列

去季节化只用训练期（前 335 个月）的月均值，与主流程一致，避免测试期信息泄漏。

跑法
----
    python code/08_dataset/build_public_dataset.py

输出
----
    final/dataset/raw_station_water_level.csv
    final/dataset/lake_monthly.csv
    final/dataset/lake_monthly_deseasonalised.csv
    final/dataset/data_dictionary.csv
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "01_shared"))
import ccm_full_pipeline as p                                    # noqa: E402

p.log = lambda msg: None
OUT = CODE.parent / "dataset"
PKL = Path(p.PKL_DIR)

SYSTEMS = [("Okanagan", ["Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake",
                         "Vaseux_Lake"]),
           ("Nelson-Winnipeg", ["Rainy_Lake", "Lake_of_the_Woods",
                                "Playgreen_Lake", "Kiskitto_Lake",
                                "Sipiwesk_Lake", "Split_Lake"])]
LAKES = [lk for _, g in SYSTEMS for lk in g]
SYSTEM_OF = {lk: n for n, g in SYSTEMS for lk in g}
LABEL = {lk: lk.replace("_Lake", "").replace("_", " ") for lk in LAKES}
LABEL["Lake_of_the_Woods"] = "Lake of the Woods"
VARS = ["RegFlow", "R", "P", "Evap", "SWE", "T"]

DICT = [
    ("lake", "-", "Lake name"),
    ("system", "-", "Hydrological system (Okanagan or Nelson-Winnipeg)"),
    ("hydrolakes_id", "-", "HydroLAKES Hylak_id"),
    ("month", "-", "First day of the calendar month (YYYY-MM-01)"),
    ("station", "-", "Water Survey of Canada (HYDAT) station number"),
    ("water_level_m", "m", "Observed water level at that station, station datum"),
    ("WL", "m",
     "Lake water level after two-stage outlier screening and anomaly-based "
     "multi-station combination (Eq. 1-2)"),
    ("RegFlow", "m3 s-1",
     "Regulated outflow, summed over the lake's regulation gauges"),
    ("T", "degC", "2 m air temperature, mean over ERA5-Land cells on the lake"),
    ("P", "mm month-1", "Total precipitation, mean over the upstream catchment"),
    ("R", "mm month-1", "Total runoff, mean over the upstream catchment"),
    ("SWE", "mm", "Snow-water equivalent, mean over the upstream catchment"),
    ("Evap", "mm month-1",
     "Evaporation, mean over ERA5-Land cells on the lake; positive means water "
     "loss from the lake"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_rows, panels = [], {}

    for lk in LAKES:
        with open(PKL / f"{lk}_result.pkl", "rb") as fh:
            cached = pickle.load(fh)

        wide = cached["wide_wl"].copy()
        wide.index = pd.DatetimeIndex(wide.index)
        wide = wide.sort_index().asfreq("MS")
        for st in wide.columns:
            s = wide[st]
            raw_rows.append(pd.DataFrame({
                "lake": LABEL[lk], "system": SYSTEM_OF[lk],
                "hydrolakes_id": cached.get("hylak_id"),
                "station": st, "month": s.index.strftime("%Y-%m-%d"),
                "water_level_m": s.values}))

        wl = p.combine_station_water_levels(
            p.clean_wide_wl(cached["wide_wl"]), method="anomaly_mean")
        wl = wl.sort_index().asfreq("MS")
        pred = cached["real_predictors"].copy()
        pred.index = pd.DatetimeIndex(pred.index)
        panel = pd.concat([wl.rename("WL"), pred.reindex(wl.index)], axis=1)
        panels[lk] = panel[["WL"] + VARS]

    raw = pd.concat(raw_rows, ignore_index=True)
    raw.to_csv(OUT / "raw_station_water_level.csv", index=False)

    def stack(dct, name):
        rows = []
        for lk, panel in dct.items():
            df = panel.reset_index(names="month")
            df.insert(0, "hydrolakes_id", None)
            df.insert(0, "system", SYSTEM_OF[lk])
            df.insert(0, "lake", LABEL[lk])
            df["month"] = pd.DatetimeIndex(df["month"]).strftime("%Y-%m-%d")
            rows.append(df.drop(columns="hydrolakes_id"))
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(OUT / name, index=False)
        return out

    lm = stack(panels, "lake_monthly.csv")

    # 去季节化：仅用训练期（前 n - 37 个月）的逐月气候态，与主流程一致
    train_end = len(next(iter(panels.values()))) - p.FORECAST_HORIZON
    des = {lk: panel.apply(lambda c: p.deseasonalize(c, train_end=train_end))
           for lk, panel in panels.items()}
    lmd = stack(des, "lake_monthly_deseasonalised.csv")

    pd.DataFrame(DICT, columns=["field", "unit", "description"]).to_csv(
        OUT / "data_dictionary.csv", index=False)

    print(f"raw_station_water_level.csv          {len(raw):>6} rows "
          f"({raw.station.nunique()} stations)")
    print(f"lake_monthly.csv                     {len(lm):>6} rows "
          f"({lm.lake.nunique()} lakes x {lm.month.nunique()} months)")
    print(f"lake_monthly_deseasonalised.csv      {len(lmd):>6} rows")
    print(f"data_dictionary.csv                  {len(DICT):>6} rows")
    print(f"\ntraining window used for the climatology: first {train_end} months")
    print("\ncompleteness of lake_monthly (% non-missing):")
    print((100 * lm.groupby("lake")[["WL"] + VARS].apply(
        lambda d: d.notna().mean())).round(1).to_string())


if __name__ == "__main__":
    main()
