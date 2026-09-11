"""Export the public monthly dataset from downloaded Modal lake panels.

The script writes station-level water levels, cleaned lake-level panels,
deseasonalised analysis panels, and a data dictionary to ``dataset/``. It does
not rerun CCM or forecasting.
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "01_analysis_core"))
import analysis_core as p  # noqa: E402
from config import LAKES  # noqa: E402

# Avoid printing the same outlier messages while exporting derived files.
p.log = lambda msg: None
OUT = CODE.parent / "dataset"
PKL = Path(p.PKL_DIR)

OKANAGAN_LAKES = {
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
}
SYSTEM_OF = {
    lake: "Okanagan" if lake in OKANAGAN_LAKES else "Nelson-Winnipeg"
    for lake in LAKES
}
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

        wl = p.combine_station_water_levels(p.clean_wide_wl(cached["wide_wl"]))
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
            df.insert(0, "system", SYSTEM_OF[lk])
            df.insert(0, "lake", LABEL[lk])
            df["month"] = pd.DatetimeIndex(df["month"]).strftime("%Y-%m-%d")
            rows.append(df)
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(OUT / name, index=False)
        return out

    lm = stack(panels, "lake_monthly.csv")

    # Match the training-only deseasonalisation used by the analysis.
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
