"""Screen Canadian HYDAT lake and regulated-outlet station pairs on Modal.
在 Modal 上筛选加拿大 HYDAT 湖泊水位站与受调控出口流量站配对。

This optional provenance step is not required to rerun the fixed ten-lake analysis.
这是可选的样本筛选溯源步骤；复现固定的十湖分析时无需运行。

Water-level stations are matched to HydroLAKES natural-lake polygons and checked for
1994–2024 completeness. Regulated outlets are matched by province, distance, and
drainage-area ratio; uncertain matches remain flagged for manual review.
水位站先匹配 HydroLAKES 天然湖泊并检查 1994–2024 年完整度；受调控出口站再按省份、
距离与集水面积比匹配，不确定的结果保留人工核查标记。

The output is ``final_lake_screening.csv`` in the ``ccm-data`` Volume.
输出为 ``ccm-data`` Volume 中的 ``final_lake_screening.csv``。
"""

import csv
import math
import re
import sqlite3
from collections import defaultdict

import shapefile
from shapely.geometry import Point
from shapely.geometry import shape as shp_shape
from shapely.prepared import prep

# Configuration / 配置
HYDAT_DB = "/data/Hydat.sqlite3"
# Resolve the nested shapefile path at runtime. / 运行时解析嵌套目录中的矢量文件路径。
HYDROLAKES_POLY_SHP = None
OUT_CSV = "/data/final_lake_screening.csv"


def resolve_hydrolakes_path(data_root="/data"):
    """Locate the HydroLAKES polygon shapefile.
    查找 HydroLAKES 湖泊面矢量文件。
    """
    import glob
    from pathlib import Path
    matches = glob.glob(str(Path(data_root) / "HydroLAKES_polys_v10_shp" / "**" / "*.shp"), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No HydroLAKES_polys_v10.shp found under {data_root}/HydroLAKES_polys_v10_shp/. "
            f"Upload it once with: modal volume put ccm-data <local-path>"
            f"HydroLAKES_polys_v10_shp HydroLAKES_polys_v10_shp"
        )
    return matches[0]

START_YEAR, END_YEAR = 1994, 2024
WL_COMPLETENESS_MIN = 0.90
RF_COMPLETENESS_MIN = 0.90
MAX_DIST_DEG = 0.05             # Near-shore matching tolerance. / 近岸匹配容差。
MAX_OUTLET_DIST_KM = 100.0      # Maximum outlet distance. / 出口站最大距离。
DRAINAGE_RATIO_LOW = 0.85
DRAINAGE_RATIO_HIGH = 1.20
TIGHT_RATIO_LOW = 0.95
TIGHT_RATIO_HIGH = 1.05
HYDROLAKES_NATURAL_TYPES = (1, 3)
SMALL_AREA_WARN_KM2 = 5.0       # Flag small polygon matches. / 标记过小的湖泊面匹配。

# Verified cross-border stations may bypass the Canada filter. / 经确认的跨境站可跳过加拿大筛选条件。
NO_COUNTRY_FILTER_STATIONS = set()  # Example: Lake Huron 02EA014. / 示例：Lake Huron 站 02EA014。

EXCLUDE_DIRECTION_RE = re.compile(r"\b(ABOVE|INLET)\b", re.I)
STOP_FOR_NAME_OVERLAP = {
    "LAKE", "LAKES", "LAC", "AT", "NEAR", "OF", "THE", "RIVER", "RIVIERE",
    "CREEK", "DU", "AU", "LA", "OUTLET", "BELOW", "ABOVE", "DAM", "BARRAGE",
    "A", "EN", "AVAL", "AMONT", "STATION", "GENERATING",
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Return great-circle distance in kilometres.
    计算两点间的大圆距离（千米）。
    """
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def tokenize(name):
    """Normalize a station name into comparable tokens.
    将测站名称标准化为可比较的词元集合。
    """
    name = re.sub(r"[().,'\-]", " ", (name or "").upper())
    return set(t for t in name.split() if t)


# HYDAT reference data / HYDAT 参考数据
def load_hydat_reference(con):
    """Load station metadata, regulation history, and data ranges.
    读取测站元数据、调控记录与数据时段。
    """
    cur = con.cursor()
    cur.row_factory = sqlite3.Row

    stations = {
        r["STATION_NUMBER"]: dict(r)
        for r in cur.execute(
            "SELECT STATION_NUMBER, STATION_NAME, PROV_TERR_STATE_LOC, LATITUDE, "
            "LONGITUDE, DRAINAGE_AREA_GROSS FROM STATIONS"
        )
    }

    regulated = {
        r[0]: r[1]
        for r in cur.execute(
            "SELECT STATION_NUMBER, MAX(REGULATED) FROM STN_REGULATION GROUP BY STATION_NUMBER"
        )
    }
    reg_history = defaultdict(list)
    for sn, yf, yt, rv in cur.execute(
        "SELECT STATION_NUMBER, YEAR_FROM, YEAR_TO, REGULATED FROM STN_REGULATION"
    ):
        reg_history[sn].append((yf, yt, rv))

    h_range = {
        r[0]: (r[1], r[2])
        for r in cur.execute("SELECT STATION_NUMBER, YEAR_FROM, YEAR_TO FROM STN_DATA_RANGE WHERE DATA_TYPE='H'")
    }
    q_range = {
        r[0]: (r[1], r[2])
        for r in cur.execute("SELECT STATION_NUMBER, YEAR_FROM, YEAR_TO FROM STN_DATA_RANGE WHERE DATA_TYPE='Q'")
    }
    return stations, regulated, reg_history, h_range, q_range


# Candidate lakes / 候选湖泊
def find_stations_on_natural_lakes(poly_shp_path, stations, h_range):
    """Match eligible level gauges to all Canadian polygons, then retain natural lakes.
    将合格水位站先匹配全部加拿大湖泊面，再保留天然湖泊类型。

    Polygon containment takes priority over the nearest match within the shoreline tolerance.
    湖泊面包含关系优先，其次采用近岸容差内的最近匹配。
    """
    h_candidates = [
        sn
        for sn, (yf, yt) in h_range.items()
        if yf is not None and yt is not None and yf <= START_YEAR and yt >= END_YEAR
        and stations.get(sn, {}).get("LATITUDE") is not None
    ]
    print(f"[pool] ALL H-type stations (name-agnostic) covering {START_YEAR}-{END_YEAR}: {len(h_candidates)}")

    pts = {sn: (stations[sn]["LONGITUDE"], stations[sn]["LATITUDE"]) for sn in h_candidates}
    lons = [p[0] for p in pts.values()]
    lats = [p[1] for p in pts.values()]
    scan_bbox = (min(lons) - 1, min(lats) - 1, max(lons) + 1, max(lats) + 1)

    sf = shapefile.Reader(poly_shp_path, encoding="latin1", encodingErrors="replace")
    fields = [f[0] for f in sf.fields[1:]]
    idx = {n: fields.index(n) for n in ["Country", "Lake_type", "Hylak_id", "Lake_name", "Lake_area"]}
    print(f"[scan] iterating {sf.numRecords} HydroLAKES polygon records (ALL types, resolve-then-filter) ...")

    # Best polygon match for each station. / 每个测站的最佳湖泊面匹配。
    resolved = {}
    n_seen = 0
    for sr in sf.iterShapeRecords():
        n_seen += 1
        rec = sr.record
        is_canada = rec[idx["Country"]] == "Canada"
        shp = sr.shape
        bx0, by0, bx1, by1 = shp.bbox
        buf = MAX_DIST_DEG
        if bx1 < scan_bbox[0] or bx0 > scan_bbox[2] or by1 < scan_bbox[1] or by0 > scan_bbox[3]:
            continue
        cand = [
            sn
            for sn, (px, py) in pts.items()
            if (bx0 - buf) <= px <= (bx1 + buf) and (by0 - buf) <= py <= (by1 + buf)
        ]
        if not is_canada:
            # Allow only verified cross-border exceptions. / 仅允许已确认的跨境例外。
            cand = [sn for sn in cand if sn in NO_COUNTRY_FILTER_STATIONS]
        if not cand:
            continue
        geom = shp_shape(shp.__geo_interface__)
        prepared = prep(geom)
        area = rec[idx["Lake_area"]] or 0
        meta = (rec[idx["Hylak_id"]], rec[idx["Lake_name"]], rec[idx["Lake_type"]], area)
        for sn in cand:
            px, py = pts[sn]
            pt = Point(px, py)
            prev = resolved.get(sn)
            if prepared.contains(pt):
                # Prefer containment, then the larger polygon. / 优先包含匹配，再选择较大的湖泊面。
                if prev is None or prev["relation"] != "inside" or area > prev["area"]:
                    resolved[sn] = {"relation": "inside", "dist_deg": 0.0, "hylak_id": meta[0],
                                     "name": meta[1], "type": meta[2], "area": meta[3]}
            else:
                if prev is not None and prev["relation"] == "inside":
                    continue  # Preserve an inside match. / 保留已有的包含匹配。
                d = geom.distance(pt)
                if d <= MAX_DIST_DEG and (prev is None or d < prev["dist_deg"]):
                    resolved[sn] = {"relation": "near", "dist_deg": round(d, 4), "hylak_id": meta[0],
                                     "name": meta[1], "type": meta[2], "area": meta[3]}
        if n_seen % 300000 == 0:
            print(f"  ... {n_seen}/{sf.numRecords} scanned, {len(resolved)} station resolutions so far")

    n_type2_excluded = sum(1 for v in resolved.values() if v["type"] not in HYDROLAKES_NATURAL_TYPES)
    print(f"[pool] stations resolved to SOME Canada polygon: {len(resolved)} "
          f"(of which {n_type2_excluded} resolved to a non-natural type, e.g. type=2 reservoirs, "
          f"and are correctly excluded here)")

    on_lake = {
        sn: (v["hylak_id"], v["name"], v["type"], v["area"], v["relation"])
        for sn, v in resolved.items()
        if v["type"] in HYDROLAKES_NATURAL_TYPES
    }
    print(f"[pool] distinct H-stations resolved to a Canada natural-type(1,3) lake polygon: {len(on_lake)}")

    tiny = {sn: v for sn, v in on_lake.items() if v[3] < SMALL_AREA_WARN_KM2}
    if tiny:
        print(f"\n[warn] {len(tiny)} station(s) resolved to a suspiciously small (< {SMALL_AREA_WARN_KM2} km^2) "
              f"polygon -- the signature of a cross-border lake: HydroLAKES tags the real polygon under "
              f"the other country, the Canada-only filter drops it, and resolution falls back to an incidental "
              f"nearby fragment. Check whether these are genuine tiny lakes or border cases; "
              f"if the latter, add the station number to NO_COUNTRY_FILTER_STATIONS and rerun:")
        for sn, v in tiny.items():
            s = stations[sn]
            print(f"    {sn:10s} {s['STATION_NAME']:45s} {s['PROV_TERR_STATE_LOC']:4s} "
                  f"-> hylak_id={v[0]} name='{v[1]}' area={v[3]}km2 relation={v[4]}")

    return on_lake


# Data completeness / 数据完整度
def yearly_month_counts(cur, table, station_ids):
    """Count complete monthly records by station and year.
    按测站和年份统计完整月记录数。
    """
    if not station_ids:
        return {}
    q = f"""
        SELECT STATION_NUMBER, YEAR,
               SUM(CASE WHEN MONTHLY_MEAN IS NOT NULL AND FULL_MONTH=1 THEN 1 ELSE 0 END)
        FROM {table}
        WHERE YEAR BETWEEN ? AND ? AND STATION_NUMBER IN ({",".join("?" * len(station_ids))})
        GROUP BY STATION_NUMBER, YEAR
    """
    out = defaultdict(dict)
    for sn, yr, mf in cur.execute(q, [START_YEAR, END_YEAR] + list(station_ids)):
        out[sn][yr] = mf
    return out


def completeness_summary(year_months, exempt_final_year=False, min_completeness=None):
    """Summarize completeness and fully missing years.
    汇总数据完整度与整年缺测情况。

    The final year exemption and completeness threshold are applied independently.
    末年豁免与完整度阈值相互独立。
    """
    if min_completeness is None:
        min_completeness = WL_COMPLETENESS_MIN
    years = list(range(START_YEAR, END_YEAR + 1))
    check_years = years[:-1] if exempt_final_year else years
    fully_missing = [y for y in check_years if year_months.get(y, 0) == 0]
    denom_years = years
    if exempt_final_year and year_months.get(END_YEAR, 0) == 0:
        denom_years = years[:-1]
    total = sum(year_months.get(y, 0) for y in denom_years)
    expected = len(denom_years) * 12
    pct = total / expected if expected else 0.0
    return {
        "pct": round(pct * 100, 1),
        "missing_years": fully_missing,
        "passes": pct >= min_completeness and not fully_missing,
    }


# Regulated-outlet matching / 受调控出口站匹配
def find_best_regulated_outlet(stations, regulated, q_range, wl_ids):
    """Select the closest drainage-area match for each eligible lake gauge.
    为每个合格湖泊水位站选择集水面积比最接近 1 的出口站。
    """
    regflow_pool = [
        sn
        for sn, v in regulated.items()
        if v == 1
        and q_range.get(sn)
        and q_range[sn][1] is not None
        and q_range[sn][1] >= START_YEAR
        and stations.get(sn, {}).get("DRAINAGE_AREA_GROSS") is not None
    ]
    print(f"[pool] REGULATED=1 Q-type stations w/ drainage area, data reaching >= {START_YEAR}: {len(regflow_pool)}")

    by_lake = {}
    for wl_sn in wl_ids:
        wl = stations[wl_sn]
        wl_da = wl["DRAINAGE_AREA_GROSS"]
        if wl_da is None or wl_da <= 0 or wl["LATITUDE"] is None:
            continue
        best = None
        for rf_sn in regflow_pool:
            if rf_sn == wl_sn:
                continue
            rf = stations[rf_sn]
            if rf["PROV_TERR_STATE_LOC"] != wl["PROV_TERR_STATE_LOC"]:
                continue
            if rf["LATITUDE"] is None:
                continue
            if EXCLUDE_DIRECTION_RE.search((rf["STATION_NAME"] or "").upper()):
                continue
            rf_da = rf["DRAINAGE_AREA_GROSS"]
            if not rf_da:
                continue
            ratio = rf_da / wl_da
            if not (DRAINAGE_RATIO_LOW <= ratio <= DRAINAGE_RATIO_HIGH):
                continue
            dist = haversine_km(wl["LATITUDE"], wl["LONGITUDE"], rf["LATITUDE"], rf["LONGITUDE"])
            if dist > MAX_OUTLET_DIST_KM:
                continue
            cand = {"rf_station": rf_sn, "rf_name": rf["STATION_NAME"], "dist_km": round(dist, 1),
                     "ratio": round(ratio, 3)}
            if best is None or abs(cand["ratio"] - 1) < abs(best["ratio"] - 1):
                best = cand
        if best:
            by_lake[wl_sn] = best

    print(f"[match] WL stations with a drainage-verified (0.85-1.20x) regulated outlet: {len(by_lake)}")
    return by_lake


# Screening pipeline / 筛选流程
def run():
    """Run the screening workflow and write the result CSV.
    运行完整筛选流程并写出结果 CSV。
    """
    poly_path = HYDROLAKES_POLY_SHP or resolve_hydrolakes_path("/data" if HYDAT_DB.startswith("/data") else ".")
    con = sqlite3.connect(HYDAT_DB)
    stations, regulated, reg_history, h_range, q_range = load_hydat_reference(con)
    print(f"[info] total HYDAT stations: {len(stations)}")
    print(f"[info] using HydroLAKES polygons at: {poly_path}")

    on_lake = find_stations_on_natural_lakes(poly_path, stations, h_range)

    cur = con.cursor()
    wl_months = yearly_month_counts(cur, "DLY_LEVELS", list(on_lake.keys()))
    wl_pass = {}
    for sn in on_lake:
        c = completeness_summary(wl_months.get(sn, {}), exempt_final_year=False,
                                 min_completeness=WL_COMPLETENESS_MIN)
        if c["passes"]:
            wl_pass[sn] = c
    print(f"[stage] pass WL completeness >={WL_COMPLETENESS_MIN:.0%} with no fully-missing year: "
          f"{len(wl_pass)} / {len(on_lake)}")

    best_outlet = find_best_regulated_outlet(stations, regulated, q_range, list(wl_pass.keys()))

    rf_ids = [p["rf_station"] for p in best_outlet.values()]
    rf_months = yearly_month_counts(cur, "DLY_FLOWS", rf_ids)

    rows = []
    for wl_sn, outlet in best_outlet.items():
        wl = stations[wl_sn]
        rf_sn = outlet["rf_station"]
        rf = stations[rf_sn]
        rf_c = completeness_summary(rf_months.get(rf_sn, {}), exempt_final_year=True,
                                    min_completeness=RF_COMPLETENESS_MIN)

        wl_toks = tokenize(wl["STATION_NAME"]) - STOP_FOR_NAME_OVERLAP
        rf_toks = tokenize(rf["STATION_NAME"]) - STOP_FOR_NAME_OVERLAP
        name_shared = bool(wl_toks & rf_toks)
        tight_ratio = TIGHT_RATIO_LOW <= outlet["ratio"] <= TIGHT_RATIO_HIGH
        confidence = "high" if (name_shared or tight_ratio) else "needs_manual_review"

        hylak_id, hl_name, hl_type, hl_area, hl_relation = on_lake[wl_sn]
        reg_hist_str = ";".join(f"{a}-{b}:{c}" for a, b, c in reg_history.get(rf_sn, []))

        rows.append({
            "wl_station": wl_sn, "wl_name": wl["STATION_NAME"], "prov": wl["PROV_TERR_STATE_LOC"],
            "wl_completeness_pct": wl_pass[wl_sn]["pct"],
            "rf_station": rf_sn, "rf_name": rf["STATION_NAME"],
            "dist_km": outlet["dist_km"], "drainage_ratio": outlet["ratio"],
            "wl_drainage_km2": wl["DRAINAGE_AREA_GROSS"], "rf_drainage_km2": rf["DRAINAGE_AREA_GROSS"],
            "rf_completeness_pct": rf_c["pct"],
            "rf_missing_years": ";".join(map(str, rf_c["missing_years"])),
            "rf_pass": rf_c["passes"],
            "hylak_id": hylak_id, "hydrolakes_lake_name": hl_name, "hydrolakes_type": hl_type,
            "hydrolakes_area_km2": hl_area, "hydrolakes_relation": hl_relation,
            "name_shared_with_outlet": name_shared, "confidence": confidence,
            "rf_regulation_history": reg_hist_str,
        })

    rows.sort(key=lambda r: (not r["rf_pass"], r["confidence"] != "high", r["wl_name"]))

    if not rows:
        # Return cleanly when no pair passes. / 没有配对通过时正常返回。
        print("[DONE] 没有任何湖泊-出流站配对通过筛选，未写出 CSV。"
              "请检查 START_YEAR/END_YEAR、完整度阈值与集水面积比阈值。")
        con.close()
        return rows

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_final = sum(1 for r in rows if r["rf_pass"])
    n_high = sum(1 for r in rows if r["rf_pass"] and r["confidence"] == "high")
    print(f"\n[DONE] wrote {len(rows)} rows to {OUT_CSV}")
    print(f"[DONE] rows passing all 4 programmatic criteria (rf_pass=True): {n_final}")
    print(f"[DONE]   of which confidence=high: {n_high}")
    print(f"[DONE]   of which confidence=needs_manual_review: {n_final - n_high} "
          f"-- hand-check these before trusting them (some prior runs found "
          f"coincidental drainage-area matches between unrelated lakes here)")

    con.close()
    return rows


# Modal entry point using the ccm-data Volume. / 使用 ccm-data Volume 的 Modal 入口。
try:
    import modal

    app = modal.App("hydat-lake-screening")

    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "pyshp==2.3.1", "shapely==2.0.4"
    )

    data_volume = modal.Volume.from_name("ccm-data", create_if_missing=True)

    @app.function(image=image, volumes={"/data": data_volume}, timeout=1800, memory=8192)
    def screen_lakes():
        """Run screening remotely and persist the CSV.
        在远程运行筛选并保存 CSV。
        """
        rows = run()
        data_volume.commit()
        return rows

    @app.local_entrypoint()
    def main():
        """Launch the remote screening task.
        启动远程筛选任务。
        """
        rows = screen_lakes.remote()
        print(f"Received {len(rows)} rows back from the remote run.")
        print(f"Download the CSV with: modal volume get ccm-data final_lake_screening.csv .")

except ImportError as exc:
    raise RuntimeError("This reproduction package runs lake screening through Modal.") from exc
