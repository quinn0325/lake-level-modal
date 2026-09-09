"""
Nationwide, programmatic screening of Canadian HYDAT lake / regulated-outlet
pairs, implementing the method we converged on:

  1. Candidate lake pool is built from HydroLAKES (Canada, Lake_type in
     {1,3}) polygons -- NOT from HYDAT station-name text. This avoids the
     LAKE/LAC language bias and misses nothing named RESERVOIR/POND/
     PONDAGE/BASIN/IMPOUNDMENT/an Indigenous name that happens to sit on a
     real natural lake.
  2. A HYDAT H-type (water level) station counts as "on the lake" if its
     coordinate falls inside the polygon, or within a small near-shore
     buffer of it (gauges are often sited a short distance from the exact
     shoreline vertex data).
  3. Criterion (2): yearly water-level completeness 1994-2024 >= 90% with
     NO fully-missing calendar year.
  4. Criterion (4) / outlet discovery: search ALL REGULATED=1 Q-type
     (flow) stations nationally (not pre-restricted to a region a human
     picked) for the best-matching regulated outlet, using:
       - same province (documented limitation: true cross-border outlets,
         e.g. a station physically gauged from the US side of a shared
         river, are excluded by this -- see NOTES.md this script prints)
       - straight-line distance <= 100 km, NO lower floor (a floor
         mechanically excludes true nearby outlets -- verified empirically
         against Kalamalka/Kootenay/Nicola/Okanagan, whose real distances
         are 0.5-61 km)
       - station name does NOT contain ABOVE/INLET (soft directional
         exclude; kept as a light filter, not the primary discovery
         signal)
       - drainage-area ratio (RF/WL) in [0.85, 1.20] -- THIS is the
         primary "verified as the lake's own outlet" signal (replaces
         station-name-token matching, which is structurally blind to most
         of Canada's largest regulated lakes: Erie, Manitoba, Diefenbaker,
         Athabasca, Split Lake, Southern Indian Lake, Williston, etc. --
         their real outlet gauges are named after the dam/town, not the
         lake)
       - best match per lake = drainage ratio closest to 1.0
     Criterion (4) completeness: yearly flow completeness 1994-2024 >= 90%
     with no fully-missing year, EXCEPT the final year 2024 is exempted
     from the "no gap" check (HYDAT publish delay).
  5. A `confidence` column is set to "high" when EITHER the WL and RF
     station names share a real (non-generic) word, OR the drainage ratio
     is inside a tight [0.95, 1.05] band. Rows scoring "needs_manual_review"
     are NOT auto-rejected -- ~3 of every batch we hand-checked turned out
     to be coincidental drainage-area matches between unrelated water
     bodies (Buffalo Pound Lake -> Wascana Creek; two different Quebec
     lakes both landing on the same neighboring dam's outlet station by
     coincidence). Keep this column and check it by hand before trusting a
     "needs_manual_review" row.

This script does NOT auto-decide the following -- they need your own
judgment pass on the output CSV:
  - Rows flagged confidence="needs_manual_review".
  - Small/near-zero-area HydroLAKES polygon matches next to a dam whose
    name says RESERVOIR/DAM (a possible sign the matched polygon is a
    leftover natural pond fragment next to an artificial reservoir, not
    the reservoir itself -- e.g. we found "Duncan Reservoir at Duncan Dam"
    matching a 0.18 km^2 type=1 polygon; that is almost certainly not
    representative of the actual reservoir).
  - Station clusters that plausibly represent ONE interconnected water
    body reported under multiple names/gauges (e.g. the Montreal-area
    Ottawa/St. Lawrence confluence: Lac Saint-Louis / Lac des
    Deux-Montagnes / the river reach at Pointe-des-Cascades all match
    similar-sized nearby polygons and sometimes the same outlet station).
  - Whether a border-adjacent gauge (PROV_TERR_STATE_LOC coded as a US
    state, e.g. "ME") should count as "within Canada" for your thesis.

INPUTS (see CONFIG below):
  - HYDAT_DB: path to Hydat.sqlite3
  - HYDROLAKES_POLY_SHP: path to HydroLAKES_polys_v10.shp (+ .dbf/.shx
    alongside it) -- the POLYGON layer, not the points layer. Get it from
    https://www.hydrosheds.org/products/hydrolakes (registration required).

OUTPUT:
  - final_lake_screening.csv -- one row per (WL station, best RF match)
    pair that passed WL completeness; rf_pass column tells you whether it
    cleared criterion (4) too. Filter rf_pass==True and confidence=="high"
    for the auto-clean subset; review the rest by hand.
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

# --------------------------------------------------------------------------- CONFIG
HYDAT_DB = "/data/Hydat.sqlite3"
# resolved via glob at runtime (see resolve_hydrolakes_path) to match the
# nested-folder layout modal_build_lake_panels.py already uses on the "ccm-data"
# volume (HydroLAKES_polys_v10_shp/**/*.shp) -- HydroLAKES is very likely
# ALREADY on that volume from your existing pipeline, no re-upload needed.
HYDROLAKES_POLY_SHP = None
OUT_CSV = "/data/final_lake_screening.csv"


def resolve_hydrolakes_path(data_root="/data"):
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
MAX_DIST_DEG = 0.05             # matches data_acquisition_lib.py's resolve_hylak_id max_dist_deg convention
MAX_OUTLET_DIST_KM = 100.0      # upper bound only -- no lower floor
DRAINAGE_RATIO_LOW = 0.85
DRAINAGE_RATIO_HIGH = 1.20
TIGHT_RATIO_LOW = 0.95
TIGHT_RATIO_HIGH = 1.05
HYDROLAKES_NATURAL_TYPES = (1, 3)
SMALL_AREA_WARN_KM2 = 5.0       # flag suspiciously tiny resolved matches for manual review

# Known cross-border-lake blind spot (same issue documented in data_acquisition_lib.py /
# modal_build_lake_panels.py's LAKE_RESOLVE_COUNTRY): HydroLAKES tags a shared lake's
# ENTIRE polygon under a single Country value. If that value isn't "Canada"
# (e.g. Lake Huron -> "United States of America", Hylak_id=8, 59399 km^2),
# a Canada-only filter misses the real polygon and falls back to a tiny,
# unrelated, incidental match nearby. Add station numbers here (with their
# resolved country value) once you've confirmed a SMALL_AREA_WARN row is a
# genuine cross-border case, and the pipeline will re-resolve them ignoring
# the country filter. Left empty by default -- populate from the printed
# "[warn] tiny match" lines after a run.
NO_COUNTRY_FILTER_STATIONS = set()  # e.g. {"02EA014"} for Lake Huron

EXCLUDE_DIRECTION_RE = re.compile(r"\b(ABOVE|INLET)\b", re.I)
STOP_FOR_NAME_OVERLAP = {
    "LAKE", "LAKES", "LAC", "AT", "NEAR", "OF", "THE", "RIVER", "RIVIERE",
    "CREEK", "DU", "AU", "LA", "OUTLET", "BELOW", "ABOVE", "DAM", "BARRAGE",
    "A", "EN", "AVAL", "AMONT", "STATION", "GENERATING",
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def tokenize(name):
    name = re.sub(r"[().,'\-]", " ", (name or "").upper())
    return set(t for t in name.split() if t)


# --------------------------------------------------------------------------- HYDAT loading
def load_hydat_reference(con):
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


# --------------------------------------------------------------------------- Step 1+2: lake pool via HydroLAKES polygons
def find_stations_on_natural_lakes(poly_shp_path, stations, h_range):
    """
    Language- and naming-agnostic candidate pool: ANY HYDAT H-type station
    (regardless of name) with nominal 1994-2024 coverage, resolved to its
    ONE correct HydroLAKES polygon, then filtered to Lake_type in {1,3}.

    IMPORTANT: resolution must consider ALL Canada polygons (any Lake_type),
    not just type-1/3 ones, and pick containment-first / nearest-second --
    mirroring data_acquisition_lib.py's resolve_hylak_id(). An earlier version of this
    script pre-filtered to type in {1,3} before matching, which is wrong:
    if a station's true, correct polygon is a type=2 reservoir (e.g. Tobin
    Lake, Sugar Lake Reservoir, Coquitlam Lake -- all independently verified
    as Lake_type=2 in the existing ccm project's RESERVOIR_LAKES set), that
    reservoir polygon was invisible to the old scan, and the algorithm would
    happily grab an unrelated, tiny, incidental type=1 pond nearby instead
    and misreport the station as sitting on a "natural lake". Resolving
    against ALL types first and checking Lake_type only at the end avoids
    that failure mode entirely.
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

    # station -> dict(relation, dist_deg, hylak_id, name, type, area)
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
            # only let a non-Canada-tagged polygon match stations explicitly
            # opted into NO_COUNTRY_FILTER_STATIONS (known cross-border cases)
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
                # containment always wins; among multiple containing polygons keep the larger one
                if prev is None or prev["relation"] != "inside" or area > prev["area"]:
                    resolved[sn] = {"relation": "inside", "dist_deg": 0.0, "hylak_id": meta[0],
                                     "name": meta[1], "type": meta[2], "area": meta[3]}
            else:
                if prev is not None and prev["relation"] == "inside":
                    continue  # never downgrade an inside match to a near-miss
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


# --------------------------------------------------------------------------- completeness
def yearly_month_counts(cur, table, station_ids):
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
    """年度完整度汇总。

    两个参数各管一件事，不可合并：
      · exempt_final_year —— 检查"整年缺测"时是否跳过 END_YEAR（HYDAT 发布延迟），
        并在末年确实为空时把它从分母中剔除；
      · min_completeness  —— 通过所需的完整度下限。

    原实现只有 exempt_final_year 一个开关，并用它同时选择阈值
    （`RF_COMPLETENESS_MIN if exempt_final_year else WL_COMPLETENESS_MIN`）。
    这在当前调用方式下恰好正确（水位不豁免末年、流量豁免），但两件事本无关联：
    一旦给水位也豁免末年，其阈值会被静默换成 RF 的；反之亦然——不报错、
    不告警，只是筛出的湖泊数悄悄变化。故拆为两个独立参数。
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


# --------------------------------------------------------------------------- Step 4: national regulated-outlet search
def find_best_regulated_outlet(stations, regulated, q_range, wl_ids):
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


# --------------------------------------------------------------------------- main pipeline
def run():
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
        # 没有任何 (WL, RF) 配对通过筛选。直接取 rows[0].keys() 会抛
        # IndexError，且错误信息完全掩盖真实原因，故显式报错。
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


# --------------------------------------------------------------------------- Modal wrapper
#
# Reuses your EXISTING "ccm-data" volume from modal_build_lake_panels.py -- HydroLAKES
# is almost certainly already uploaded there (its own docstring's setup step
# puts it at HydroLAKES_polys_v10_shp/ on that volume). The only new upload
# this script needs is Hydat.sqlite3, which nothing in modal_build_lake_panels.py /
# data_acquisition_lib.py currently references:
#
#   modal volume put ccm-data <local-path> Hydat.sqlite3
#
# (skip the HydroLAKES put entirely if `modal volume ls ccm-data` already
# shows HydroLAKES_polys_v10_shp/ -- re-uploading ~1.1GB for nothing is slow)
#
#   modal run modal_lake_screening.py
#   modal volume get ccm-data final_lake_screening.csv .
try:
    import modal

    app = modal.App("hydat-lake-screening")

    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "pyshp==2.3.1", "shapely==2.0.4"
    )

    data_volume = modal.Volume.from_name("ccm-data", create_if_missing=True)

    @app.function(image=image, volumes={"/data": data_volume}, timeout=1800, memory=8192)
    def screen_lakes():
        rows = run()
        data_volume.commit()
        return rows

    @app.local_entrypoint()
    def main():
        rows = screen_lakes.remote()
        print(f"Received {len(rows)} rows back from the remote run.")
        print(f"Download the CSV with: modal volume get ccm-data final_lake_screening.csv .")

except ImportError as exc:
    raise RuntimeError("This reproduction package runs lake screening through Modal.") from exc
