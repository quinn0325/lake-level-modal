"""Acquire hydrological and climate data and prepare inputs for CCM.
获取水文与气候数据，并构建 CCM 所需输入。

This is a function library used by the Modal data stage, not a second executable route.
本文件是 Modal 数据阶段调用的函数库，不是另一条可执行路线。

Monthly series retain the complete calendar; missing months remain NaN before
time-delay embedding. Configuration is imported from modal_build_lake_panels.
月序列在延迟嵌入前保留完整日历，缺测月份保留为 NaN；配置来自
modal_build_lake_panels。
"""
import io

import geopandas as gpd
import numpy as np
import pandas as pd
import pyEDM
import requests
from shapely.geometry import Point, box

from modal_build_lake_panels import (
    LAKES, REGULATION_STATIONS, REGULATION_SUBPERIODS,
    START_YEAR, END_YEAR, STATION_COORDS, LAKE_OUTLET_STATION,
)

WSC_BASE_URL = "https://wateroffice.ec.gc.ca/services/monthly_data/csv/inline"

CDS_VARIABLES = {
    "T": "2m_temperature", "P": "total_precipitation", "Evap": "total_evaporation",
    "SWE": "snow_depth_water_equivalent", "R": "runoff",
}


# WSC water levels / WSC 水位

def fetch_wsc_station_level(station_id, start_year=START_YEAR, end_year=END_YEAR, timeout=30):
    """Fetch monthly mean water level for one WSC station.
    获取单个 WSC 站点的月平均水位。"""
    params = {"stations[]": station_id, "parameters[]": "level", "start_year": start_year, "end_year": end_year}
    r = requests.get(WSC_BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), encoding_errors="ignore")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "ID": "station_id", "Year/Année": "year", "Month/Mois": "month",
        "Value/Valeur": "water_level", "Symbol/Symbole": "symbol",
    })
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df[["date", "station_id", "water_level", "symbol"]].sort_values("date").reset_index(drop=True)


def combine_station_water_levels(wide):
    """Combine gauges after centring each by its available-period mean.
    各测站减去自身可用时段均值后合成，并恢复总体平均水位。"""
    wide = wide.sort_index()
    station_means = wide.mean(axis=0, skipna=True)
    combined = (wide - station_means).mean(axis=1, skipna=True) + station_means.mean(skipna=True)
    return combined.where(wide.notna().any(axis=1))


def fetch_lake_water_level(lake_name, start_year=START_YEAR, end_year=END_YEAR):
    """Fetch, align and combine all configured water-level gauges for a lake.
    获取、对齐并合成湖泊配置中的全部水位站。"""
    stations = LAKES[lake_name]["stations"]
    station_dfs = {}
    for sid in stations:
        try:
            station_dfs[sid] = fetch_wsc_station_level(sid, start_year, end_year)
        except Exception as e:
            print(f"  [WARN] station {sid} fetch failed: {e}")
    if not station_dfs:
        raise RuntimeError(f"No WSC data retrieved for {lake_name}")

    wide = pd.concat({sid: df.set_index("date")["water_level"] for sid, df in station_dfs.items()}, axis=1).sort_index()
    full_index = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    wide = wide.reindex(full_index)

    n_stations_used = wide.notna().sum(axis=1)
    coverage_df = pd.DataFrame({
        "n_stations_reporting": n_stations_used,
        "coverage_fraction": n_stations_used / wide.shape[1],
    })
    combined = combine_station_water_levels(wide)
    return combined, wide, coverage_df


# WSC regulation-flow series / WSC 调控流量序列

def fetch_wsc_station_flow(station_id, start_year=START_YEAR, end_year=END_YEAR, timeout=30):
    """Fetch monthly mean discharge for one configured WSC station.
    获取单个已配置 WSC 站点的月平均流量。"""
    params = {"stations[]": station_id, "parameters[]": "flow", "start_year": start_year, "end_year": end_year}
    r = requests.get(WSC_BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), encoding_errors="ignore")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "ID": "station_id", "Year/Année": "year", "Month/Mois": "month",
        "Value/Valeur": "regulated_flow", "Symbol/Symbole": "symbol",
    })
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df[["date", "station_id", "regulated_flow", "symbol"]].sort_values("date").reset_index(drop=True)


def fetch_lake_regulation_flow(lake_name, start_year=START_YEAR, end_year=END_YEAR):
    """Build a lake's regulation-flow series from its configured gauges.
    根据已配置测站构建湖泊调控流量；多站按月求和，未配置时返回 None。"""
    station_ids = REGULATION_STATIONS.get(lake_name)
    if not station_ids:
        return None

    full_index = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    station_series = {}
    for sid in station_ids:
        try:
            df = fetch_wsc_station_flow(sid, start_year, end_year)
            station_series[sid] = df.set_index("date")["regulated_flow"].reindex(full_index)
        except Exception as e:
            print(f"  [WARN] regulation flow station {sid} fetch failed: {e}")
    if not station_series:
        return None

    wide = pd.DataFrame(station_series)
    total_flow = wide.sum(axis=1, skipna=False)

    sub = REGULATION_SUBPERIODS.get(lake_name)
    if sub is not None:
        valid_start, valid_end = sub
        mask = (total_flow.index.year < valid_start) | (total_flow.index.year > valid_end)
        total_flow = total_flow.mask(mask)

    return total_flow


# ERA5-Land retrieval / ERA5-Land 获取

def build_cds_request(area_bbox, years, variables=None):
    """Build a CDS request for monthly ERA5-Land variables.
    构建 ERA5-Land 月度变量的 CDS 请求。"""
    variables = variables or list(CDS_VARIABLES.values())
    return {
        "product_type": ["monthly_averaged_reanalysis"], "variable": variables,
        "year": [str(y) for y in years], "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"], "area": area_bbox, "data_format": "netcdf",
    }


def fetch_era5_land_monthly(area_bbox, years, out_nc_path, variables=None):
    """Download monthly ERA5-Land data to a NetCDF file.
    下载 ERA5-Land 月度数据并保存为 NetCDF。"""
    import cdsapi
    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-land-monthly-means", build_cds_request(area_bbox, years, variables), str(out_nc_path))
    return out_nc_path


def try_fetch_era5_land(lake_name, area_bbox, era5_dir, years=range(START_YEAR, END_YEAR + 1)):
    """Return a cached ERA5-Land file or download it; return None on failure.
    优先返回缓存文件，否则下载；失败时返回 None。"""
    out_path = era5_dir / f"{lake_name}_era5land_monthly.nc"
    if out_path.exists():
        print(f"[CACHED] {lake_name} 的ERA5-Land数据已在Volume上: {out_path}")
        return out_path
    try:
        fetch_era5_land_monthly(area_bbox, years, out_path)
        print(f"[OK] ERA5-Land 下载完成: {out_path}")
        return out_path
    except Exception as e:
        print(f"[SKIP] ERA5-Land 下载未执行 ({lake_name}): {e}")
        return None


# Spatial masks and HydroLAKES matching / 空间掩膜与 HydroLAKES 匹配

def grid_cells_intersecting_polygon(poly, res=0.1, buffer_deg=0.15):
    """Return ERA5-Land grid cells that intersect a polygon.
    返回与多边形相交的 ERA5-Land 网格。"""
    minx, miny, maxx, maxy = poly.bounds
    lon0, lon1 = np.floor((minx - buffer_deg) / res) * res, np.ceil((maxx + buffer_deg) / res) * res
    lat0, lat1 = np.floor((miny - buffer_deg) / res) * res, np.ceil((maxy + buffer_deg) / res) * res
    cells = []
    for lo in np.arange(lon0, lon1, res):
        for la in np.arange(lat0, lat1, res):
            cell = box(lo, la, lo + res, la + res)
            if cell.intersects(poly):
                cells.append({"lon_center": round(lo + res / 2, 3), "lat_center": round(la + res / 2, 3), "geometry": cell})
    return gpd.GeoDataFrame(cells, crs="EPSG:4326")


def get_lake_mask_cells(hydrolakes_gdf, hylak_id, id_col="Hylak_id"):
    """Return lake-mask cells used for temperature and evaporation.
    返回温度和蒸发变量使用的湖面掩膜网格。"""
    poly = hydrolakes_gdf.loc[hydrolakes_gdf[id_col] == hylak_id, "geometry"].iloc[0]
    return grid_cells_intersecting_polygon(poly), poly


def trace_upstream_basin(hydrobasins_gdf, outlet_point, id_col="HYBAS_ID", next_down_col="NEXT_DOWN"):
    """Trace and merge all HydroBASINS polygons upstream of an outlet.
    从出口反向追溯并合并全部上游 HydroBASINS 子流域。"""
    containing = hydrobasins_gdf[hydrobasins_gdf.geometry.contains(outlet_point)]
    if containing.empty:
        raise ValueError("出口点不在任何子流域多边形内，请检查坐标或 shapefile 覆盖范围")
    outlet_id = containing.iloc[0][id_col]
    children = {}
    for _, row in hydrobasins_gdf.iterrows():
        children.setdefault(row[next_down_col], []).append(row[id_col])
    upstream_ids, frontier = set(), [outlet_id]
    while frontier:
        current = frontier.pop()
        if current in upstream_ids:
            continue
        upstream_ids.add(current)
        frontier.extend(children.get(current, []))
    sub = hydrobasins_gdf[hydrobasins_gdf[id_col].isin(upstream_ids)]
    return outlet_id, upstream_ids, sub.geometry.union_all()


def get_basin_mask_cells(hydrobasins_gdf, outlet_lonlat):
    """Return upstream-basin grid cells, geometry and sub-basin identifiers.
    返回上游流域网格、合并后的边界及子流域编号。"""
    outlet_id, upstream_ids, basin_poly = trace_upstream_basin(hydrobasins_gdf, Point(outlet_lonlat))
    return grid_cells_intersecting_polygon(basin_poly), basin_poly, upstream_ids


def extract_masked_series(nc_path, var_name, cells_gdf):
    """Extract a spatially averaged monthly series using paired grid points.
    按成对网格坐标提取空间平均月序列，避免经纬度笛卡尔积。"""
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_da = xr.DataArray(cells_gdf["lat_center"].values, dims="points")
    lon_da = xr.DataArray(cells_gdf["lon_center"].values, dims="points")
    sel = ds[var_name].sel({lat_name: lat_da, lon_name: lon_da}, method="nearest")
    s = sel.mean(dim="points").to_series()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return s


def resolve_lake_area_bbox(lake_name, hydrolakes_gdf, hydrobasins_gdf):
    """Resolve a lake and return its lake/basin masks and ERA5 bounding box.
    匹配湖泊，并返回湖面掩膜、流域掩膜和 ERA5 下载边界框。"""
    ref_lonlat = STATION_COORDS[LAKE_OUTLET_STATION.get(lake_name) or LAKES[lake_name]["stations"][0]]
    # Match the Canadian HydroLAKES record associated with the configured gauge.
    # 匹配配置测站对应的加拿大 HydroLAKES 记录。
    country = "Canada"
    hylak_id, _ = resolve_hylak_id(hydrolakes_gdf, ref_lonlat, country=country)
    if hylak_id is None:
        return None, None, None, None
    pour_row = hydrolakes_gdf.loc[hydrolakes_gdf["Hylak_id"] == hylak_id].iloc[0]
    outlet_lonlat = (pour_row["Pour_long"], pour_row["Pour_lat"])
    lake_cells, _lake_poly = get_lake_mask_cells(hydrolakes_gdf, hylak_id)
    basin_cells, _basin_poly, _upstream_ids = get_basin_mask_cells(hydrobasins_gdf, outlet_lonlat)
    area_bbox = [
        max(lake_cells["lat_center"].max(), basin_cells["lat_center"].max()) + 0.1,
        min(lake_cells["lon_center"].min(), basin_cells["lon_center"].min()) - 0.1,
        min(lake_cells["lat_center"].min(), basin_cells["lat_center"].min()) - 0.1,
        max(lake_cells["lon_center"].max(), basin_cells["lon_center"].max()) + 0.1,
    ]
    return hylak_id, lake_cells, basin_cells, area_bbox


def resolve_hylak_id(hydrolakes_gdf, ref_lonlat, country="Canada",
                     search_buffer_deg=0.5, max_dist_deg=0.05):
    """Resolve Hylak_id by containment, then by nearest distance in degrees.
    先按空间包含关系、再按度数距离匹配 Hylak_id；匹配前不筛选 Lake_type。"""
    lon, lat = ref_lonlat
    ref_pt = Point(ref_lonlat)
    subset = hydrolakes_gdf.cx[lon - search_buffer_deg: lon + search_buffer_deg, lat - search_buffer_deg: lat + search_buffer_deg]
    if country:
        subset = subset[subset["Country"] == country]
    if subset.empty:
        return None, subset
    containing = subset[subset.geometry.contains(ref_pt)]
    if not containing.empty:
        return int(containing.iloc[0]["Hylak_id"]), containing
    subset = subset.copy()
    subset["dist_to_ref"] = subset.geometry.distance(ref_pt)
    subset = subset.sort_values("dist_to_ref")
    best = subset.iloc[0]
    if best["dist_to_ref"] > max_dist_deg:
        return None, subset.head(10)
    return int(best["Hylak_id"]), subset.head(10)


# Unit conversion, panels and embedding / 单位换算、面板与嵌入参数

def _days_in_month(series):
    """Return the number of days represented by each monthly observation.
    返回每个月度观测对应的自然月天数。"""
    idx = pd.DatetimeIndex(series.index)
    return pd.Series(idx.days_in_month, index=series.index, dtype=float)


def convert_era5_units(var_name, series):
    """Convert ERA5-Land units to degC or monthly millimetres.
    将 ERA5-Land 单位转换为摄氏度或月累计毫米；蒸发取正值。"""
    series = series.astype(float)
    if var_name == "T":
        return series - 273.15
    if var_name == "SWE":
        return series * 1000.0
    if var_name in ("P", "R"):
        return series * 1000.0 * _days_in_month(series)
    if var_name == "Evap":
        return -series * 1000.0 * _days_in_month(series)
    raise ValueError(f"unknown variable {var_name}")


def deseasonalize(series, train_end=None):
    """Subtract monthly climatology estimated before train_end.
    减去 train_end 之前估计的逐月气候态；未指定时使用完整序列。"""
    train_series = series if train_end is None else series.iloc[:train_end]
    monthly_clim = train_series.groupby(train_series.index.month).mean()
    month_of_each_point = pd.Series(series.index.month, index=series.index)
    return series - month_of_each_point.map(monthly_clim)


def build_variable_panel(wl_series, era5_vars=None, forecast_horizon=None):
    """Align variables to a complete monthly calendar and deseasonalize them.
    将变量对齐到完整月历并去季节化，返回原始面板和距平面板；缺测保留为 NaN。"""
    era5_vars = era5_vars or {}
    panel = pd.DataFrame({"WL": wl_series})
    for name, s in era5_vars.items():
        panel[name] = s
    panel.index = pd.DatetimeIndex(panel.index)
    panel = panel.sort_index()
    panel = panel.reindex(pd.date_range(panel.index.min(), panel.index.max(), freq="MS"))
    train_end = len(panel) - forecast_horizon if forecast_horizon else None
    return panel, panel.apply(lambda col: deseasonalize(col, train_end=train_end))


def simplex_self_predict_rho(values, E, tau, exclusion_radius=None):
    """Return Simplex self-prediction rho for one E/tau combination.
    计算一个 E/tau 组合的 Simplex 自预测相关系数；失败或有效配对不足时返回 NaN。"""
    n = len(values)
    df = pd.DataFrame({"time": np.arange(n), "v": values})
    full = f"1 {n}"
    exclusion_radius = exclusion_radius if exclusion_radius is not None else max(tau, 1)
    try:
        res = pyEDM.Simplex(dataFrame=df, columns="v", target="v", lib=full, pred=full,
                             E=E, tau=-tau, exclusionRadius=exclusion_radius, embedded=False)
    except Exception:
        return np.nan
    obs, pred = res["Observations"], res["Predictions"]
    mask = obs.notna() & pred.notna()
    if mask.sum() < 2:
        return np.nan
    return np.corrcoef(obs[mask], pred[mask])[0, 1]


def select_E(values, tau, candidate_E=range(2, 11), fallback_E=2):
    """Select E by maximum valid Simplex rho, with a documented fallback.
    按有效 Simplex rho 最大值选择 E；全部失败时使用并标记回退值。"""
    rows = []
    for E in candidate_E:
        rho = simplex_self_predict_rho(values, E=E, tau=tau)
        rows.append({"E": E, "rho": rho})
    curve = pd.DataFrame(rows)
    if curve["rho"].notna().any():
        best_row = curve.loc[curve["rho"].idxmax()]
        return int(best_row["E"]), curve
    curve = curve.copy()
    curve["note"] = "all_candidate_E_failed_insufficient_contiguous_data_using_fallback"
    return fallback_E, curve


def get_embedding_params(values, var_name, verbose=True, tau=None, candidate_E=None):
    """Use a fixed tau and select E from calendar-preserving monthly data.
    使用固定 tau，并从保留完整月历和 NaN 的序列中选择 E。"""
    if tau is None:
        tau = _config_embed_tau()
    if candidate_E is None:
        candidate_E = _config_embed_E_candidates()
    E, e_curve = select_E(values, tau, candidate_E=candidate_E)
    if verbose:
        print(f"[{var_name}] tau={tau} (fixed), E={E}")
    return {"variable": var_name, "tau": tau, "E": E,
            "tau_info": {"method": "fixed", "value": tau}, "E_curve": e_curve}


def _config_embed_tau():
    """Read tau from config, falling back to 1.
    从 config 读取 tau，失败时回退为 1。"""
    try:
        import config
        return config.EMBED_TAU
    except Exception:
        return 1


def _config_embed_E_candidates():
    """Read candidate E values from config, falling back to 2 through 10.
    从 config 读取候选 E，失败时回退为 2 至 10。"""
    try:
        import config
        return config.EMBED_E_CANDIDATES
    except Exception:
        return range(2, 11)
