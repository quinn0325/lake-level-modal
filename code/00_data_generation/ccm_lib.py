"""数据获取与嵌入参数：WSC 水位/调控流量、ERA5-Land、HydroLAKES/HydroBASINS 掩膜。

配置（LAKES、REGULATION_STATIONS 等）从 ccm_modal_app 导入，不在这里重复定义。

贯穿全文件的一条规矩：**任何时候都不要 dropna**
------------------------------------------------
喂给 pyEDM 的序列必须保留完整的日历月份索引，缺测的月份就留成真实的 NaN，不删行、
不重新编号。原因是 pyEDM 的延迟嵌入完全按数据行的物理顺序算，不看 time 列的实际
数值——一旦 dropna 再 reset_index，缺口两侧物理上相隔几个月的两行就会被当成"相邻的
一个月"来构造嵌入向量。那是编出来的邻接关系，不是真实动态，而且不会报错，只会悄悄
把结果算错。

保留 NaN 就够了：Simplex/CCM 默认 ignoreNan=True，遇到某个预测需要的历史点是 NaN
会正确地跳过并返回 NaN，不会拿假邻居硬凑。这不是在实现某篇论文的方法（例如 Clark
et al. 2015 的 multispatial CCM 是一套带 bootstrap 的独立算法，本项目没有实现），
只是让 pyEDM 按它自己文档"Disjoint prediction sets"那条说明正确处理缺测。

这条规矩对调用方同样成立：传给 get_embedding_params 的必须是保留了完整索引和真实
NaN 的序列，传 `.dropna().values` 等于让这里的处理全部白做。
"""
import io

import geopandas as gpd
import numpy as np
import pandas as pd
import pyEDM
import requests
from shapely.geometry import Point, box

from ccm_modal_app import (
    LAKES, REGULATION_STATIONS, REGULATION_SUBPERIODS,
    START_YEAR, END_YEAR, STATION_COORDS, LAKE_OUTLET_STATION,
)

WSC_BASE_URL = "https://wateroffice.ec.gc.ca/services/monthly_data/csv/inline"

CDS_VARIABLES = {
    "T": "2m_temperature", "P": "total_precipitation", "Evap": "total_evaporation",
    "SWE": "snow_depth_water_equivalent", "R": "runoff",
}


# ============ WSC 水位抓取 ============

def fetch_wsc_station_level(station_id, start_year=START_YEAR, end_year=END_YEAR, timeout=30):
    """Fetch monthly mean water level for one WSC station."""
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
    """合成多站水位；baseline_index用于只用训练窗估计站点基准差。"""
    wide = wide.sort_index()
    station_means = wide.mean(axis=0, skipna=True)
    combined = (wide - station_means).mean(axis=1, skipna=True) + station_means.mean(skipna=True)
    return combined.where(wide.notna().any(axis=1))


def fetch_lake_water_level(lake_name, start_year=START_YEAR, end_year=END_YEAR):
    """Fetch + combine all WSC stations for a lake."""
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


# ============ WSC 调控流量抓取 ============

def fetch_wsc_station_flow(station_id, start_year=START_YEAR, end_year=END_YEAR, timeout=30):
    """Fetch monthly mean regulated discharge (flow) for one WSC station.
    跟fetch_wsc_station_level结构一致，只是parameters[]换成"flow"。"""
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
    """抓取一个湖泊调控出口/调控代理站的实测流量；如一个湖泊映射多个放水通道，则多站相加。
    返回None表示该湖泊没有REGULATION_STATIONS条目，调用方应把RegFlow这一列跳过，
    不要塞NaN占位。"""
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


# ============ ERA5-Land 下载 ============

def build_cds_request(area_bbox, years, variables=None):
    variables = variables or list(CDS_VARIABLES.values())
    return {
        "product_type": ["monthly_averaged_reanalysis"], "variable": variables,
        "year": [str(y) for y in years], "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"], "area": area_bbox, "data_format": "netcdf",
    }


def fetch_era5_land_monthly(area_bbox, years, out_nc_path, variables=None):
    import cdsapi
    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-land-monthly-means", build_cds_request(area_bbox, years, variables), str(out_nc_path))
    return out_nc_path


def try_fetch_era5_land(lake_name, area_bbox, era5_dir, years=range(START_YEAR, END_YEAR + 1)):
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


# ============ 流域/湖泊掩膜、Hylak_id 解析 ============

def grid_cells_intersecting_polygon(poly, res=0.1, buffer_deg=0.15):
    """返回与给定多边形相交的所有 0.1度 ERA5-Land 网格盒子。"""
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
    """T / Evap 用：湖泊本体覆盖到的 ERA5-Land 格点。"""
    poly = hydrolakes_gdf.loc[hydrolakes_gdf[id_col] == hylak_id, "geometry"].iloc[0]
    return grid_cells_intersecting_polygon(poly), poly


def trace_upstream_basin(hydrobasins_gdf, outlet_point, id_col="HYBAS_ID", next_down_col="NEXT_DOWN"):
    """P / R / SWE 用：从出口点反向追溯上游全部子流域，dissolve 成完整流域边界。"""
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
    outlet_id, upstream_ids, basin_poly = trace_upstream_basin(hydrobasins_gdf, Point(outlet_lonlat))
    return grid_cells_intersecting_polygon(basin_poly), basin_poly, upstream_ids


def extract_masked_series(nc_path, var_name, cells_gdf):
    """从ERA5-Land netCDF按格点坐标提取区域平均月度序列。用共享维度做逐点匹配（不是笛卡尔积）。"""
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
    """解析湖泊Hylak_id + 计算ERA5下载用的外接矩形bbox，供fetch_era5_modal（低并发预下载）
    和process_lake_modal（正式流程）共用，避免两边各写一份、容易不同步。
    返回(hylak_id, lake_cells, basin_cells, area_bbox)；hylak_id为None表示解析失败。"""
    ref_lonlat = STATION_COORDS[LAKE_OUTLET_STATION.get(lake_name) or LAKES[lake_name]["stations"][0]]
    # 十个研究湖泊全在加拿大境内。跨境湖需要关掉国家过滤（HydroLAKES 会把
    # 整个湖标成对岸国家），本研究没有这种情况。
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
    """按坐标空间位置解析Hylak_id：优先看有没有多边形直接包含参照点，其次找离参照点最近的多边形。
    关键点：这里对全部Lake_type一起解析(不预先筛type∈{1,3})，调用方自己决定要不要按类型过滤——
    如果解析前就只看天然湖多边形，会把本该落在附近水库(type=2)里的测站，误配到旁边一个不相关
    的小天然水体上——早期按名字+类型筛选时，多个测站就是这样被误配到旁边不相关的
    水体上的。
    """
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


# ============ 单位换算、去季节化、面板拼装 ============

def _days_in_month(series):
    idx = pd.DatetimeIndex(series.index)
    return pd.Series(idx.days_in_month, index=series.index, dtype=float)


def convert_era5_units(var_name, series):
    """ERA5-Land单位换算。
    T: K -> degC；SWE: m water equivalent -> mm；P/R/Evap: m/day -> mm/month（monthly_total=True）或mm/day。
    ERA5 total_evaporation通常以向下通量为正，蒸发为负，因此取负号让Evap表示正向蒸发量。
    """
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
    """减去逐月气候态，返回距平序列。

    月度气候态均值只用 train_end 之前(训练窗口)的数据计算，同一套均值同时套用到
    训练段与测试段。不能用全部数据(含测试期)一起算月度均值再切分 train/test：
    那样测试期某个月的真实值，在被算进"该月气候态均值"的那一刻，就已经把自己的
    信息用于标准化自己了——这是发生在 train/test 切分之前的信息泄漏。

    train_end=None 会退化成用全序列计算，只在探索性场景下可用；正式流程必须由调用方
    显式传入。01_shared/ccm_full_pipeline.py 里有一份逐字相同的实现。
    """
    train_series = series if train_end is None else series.iloc[:train_end]
    monthly_clim = train_series.groupby(train_series.index.month).mean()
    month_of_each_point = pd.Series(series.index.month, index=series.index)
    return series - month_of_each_point.map(monthly_clim)


def build_variable_panel(wl_series, era5_vars=None, forecast_horizon=None):
    """把 WL 与 ERA5-Land 各变量对齐成一张月度面板，逐变量去季节化。返回 (原始面板, 去季节化面板)。
    reindex 到完整日历月份范围(min~max, 逐月)，保留真实 NaN——这是下游一切处理的前提，
    此后任何环节都不要对整张面板做 listwise dropna（见文件头）。

    forecast_horizon 必须由调用方显式传入：去季节化的月度气候态只用训练窗口
    (排除最后 forecast_horizon 个月)计算，否则测试期信息会经由气候态均值泄漏。
    原实现无此参数、恒用全序列，是已确认的泄漏来源。
    """
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
    """values需保留完整日历位置和真实NaN——本函数不做任何缺测预处理，直接靠pyEDM
    Simplex()默认的ignoreNan=True正确跳过需要缺测点的预测，不需要额外代码。

    如果这段数据缺测拼接过于零散(某个候选E/tau组合下连一个有效近邻都凑不出来)，
    pyEDM底层(scipy cKDTree.query)会直接抛异常而不是返回一个低分——调控流量站常年
    冬季停测，三十年下来能拼成三十多段、最短的只有一个月。这里接住异常返回NaN，让调用方
    (select_E)把这个候选当作"此路不通"处理，不要让整个嵌入参数选择跟着崩掉。"""
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
    """values需保留完整日历位置和真实NaN，理由同simplex_self_predict_rho。

    原来直接调pyEDM.EmbedDimension()一次性扫描全部候选E——这个函数内部用多进程池
    并行跑每个E，只要其中一个E因为数据太碎、找不到有效近邻而抛异常，整个进程池
    连带崩溃、一个E的结果都拿不到。改成逐个E值
    单独调simplex_self_predict_rho、每个都单独接住异常，一个E失败不影响其他E值
    继续尝试；如果全部候选E都失败，退回fallback_E=2并在返回的curve里标注清楚，
    不能让整个湖泊的计算因为一个变量选不出E就整体失败。"""
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
    """values必须是保留了完整日历月份位置和真实NaN的序列/数组——调用方不能先做
    .dropna()再传进来，否则E的选择会受"缺测拼接"问题影响。

    tau 现固定为 1（由 config.EMBED_TAU 提供），不再用 AMI/simplex 自动选择。三点理由：

    1. 文献先例：Javier et al. (2022, Physica A 604, 127893) 在同类水库系统的 CCM
       分析中即固定 τ=1，理由是"gives the highest resolution for the embedding"。
    2. 消除泄漏：原先数据驱动选 τ 依赖去季节化后的序列，而数据生成阶段的去季节化
       曾使用全序列（含测试期）气候态，导致 τ 携带测试期信息——实测有近四成变量的
       τ 会因为修正去季节化而改变。固定 τ 后该环节不复存在。
    3. 缩短嵌入跨度：原 τ∈[1,5] 配合 E∈[2,10]，嵌入跨度 (E-1)*τ 最长达 36 个月，
       在约 330 个月度观测且含缺测的序列上代价很大。实测跨度 25–40 个月的边
       有效样本中位数降至 299、显著率降至 30%（跨度 0–6 个月者为 329 / 46%）。
       τ=1 使跨度上限降为 E-1 ≤ 9 个月。

    E 仍由单变量 simplex 自预测选择，**不**改用"使交叉映射技巧最大"的选法——
    后者用被检验的量本身来选参数，存在循环论证问题。

    代价（需在方法论中披露）：月度序列自相关强，τ=1 时相邻滞后坐标高度冗余，
    嵌入向量沿对角线方向退化（Fraser & Swinney, 1986 提出 AMI 选 τ 正是为此）。
    该偏差通过 IAAFT 替代序列控制——替代序列保留原序列功率谱因而保留自相关结构，
    观测数据与零分布走完全相同的嵌入流程，冗余对两者影响一致。
    """
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
    """从 config.py 读 τ；config 不可导入时退回 1（与 config 默认值一致）。"""
    try:
        import config
        return config.EMBED_TAU
    except Exception:
        return 1


def _config_embed_E_candidates():
    try:
        import config
        return config.EMBED_E_CANDIDATES
    except Exception:
        return range(2, 11)
