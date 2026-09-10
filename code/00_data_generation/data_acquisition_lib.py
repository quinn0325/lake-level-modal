"""Acquire hydrological and climate data and prepare inputs for CCM.

This function library is used by the Modal data stage; it is not a separate
executable route. Monthly series retain the complete calendar, with missing
months represented as NaN before time-delay embedding. Configuration is
imported from ``modal_build_lake_panels``.
"""
import io

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, box

from modal_build_lake_panels import (
    LAKES, REGULATION_STATIONS, REGULATION_SUBPERIODS,
    START_YEAR, END_YEAR, LAKE_REFERENCE_COORDS,
)

WSC_BASE_URL = "https://wateroffice.ec.gc.ca/services/monthly_data/csv/inline"

CDS_VARIABLES = {
    "T": "2m_temperature", "P": "total_precipitation", "Evap": "total_evaporation",
    "SWE": "snow_depth_water_equivalent", "R": "runoff",
}


# WSC water levels

def fetch_wsc_station_level(station_id, start_year=START_YEAR, end_year=END_YEAR, timeout=30):
    """Fetch monthly mean water levels for one WSC station."""
    params = {"stations[]": station_id, "parameters[]": "level", "start_year": start_year, "end_year": end_year}
    r = requests.get(WSC_BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), encoding_errors="ignore")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Year/Année": "year", "Month/Mois": "month",
        "Value/Valeur": "water_level",
    })
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df[["date", "water_level"]].sort_values("date").reset_index(drop=True)


def fetch_lake_water_level(lake_name, start_year=START_YEAR, end_year=END_YEAR):
    """Fetch and align all configured water-level gauges for a lake."""
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

    coverage_mean = float(wide.notna().mean(axis=1).mean())
    return wide, coverage_mean


# WSC regulation-flow series

def fetch_wsc_station_flow(station_id, start_year=START_YEAR, end_year=END_YEAR, timeout=30):
    """Fetch monthly mean discharge for one configured WSC station."""
    params = {"stations[]": station_id, "parameters[]": "flow", "start_year": start_year, "end_year": end_year}
    r = requests.get(WSC_BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), encoding_errors="ignore")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Year/Année": "year", "Month/Mois": "month",
        "Value/Valeur": "regulated_flow",
    })
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df[["date", "regulated_flow"]].sort_values("date").reset_index(drop=True)


def fetch_lake_regulation_flow(lake_name, start_year=START_YEAR, end_year=END_YEAR):
    """Build a lake's regulation-flow series from its configured gauges.

    Multiple gauges are summed by month. Return None when no gauges are
    configured or no gauge data can be retrieved.
    """
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


# ERA5-Land retrieval

def build_cds_request(area_bbox, years, variables=None):
    """Build a CDS request for monthly ERA5-Land variables."""
    variables = variables or list(CDS_VARIABLES.values())
    return {
        "product_type": ["monthly_averaged_reanalysis"], "variable": variables,
        "year": [str(y) for y in years], "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"], "area": area_bbox, "data_format": "netcdf",
    }


def fetch_era5_land_monthly(area_bbox, years, out_nc_path, variables=None):
    """Download monthly ERA5-Land data to a NetCDF file."""
    import cdsapi
    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-land-monthly-means", build_cds_request(area_bbox, years, variables), str(out_nc_path))
    return out_nc_path


def try_fetch_era5_land(lake_name, area_bbox, era5_dir, years=range(START_YEAR, END_YEAR + 1)):
    """Return a cached ERA5-Land file or download it; return None on failure."""
    out_path = era5_dir / f"{lake_name}_era5land_monthly.nc"
    if out_path.exists():
        print(f"[CACHED] ERA5-Land data for {lake_name} already exists in the Volume: {out_path}")
        return out_path
    try:
        fetch_era5_land_monthly(area_bbox, years, out_path)
        print(f"[OK] ERA5-Land download completed: {out_path}")
        return out_path
    except Exception as e:
        print(f"[SKIP] ERA5-Land download failed for {lake_name}: {e}")
        return None


# Spatial masks and HydroLAKES matching

def grid_cells_intersecting_polygon(poly, res=0.1, buffer_deg=0.15):
    """Return ERA5-Land grid cells that intersect a polygon."""
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
    """Return lake-mask cells used for temperature and evaporation."""
    poly = hydrolakes_gdf.loc[hydrolakes_gdf[id_col] == hylak_id, "geometry"].iloc[0]
    return grid_cells_intersecting_polygon(poly)


def trace_upstream_basin(hydrobasins_gdf, outlet_point, id_col="HYBAS_ID", next_down_col="NEXT_DOWN"):
    """Trace and merge all HydroBASINS polygons upstream of an outlet."""
    containing = hydrobasins_gdf[hydrobasins_gdf.geometry.contains(outlet_point)]
    if containing.empty:
        raise ValueError(
            "The outlet point is not inside any sub-basin polygon; "
            "check the coordinates and shapefile coverage."
        )
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
    return sub.geometry.union_all()


def get_basin_mask_cells(hydrobasins_gdf, outlet_lonlat):
    """Return ERA5-Land grid cells for the upstream basin."""
    basin_poly = trace_upstream_basin(hydrobasins_gdf, Point(outlet_lonlat))
    return grid_cells_intersecting_polygon(basin_poly)


def extract_masked_series(nc_path, var_name, cells_gdf):
    """Extract a spatial mean from paired grid points without a coordinate product."""
    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lat_da = xr.DataArray(cells_gdf["lat_center"].values, dims="points")
        lon_da = xr.DataArray(cells_gdf["lon_center"].values, dims="points")
        sel = ds[var_name].sel({lat_name: lat_da, lon_name: lon_da}, method="nearest")
        s = sel.mean(dim="points").to_series()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return s


def resolve_lake_area_bbox(lake_name, hydrolakes_gdf, hydrobasins_gdf):
    """Resolve a lake and return its lake/basin masks and ERA5 bounding box."""
    ref_lonlat = LAKE_REFERENCE_COORDS[lake_name]
    # Match the Canadian HydroLAKES record associated with the configured gauge.
    hylak_id = resolve_hylak_id(hydrolakes_gdf, ref_lonlat, country="Canada")
    if hylak_id is None:
        return None, None, None, None
    pour_row = hydrolakes_gdf.loc[hydrolakes_gdf["Hylak_id"] == hylak_id].iloc[0]
    outlet_lonlat = (pour_row["Pour_long"], pour_row["Pour_lat"])
    lake_cells = get_lake_mask_cells(hydrolakes_gdf, hylak_id)
    basin_cells = get_basin_mask_cells(hydrobasins_gdf, outlet_lonlat)
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

    Lake_type is not filtered before matching.
    """
    lon, lat = ref_lonlat
    ref_pt = Point(ref_lonlat)
    subset = hydrolakes_gdf.cx[lon - search_buffer_deg: lon + search_buffer_deg, lat - search_buffer_deg: lat + search_buffer_deg]
    if country:
        subset = subset[subset["Country"] == country]
    if subset.empty:
        return None
    containing = subset[subset.geometry.contains(ref_pt)]
    if not containing.empty:
        return int(containing.iloc[0]["Hylak_id"])
    subset = subset.copy()
    subset["dist_to_ref"] = subset.geometry.distance(ref_pt)
    subset = subset.sort_values("dist_to_ref")
    best = subset.iloc[0]
    if best["dist_to_ref"] > max_dist_deg:
        return None
    return int(best["Hylak_id"])


# Unit conversion and embedding

def _days_in_month(series):
    """Return the number of days represented by each monthly observation."""
    idx = pd.DatetimeIndex(series.index)
    return pd.Series(idx.days_in_month, index=series.index, dtype=float)


def convert_era5_units(var_name, series):
    """Convert ERA5-Land units to degrees Celsius or monthly millimetres.

    Evaporation is converted to a positive magnitude.
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


def simplex_self_predict_rho(values, E, tau, exclusion_radius=None):
    """Return Simplex self-prediction rho for one E/tau combination.

    Return NaN if Simplex fails or produces fewer than two valid pairs.
    """
    import pyEDM

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
    """Select E by maximum valid Simplex rho, with a marked fallback."""
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
