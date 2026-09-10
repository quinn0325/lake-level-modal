"""Build monthly lake panels from WSC and ERA5-Land data on Modal.
"""
import modal

app = modal.App("ccm-lakes")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgdal-dev", "gdal-bin")
    .pip_install(
        # Pin compatible serialization versions.
        "pandas==2.2.2", "numpy==1.26.4",
        "geopandas", "shapely", "requests",
        "xarray", "netcdf4", "cdsapi",
    )
    .add_local_python_source("modal_build_lake_panels")  # Share configuration.
    .add_local_python_source("data_acquisition_lib")
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=True)
DATA_ROOT = "/data"

START_YEAR, END_YEAR = 1994, 2024
# Bump this version after panel-building changes to invalidate checkpoints.
PIPELINE_VERSION = "reproduction_v1"
# Fixed holdout for the approximate 9:1 split.
FORECAST_HORIZON = 37

# Fixed study lakes and water-level gauges.
LAKES = {
    "Lake_of_the_Woods": {"stations": ["05PD029", "05PD008", "05PE014", "05PD011"]},
    "Rainy_Lake": {"stations": ["05PB024", "05PB007"]},
    "Okanagan_Lake": {"stations": ["08NM083"]},
    "Vaseux_Lake": {"stations": ["08NM243"]},
    "Kalamalka_Lake": {"stations": ["08NM143"]},
    "Skaha_Lake": {"stations": ["08NM084"]},
    "Split_Lake": {"stations": ["05UF003"]},
    "Playgreen_Lake": {"stations": ["05UB005"]},
    "Kiskitto_Lake": {"stations": ["05UB013"]},
    "Sipiwesk_Lake": {"stations": ["05UD006"]},
}

# Reference coordinates used to match each study lake to HydroLAKES.
LAKE_REFERENCE_COORDS = {
    "Lake_of_the_Woods": (-94.5537185668945, 49.762939453125),       # WSC 05PE014
    "Rainy_Lake": (-93.3206787109375, 48.6491203308106),            # WSC 05PB007
    "Okanagan_Lake": (-119.499656677246, 49.8861312866211),         # WSC 08NM083
    "Vaseux_Lake": (-119.525512695313, 49.2731895446777),           # WSC 08NM243
    "Kalamalka_Lake": (-119.274421691895, 50.2299194335938),        # WSC 08NM143
    "Skaha_Lake": (-119.57508087158205, 49.42641830444336),         # WSC 08NM084
    "Split_Lake": (-96.08728790283205, 56.24385833740234),          # WSC 05UF003
    "Playgreen_Lake": (-97.9749984741211, 53.90277862548828),       # WSC 05UB005
    "Kiskitto_Lake": (-98.4383316040039, 54.30305862426758),        # WSC 05UB013
    "Sipiwesk_Lake": (-97.5, 55.09444046020508),                    # WSC 05UD006
}

# Regulation-flow gauges.
REGULATION_STATIONS = {
    "Lake_of_the_Woods":    ["05PE011", "05PE006"],  # Sum both outlets.
    "Rainy_Lake":           ["05PC019"],
    "Okanagan_Lake":        ["08NM050"],
    "Vaseux_Lake":          ["08NM247"],             # Available for 2012–2024.
    "Kalamalka_Lake":       ["08NM065"],             # Outlet gauge.
    "Skaha_Lake":           ["08NM002"],             # Matched drainage area.
    "Split_Lake":           ["05UF006"],             # Downstream Kettle proxy.
    "Playgreen_Lake":       ["05UB009"],             # Shared Jenpeg control.
    "Kiskitto_Lake":        ["05UB009"],             # Shared Jenpeg control.
    "Sipiwesk_Lake":        ["05UE005"],             # Downstream Kelsey proxy.
}
# Valid regulation-flow periods.
REGULATION_SUBPERIODS = {
    "Vaseux_Lake": (2012, 2024),  # Records begin in 2012.
}


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    secrets=[modal.Secret.from_name("cds-api")],
    timeout=3600,
    max_containers=4,  # Limit CDS concurrency.
)
def fetch_era5_modal(lake_name: str) -> dict:
    """Download and cache ERA5-Land with limited concurrency and retry."""
    import os
    import glob
    import time
    from pathlib import Path

    import geopandas as gpd

    cds_rc = Path.home() / ".cdsapirc"
    cds_rc.write_text(f"url: {os.environ['CDSAPI_URL']}\nkey: {os.environ['CDSAPI_KEY']}\n")

    ERA5_DIR = Path(DATA_ROOT) / "era5_downloads"
    ERA5_DIR.mkdir(exist_ok=True, parents=True)

    from data_acquisition_lib import resolve_lake_area_bbox, fetch_era5_land_monthly

    hydrolakes_matches = glob.glob(str(Path(DATA_ROOT) / "HydroLAKES_polys_v10_shp" / "**" / "*.shp"), recursive=True)
    hydrobasins_matches = glob.glob(str(Path(DATA_ROOT) / "hybas_na_lev12_v1c" / "*.shp"))
    hydrolakes_gdf = gpd.read_file(hydrolakes_matches[0])
    hydrobasins_gdf = gpd.read_file(hydrobasins_matches[0])

    # Isolate basin-resolution failures by lake.
    try:
        hylak_id, _, _, area_bbox = resolve_lake_area_bbox(lake_name, hydrolakes_gdf, hydrobasins_gdf)
    except Exception as e:
        return {"lake_name": lake_name, "status": f"FAILED_exception_in_resolve_lake_area_bbox: {type(e).__name__}: {e}"}
    if hylak_id is None:
        return {"lake_name": lake_name, "status": "FAILED_hylak_id_not_resolved"}

    out_path = ERA5_DIR / f"{lake_name}_era5land_monthly.nc"
    if out_path.exists():
        print(f"[CACHED] ERA5-Land data for {lake_name}: {out_path}")
        return {"lake_name": lake_name, "status": "OK"}

    last_err = None
    for attempt in range(1, 5):
        try:
            fetch_era5_land_monthly(area_bbox, range(START_YEAR, END_YEAR + 1), out_path)
            print(f"[OK] ERA5-Land download completed: {out_path}")
            volume.commit()
            return {"lake_name": lake_name, "status": "OK"}
        except Exception as e:
            last_err = e
            wait_s = 30 * attempt
            print(f"  [RETRY {attempt}/4] CDS request failed for {lake_name}: {e}")
            if attempt < 4:
                time.sleep(wait_s)

    print(f"[FAILED] ERA5-Land download failed after four attempts for {lake_name}: {last_err}")
    return {"lake_name": lake_name, "status": f"FAILED_era5_download: {last_err}"}


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    secrets=[modal.Secret.from_name("cds-api")],
    timeout=7200,
    max_containers=10,  # Process cached lakes in parallel.
    retries=1,
)
def process_lake_modal(lake_name: str) -> dict:
    """Build and save one lake's water-level and predictor data."""
    import os
    import glob
    import hashlib
    import json
    import pickle
    import zipfile
    from pathlib import Path

    import geopandas as gpd
    import pandas as pd

    # Write Modal credentials in CDS API format for a fallback download.
    cds_rc = Path.home() / ".cdsapirc"
    cds_rc.write_text(f"url: {os.environ['CDSAPI_URL']}\nkey: {os.environ['CDSAPI_KEY']}\n")

    RESULTS_DIR = Path(DATA_ROOT) / "lake_results"
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    ERA5_DIR = Path(DATA_ROOT) / "era5_downloads"
    ERA5_DIR.mkdir(exist_ok=True, parents=True)

    # Fingerprint the lake-specific configuration.
    config_fingerprint = hashlib.sha256(json.dumps({
        "pipeline_version": PIPELINE_VERSION,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "forecast_horizon": FORECAST_HORIZON,
        "stations": LAKES.get(lake_name, {}).get("stations"),
        "reference_coordinates": LAKE_REFERENCE_COORDS.get(lake_name),
        "regulation_stations": REGULATION_STATIONS.get(lake_name),
        "regulation_subperiod": REGULATION_SUBPERIODS.get(lake_name),
    }, sort_keys=True, default=str).encode()).hexdigest()[:16]

    checkpoint_path = RESULTS_DIR / f"{lake_name}_result.pkl"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("config_fingerprint") == config_fingerprint and cached.get("status") == "OK":
                return cached
        except Exception as e:
            print(f"  [WARN] Unreadable checkpoint; recomputing {lake_name}: {e}")

    # Load lake and basin shapefiles from the Volume.
    hydrolakes_matches = glob.glob(str(Path(DATA_ROOT) / "HydroLAKES_polys_v10_shp" / "**" / "*.shp"), recursive=True)
    hydrobasins_matches = glob.glob(str(Path(DATA_ROOT) / "hybas_na_lev12_v1c" / "*.shp"))
    hydrolakes_gdf = gpd.read_file(hydrolakes_matches[0])
    hydrobasins_gdf = gpd.read_file(hydrobasins_matches[0])

    from data_acquisition_lib import fetch_lake_water_level, fetch_lake_regulation_flow, resolve_lake_area_bbox, \
        try_fetch_era5_land, extract_masked_series, \
        convert_era5_units

    result = {"lake_name": lake_name, "status": "started", "pipeline_version": PIPELINE_VERSION,
              "config_fingerprint": config_fingerprint}
    try:
        wide_wl, coverage_mean = fetch_lake_water_level(lake_name)
        result["coverage_mean"] = coverage_mean
        result["wide_wl"] = wide_wl

        regflow_series = fetch_lake_regulation_flow(lake_name)
        result["has_regflow"] = regflow_series is not None

        hylak_id, lake_cells, basin_cells, area_bbox = resolve_lake_area_bbox(lake_name, hydrolakes_gdf, hydrobasins_gdf)
        if hylak_id is None:
            result["status"] = "FAILED_hylak_id_not_resolved"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result
        result["hylak_id"] = hylak_id

        # Reuse the ERA5-Land cache created by fetch_only.
        nc_path = try_fetch_era5_land(lake_name, area_bbox, era5_dir=ERA5_DIR)
        if nc_path is None:
            result["status"] = "FAILED_era5_download"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result

        nc_path = Path(nc_path)
        with nc_path.open("rb") as f:
            is_zip = f.read(4) == b"PK\x03\x04"
        if is_zip:
            extract_dir = nc_path.parent / (nc_path.stem + "_extracted")
            if not extract_dir.exists():
                with zipfile.ZipFile(nc_path) as zf:
                    zf.extractall(extract_dir)
            nc_path = sorted(extract_dir.glob("*.nc"))[0]

        real_predictors = pd.DataFrame({
            "T": convert_era5_units("T", extract_masked_series(nc_path, "t2m", lake_cells)),
            "Evap": convert_era5_units("Evap", extract_masked_series(nc_path, "e", lake_cells)),
            "P": convert_era5_units("P", extract_masked_series(nc_path, "tp", basin_cells)),
            "R": convert_era5_units("R", extract_masked_series(nc_path, "ro", basin_cells)),
            "SWE": convert_era5_units("SWE", extract_masked_series(nc_path, "sd", basin_cells)),
        })
        if regflow_series is not None:
            real_predictors["RegFlow"] = regflow_series.reindex(real_predictors.index)
        result["real_predictors"] = real_predictors

        # Check water-level availability without listwise deletion.
        water_level_available = wide_wl.notna().any(axis=1)
        if int(water_level_available.sum()) < 60:
            result["status"] = "FAILED_too_few_water_level_months"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result

        # Reserve the final months for forecasting.
        if int(water_level_available.iloc[:-FORECAST_HORIZON].sum()) < 60:
            result["status"] = "FAILED_too_few_training_rows_for_ccm"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result

        result["status"] = "OK"
    except Exception as e:
        import traceback
        result["status"] = f"FAILED_exception: {type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"])

    with open(checkpoint_path, "wb") as f:
        pickle.dump(result, f)
    volume.commit()
    return result


@app.local_entrypoint()
def fetch_only(lakes: str = ""):
    """Run stage 1: download and cache ERA5-Land."""
    target_lakes = lakes.split(",") if lakes else list(LAKES.keys())
    print(f"Downloading ERA5-Land for {len(target_lakes)} lake(s): {target_lakes}")
    era5_results = list(fetch_era5_modal.map(target_lakes))
    print("\n=== ERA5-Land download stage completed ===")
    for r in era5_results:
        print(f"  {r['lake_name']}: {r['status']}")
    failed = [r["lake_name"] for r in era5_results if r["status"] != "OK"]
    if failed:
        print(f"\n[WARN] ERA5-Land download failed for {len(failed)} lake(s): {failed}")
        print("Rerun the failed lakes before starting process_only.")
    else:
        print("\nAll ERA5-Land files are ready; process_only can now be run.")


@app.local_entrypoint()
def process_only(lakes: str = ""):
    """Run stage 2: build lake panels from the cached inputs."""
    target_lakes = lakes.split(",") if lakes else list(LAKES.keys())
    print(f"Building panels for {len(target_lakes)} lake(s): {target_lakes}")
    results = list(process_lake_modal.map(target_lakes))
    print("\n=== Panel-building stage completed ===")
    for r in results:
        print(f"  {r['lake_name']}: {r['status']}")
