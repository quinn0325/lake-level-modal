"""Build monthly lake panels from WSC and ERA5-Land data on Modal.
在 Modal 上利用 WSC 与 ERA5-Land 数据构建月度湖泊面板。

Run ``fetch_only`` before ``process_only``; outputs are stored in ``ccm-data/lake_results``.
先运行 ``fetch_only``，再运行 ``process_only``；结果存入 ``ccm-data/lake_results``。

Data-acquisition and panel-building functions are implemented in
``data_acquisition_lib.py``; this file defines the Modal resources and execution steps.
数据获取与面板构建函数位于 ``data_acquisition_lib.py``；本文件定义 Modal 资源与执行步骤。
"""
import modal

app = modal.App("ccm-lakes")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgdal-dev", "gdal-bin")
    .pip_install(
        # Pin compatible serialization versions. / 固定兼容的序列化版本。
        "pandas==2.2.2", "numpy==1.26.4",
        "geopandas", "shapely", "requests",
        "pmdarima", "statsmodels", "networkx", "xarray", "netcdf4",
        "cdsapi", "matplotlib", "pyEDM", "scipy", "xgboost",
    )
    .add_local_python_source("modal_build_lake_panels")  # Share configuration. / 共享配置。
    .add_local_python_source("data_acquisition_lib")  # Include acquisition helpers. / 包含数据获取函数。
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=True)
DATA_ROOT = "/data"

START_YEAR, END_YEAR = 1994, 2024
# Bump after panel-building changes to invalidate checkpoints. / 面板构建逻辑变更后升级版本，使检查点失效。
PIPELINE_VERSION = "2026-08-09_v14_forecast_horizon_37mo_revert"
# Fixed holdout for the 9:1 split. / 9:1 切分使用固定的测试期。
FORECAST_HORIZON = 37

# Study lakes and water-level gauges. / 研究湖泊与水位站。
LAKES = {
    "Lake_of_the_Woods":    dict(group="Boreal Shield", rank=1, stations=["05PD029", "05PD008", "05PE014", "05PD011"]),
    "Rainy_Lake":           dict(group="Boreal Shield", rank=2, stations=["05PB024", "05PB007"]),
    "Okanagan_Lake":        dict(group="Montane Cordillera", rank=1, stations=["08NM083"]),
    "Vaseux_Lake":          dict(group="Montane Cordillera", rank=2, stations=["08NM243"]),
    "Kalamalka_Lake":       dict(group="Montane Cordillera", rank=4, stations=["08NM143"]),
    "Skaha_Lake":           dict(group="New_2026", rank=7, stations=["08NM084"]),
    "Split_Lake":           dict(group="New_2026", rank=8, stations=["05UF003"]),
    # Nelson River reservoir chain. / Nelson River 水库链。
    "Playgreen_Lake":       dict(group="Nelson_River_Chain", rank=1, stations=["05UB005"]),
    "Kiskitto_Lake":        dict(group="Nelson_River_Chain", rank=2, stations=["05UB013"]),
    "Sipiwesk_Lake":        dict(group="Nelson_River_Chain", rank=3, stations=["05UD006"]),
}

STATION_COORDS = {
    "05PB007": (-93.3206787109375, 48.6491203308106),   # Rainy Lake, Fort Frances / Rainy Lake，Fort Frances
    "05PB024": (-92.9583587646484, 48.7004699707031),   # Rainy Lake, Bear Pass / Rainy Lake，Bear Pass
    "05PD008": (-94.2834930419922, 49.1328010559082),   # Hanson Bay / Hanson Bay
    "05PD011": (-94.8102493286133, 49.7126388549805),   # Clearwater Bay / Clearwater Bay
    "05PD029": (-94.8531112670898, 49.3284683227539),   # Cyclone Island / Cyclone Island
    "05PE014": (-94.5537185668945, 49.762939453125),    # Keewatin outlet / Keewatin 出口
    "08NM083": (-119.499656677246, 49.8861312866211),   # Okanagan Lake, Kelowna / Okanagan Lake，Kelowna
    "08NM143": (-119.274421691895, 50.2299194335938),   # Kalamalka Lake, Vernon / Kalamalka Lake，Vernon
    "08NM243": (-119.525512695313, 49.2731895446777),   # Vaseux Lake outlet / Vaseux Lake 出口
    "05PE011": (-94.52446746826172, 49.771968841552734),  # Norman Dam flow / Norman Dam 流量
    "05PE006": (-94.50308227539062, 49.77252960205078),   # Kenora Powerhouse flow / Kenora Powerhouse 流量
    "05PC019": (-93.4034423828125, 48.60852813720703),    # Rainy River flow / Rainy River 流量
    "08NM050": (-119.6153564453125, 49.49851989746094),   # Okanagan River flow / Okanagan River 流量
    "08NM247": (-119.5280303955078, 49.25683975219727),   # McIntyre Dam flow / McIntyre Dam 流量
    "08NM065": (-119.2668914794922, 50.238468170166016),  # Kalamalka outlet flow / Kalamalka 出口流量
    # Coordinates from HYDAT STATIONS. / 坐标来自 HYDAT STATIONS。
    "08NM084": (-119.57508087158205, 49.42641830444336),  # Skaha Lake / Skaha Lake
    "08NM002": (-119.58040618896484, 49.34204864501953),  # Okanagan Falls flow / Okanagan Falls 流量
    "05UF003": (-96.08728790283205, 56.24385833740234),   # Split Lake / Split Lake
    "05UF006": (-94.6338882446289, 56.380279541015625),   # Kettle station flow / Kettle 站流量
    # Coordinates from MSC GeoMet. / 坐标来自 MSC GeoMet。
    "05UB005": (-97.9749984741211, 53.90277862548828),    # Playgreen Lake / Playgreen Lake
    "05UB013": (-98.4383316040039, 54.30305862426758),    # Kiskitto Lake / Kiskitto Lake
    "05UB009": (-98.04805755615234, 54.4980583190918),    # Shared Jenpeg flow / 共用 Jenpeg 流量
    "05UD006": (-97.5, 55.09444046020508),                # Sipiwesk Lake / Sipiwesk Lake
    "05UE005": (-96.5250015258789, 56.03889083862305),    # Kelsey flow proxy / Kelsey 流量代理
}

# Outlet references for basin tracing. / 流域追溯使用的出口参照站。
LAKE_OUTLET_STATION = {
    "Lake_of_the_Woods": "05PE014",
    "Rainy_Lake": "05PB007",
}

# Regulation-flow gauges; absent lakes omit RegFlow. / 调控流量站；未列出的湖泊不添加 RegFlow。
REGULATION_STATIONS = {
    "Lake_of_the_Woods":    ["05PE011", "05PE006"],  # Sum both outlets. / 两个出口流量相加。
    "Rainy_Lake":           ["05PC019"],
    "Okanagan_Lake":        ["08NM050"],
    "Vaseux_Lake":          ["08NM247"],             # Available for 2012–2024. / 仅 2012–2024 年可用。
    "Kalamalka_Lake":       ["08NM065"],             # Outlet gauge. / 出口站。
    "Skaha_Lake":           ["08NM002"],              # Matched drainage area. / 集水面积匹配。
    "Split_Lake":           ["05UF006"],              # Downstream Kettle proxy. / 下游 Kettle 代理站。
    "Playgreen_Lake":       ["05UB009"],   # Shared Jenpeg control. / 共用 Jenpeg 调控站。
    "Kiskitto_Lake":        ["05UB009"],   # Shared Jenpeg control. / 共用 Jenpeg 调控站。
    "Sipiwesk_Lake":        ["05UE005"],   # Downstream Kelsey proxy. / 下游 Kelsey 代理站。
}
# Valid regulation-flow periods. / 调控流量的有效时段。
REGULATION_SUBPERIODS = {
    "Vaseux_Lake": (2012, 2024),   # Records begin in 2012. / 记录始于 2012 年。
}


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    secrets=[modal.Secret.from_name("cds-api")],
    timeout=3600,  # Allow downloads and retries. / 为下载与重试预留时间。
    max_containers=4,  # Limit CDS concurrency. / 限制 CDS 并发数。
)
def fetch_era5_modal(lake_name: str) -> dict:
    """Download and cache ERA5-Land with limited concurrency and retry.
    以受限并发和重试机制下载并缓存 ERA5-Land。
    """
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

    # Isolate basin-resolution failures by lake. / 按湖隔离流域解析失败。
    try:
        hylak_id, lake_cells, basin_cells, area_bbox = resolve_lake_area_bbox(lake_name, hydrolakes_gdf, hydrobasins_gdf)
    except Exception as e:
        return {"lake_name": lake_name, "status": f"FAILED_exception_in_resolve_lake_area_bbox: {type(e).__name__}: {e}"}
    if hylak_id is None:
        return {"lake_name": lake_name, "status": "FAILED_hylak_id_not_resolved"}

    out_path = ERA5_DIR / f"{lake_name}_era5land_monthly.nc"
    if out_path.exists():
        print(f"[CACHED] {lake_name} 的ERA5-Land数据已在Volume上: {out_path}")
        return {"lake_name": lake_name, "status": "OK"}

    last_err = None
    for attempt in range(1, 5):
        try:
            fetch_era5_land_monthly(area_bbox, range(START_YEAR, END_YEAR + 1), out_path)
            print(f"[OK] ERA5-Land 下载完成: {out_path}")
            volume.commit()
            return {"lake_name": lake_name, "status": "OK"}
        except Exception as e:
            last_err = e
            wait_s = 30 * attempt  # Linear backoff. / 线性退避。
            print(f"  [RETRY {attempt}/4] {lake_name} CDS请求失败({e})，{wait_s}秒后重试")
            if attempt < 4:
                time.sleep(wait_s)

    print(f"[SKIP] ERA5-Land 下载未执行 ({lake_name})，重试4次仍失败: {last_err}")
    return {"lake_name": lake_name, "status": f"FAILED_era5_download: {last_err}"}


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    secrets=[modal.Secret.from_name("cds-api")],
    timeout=7200,
    max_containers=10,  # Process cached lakes in parallel. / 并行处理已缓存的湖泊。
    retries=1,
)
def process_lake_modal(lake_name: str) -> dict:
    """Build and save one lake's water-level and predictor data.
    构建并保存单湖的水位与驱动变量数据。
    """
    import os
    import glob
    import hashlib
    import io
    import json
    import pickle
    import zipfile
    from pathlib import Path

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import requests

    # Write Modal credentials in CDS API format. / 将 Modal 凭证写成 CDS API 格式。
    cds_rc = Path.home() / ".cdsapirc"
    cds_rc.write_text(f"url: {os.environ['CDSAPI_URL']}\nkey: {os.environ['CDSAPI_KEY']}\n")

    RESULTS_DIR = Path(DATA_ROOT) / "lake_results"
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    ERA5_DIR = Path(DATA_ROOT) / "era5_downloads"
    ERA5_DIR.mkdir(exist_ok=True, parents=True)

    # Fingerprint the lake-specific configuration. / 为单湖配置生成指纹。
    config_fingerprint = hashlib.sha256(json.dumps({
        "pipeline_version": PIPELINE_VERSION,
        "stations": LAKES.get(lake_name, {}).get("stations"),
        "regulation_stations": REGULATION_STATIONS.get(lake_name),
        "regulation_subperiod": REGULATION_SUBPERIODS.get(lake_name),
    }, sort_keys=True, default=str).encode()).hexdigest()[:16]

    checkpoint_path = RESULTS_DIR / f"{lake_name}_result.pkl"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("config_fingerprint") == config_fingerprint and cached["status"] in ("OK", "OK_no_candidates"):
                return cached
        except Exception as e:
            # Recompute unreadable checkpoints. / 无法读取的检查点重新计算。
            print(f"  [WARN] 旧checkpoint读取失败（可能是版本不兼容），忽略缓存重新计算: {e}")

    # Load basin shapefiles from the Volume. / 从 Volume 读取流域矢量文件。
    hydrolakes_matches = glob.glob(str(Path(DATA_ROOT) / "HydroLAKES_polys_v10_shp" / "**" / "*.shp"), recursive=True)
    hydrobasins_matches = glob.glob(str(Path(DATA_ROOT) / "hybas_na_lev12_v1c" / "*.shp"))
    hydrolakes_gdf = gpd.read_file(hydrolakes_matches[0])
    hydrobasins_gdf = gpd.read_file(hydrobasins_matches[0])

    from data_acquisition_lib import fetch_lake_water_level, fetch_lake_regulation_flow, resolve_lake_area_bbox, \
        try_fetch_era5_land, extract_masked_series, \
        convert_era5_units, build_variable_panel

    result = {"lake_name": lake_name, "status": "started", "pipeline_version": PIPELINE_VERSION,
              "config_fingerprint": config_fingerprint}
    try:
        combined_wl, wide_wl, coverage_wl = fetch_lake_water_level(lake_name)
        result["coverage_mean"] = coverage_wl["coverage_fraction"].mean()
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

        # Reuse the ERA5-Land cache created by fetch_only. / 复用 fetch_only 创建的 ERA5-Land 缓存。
        nc_path = try_fetch_era5_land(lake_name, area_bbox, era5_dir=ERA5_DIR)
        if nc_path is None:
            result["status"] = "FAILED_era5_download"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result

        nc_path = Path(nc_path)
        if nc_path.read_bytes()[:4] == b"PK\x03\x04":
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

        _, panel_deseason = build_variable_panel(combined_wl, {c: real_predictors[c] for c in real_predictors.columns}, forecast_horizon=FORECAST_HORIZON)
        # Check water-level coverage without listwise deletion. / 不做整表删缺，仅检查水位覆盖率。
        if panel_deseason["WL"].dropna().shape[0] < 60:
            result["status"] = "FAILED_too_few_overlapping_months"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result

        # Reserve the final months without listwise deletion. / 不做整表删缺，保留末段月份作测试期。
        ccm_train_panel = panel_deseason.iloc[:-FORECAST_HORIZON]
        if ccm_train_panel["WL"].dropna().shape[0] < 60:
            result["status"] = "FAILED_too_few_training_rows_for_ccm"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result

        # Save inputs for embedding selection and downstream analysis. / 保存嵌入参数选择与后续分析所需输入。
        result.update({"status": "OK"})
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
    """Run stage 1: download and cache ERA5-Land.
    运行第一阶段：下载并缓存 ERA5-Land。
    """
    target_lakes = lakes.split(",") if lakes else list(LAKES.keys())
    print(f"低并发预下载/缓存 ERA5，{len(target_lakes)} 个湖泊: {target_lakes}")
    era5_results = list(fetch_era5_modal.map(target_lakes))
    print("\n=== ERA5预下载完成 ===")
    for r in era5_results:
        print(f"  {r['lake_name']}: {r['status']}")
    failed = [r["lake_name"] for r in era5_results if r["status"] != "OK"]
    if failed:
        print(f"\n[WARN] {len(failed)}个湖泊没成功: {failed}")
        print("       明天先单独重跑这几个 --lakes，确认都OK了再跑process_only。")
    else:
        print("\n全部成功，可以跑 modal run --detach modal_build_lake_panels.py::process_only 了。")


@app.local_entrypoint()
def process_only(lakes: str = ""):
    """Run stage 2: build lake panels from the cached inputs.
    运行第二阶段：利用缓存输入构建湖泊面板。
    """
    target_lakes = lakes.split(",") if lakes else list(LAKES.keys())
    print(f"{len(target_lakes)} 个湖泊全速并行构建面板...")
    results = list(process_lake_modal.map(target_lakes))
    print("\n=== 全部完成 ===")
    for r in results:
        print(f"  {r['lake_name']}: {r['status']}")
