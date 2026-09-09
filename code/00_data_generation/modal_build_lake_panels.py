"""数据获取阶段：把十个湖泊的 WSC 水位/调控流量与 ERA5-Land 气候变量拼成月度面板。

产出 `{lake}_result.pkl`（wide_wl、real_predictors、embed_params），供 02_/03_ 的 CCM
和 04_ 的预测读取。CCM 与预测本身不在这里跑。

用法
----
    modal run code/00_data_generation/modal_build_lake_panels.py::fetch_only
    modal run code/00_data_generation/modal_build_lake_panels.py::process_only

分两步手动执行：
  1. fetch_era5_modal   低并发预下载 ERA5 并缓存到 Volume
  2. process_lake_modal 逐湖并行构建面板与嵌入参数

第 1 步必须低并发：十个湖同时打 CDS API 会触发账号级限流
("Number queued requests for this dataset is temporarily limited")。

前置条件（在终端自己做，不要把密钥贴进任何地方）
------------------------------------------------
    modal setup
    modal secret create cds-api CDSAPI_URL=... CDSAPI_KEY=...

首次运行前把 HydroLAKES / HydroBASINS shapefile 上传到 Volume：
    modal volume create ccm-data
    modal volume put ccm-data <local-path> HydroLAKES_polys_v10_shp
    modal volume put ccm-data /path/to/hybas_na_lev12_v1c hybas_na_lev12_v1c
"""
import modal

app = modal.App("ccm-lakes")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgdal-dev", "gdal-bin")
    .pip_install(
        "pandas==2.2.2", "numpy==1.26.4",  # 锁版本跟本机(3.10)一致，否则.map()结果传回本地反序列化会因为
        # pandas内部dtype表示不兼容而报DeserializationError（已经踩过这个坑）
        "geopandas", "shapely", "requests",
        "pmdarima", "statsmodels", "networkx", "xarray", "netcdf4",
        "cdsapi", "matplotlib", "pyEDM", "scipy", "xgboost",
    )
    .add_local_python_source("modal_build_lake_panels")  # data_acquisition_lib 会读取这里的配置常量
    .add_local_python_source("data_acquisition_lib")  # 显式打进镜像，不依赖自动挂载
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=True)
DATA_ROOT = "/data"

VARS = ["WL", "T", "P", "R", "SWE", "Evap", "RegFlow"]
START_YEAR, END_YEAR = 1994, 2024
PIPELINE_VERSION = "2026-08-09_v14_forecast_horizon_37mo_revert"
# 只作为缓存指纹的一部分（见 process_lake_modal 里的 config_fingerprint）。改动任何
# 影响面板构建的东西时把它升一版，全部湖泊的 checkpoint 就会失效重算。
FORECAST_HORIZON = 37  # 9:1切分，固定月份数、不按各湖面板长度动态算

# 研究用的十个受调控湖泊，与 code/config.py 的 LAKES 一致。
LAKES = {
    "Lake_of_the_Woods":    dict(group="Boreal Shield", rank=1, stations=["05PD029", "05PD008", "05PE014", "05PD011"]),
    "Rainy_Lake":           dict(group="Boreal Shield", rank=2, stations=["05PB024", "05PB007"]),
    "Okanagan_Lake":        dict(group="Montane Cordillera", rank=1, stations=["08NM083"]),
    "Vaseux_Lake":          dict(group="Montane Cordillera", rank=2, stations=["08NM243"]),
    "Kalamalka_Lake":       dict(group="Montane Cordillera", rank=4, stations=["08NM143"]),
    "Skaha_Lake":           dict(group="New_2026", rank=7, stations=["08NM084"]),
    "Split_Lake":           dict(group="New_2026", rank=8, stations=["05UF003"]),
    # 尼尔森河梯级链，由 Manitoba Hydro 在 Jenpeg 大坝统一调度。
    "Playgreen_Lake":       dict(group="Nelson_River_Chain", rank=1, stations=["05UB005"]),
    "Kiskitto_Lake":        dict(group="Nelson_River_Chain", rank=2, stations=["05UB013"]),
    "Sipiwesk_Lake":        dict(group="Nelson_River_Chain", rank=3, stations=["05UD006"]),
}

STATION_COORDS = {
    "05PB007": (-93.3206787109375, 48.6491203308106),   # RAINY LAKE NEAR FORT FRANCES
    "05PB024": (-92.9583587646484, 48.7004699707031),   # RAINY LAKE NEAR BEAR PASS
    "05PD008": (-94.2834930419922, 49.1328010559082),   # LAKE OF THE WOODS AT HANSON BAY
    "05PD011": (-94.8102493286133, 49.7126388549805),   # LAKE OF THE WOODS AT CLEARWATER BAY
    "05PD029": (-94.8531112670898, 49.3284683227539),   # LAKE OF THE WOODS AT CYCLONE ISLAND
    "05PE014": (-94.5537185668945, 49.762939453125),    # LAKE OF THE WOODS AT KEEWATIN (出口站)
    "08NM083": (-119.499656677246, 49.8861312866211),   # OKANAGAN LAKE AT KELOWNA
    "08NM143": (-119.274421691895, 50.2299194335938),   # KALAMALKA LAKE AT VERNON PUMPHOUSE
    "08NM243": (-119.525512695313, 49.2731895446777),   # VASEUX LAKE NEAR THE OUTLET
    "05PE011": (-94.52446746826172, 49.771968841552734),  # LAKE OF THE WOODS WESTERN OUTLET ABOVE NORMAN DAM (RegFlow)
    "05PE006": (-94.50308227539062, 49.77252960205078),   # LAKE OF THE WOODS EASTERN OUTLET AT KENORA POWERHOUSE (RegFlow)
    "05PC019": (-93.4034423828125, 48.60852813720703),    # RAINY RIVER AT FORT FRANCES (RegFlow)
    "08NM050": (-119.6153564453125, 49.49851989746094),   # OKANAGAN RIVER AT PENTICTON (RegFlow)
    "08NM247": (-119.5280303955078, 49.25683975219727),   # OKANAGAN RIVER BELOW MCINTYRE DAM (RegFlow)
    "08NM065": (-119.2668914794922, 50.238468170166016),  # VERNON CREEK AT OUTLET OF KALAMALKA LAKE (RegFlow)
    # --- 第11-14个候选，坐标取自HYDAT STATIONS表LATITUDE/LONGITUDE字段 ---
    "08NM084": (-119.57508087158205, 49.42641830444336),  # SKAHA LAKE AT OKANAGAN FALLS
    "08NM002": (-119.58040618896484, 49.34204864501953),  # OKANAGAN RIVER AT OKANAGAN FALLS (RegFlow)
    "05UF003": (-96.08728790283205, 56.24385833740234),   # SPLIT LAKE AT SPLIT LAKE
    "05UF006": (-94.6338882446289, 56.380279541015625),   # NELSON RIVER AT KETTLE GENERATING STATION (RegFlow)
    # --- 第18-21个候选：尼尔森河梯级链，坐标取自MSC GeoMet hydrometric-stations API ---
    "05UB005": (-97.9749984741211, 53.90277862548828),    # PLAYGREEN LAKE AT ENTRANCE TO EAST NELSON RIVER
    "05UB013": (-98.4383316040039, 54.30305862426758),    # KISKITTO LAKE NEAR NORWAY HOUSE
    "05UB009": (-98.04805755615234, 54.4980583190918),    # NELSON RIVER (WEST CHANNEL) AT JENPEG (RegFlow, Playgreen_Lake/Kiskitto_Lake共用)
    "05UD006": (-97.5, 55.09444046020508),                # SIPIWESK LAKE AT FORESTRY DOCK
    "05UE005": (-96.5250015258789, 56.03889083862305),    # NELSON RIVER AT KELSEY GENERATING STATION (RegFlow, Sipiwesk_Lake下游代理)
}

# 流域追溯起点的参照坐标。多站湖泊里指明哪一个是出水口；未列出的湖泊由代码取第一个站。
LAKE_OUTLET_STATION = {
    "Lake_of_the_Woods": "05PE014",
    "Rainy_Lake": "05PB007",
}

# 调控出流站。湖泊不在这个字典里时 process_lake_modal 会自动跳过 RegFlow 这一列，
# 不塞 NaN 占位。
REGULATION_STATIONS = {
    "Lake_of_the_Woods":    ["05PE011", "05PE006"],  # Norman Dam + Kenora Powerhouse，两站相加才是总放水量
    "Rainy_Lake":           ["05PC019"],
    "Okanagan_Lake":        ["08NM050"],
    "Vaseux_Lake":          ["08NM247"],             # 仅2012-2024子区间可用
    "Kalamalka_Lake":       ["08NM065"],             # 正好在Kalamalka Lake出口
    "Skaha_Lake":           ["08NM002"],              # 集水面积 6686 vs 6684 km²
    "Split_Lake":           ["05UF006"],              # 无自有坝，借用下游 Kettle 发电站
    "Playgreen_Lake":       ["05UB009"],   # Jenpeg大坝回水同时顶托Playgreen_Lake和下游Cross Lake
    "Kiskitto_Lake":        ["05UB009"],   # 同一西支流，同样经Jenpeg控制
    "Sipiwesk_Lake":        ["05UE005"],   # Sipiwesk_Lake自己没有坝，借用下游Kelsey发电站出流代理
}
# 调控流量只在这些年份区间内有效，区间外一律置为缺测。
REGULATION_SUBPERIODS = {
    "Vaseux_Lake": (2012, 2024),   # 测流站 2012 年才开始记录
}


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    secrets=[modal.Secret.from_name("cds-api")],
    timeout=3600,  # 容纳下面最多 4 次、累计约 5 分钟的退避重试，外加下载本身
    max_containers=4,  # 全开会撞 CDS 账号级限流。4 是经验值，不是量出来的最优值：
                       # 若仍频繁触发下面的退避重试，往下调。
)
def fetch_era5_modal(lake_name: str) -> dict:
    """只做ERA5-Land下载+缓存到Volume，不跑CCM。低并发运行，避免撞CDS账号级限流。

    注意：try_fetch_era5_land内部会把cdsapi抛出的异常自己try/except吞掉、返回None，
    不会往外抛异常——这意味着Modal装饰器上的retries=N对这种失败完全不起作用(retries
    只在函数本身抛异常时触发，正常return不算)。所以这里不走try_fetch_era5_land的
    封装，直接调用更底层的fetch_era5_land_monthly，自己捕获异常、按退避策略重试。
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

    # resolve_lake_area_bbox内部可能抛异常(比如trace_upstream_basin在参照点没落进任何
    # 子流域多边形时会raise ValueError)——这里必须兜住，否则Modal的.map()会把这一个湖泊的
    # 异常原样往上抛，导致整批其余湖泊(包括本来能秒回缓存的)全部被取消(已经在10湖泊批量跑
    # Prosperous_Lake时踩过一次：其余9个湖泊全部收到cancellation signal)。
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
            wait_s = 30 * attempt  # 30s, 60s, 90s, 120s
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
    max_containers=10,  # 十个湖可以全速并行：ERA5 已由 fetch_era5_modal 缓存好，
                        # 这里命中 Volume 缓存直接返回，不会再打 CDS。
    retries=1,
)
def process_lake_modal(lake_name: str) -> dict:
    """抓取一个湖泊的水位/调控流量/ERA5 变量，拼成月度面板，选嵌入参数，写 pkl。

    数据源是 Modal Volume，ERA5 凭证来自 Modal Secret（写成容器内 ~/.cdsapirc，
    cdsapi 库照常读取）。
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

    # ---- CDS凭证：从Modal Secret环境变量写成cdsapi库要求的文件格式 ----
    cds_rc = Path.home() / ".cdsapirc"
    cds_rc.write_text(f"url: {os.environ['CDSAPI_URL']}\nkey: {os.environ['CDSAPI_KEY']}\n")

    RESULTS_DIR = Path(DATA_ROOT) / "lake_results"
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    ERA5_DIR = Path(DATA_ROOT) / "era5_downloads"
    ERA5_DIR.mkdir(exist_ok=True, parents=True)

    # ---- 缓存失效判据 ----
    # 指纹 = PIPELINE_VERSION + 这个湖自己的水位站/调控站/子区间配置。任何一项变了，
    # 这个湖的 checkpoint 自动失效重算，不依赖人记得手动升版本号。
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
            # 旧checkpoint可能是不同pandas/numpy版本序列化的，读不出来就当没有缓存，重新跑，
            # 不要让一个读不了的旧文件把整个函数炸掉。
            print(f"  [WARN] 旧checkpoint读取失败（可能是版本不兼容），忽略缓存重新计算: {e}")

    # ---- HydroLAKES / HydroBASINS：从Volume读（提前用modal volume put上传好） ----
    hydrolakes_matches = glob.glob(str(Path(DATA_ROOT) / "HydroLAKES_polys_v10_shp" / "**" / "*.shp"), recursive=True)
    hydrobasins_matches = glob.glob(str(Path(DATA_ROOT) / "hybas_na_lev12_v1c" / "*.shp"))
    hydrolakes_gdf = gpd.read_file(hydrolakes_matches[0])
    hydrobasins_gdf = gpd.read_file(hydrobasins_matches[0])

    from data_acquisition_lib import fetch_lake_water_level, fetch_lake_regulation_flow, resolve_lake_area_bbox, \
        try_fetch_era5_land, extract_masked_series, \
        convert_era5_units, build_variable_panel, get_embedding_params

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

        # ERA5数据已经由main()先调用fetch_era5_modal低并发预下载/缓存过，这里命中Volume缓存
        # 直接返回，不会再触发 CDS 请求（并发全开打 CDS 会被限流，见 fetch_era5_modal）。
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
        # 不要对 panel_deseason 做 listwise dropna：那会把整张面板砍到 RegFlow 能用的
        # 那个短窗口，下游每条边的可用样本量就全被拉平了。只用 WL 自己的有效月数做检查。
        if panel_deseason["WL"].dropna().shape[0] < 60:
            result["status"] = "FAILED_too_few_overlapping_months"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result

        # 同样不 dropna，只按日期切掉最后 FORECAST_HORIZON 个月留作测试期。不因为某个
        # 变量（通常是 RegFlow）局部缺测就连带砍掉其他变量的完整数据（Little & Rubin, 2002）。
        ccm_train_panel = panel_deseason.iloc[:-FORECAST_HORIZON]
        if ccm_train_panel["WL"].dropna().shape[0] < 60:
            result["status"] = "FAILED_too_few_training_rows_for_ccm"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(result, f)
            volume.commit()
            return result

        available_vars = [v for v in VARS if v in ccm_train_panel.columns]
        # 传完整日历序列、保留真实 NaN，不要 dropna（原因见 data_acquisition_lib.py 文件头）。
        embed_params = {v: get_embedding_params(ccm_train_panel[v].values, v, verbose=False) for v in available_vars}
        # 这个阶段只负责把面板和嵌入参数存进 pkl。CCM 本身由 02_/03_ 跑，
        # 预测由 04_ 跑，两者都从后续 Modal Volume 结果读，不经过这里。
        result.update({"status": "OK", "embed_params": embed_params})
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
    """只跑第 1 步：低并发预下载并缓存 ERA5。

        caffeinate -i modal run --detach modal_build_lake_panels.py::fetch_only

    过夜跑要把两步分开手动触发。main() 里"等第 1 步跑完再派发第 2 步"的调度是本地
    Python 在做的，笔记本一旦睡眠/断开，第 2 步就不会被触发——--detach 只保证已经派
    发出去的云端任务不被杀掉，不能让本地进程凭空继续跑。
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
    """只跑第 2 步：并行构建面板，假设 ERA5 已由 fetch_only 缓存好。

        caffeinate -i modal run --detach modal_build_lake_panels.py::process_only

    某个湖的 ERA5 若还没缓存，这里会临时下载一次——走的是本函数自己的高并发通道、
    不带退避重试，可能又撞限流。先确认 fetch_only 全部 OK 再跑这个。
    """
    target_lakes = lakes.split(",") if lakes else list(LAKES.keys())
    print(f"{len(target_lakes)} 个湖泊全速并行构建面板...")
    results = list(process_lake_modal.map(target_lakes))
    print("\n=== 全部完成 ===")
    for r in results:
        print(f"  {r['lake_name']}: {r['status']}")
