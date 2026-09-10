"""Modal execution stage for computing the authoritative embedding parameters.
在 Modal 上计算正式使用的湖泊嵌入参数。

Parameters are selected from cleaned, training-only panels and written to
``embed_params_corrected.json`` in the Volume. The source lake PKLs remain unchanged.
参数从清理后的训练期面板中选择，并写入 Volume 中的
``embed_params_corrected.json``；源湖泊 PKL 保持不变。

This file defines the Modal execution stage. It reuses panel preparation from
``01_analysis_core/analysis_core.py`` and embedding selection from
``data_acquisition_lib.py``; it is not a second analysis implementation.
本文件只定义 Modal 执行阶段；面板预处理复用 ``01_analysis_core/analysis_core.py``，
嵌入参数选择复用 ``data_acquisition_lib.py``，并非另一套分析实现。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

CODE_DIR = Path(__file__).resolve().parent                  # Data-stage code / 数据阶段代码
sys.path.insert(0, str(CODE_DIR))                            # Acquisition helpers / 数据获取函数
sys.path.insert(0, str(CODE_DIR.parent))                     # Shared configuration / 共享配置
sys.path.insert(0, str(CODE_DIR.parent / "01_analysis_core"))  # Analysis core / 分析核心

app = modal.App("compute-embedding-params")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas==2.2.2", "numpy==1.26.4", "scipy", "networkx", "pyEDM==2.4.0")
    .add_local_python_source("data_acquisition_lib")
    .add_local_python_source("analysis_core")
    .add_local_python_source("config")
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=False)
DATA_ROOT = "/data"
# Cached lake inputs. / 缓存的湖泊输入。
PKL_DIR_REMOTE = f"{DATA_ROOT}/lake_results"
# Output path relative to the Volume root. / 相对于 Volume 根目录的输出路径。
OUT_REMOTE = "lake_results/full_pipeline_v2/embed_params_corrected.json"


@app.function(image=image, volumes={DATA_ROOT: volume}, cpu=2.0, memory=4096,
              timeout=60 * 30)
def compute_one_lake(lake: str) -> dict:
    """Compute one lake's embedding parameters from its training panel.
    利用单湖训练面板计算各变量的嵌入参数。
    """
    import pickle
    import numpy as np
    import pandas as pd

    import config
    import analysis_core as p

    # Load only the required helpers without geospatial dependencies. / 仅加载所需函数，避免引入地理依赖。
    import ast as _ast
    _ns = {"np": np, "pd": pd}
    # Support Modal and local source paths. / 同时支持 Modal 与本地源码路径。
    _cands = ["/root/data_acquisition_lib.py", str(CODE_DIR / "data_acquisition_lib.py")]
    _lib = next((c for c in _cands if os.path.exists(c)), None)
    assert _lib, f"找不到 data_acquisition_lib.py，已查找: {_cands}"
    _src = open(_lib, encoding="utf-8").read()
    _want = {"simplex_self_predict_rho", "select_E"}
    _got = set()
    for _n in _ast.parse(_src).body:
        if isinstance(_n, _ast.FunctionDef) and _n.name in _want:
            exec(compile(_ast.Module([_n], []), "data_acquisition_lib.py", "exec"), _ns)
            _got.add(_n.name)
    missing = _want - _got
    assert not missing, f"未能从 data_acquisition_lib.py 抽取: {missing}"

    p.PKL_DIR = PKL_DIR_REMOTE

    with open(os.path.join(PKL_DIR_REMOTE, f"{lake}_result.pkl"), "rb") as f:
        cached = pickle.load(f)

    # Reproduce the downstream training-panel workflow. / 复用下游训练面板构建流程。
    clean_wide = p.clean_wide_wl(cached["wide_wl"], log_prefix=f"[{lake}] ")
    combined_wl = p.combine_station_water_levels(clean_wide, method="anomaly_mean")
    real_predictors = cached["real_predictors"]
    _, panel_deseason = p.build_variable_panel(
        combined_wl,
        {c: real_predictors[c] for c in real_predictors.columns},
        forecast_horizon=config.FORECAST_HORIZON,
    )
    panel_train = panel_deseason.iloc[:-config.FORECAST_HORIZON]

    out = {}
    for var in config.VARIABLES:
        if var not in panel_train.columns:
            continue
        values = panel_train[var].values          # Preserve calendar gaps. / 保留日历缺测位置。
        n_obs = int(np.sum(~np.isnan(values)))
        if n_obs < 30:
            out[var] = {"status": "insufficient_observations", "n_obs": n_obs}
            continue
        try:
            E, _ = _ns["select_E"](
                values,
                tau=config.EMBED_TAU,
                candidate_E=config.EMBED_E_CANDIDATES,
            )
            out[var] = {
                "E": int(E), "tau": int(config.EMBED_TAU), "n_obs": n_obs,
            }
        except Exception as exc:                  # Isolate failures by variable. / 按变量隔离失败。
            out[var] = {"status": f"ERROR: {type(exc).__name__}: {exc}", "n_obs": n_obs}

    return {"lake": lake, "params": out}


@app.local_entrypoint()
def main():
    """Compute all lakes and upload the merged JSON result.
    计算全部湖泊并上传合并后的 JSON 结果。
    """
    import config

    results = list(compute_one_lake.map(config.LAKES))

    merged = {r["lake"]: r["params"] for r in results}

    print(f"\n{'湖泊':<20}{'变量':<9}{'E':>5}{'τ':>5}  {'n_obs':>6}")
    print("-" * 48)
    n_total = 0
    for lake in config.LAKES:
        for var, d in merged.get(lake, {}).items():
            if "E" not in d:
                print(f"{lake:<20}{var:<9}  {d.get('status', '?')}")
                continue
            n_total += 1
            print(f"{lake:<20}{var:<9}{d['E']:>5}{d['tau']:>5}  {d['n_obs']:>6}")
    print("-" * 48)
    print(f"共计算 {n_total} 个变量的嵌入参数")

    payload = json.dumps(merged, ensure_ascii=False, indent=2)

    with volume.batch_upload(force=True) as batch:
        import io
        batch.put_file(io.BytesIO(payload.encode("utf-8")), OUT_REMOTE)
    print(f"\n已写入 Volume: {OUT_REMOTE}")
