"""重算全部湖泊、全部变量的嵌入参数，写入 Modal Volume。

为什么需要这一步
----------------
Modal Volume 中 `lake_results/*_result.pkl` 里缓存的 `embed_params` 由数据生成阶段算出，当时
`build_variable_panel()` 未传 `forecast_horizon`，去季节化用了含测试期的全序列
气候态，因此 τ 与 E 携带了测试期信息。

下游两个面板构建函数（`load_lake_panel_for_ccm`、`load_pair_panel_for_connectivity`）
本身早已修复为只用训练期去季节化，**唯一残留的泄漏点**就是它们仍从 PKL 读取
`embed_params`。因此不需要重建 PKL、不需要重新下载任何原始数据，只需用正确的
训练期面板重算嵌入参数。

本脚本不回写 PKL——PKL 内还存着 `wide_wl` / `real_predictors` 等原始数据，
回写有损坏风险。改为写出独立 JSON，便于与旧值逐项对照审计。

本次同时固定 τ=1（见 config.EMBED_TAU 与 data_acquisition_lib.get_embedding_params 的说明），
因此实际只需重算 E。

跑法
----
    modal run code/00_data_generation/recompute_embed_params.py

输出
----
    lake_results/full_pipeline_v2/embed_params_corrected.json
        {lake: {var: {"E": int, "tau": int, "n_obs": int}}}
    同时打印新旧 E 的逐项对照，便于核对改动幅度。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

CODE_DIR = Path(__file__).resolve().parent                  # code/00_data_generation
sys.path.insert(0, str(CODE_DIR))                            # data_acquisition_lib
sys.path.insert(0, str(CODE_DIR.parent))                     # config
sys.path.insert(0, str(CODE_DIR.parent / "01_shared"))       # ccm_forecast_core

app = modal.App("recompute-embed-params")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas==2.2.2", "numpy==1.26.4", "scipy", "networkx", "pyEDM==2.4.0")
    .add_local_python_source("data_acquisition_lib")
    .add_local_python_source("ccm_forecast_core")
    .add_local_python_source("config")
)

volume = modal.Volume.from_name("ccm-data", create_if_missing=False)
DATA_ROOT = "/data"
# PKL 实际位于 lake_results/ 根下（已用 modal volume ls 核实），不在 lake_pkls 子目录
PKL_DIR_REMOTE = f"{DATA_ROOT}/lake_results"
# batch_upload 的 remote path 相对 volume 根，不带 /data 前缀
OUT_REMOTE = "lake_results/full_pipeline_v2/embed_params_corrected.json"


@app.function(image=image, volumes={DATA_ROOT: volume}, cpu=2.0, memory=4096,
              timeout=60 * 30)
def recompute_one_lake(lake: str) -> dict:
    """用**只在训练期去季节化**的面板重算该湖每个变量的嵌入参数。"""
    import pickle
    import numpy as np
    import pandas as pd

    import config
    import ccm_forecast_core as p

    # 不能直接 import data_acquisition_lib：它在模块级 import geopandas / shapely / requests
    # 以及 modal_build_lake_panels（选湖与数据下载用），本作业一个都不需要。
    # 改为用 AST 从源文件原样抽取所需的三个函数后执行——保证与正式实现逐字一致，
    # 不重新实现，也不触发重依赖。
    import ast as _ast
    _ns = {"np": np, "pd": pd, "pyEDM": __import__("pyEDM")}
    # data_acquisition_lib.py 的位置：Modal 容器里由 add_local_python_source 放在 /root/；
    # 本地运行时与本文件同目录。两种情形都要能找到，否则本地跑会 FileNotFoundError。
    _cands = ["/root/data_acquisition_lib.py", str(CODE_DIR / "data_acquisition_lib.py")]
    _lib = next((c for c in _cands if os.path.exists(c)), None)
    assert _lib, f"找不到 data_acquisition_lib.py，已查找: {_cands}"
    _src = open(_lib, encoding="utf-8").read()
    _want = {"simplex_self_predict_rho", "select_E", "get_embedding_params",
             "_config_embed_tau", "_config_embed_E_candidates"}
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

    # 与 load_lake_panel_for_ccm 完全相同的面板构建路径：
    # 清洗异常值 → 多站距平合成 → 补全月历 → 仅用训练期去季节化 → 切出训练段
    clean_wide = p.clean_wide_wl(cached["wide_wl"], log_prefix=f"[{lake}] ")
    combined_wl = p.combine_station_water_levels(clean_wide, method="anomaly_mean")
    real_predictors = cached["real_predictors"]
    _, panel_deseason = p.build_variable_panel(
        combined_wl,
        {c: real_predictors[c] for c in real_predictors.columns},
        forecast_horizon=config.FORECAST_HORIZON,
    )
    panel_train = panel_deseason.iloc[:-config.FORECAST_HORIZON]

    old = cached.get("embed_params", {})
    out = {}
    for var in config.VARIABLES:
        if var not in panel_train.columns:
            continue
        values = panel_train[var].values          # 保留完整日历位置与真实 NaN
        n_obs = int(np.sum(~np.isnan(values)))
        if n_obs < 30:
            out[var] = {"status": "insufficient_observations", "n_obs": n_obs}
            continue
        try:
            params = _ns["get_embedding_params"](
                values, var, verbose=False,
                tau=config.EMBED_TAU, candidate_E=config.EMBED_E_CANDIDATES,
            )
            out[var] = {
                "E": int(params["E"]), "tau": int(params["tau"]), "n_obs": n_obs,
                "old_E": old.get(var, {}).get("E"),
                "old_tau": old.get(var, {}).get("tau"),
            }
        except Exception as exc:                  # 单个变量失败不拖垮整个湖泊
            out[var] = {"status": f"ERROR: {type(exc).__name__}: {exc}", "n_obs": n_obs}

    return {"lake": lake, "params": out}


@app.local_entrypoint()
def main():
    import config

    results = list(recompute_one_lake.map(config.LAKES))

    merged = {r["lake"]: r["params"] for r in results}

    print(f"\n{'湖泊':<20}{'变量':<9}{'旧E':>5}{'新E':>5}{'旧τ':>5}{'新τ':>5}  {'n_obs':>6}")
    print("-" * 62)
    n_e_changed = n_tau_changed = n_total = 0
    for lake in config.LAKES:
        for var, d in merged.get(lake, {}).items():
            if "E" not in d:
                print(f"{lake:<20}{var:<9}  {d.get('status', '?')}")
                continue
            n_total += 1
            if d.get("old_E") is not None and d["old_E"] != d["E"]:
                n_e_changed += 1
            if d.get("old_tau") is not None and d["old_tau"] != d["tau"]:
                n_tau_changed += 1
            print(f"{lake:<20}{var:<9}{str(d.get('old_E')):>5}{d['E']:>5}"
                  f"{str(d.get('old_tau')):>5}{d['tau']:>5}  {d['n_obs']:>6}")
    print("-" * 62)
    print(f"共 {n_total} 个变量：E 改变 {n_e_changed} 个，τ 改变 {n_tau_changed} 个")

    payload = json.dumps(merged, ensure_ascii=False, indent=2)

    with volume.batch_upload(force=True) as batch:
        import io
        batch.put_file(io.BytesIO(payload.encode("utf-8")), OUT_REMOTE)
    print(f"\n已写入 Volume: {OUT_REMOTE}")

    print("本复现包的官方输出保存在 Modal Volume；需要本地副本时请用 modal volume get 下载。")
