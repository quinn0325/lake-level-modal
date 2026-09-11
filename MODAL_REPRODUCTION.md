# Modal Command Sheet

This is the compact command sheet for the single official reproduction route.
For context and expected outputs, see `README.md`.

For the analysis stages, run only the `modal_*.py` files listed below. The shared algorithms in
`code/01_analysis_core/analysis_core.py` are imported by those stages and are
not a separate command or a local alternative.

Run this route only with a newly created, empty Modal Volume named `ccm-data`.
Do not reuse a Volume containing shards from an earlier code version or a run
with different parameters. Merge only when the two shard counts below are
exactly 420 and 90, respectively.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal setup

modal volume create ccm-data
modal secret create cds-api CDSAPI_URL=https://cds.climate.copernicus.eu/api CDSAPI_KEY=YOUR_PERSONAL_ACCESS_TOKEN
# Download and extract the exact source products listed in README.md first.
modal volume put ccm-data /absolute/path/to/source_data/HydroLAKES_polys_v10_shp HydroLAKES_polys_v10_shp
modal volume put ccm-data /absolute/path/to/source_data/hybas_na_lev12_v1c hybas_na_lev12_v1c

modal run code/00_data_generation/modal_build_lake_panels.py::fetch_only
modal run code/00_data_generation/modal_build_lake_panels.py::process_only

modal run code/00_data_generation/modal_compute_embedding_params.py

modal run --detach code/02_within_lake_ccm/modal_within_lake_ccm.py
modal volume ls ccm-data lake_results/final_v3/edges | grep -c json
modal run code/02_within_lake_ccm/modal_within_lake_ccm.py --merge-only

modal run --detach code/03_inter_lake_ccm/modal_inter_lake_ccm.py
modal volume ls ccm-data lake_results/final_v3/inter_edges | grep -c json
modal run code/03_inter_lake_ccm/modal_inter_lake_ccm.py --merge-only

modal run --detach code/04_forecast/modal_forecast_synchrony_filtered.py::detached

mkdir -p lake_pkls results
modal volume get ccm-data lake_results/Kalamalka_Lake_result.pkl lake_pkls/
modal volume get ccm-data lake_results/Okanagan_Lake_result.pkl lake_pkls/
modal volume get ccm-data lake_results/Skaha_Lake_result.pkl lake_pkls/
modal volume get ccm-data lake_results/Vaseux_Lake_result.pkl lake_pkls/
modal volume get ccm-data lake_results/Rainy_Lake_result.pkl lake_pkls/
modal volume get ccm-data lake_results/Lake_of_the_Woods_result.pkl lake_pkls/
modal volume get ccm-data lake_results/Playgreen_Lake_result.pkl lake_pkls/
modal volume get ccm-data lake_results/Kiskitto_Lake_result.pkl lake_pkls/
modal volume get ccm-data lake_results/Sipiwesk_Lake_result.pkl lake_pkls/
modal volume get ccm-data lake_results/Split_Lake_result.pkl lake_pkls/

modal volume get ccm-data lake_results/full_pipeline_v2/embed_params_corrected.json results/
modal volume get ccm-data lake_results/final_v3/ccm_all_edges_merged_fdr.csv results/
modal volume get ccm-data lake_results/final_v3/connectivity_full_pairwise_ccm_results.csv results/
modal volume get ccm-data lake_results/final_v3/connectivity_connected_vs_unconnected_summary.csv results/
modal volume get ccm-data lake_results/final_v3/forecast_synchrony_filtered_full_results.csv results/
modal volume get ccm-data lake_results/final_v3/forecast_synchrony_filtered_rolling_results.csv results/
modal volume get ccm-data lake_results/final_v3/forecast_synchrony_filtered_dm_results.csv results/
modal volume get ccm-data lake_results/final_v3/forecast_synchrony_filtered_selected_lags.csv results/
modal volume get ccm-data lake_results/final_v3/xgboost_tuning_results.csv results/

python code/05_postprocess_local/build_outputs.py
python code/99_check_outputs/check_modal_outputs.py
```
