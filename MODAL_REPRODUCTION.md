# Modal Command Sheet

This is the compact command sheet for the single official reproduction route.
For context and expected outputs, see `README.md`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal setup

modal volume create ccm-data
modal secret create cds-api CDSAPI_URL=... CDSAPI_KEY=...
modal volume put ccm-data /path/to/HydroLAKES_polys_v10_shp HydroLAKES_polys_v10_shp
modal volume put ccm-data /path/to/hybas_na_lev12_v1c hybas_na_lev12_v1c

modal run code/00_data_generation/modal_build_lake_panels.py::fetch_only
modal run code/00_data_generation/modal_build_lake_panels.py::process_only

modal run code/00_data_generation/recompute_embed_params.py

modal run --detach code/02_within_lake_ccm/run_within_lake_ccm.py
modal volume ls ccm-data lake_results/final_v3/edges | grep -c json
modal run code/02_within_lake_ccm/run_within_lake_ccm.py --merge-only

modal run --detach code/03_inter_lake_ccm/run_inter_lake_ccm.py
modal volume ls ccm-data lake_results/final_v3/inter_edges | grep -c json
modal run code/03_inter_lake_ccm/run_inter_lake_ccm.py --merge-only

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

python code/06_postprocess_local/build_outputs.py
python code/99_check_outputs/check_modal_outputs.py
```
