# Lake-Level CCM Modal Reproduction

This repository is a single-route reproduction package for the MSc dissertation
*Causal Exploration and Predictability of Lake Water Level: Evidence from Two
Regulated Canadian River Basins*.

There is one official route:

1. Prepare public source data locally.
2. Upload required geospatial files and credentials to Modal.
3. Run data generation, embedding selection, CCM and forecasting on Modal.
4. Download Modal outputs.
5. Run local post-processing only for tables, figures, datasets and checks.

The local post-processing scripts do not rerun CCM or forecasting. The local
analysis driver from the earlier repository has been removed to avoid a second
reproduction route.

## Repository Layout

```text
code/
  00_data_generation/       Modal source-data acquisition and panel building
  01_shared/                shared modelling library used inside Modal jobs
  02_within_lake_ccm/       Modal within-lake CCM, 420 directed edges
  03_inter_lake_ccm/        Modal between-lake CCM, 90 directed edges
  04_forecast/              Modal forecasting, server-side orchestration
  06_figures/               local post-processing from downloaded CSV outputs
  06_postprocess_local/     one local entry point for post-processing
  07_tables/                local deterministic table builders
  08_dataset/               local dataset builder from downloaded lake PKLs
  99_check_outputs/         checks downloaded Modal outputs
reference/
  results/                  committed reference CSV/JSON outputs
  lake_pkls/                committed reference panel pickles
  figures/                  committed reference figures and map cache layers
  appendices/               committed reference appendix CSVs
  dataset/                  committed reference public dataset
```

Runtime directories such as `results/`, `lake_pkls/`, `figures/`, `appendices/`
and `dataset/` are generated after the Modal outputs are downloaded.

## Environment

Use Python 3.11, matching the Modal images:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal setup
```

## Modal Setup

Create the shared Volume and the CDS API secret:

```bash
modal volume create ccm-data
modal secret create cds-api CDSAPI_URL=... CDSAPI_KEY=...
```

Upload HydroLAKES and HydroBASINS from your local downloads:

```bash
modal volume put ccm-data /path/to/HydroLAKES_polys_v10_shp HydroLAKES_polys_v10_shp
modal volume put ccm-data /path/to/hybas_na_lev12_v1c hybas_na_lev12_v1c
```

If you want to reproduce the candidate-lake screening step, also upload HYDAT:

```bash
modal volume put ccm-data /path/to/Hydat.sqlite3 Hydat.sqlite3
modal run code/00_data_generation/modal_lake_screening.py
```

The final ten-lake analysis below does not require `Hydat.sqlite3`; water-level
and regulated-flow station series are fetched by the data-generation code.

## Official Reproduction Route

Run all commands from the repository root.

### 1. Build Monthly Lake Panels On Modal

```bash
modal run code/00_data_generation/ccm_modal_app.py::fetch_only
modal run code/00_data_generation/ccm_modal_app.py::process_only
```

Expected Modal Volume outputs:

```text
lake_results/<lake>_result.pkl
era5_downloads/<lake>_era5land_monthly.nc
```

### 2. Recompute Corrected Embedding Parameters On Modal

```bash
modal run code/00_data_generation/recompute_embed_params.py
```

Expected Modal Volume output:

```text
lake_results/full_pipeline_v2/embed_params_corrected.json
```

### 3. Run Within-Lake CCM On Modal

```bash
modal run --detach code/02_within_lake_ccm/run_within_lake_ccm.py
```

Progress check:

```bash
modal volume ls ccm-data lake_results/final_v3/edges | grep -c json
```

After the count reaches 420:

```bash
modal run code/02_within_lake_ccm/run_within_lake_ccm.py --merge-only
```

Expected Modal Volume output:

```text
lake_results/final_v3/ccm_all_edges_merged_fdr.csv
```

### 4. Run Between-Lake CCM On Modal

```bash
modal run --detach code/03_inter_lake_ccm/run_inter_lake_ccm.py
```

Progress check:

```bash
modal volume ls ccm-data lake_results/final_v3/inter_edges | grep -c json
```

After the count reaches 90:

```bash
modal run code/03_inter_lake_ccm/run_inter_lake_ccm.py --merge-only
```

Expected Modal Volume outputs:

```text
lake_results/final_v3/connectivity_full_pairwise_ccm_results.csv
lake_results/final_v3/connectivity_connected_vs_unconnected_summary.csv
```

### 5. Run Forecasting On Modal

Use the detached server-side orchestration entry point:

```bash
modal run --detach code/04_forecast/modal_forecast_synchrony_filtered.py::detached
```

Expected Modal Volume outputs:

```text
lake_results/final_v3/forecast_synchrony_filtered_full_results.csv
lake_results/final_v3/forecast_synchrony_filtered_rolling_results.csv
lake_results/final_v3/forecast_synchrony_filtered_dm_results.csv
lake_results/final_v3/forecast_synchrony_filtered_selected_lags.csv
```

## Download Modal Outputs

Create runtime output directories:

```bash
mkdir -p lake_pkls results
```

Download the ten lake panel pickles:

```bash
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
```

Download analysis outputs:

```bash
modal volume get ccm-data lake_results/full_pipeline_v2/embed_params_corrected.json results/
modal volume get ccm-data lake_results/final_v3/ccm_all_edges_merged_fdr.csv results/
modal volume get ccm-data lake_results/final_v3/connectivity_full_pairwise_ccm_results.csv results/
modal volume get ccm-data lake_results/final_v3/connectivity_connected_vs_unconnected_summary.csv results/
modal volume get ccm-data lake_results/final_v3/forecast_synchrony_filtered_full_results.csv results/
modal volume get ccm-data lake_results/final_v3/forecast_synchrony_filtered_rolling_results.csv results/
modal volume get ccm-data lake_results/final_v3/forecast_synchrony_filtered_dm_results.csv results/
modal volume get ccm-data lake_results/final_v3/forecast_synchrony_filtered_selected_lags.csv results/
```

## Local Post-Processing

After the Modal outputs are downloaded, build deterministic tables, figures and
dataset files locally:

```bash
python code/06_postprocess_local/build_outputs.py
```

Check the downloaded Modal outputs and regenerated local products:

```bash
python code/99_check_outputs/check_modal_outputs.py
```

## Expected Output Sizes

| Output | Expected size |
| --- | ---: |
| `lake_pkls/*_result.pkl` | 10 files |
| `results/embed_params_corrected.json` | 10 top-level lakes |
| `results/ccm_all_edges_merged_fdr.csv` | 420 rows |
| `results/connectivity_full_pairwise_ccm_results.csv` | 90 rows |
| `results/connectivity_connected_vs_unconnected_summary.csv` | 2 rows |
| `results/forecast_synchrony_filtered_full_results.csv` | 122 rows |
| `results/forecast_synchrony_filtered_rolling_results.csv` | 372 rows |
| `results/forecast_synchrony_filtered_dm_results.csv` | 341 rows |
| `results/forecast_synchrony_filtered_selected_lags.csv` | 38 rows |

The forecasting full-results table is expected to contain some method-level
SARIMAX error rows where exogenous-variable gaps prevent block forecasting. The
rolling and DM outputs use the valid fitted methods.

## Reproducibility Notes

The CCM stages use fixed surrogate seeds and Modal images with pinned
`pyEDM==2.4.0`, `pandas==2.2.2` and `numpy==1.26.4`.

The original Modal run did not preserve a full lock file for `xgboost`,
`statsmodels` or `pmdarima`. Forecast metrics are therefore expected to be
stable for the dissertation conclusions but should not be described as
byte-identical across future image rebuilds.
