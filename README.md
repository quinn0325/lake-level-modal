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
  00_data_generation/       source-data functions and Modal data-stage entry points
  01_analysis_core/         shared scientific algorithms; not run directly
  02_within_lake_ccm/       Modal stage for 420 within-lake CCM edges
  03_inter_lake_ccm/        Modal stage for 90 between-lake CCM edges
  04_forecast/              Modal stage for the forecasting experiment
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

`code/01_analysis_core/analysis_core.py` is the shared scientific algorithm
library and is not run directly. Scripts whose names begin with `modal_` are the
only executable analysis stages: they set the experiment-specific inputs and
parameters, call the shared algorithms, schedule work on Modal, and merge or
save the outputs in the `ccm-data` Volume. Together they form one implementation
route, not parallel local and cloud versions. Scripts under `06`–`08` only turn
downloaded outputs into figures, tables and datasets; they do not rerun CCM or
forecasting.

Runtime directories such as `results/` (including `results/figures/`),
`lake_pkls/`, `ch4_tables/`, `appendices/` and `dataset/` are generated after
the Modal outputs are downloaded.

## Environment

Use Python 3.11, matching the Modal images:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal setup
```

## Source Data

The analysis covers January 1994 through December 2024. Download and extract
the two required geospatial products before starting the Modal run:

| Input | Required product | Role |
| --- | --- | --- |
| [HydroLAKES](https://www.hydrosheds.org/products/hydrolakes) | Version 1.0 lake polygons, shapefile distribution (`HydroLAKES_polys_v10_shp`) | lake polygons and lake-grid masks |
| [HydroBASINS](https://www.hydrosheds.org/products/hydrobasins) | Version 1c, standard North America level 12 polygons (`hybas_na_lev12_v1c`) | upstream-basin masks |

After extraction, arrange the shapefile components as follows. A shapefile is
not only its `.shp` file: keep its accompanying `.dbf`, `.shx`, `.prj` and
other files in the same directory.

```text
source_data/
  HydroLAKES_polys_v10_shp/
    .../HydroLAKES_polys_v10.shp
    .../HydroLAKES_polys_v10.dbf
    ...
  hybas_na_lev12_v1c/
    hybas_na_lev12_v1c.shp
    hybas_na_lev12_v1c.dbf
    hybas_na_lev12_v1c.shx
    ...
```

The official ten-lake route does not require a local HYDAT database. Monthly
water-level and regulated-flow series are fetched directly from Water Survey
of Canada by `modal_build_lake_panels.py`. HYDAT is used only for the optional
candidate-lake screening step. To reproduce that step, download the current
SQLite release (`Hydat.sqlite3`) from the [National Water Data Archive:
HYDAT](https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html).
HYDAT is updated quarterly, so record the download date; a later release may
produce a slightly different screening table. The selected ten-lake set used
by the analysis is fixed in the repository code.

ERA5-Land is not downloaded manually. The Modal data-generation job retrieves
the [ERA5-Land monthly averaged dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means)
through the CDS API. Create a CDS account, accept the dataset terms and obtain
the URL and personal access token using the [official CDS API setup
instructions](https://cds.climate.copernicus.eu/how-to-api).

## Modal Setup

This reproduction route must start with a newly created, empty Modal Volume
named `ccm-data`. Do not reuse a Volume containing outputs produced by an
earlier code version or different parameters. The shard files are used for
resume support and are identified by filename, so reusing an old Volume could
mix results from different runs.

Create the empty Volume and the CDS API secret:

```bash
modal volume create ccm-data
modal secret create cds-api \
  CDSAPI_URL=https://cds.climate.copernicus.eu/api \
  CDSAPI_KEY=YOUR_PERSONAL_ACCESS_TOKEN
```

Upload the two extracted geospatial directories. The HydroBASINS `.shp` file
must be directly inside the uploaded `hybas_na_lev12_v1c/` directory.

```bash
modal volume put ccm-data /absolute/path/to/source_data/HydroLAKES_polys_v10_shp HydroLAKES_polys_v10_shp
modal volume put ccm-data /absolute/path/to/source_data/hybas_na_lev12_v1c hybas_na_lev12_v1c
```

If you want to reproduce the candidate-lake screening step, also upload HYDAT:

```bash
modal volume put ccm-data /absolute/path/to/Hydat.sqlite3 Hydat.sqlite3
modal run code/00_data_generation/modal_lake_screening.py
```

The final ten-lake analysis below does not require `Hydat.sqlite3`; water-level
and regulated-flow station series are fetched by the data-generation code.

## Official Reproduction Route

Run all commands from the repository root.

### 1. Build Monthly Lake Panels On Modal

```bash
modal run code/00_data_generation/modal_build_lake_panels.py::fetch_only
modal run code/00_data_generation/modal_build_lake_panels.py::process_only
```

Expected Modal Volume outputs:

```text
lake_results/<lake>_result.pkl
era5_downloads/<lake>_era5land_monthly.nc
```

### 2. Compute Embedding Parameters On Modal

```bash
modal run code/00_data_generation/modal_compute_embedding_params.py
```

Expected Modal Volume output:

```text
lake_results/full_pipeline_v2/embed_params_corrected.json
```

### 3. Run Within-Lake CCM On Modal

```bash
modal run --detach code/02_within_lake_ccm/modal_within_lake_ccm.py
```

Progress check:

```bash
modal volume ls ccm-data lake_results/final_v3/edges | grep -c json
```

Only after the count reaches exactly 420:

```bash
modal run code/02_within_lake_ccm/modal_within_lake_ccm.py --merge-only
```

Expected Modal Volume output:

```text
lake_results/final_v3/ccm_all_edges_merged_fdr.csv
```

### 4. Run Between-Lake CCM On Modal

```bash
modal run --detach code/03_inter_lake_ccm/modal_inter_lake_ccm.py
```

Progress check:

```bash
modal volume ls ccm-data lake_results/final_v3/inter_edges | grep -c json
```

Only after the count reaches exactly 90:

```bash
modal run code/03_inter_lake_ccm/modal_inter_lake_ccm.py --merge-only
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

This writes chapter tables to `ch4_tables/`, figures to `results/figures/`,
appendix files to `appendices/` and the public dataset to `dataset/`.

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
