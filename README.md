# Lake Level CCM Modal Reproduction

Code and reference outputs for the MSc dissertation Causal Exploration and
Predictability of Lake Water Level: Evidence from Two Regulated Canadian River
Basins.

The analysis covers ten regulated Canadian lakes from January 1994 to December
2024. Monthly lake panels are assembled from Water Survey of Canada records,
ERA5-Land, HydroLAKES and HydroBASINS. Embedding selection, within-lake CCM,
between-lake CCM and forecasting run on Modal. Figures, tables, appendices and
the derived dataset are then built locally.

## Repository Contents

`code/` contains the executable analysis and output-building scripts:

| Directory | Purpose |
| --- | --- |
| `00_data_generation/` | retrieve WSC and ERA5-Land data and build lake panels |
| `01_analysis_core/` | shared preprocessing, CCM and forecasting functions |
| `02_within_lake_ccm/` | run 420 directed within-lake CCM tests |
| `03_inter_lake_ccm/` | run 90 directed between-lake CCM tests |
| `04_forecast/` | run the forecasting experiment |
| `05_postprocess_local/` | build all local outputs |
| `06_figures/` | figure scripts |
| `07_tables/` | table and appendix scripts |
| `08_dataset/` | derived-dataset builder |
| `99_check_outputs/` | output-completeness checks |

`reference/` contains the outputs used in the dissertation:

| Directory | Contents |
| --- | --- |
| `results/` | embedding, CCM, connectivity, forecasting and tuning results |
| `lake_pkls/` | processed monthly lake panels |
| `figures/` | dissertation figures and the map layers used by Figure 3.1 |
| `tables/` | main-text table output |
| `appendices/` | appendix tables |
| `dataset/` | derived monthly datasets and their data dictionary |

Files produced during a new run are written to `lake_pkls/`, `results/`,
`ch4_tables/`, `appendices/` and `dataset/`.

## Requirements

- Python 3.11
- a Modal account and command-line client
- a Copernicus Climate Data Store account and API token
- HydroLAKES v1.0 lake polygons
- HydroBASINS North America level 12 v1c polygons

Clone the repository and create the local environment:

```bash
git clone https://github.com/quinn0325/lake-level-modal.git
cd lake-level-modal

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
modal setup
```

The commands below use the current Modal environment. When using a named
environment, add the same `--env=<name>` option to every `modal run`,
`modal volume` and `modal secret` command.

## Prepare the Source Data

Download and extract the two geospatial products:

| Dataset | Required file | Use |
| --- | --- | --- |
| HydroLAKES v1.0 | [HydroLAKES_polys_v10_shp.zip](https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip) | lake polygons and ERA5-Land grid masks |
| HydroBASINS v1c | [North America level 12](https://data.hydrosheds.org/file/hydrobasins/standard/hybas_na_lev12_v1c.zip) | upstream-basin masks |

Keep all components of each shapefile together. The extracted directories
should have the following form:

```text
source_data/
  HydroLAKES_polys_v10_shp/
    .../HydroLAKES_polys_v10.shp
    .../HydroLAKES_polys_v10.dbf
    .../HydroLAKES_polys_v10.shx
    .../HydroLAKES_polys_v10.prj
  hybas_na_lev12_v1c/
    hybas_na_lev12_v1c.shp
    hybas_na_lev12_v1c.dbf
    hybas_na_lev12_v1c.shx
    hybas_na_lev12_v1c.prj
```

The HydroLAKES shapefile may be inside a nested directory. The HydroBASINS
shapefile must be directly inside `hybas_na_lev12_v1c/`.

Water level and regulated flow observations are requested from the Water
Survey of Canada historical hydrometric service during panel construction. No
local HYDAT database is required.

ERA5-Land is requested through the Copernicus CDS API. Before running the data
stage, accept the licence for the
[ERA5-Land monthly averaged dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means)
and obtain a personal access token.

The fixed lake and station assignments are listed in
`reference/appendices/A1_lakes_and_stations.csv`. Data coverage is reported in
`reference/appendices/B2_data_availability.csv`.

## Configure Modal

Create the Volume and CDS secret:

```bash
modal volume create ccm-data

modal secret create cds-api \
  CDSAPI_URL=https://cds.climate.copernicus.eu/api \
  CDSAPI_KEY=YOUR_PERSONAL_ACCESS_TOKEN
```

Upload the two extracted directories:

```bash
modal volume put ccm-data \
  /absolute/path/to/source_data/HydroLAKES_polys_v10_shp \
  HydroLAKES_polys_v10_shp

modal volume put ccm-data \
  /absolute/path/to/source_data/hybas_na_lev12_v1c \
  hybas_na_lev12_v1c

modal volume ls ccm-data
```

## Run the Analysis

Run all commands from the repository root.

### 1. Lake panels

Download and cache ERA5-Land, then build the ten monthly panels:

```bash
modal run code/00_data_generation/modal_build_lake_panels.py::fetch_only
modal run code/00_data_generation/modal_build_lake_panels.py::process_only
```

Each lake should finish with `<lake>: OK`.

### 2. Embedding parameters

```bash
modal run code/00_data_generation/modal_compute_embedding_params.py
```

### 3. Within-lake CCM

```bash
modal run --detach code/02_within_lake_ccm/modal_within_lake_ccm.py

modal run code/02_within_lake_ccm/modal_within_lake_ccm.py --status

modal run code/02_within_lake_ccm/modal_within_lake_ccm.py --merge-only
```

Run `--merge-only` after the status reaches 420 completed edges.

### 4. Between-lake CCM

```bash
modal run --detach code/03_inter_lake_ccm/modal_inter_lake_ccm.py

modal run code/03_inter_lake_ccm/modal_inter_lake_ccm.py --status

modal run code/03_inter_lake_ccm/modal_inter_lake_ccm.py --merge-only
```

Run `--merge-only` after the status reaches 90 completed edges.

### 5. Forecasting

```bash
modal run --detach \
  code/04_forecast/modal_forecast_synchrony_filtered.py::detached
```

This submits the forecasting orchestration to Modal. It continues remotely if
the local terminal is closed.

## Interrupted Runs

The CCM and forecasting stages save successful tasks as separate shards. To
resume an interrupted run, repeat the same command in the same Modal
environment and Volume. Completed shards are skipped and missing tasks are
submitted again.

Modal may occasionally report errors caused by resource allocation, including
`Container terminated due to preemption`. This means that Modal stopped a
container while reallocating compute resources. It does not indicate an error
in the analysis code. The retry policy restarts the affected task, while
completed shards already written to the Volume are retained.

## Download the Results

Create the local output directories:

```bash
mkdir -p lake_pkls results
```

Download the ten lake panels:

```bash
for lake in \
  Kalamalka_Lake Okanagan_Lake Skaha_Lake Vaseux_Lake Rainy_Lake \
  Lake_of_the_Woods Playgreen_Lake Kiskitto_Lake Sipiwesk_Lake Split_Lake
do
  modal volume get ccm-data \
    "lake_results/${lake}_result.pkl" \
    lake_pkls/
done
```

Download the embedding parameters:

```bash
modal volume get ccm-data \
  lake_results/full_pipeline_v2/embed_params_corrected.json \
  results/
```

Download the CCM and forecasting files:

```bash
for file in \
  ccm_all_edges_merged_fdr.csv \
  connectivity_full_pairwise_ccm_results.csv \
  connectivity_connected_vs_unconnected_summary.csv \
  forecast_synchrony_filtered_full_results.csv \
  forecast_synchrony_filtered_rolling_results.csv \
  forecast_synchrony_filtered_dm_results.csv \
  forecast_synchrony_filtered_selected_lags.csv \
  xgboost_tuning_results.csv
do
  modal volume get ccm-data \
    "lake_results/final_v3/${file}" \
    results/
done
```

## Build and Check the Outputs

After all Modal files have been downloaded:

```bash
python code/05_postprocess_local/build_outputs.py
python code/99_check_outputs/check_modal_outputs.py
```

The first command writes:

```text
ch4_tables/        chapter tables and intermediate summaries
results/figures/   Figures 3.1, 4.1–4.4 and B1
appendices/        Appendix Tables A1–A2, B1–B5 and C1–C4
dataset/           derived monthly datasets and data dictionary
```

The checker validates the ten lake panels, embedding coverage, 420 within-lake
edges, 90 between-lake edges, forecasting outputs and report files. A complete
run ends with:

```text
Output checks passed.
```

The files in `reference/` may be used for additional numerical comparison.
Forecast row counts are not treated as fixed structural requirements because
model availability can vary when the remote forecasting environment is
rebuilt.

## Runtime Notes

The complete analysis may take several hours. Runtime depends on Modal account
limits, queueing and container preemption. The 420-edge within-lake CCM stage is
usually the longest.

The data generation and CCM images use Python 3.11 with `pandas==2.2.2`,
`numpy==1.26.4` and `pyEDM==2.4.0`. The forecasting image uses Python 3.12 and
`xgboost==3.4.1`. Exact `statsmodels` and `pmdarima` versions were not preserved
for the original run, so later SARIMA and SARIMAX image rebuilds may produce
small numerical differences while retaining the same workflow and output
structure.
