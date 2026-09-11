# Lake Level CCM Modal Reproduction

This repository provides the code, execution instructions and reference outputs
for the MSc dissertation *Causal Exploration and Predictability of Lake Water
Level: Evidence from Two Regulated Canadian River Basins*.

This README is the technical guide for environment setup, input preparation,
execution, output download and verification.

## Reproduction Workflow

```text
Prepare HydroLAKES, HydroBASINS and CDS credentials
                         ↓
Build 10 monthly lake panels on Modal
                         ↓
Compute embedding parameters on Modal
                         ↓
Run 420 within-lake CCM edges on Modal
                         ↓
Run 90 between-lake CCM edges on Modal
                         ↓
Run forecasting on Modal
                         ↓
Download the Modal outputs
                         ↓
Build tables, figures, appendices and the public dataset locally
                         ↓
Check output completeness
```

Data generation, embedding selection, CCM and forecasting run on Modal. Local
scripts only transform the downloaded outputs into the final dissertation
materials.

## Repository Layout

```text
code/
  00_data_generation/       data acquisition and lake-panel construction
  01_analysis_core/         shared scientific algorithms; not run directly
  02_within_lake_ccm/       within-lake CCM Modal entry point
  03_inter_lake_ccm/        between-lake CCM Modal entry point
  04_forecast/              forecasting Modal entry point
  05_postprocess_local/     single local post-processing entry point
  06_figures/               local figure builders
  07_tables/                local table and appendix builders
  08_dataset/               local public-dataset builder
  99_check_outputs/         output-completeness checker

reference/
  results/                  dissertation analysis outputs
  lake_pkls/                dissertation lake panels
  figures/                  dissertation figures and map cache layers
  tables/                   dissertation main-text table
  appendices/               dissertation appendix tables
  dataset/                  dissertation public dataset
```

The committed `reference/` directory contains the outputs used in the
dissertation and provides a numerical baseline for a new full reproduction.

## Local Setup

Clone the repository:

```bash
git clone https://github.com/quinn0325/lake-level-modal.git
cd lake-level-modal
```

Create a Python 3.11 environment and install the local dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Authenticate the Modal command-line client:

```bash
modal setup
```

All commands below assume the current Modal environment. If a named environment
is used, specify the same `--env` on every `modal run`, `modal volume` and
`modal secret` command.

## Source Data

Download and extract the two required geospatial products:

| Dataset | Version | Download | Use |
| --- | --- | --- | --- |
| HydroLAKES | v1.0 | [`HydroLAKES_polys_v10_shp.zip`](https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip) | lake polygons and lake-grid masks |
| HydroBASINS | North America level 12, v1c | [`hybas_na_lev12_v1c.zip`](https://data.hydrosheds.org/file/hydrobasins/standard/hybas_na_lev12_v1c.zip) | upstream-basin masks |

Keep every shapefile component (`.shp`, `.dbf`, `.shx`, `.prj` and associated
files) in the same directory:

```text
source_data/
  HydroLAKES_polys_v10_shp/
    .../
      HydroLAKES_polys_v10.shp
      HydroLAKES_polys_v10.dbf
      HydroLAKES_polys_v10.shx
      HydroLAKES_polys_v10.prj
      ...

  hybas_na_lev12_v1c/
    hybas_na_lev12_v1c.shp
    hybas_na_lev12_v1c.dbf
    hybas_na_lev12_v1c.shx
    hybas_na_lev12_v1c.prj
    ...
```

HydroLAKES may contain a nested shapefile directory. The HydroBASINS `.shp`
file must be directly inside `hybas_na_lev12_v1c/`.

Monthly water-level and regulated-flow observations are retrieved directly from
the Water Survey of Canada historical hydrometric service. A local HYDAT
database is not required.

ERA5-Land monthly data are retrieved through the Copernicus CDS API. Before
running the data stage:

1. create a [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
   account;
2. accept the licence for the
   [ERA5-Land monthly averaged dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means);
3. obtain a personal CDS API access token.

The ten lake and station assignments are fixed in the code. Their identities
and observed-data coverage are recorded in:

```text
reference/appendices/A1_lakes_and_stations.csv
reference/appendices/B2_data_availability.csv
```

## Modal Setup

Create a new, empty Volume named `ccm-data`:

```bash
modal volume create ccm-data
```

Do not mix shards produced by different code versions or analysis parameters in
the same Volume.

Create the CDS API secret:

```bash
modal secret create cds-api \
  CDSAPI_URL=https://cds.climate.copernicus.eu/api \
  CDSAPI_KEY=YOUR_PERSONAL_ACCESS_TOKEN
```

Upload HydroLAKES:

```bash
modal volume put ccm-data \
  /absolute/path/to/source_data/HydroLAKES_polys_v10_shp \
  HydroLAKES_polys_v10_shp
```

Upload HydroBASINS:

```bash
modal volume put ccm-data \
  /absolute/path/to/source_data/hybas_na_lev12_v1c \
  hybas_na_lev12_v1c
```

Inspect the uploaded files:

```bash
modal volume ls ccm-data
```

## Reproduction Route

Run every command from the repository root.

### 1. Build Monthly Lake Panels

Download and cache ERA5-Land data:

```bash
modal run \
  code/00_data_generation/modal_build_lake_panels.py::fetch_only
```

Build all ten monthly lake panels:

```bash
modal run \
  code/00_data_generation/modal_build_lake_panels.py::process_only
```

Expected Volume outputs:

```text
lake_results/<lake>_result.pkl
era5_downloads/<lake>_era5land_monthly.nc
```

Every lake should finish with `<lake>: OK`.

### 2. Compute Embedding Parameters

```bash
modal run \
  code/00_data_generation/modal_compute_embedding_params.py
```

Expected Volume output:

```text
lake_results/full_pipeline_v2/embed_params_corrected.json
```

The JSON file must cover all ten lakes and seven variables.

### 3. Run Within-Lake CCM

Submit the 420 directed edges:

```bash
modal run --detach \
  code/02_within_lake_ccm/modal_within_lake_ccm.py
```

Check progress:

```bash
modal run \
  code/02_within_lake_ccm/modal_within_lake_ccm.py \
  --status
```

The status has the following form:

```text
Total edges: 420 | completed: <count> | remaining: <count>
```

Merge only after all 420 edges have completed:

```bash
modal run \
  code/02_within_lake_ccm/modal_within_lake_ccm.py \
  --merge-only
```

Expected Volume output:

```text
lake_results/final_v3/ccm_all_edges_merged_fdr.csv
```

### 4. Run Between-Lake CCM

Submit the 90 directed edges:

```bash
modal run --detach \
  code/03_inter_lake_ccm/modal_inter_lake_ccm.py
```

Check progress:

```bash
modal run \
  code/03_inter_lake_ccm/modal_inter_lake_ccm.py \
  --status
```

Merge only after all 90 edges have completed:

```bash
modal run \
  code/03_inter_lake_ccm/modal_inter_lake_ccm.py \
  --merge-only
```

Expected Volume outputs:

```text
lake_results/final_v3/connectivity_full_pairwise_ccm_results.csv
lake_results/final_v3/connectivity_connected_vs_unconnected_summary.csv
```

### 5. Run Forecasting

Submit the server-side forecasting orchestration:

```bash
modal run --detach \
  code/04_forecast/modal_forecast_synchrony_filtered.py::detached
```

The command returns after submission. Closing the local terminal does not stop
the server-side Modal job.

Expected Volume outputs:

```text
lake_results/final_v3/forecast_synchrony_filtered_full_results.csv
lake_results/final_v3/forecast_synchrony_filtered_rolling_results.csv
lake_results/final_v3/forecast_synchrony_filtered_dm_results.csv
lake_results/final_v3/forecast_synchrony_filtered_selected_lags.csv
lake_results/final_v3/xgboost_tuning_results.csv
```

## Resuming Interrupted Runs

The within-lake CCM, between-lake CCM and forecasting stages save successful
shards independently. If a run is interrupted by account limits, preemption,
terminal closure or a temporary error, rerun the same command in the same
Modal environment with the same `ccm-data` Volume.

If Modal displays `Container terminated due to preemption`, the container was
interrupted by resource scheduling rather than by an analysis-code error. The
interrupted task is retried according to the configured retry policy, while
successful shards already written to the Modal Volume are retained.

On a resumed run:

- valid completed shards are skipped;
- missing or failed tasks are submitted again; and
- final outputs are merged only after every expected shard passes validation.

## Download Modal Outputs

Create the local runtime directories:

```bash
mkdir -p lake_pkls results
```

Download the ten lake panels:

```bash
for lake in \
  Kalamalka_Lake \
  Okanagan_Lake \
  Skaha_Lake \
  Vaseux_Lake \
  Rainy_Lake \
  Lake_of_the_Woods \
  Playgreen_Lake \
  Kiskitto_Lake \
  Sipiwesk_Lake \
  Split_Lake
do
  modal volume get \
    ccm-data \
    "lake_results/${lake}_result.pkl" \
    lake_pkls/
done
```

Download the embedding parameters:

```bash
modal volume get \
  ccm-data \
  lake_results/full_pipeline_v2/embed_params_corrected.json \
  results/
```

Download the CCM and forecasting outputs:

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
  modal volume get \
    ccm-data \
    "lake_results/final_v3/${file}" \
    results/
done
```

## Build Local Outputs

After downloading the complete Modal outputs, run:

```bash
python code/05_postprocess_local/build_outputs.py
```

This writes:

```text
ch4_tables/        chapter tables and validated intermediate tables
results/figures/   dissertation figures
appendices/        appendix tables
dataset/           public derived dataset and data dictionary
```

The report-aligned outputs are:

- Figures 3.1, 4.1, 4.2, 4.3, 4.4 and B1;
- Table 4.1;
- Appendix Tables A1–A2, B1–B5 and C1–C4; and
- the public derived dataset.

`figure_4_3_interlake_network.py` renders both Figure 4.3 and Figure 4.4.

## Verify Outputs

Run the output checker after local post-processing:

```bash
python code/99_check_outputs/check_modal_outputs.py
```

The checker verifies that:

- all ten lake panels are present;
- the embedding parameters cover ten lakes and seven variables;
- the within-lake CCM output contains the 420 expected unique edges, all with
  `status=OK`;
- the between-lake CCM output contains the 90 expected unique edges, all with
  `status=OK`;
- every forecasting output has the required columns and covers all ten lakes;
- the rolling-forecast and DM outputs cover the 1-, 3-, 6- and 12-month
  horizons;
- the XGBoost tuning output contains six candidates and one selected
  configuration; and
- the expected tables, Figures 3.1, 4.1–4.4 and B1, appendices and dataset files
  were generated.

A successful check ends with:

```text
Output checks passed.
```

The dissertation outputs in `reference/` may be used for additional numerical
comparison. Forecast row counts are not treated as fixed structural
requirements because model availability can vary when remote forecasting
dependencies are rebuilt.

## Runtime and Environment Notes

The full Modal analysis is computationally intensive and may require several
hours. Runtime depends on account concurrency, queueing and container
preemption. The 420-edge within-lake CCM stage is normally the longest stage.

The data-generation and CCM images use Python 3.11 with the principal versions
`pandas==2.2.2`, `numpy==1.26.4` and `pyEDM==2.4.0`. The forecasting image uses
Python 3.12 and `xgboost==3.4.1`. XGBoost is installed remotely and is not
required for local post-processing.

The original Modal run did not preserve exact `statsmodels` and `pmdarima`
versions. Future SARIMA and SARIMAX image rebuilds may therefore produce small
numerical differences, while retaining the same workflow and output structure.
