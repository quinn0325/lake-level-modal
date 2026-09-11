# Code Layout

Directory order follows the single official reproduction route.

| Directory | Role |
| --- | --- |
| `00_data_generation/` | source-data functions plus Modal entry points for panel construction and embedding selection |
| `01_analysis_core/` | shared scientific preprocessing, CCM and forecasting algorithms; imported as a library and not run directly |
| `02_within_lake_ccm/` | executable Modal stage for within-lake CCM, with one JSON shard per directed edge |
| `03_inter_lake_ccm/` | executable Modal stage for between-lake CCM, with one JSON shard per directed edge |
| `04_forecast/` | executable Modal stage for the forecasting experiment |
| `05_postprocess_local/` | the only local post-processing entry point |
| `06_figures/` | deterministic local figure rendering from downloaded Modal outputs |
| `07_tables/` | deterministic local table and appendix rendering from downloaded Modal outputs |
| `08_dataset/` | deterministic local public dataset builder from downloaded Modal lake PKLs |
| `99_check_outputs/` | output checks; does not rerun CCM or forecasting |

There is deliberately no local CCM or local forecasting driver in this
reproduction package. Files prefixed with `modal_` define the Modal execution
stages: they configure this experiment, call the shared algorithms and save or
merge the results. `01_analysis_core/analysis_core.py` is their shared algorithm
library. These files are complementary parts of one route, not separate local
and cloud implementations. Stage 05 orchestrates the local scripts in stages
06–08, which only render products from downloaded results.
