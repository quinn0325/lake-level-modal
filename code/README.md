# Code Layout

Directory order follows the single official reproduction route.

| Directory | Role |
| --- | --- |
| `00_data_generation/` | Modal data acquisition, lake screening, panel construction and corrected embedding parameters |
| `01_shared/` | shared modelling library imported by Modal jobs and post-processing scripts |
| `02_within_lake_ccm/` | Modal within-lake CCM with one JSON shard per directed edge |
| `03_inter_lake_ccm/` | Modal between-lake CCM with one JSON shard per directed edge |
| `04_forecast/` | Modal forecasting with server-side orchestration |
| `06_figures/` | deterministic local figure rendering from downloaded Modal outputs |
| `06_postprocess_local/` | the only local post-processing entry point |
| `07_tables/` | deterministic local table and appendix rendering from downloaded Modal outputs |
| `08_dataset/` | deterministic local public dataset builder from downloaded Modal lake PKLs |
| `99_check_outputs/` | output checks; does not rerun CCM or forecasting |

There is deliberately no local CCM or local forecasting driver in this
reproduction package.
