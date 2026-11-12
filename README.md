# AgniScope

AgniScope is a research codebase for studying NASA VIIRS thermal detections, seasonal spatial clustering, and fixed-grid recurrence in Uttarakhand, India. The repository contains analysis code and methods documentation. It does not include source observations, derived scientific datasets, model fits, GIS projects, or dashboard data.

## Quick start (code and tests)

The exact Conda lock in this repository is for Windows x64. On Windows, install Miniforge/Conda, then create the staging-local environment and install the scientific package and hashed dashboard dependencies:

```powershell
.\scripts\setup_environment.ps1
conda run -n agniscope-public-source python -B -m pytest -p no:cacheprovider --basetemp=.cache/pytest -q tests/test_acquisition.py tests/test_dbscan.py tests/test_hdbscan.py tests/test_spatial.py tests/test_recurrence.py tests/test_planning.py tests/test_landcover.py -k "not actual_local_boundaries and not original_tiles_sample_actual_district_representative_points"
```

`environment.lock.win-64.txt` locks the scientific runtime; `dashboard/requirements.lock.txt` is the separate hashed dashboard dependency lock; and `dashboard/constraints-scientific.txt` records scientific versions that must remain compatible. The focused test command uses synthetic fixtures and does not download data. Two tests are excluded because they require the omitted real boundary and WorldCover inputs. The full legacy suite additionally requires omitted annual products, dashboard data, and sealed phase manifests. The exact lock is Windows-specific and historical; it is not a promise of cross-platform GIS compatibility. `environment.yml` is provided as a Conda recipe for environments where the explicit Windows lock is not applicable.

The dashboard source is included; scientific datasets and generated outputs are excluded. To run it with study results, first prepare the complete observation Parquet, annual/monthly/district and coverage summaries, seasonal summaries, saved DBSCAN/HDBSCAN assignments/statistics/comparisons, recurrence grids/intersections/sensitivity summaries, district boundaries, and optional figures. Bind every file in `configs/dashboard.toml` and `dashboard/catalog.json`, including matching hashes, sizes, and a verified catalogue status. The dashboard checks these bindings at startup and on data access; it cannot run with the intentionally empty catalogue shipped here. Raw data or arbitrary CSVs are not dashboard substitutes.

After preparing and verifying that complete local bundle, launch with `conda run -n agniscope-public-source streamlit run dashboard/app.py`. This copy intentionally shows an unverified-catalogue state until those data bindings are supplied.

The inherited dashboard monitor stops below 200 MiB available system RAM, warns below 512 MiB, and treats 256 MiB as critically low. The 200 MiB threshold is a low-memory operating condition, not a promise of stability; Windows can page, slow down, freeze, or fail between monitor checks. Close other applications and review `configs/dashboard_resources.toml` before use.

## Data and reproduction

Read [data sources, access and reproduction](docs/data_sources.md) before collecting any observations. The study design uses NASA FIRMS `VIIRS_SNPP_SP` Standard Processing, a declared Uttarakhand study footprint, and static ESA WorldCover 2021 context. Obtain inputs directly from their authoritative sources, retain source terms and provenance, and review completeness, product processing lineage, and geography before analysis. Never commit credentials or downloaded inputs.

The analysis pipeline is staged: validate/licence boundary inputs, acquire appropriately bounded observations, preserve raw bytes and provenance, run row-level QC, construct analysis-ready data, then perform clustering and recurrence only after the scientific inputs and coverage have been reviewed. The scripts reflect a multi-stage study workflow and require intermediate products from earlier stages; they are not a one-command public-data downloader. See the methods documents in `docs/` for definitions, assumptions, and limitations.

The complete data-preparation and catalogue-building workflow is not included as a ready-to-run public-data recipe in this source-only release. Do not claim a populated dashboard is reproducible from this repository alone.

## Interpretation limits

Satellite thermal detections are not independent, confirmed forest-fire incidents. Successful data retrieval does not prove complete satellite observation opportunity. Recurring grid support is not burned area, susceptibility, or a forecast of future fire risk. Clusters depend on the input population and chosen parameters; agreement between methods is not accuracy against independent ground truth. Boundary vintage, geolocation, land-cover date, cloud, overpass timing, and upstream processing affect interpretation. See [spatial methodology](docs/spatial_methodology.md) and [land-cover methodology](docs/landcover_methodology.md).

## Attribution and licensing

Source attribution and data-specific terms are summarized in [docs/data_sources.md](docs/data_sources.md). In brief, NASA FIRMS data require appropriate acknowledgement and citation; the cited geoBoundaries district source is stated as ODbL 1.0; ESA WorldCover 2021 v200 is stated as CC BY 4.0. The software is licensed under MIT; these data terms remain separate. Do not infer permission to redistribute omitted inputs or derived datasets from this code repository.

## Repository contents

- `src/agniscope/` — data validation, acquisition, spatial, clustering, and recurrence modules.
- `scripts/` — staged analysis and dashboard launch utilities.
- `tests/` — regression tests, including synthetic fixtures.
- `dashboard/` — six-page Streamlit application source and dependency locks; results data are omitted.
- `configs/` — scientific workflow settings and dashboard/resource config. Dashboard source bindings must be added for a verified local data bundle.
- `docs/` — scientific methodology, source notes, and study protocol.

## Security

Use local environment variables for credentials when an authorized acquisition requires them. `.env` files and data folders are ignored by Git. Never commit API keys, credential-bearing URLs, raw downloads, generated outputs, local logs, or machine-specific paths.
