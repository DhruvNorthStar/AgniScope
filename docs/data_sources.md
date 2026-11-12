# Data sources, acquisition and reuse

This repository copy contains no empirical data, GIS handoff, processed datasets, model assignments, maps, or dashboard results. Source names and terms below are those recorded by the project documentation. Verify current upstream metadata, citations, and terms before acquisition or redistribution.

| Source | Intended role | Recorded attribution or terms | Review needed |
|---|---|---|---|
| NASA FIRMS VIIRS S-NPP 375 m Standard Processing (`VIIRS_SNPP_SP`) | Thermal detections | NASA/FIRMS acknowledgement and product-specific citation | Confirm current API access, product lineage, use terms, and date completeness. |
| geoBoundaries Uttarakhand ADM2, 2021 vintage | Primary district footprint | Pathways Data / lgdirectory.gov.in; ODbL 1.0 | Preserve notices; review applicable adapted-database share-alike duties. |
| geoBoundaries Uttarakhand ADM1, 2011 vintage | Secondary state reference | DataMeet / Election Commission of India; CC BY 2.5 India | Preserve attribution and source notice. |
| ESA WorldCover 2021 v200 | Static land-cover context | CC BY 4.0 | Attribute product; do not treat as a time-varying vegetation history. |

## Reproduction workflow

1. Review the current official NASA FIRMS source and document the product, date range, retrieval time, extent, and processing lineage. Do not silently replace Standard Processing with another satellite or processing stream.
2. Obtain boundary and land-cover data from authoritative sources. Record URLs, retrieval dates, hashes, metadata, licences, CRS, and transformations.
3. Keep credentials in a local environment variable such as `FIRMS_MAP_KEY`. Never commit `.env`, keys, or credential-bearing URLs.
4. Preserve downloaded source bytes. Validate schemas, coordinates, dates, source/platform, duplicates, study-footprint membership, row accounting, and observation coverage before analysis.
5. Generate derived tables, model results, and dashboard catalogue as separate artifacts. Review both source terms and derived database obligations before sharing any data or results.

The included acquisition code implements bounded, historically authorized workflows. It is not a general-purpose bulk downloader; inspect its scope and limits before use. Tests use synthetic fixtures and do not acquire data.

## Interpretation

Thermal detections are observations, not confirmed independent incidents. Successful retrieval does not prove complete satellite observation opportunity. No detection is not evidence of no fire. Recurrence depends on resolution, grid alignment, thresholds, and observation opportunity; it is not burned area or a forecast. Clustering agreement is not validation against independent ground truth. Static WorldCover context is not a universal forest-fire label.

Consult [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/), [NASA Earthdata data-use guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy), [geoBoundaries](https://www.geoboundaries.org/), the [ODbL summary](https://opendatacommons.org/licenses/odbl/summary/), and [ESA WorldCover](https://worldcover2021.esa.int/) for current information. This code is under the MIT License; data terms do not license or replace the software licence.
