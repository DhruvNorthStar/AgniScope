# Spatial methodology — Phase 0

## Decision

Use the union of the thirteen supplied geoBoundaries **2021 district polygons** as the primary analytical footprint. Keep the supplied geoBoundaries **2011 state polygon** unchanged as a secondary reference. This choice supports district-complete statewide summaries using one fixed geography throughout the proposed 2019–2025 series. It does not assert that the district union is an official or more accurately surveyed state border.

No original boundary is repaired, snapped, rewritten or replaced. The derived union is saved separately at `data/processed/analysis_footprint.gpkg`, layer `analysis_footprint`, CRS EPSG:32644. Its database provenance and licence are **ODbL 1.0**, inherited from the district source.

## Measured disagreement

The executed spatial validation reproduced these planar measurements in EPSG:32644:

| Quantity | km² |
|---|---:|
| State reference area | 53,417.947780 |
| Primary district-union area | 53,232.286997 |
| State reference outside district union | 404.400677 |
| District union outside state reference | 218.739894 |
| Symmetric difference | 623.140572 |
| Positive-area overlaps between districts | 0 |

All fourteen source polygons are nonempty and valid. The discrepancy is not explained by a CRS mismatch: both source files already use EPSG:32644. Different upstream sources, representation dates and delineations are documented; the exact contribution of each cause has not been established. Valid individual geometries do not imply identical external outlines.

The primary footprint bounds in WGS84 are approximately 77.57563–81.04568° E and 28.70948–31.45230° N. The configured acquisition box fully encloses it. Areas above are computed diagnostics, not official administrative area figures. Uttarakhand extends slightly west of UTM zone 44's nominal western edge; scale variation should be compared with geodesic distances before final clustering parameters are fixed.

## Assignment and edge rules

1. Retain the original FIRMS coordinate fields. Construct WGS84 points and transform coordinates into the analysis CRS with explicit CRS metadata.
2. Include points covered by the primary footprint, including its exterior boundary. Apply no additional state-reference clip.
3. A point covered by exactly one district gets that district's unchanged original label. Record whether it is in the interior or on a single district edge.
4. A point covered by multiple districts is retained statewide but marked `ambiguous_shared_edge`; store all candidate original labels and leave the single district field null.
5. Any point covered by the union but not matched to a district is retained in an `unassigned_inside` QA bucket and requires investigation. It is not reassigned to the nearest district.
6. Points outside the footprint remain in the audit ledger as a scientific geographical exclusion, distinct from invalid input.
7. No rounding tolerance, snapping or nearest-neighbour district assignment is used. Floating-point coordinate transformations can affect mathematically exact edge cases. A separate 375 m near-outer-edge flag is diagnostic only; actual VIIRS scan/track sizes and boundary uncertainty remain relevant.

For every accepted dataset:

**statewide retained rows = uniquely assigned district rows + ambiguous/unassigned rows.**

District summaries must show the unassigned/ambiguous bucket, even when it is zero. Future area-normalized measures must use these same district geometries, rather than mixing state-reference and district-union areas. Observation counts are detections, not event counts.

## Names and provenance

Preserve `district` and `district_original` unchanged. Separately configured display aliases are `Garhwal → Pauri Garhwal` and `Hardwar → Haridwar`. The current display names were checked against the [Government of Uttarakhand district portal](https://tcp.uk.gov.in/district-portal/) and [Garhwal division website](https://garhwal.uk.gov.in/). This verifies naming conventions, not the accuracy of the copied geometry.

The source state boundary represents **2011**, CC BY 2.5 India, DataMeet/Election Commission of India; the source districts represent **2021**, ODbL 1.0, Pathways Data/lgdirectory.gov.in. Current official geoBoundaries records match the upstream identifiers. Source derivations should be independently reviewed against the upstream geometry files before making new claims.

The two input boundaries represent different vintages: the state reference is 2011 and the district source is 2021. This repository copy omits the machine-readable validation results and source geometries; regenerate them from reviewed, licensed upstream inputs before analysis.
