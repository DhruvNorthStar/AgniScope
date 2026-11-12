# Land-cover methodology — Phase 0

## Source and role

Use the five original **ESA WorldCover 2021 v200** tiles as contextual land-cover evidence. Each is a single-band UInt8 EPSG:4326 GeoTIFF, 36,000 × 36,000 cells, with nodata code 0 and nominal 10 m product resolution. Geographic spacing is 1/12,000 degree; ground spacing varies with latitude. Their combined extents cover the entire primary district-union footprint and the secondary state reference.

The CSV legend matches all five embedded TIFF legends: 10 tree cover, 20 shrubland, 30 grassland, 40 cropland, 50 built-up, 60 bare/sparse vegetation, 70 snow and ice, 80 permanent water bodies, 90 herbaceous wetland, 95 mangroves, 100 moss and lichen. **Class 10 is tree cover**, not verified forest tenure, fuel condition or proof of a forest fire.

The source product describes 2021. Applying it to 2019–2025 does not reconstruct annual land-cover change. Forest/tree-cover selection must be a later sensitivity comparison against the unfiltered detection baseline.

## Existing processed raster

The immutable processed reference uses EPSG:32644, 30 m cells, 11,123 columns × 10,135 rows, nodata 0, and bounds `(170670, 3177720, 504360, 3481770)` metres. It contains valid pixels outside the state polygon and excludes **7.809559 km²** of the primary district footprint.

The initial audit's full scan found approximately 47.03% of valid pixels outside the state reference. Phase 0 independently used a bounded **512 × 512 nearest-neighbour diagnostic sample**, obtaining approximately 47.01%. This is a sampled recheck, not a second full-resolution census. The original 30 m resampling recipe is not present locally. Do not use this reference as a ready-made state mask.

## Implemented extraction

`sample_landcover` transforms detection-centre points to EPSG:4326 and samples the appropriate original tile. Tiles are opened read-only, sequentially, with a 32 MiB GDAL cache and persistent auxiliary metadata disabled. There is no mosaic, continuous interpolation, full-tile array load or source rewrite.

Store the extracted class, source tile and extraction status alongside the original observation. Nodata stays unknown; a point outside all tiles is explicitly marked `outside_tiles`. Neither case is treated as nonforest or silently dropped. Exact tile-edge indexing uses left-inclusive/right-exclusive and top-inclusive/bottom-exclusive cells, consistent with raster row/column indexing. Tests include categorical classes, nodata and actual representative points from all thirteen local districts.

A centre-pixel class is contextual and can differ from land cover across a VIIRS footprint, whose scan/track dimensions vary. No subpixel forest fraction or exact burned area is inferred. A footprint-based land-cover comparison can be considered later if it is scientifically warranted.

## Reproducible masking recipe, if a raster output becomes necessary

1. Select the primary district-union polygon and preserve its version/hash.
2. Transform that polygon into each source tile's CRS before deriving intersecting windows.
3. Read bounded windows only. Apply a polygon mask with cell-centre inclusion (`all_touched=False`) and an explicit nodata value of 0. Report the edge rule and preserve metadata.
4. Keep native categorical codes. If reprojection or a 30 m display product is needed, use **nearest-neighbour** resampling; use a mode aggregation only as a separately documented analytical choice. Never use bilinear or cubic interpolation for class IDs.
5. Use one explicit destination grid and transform the polygon to that grid for the final mask. Retain cells based on the stated cell-centre rule, recording nodata and edge effects.
6. Write only to a new derived path outside the immutable source folder. Validate bounds, CRS, class codes, nodata and source hashes afterwards.

This masking recipe is documented, not a claim that a new raster was generated. Point sampling is adequate for the current detection-centre contextual analysis, so no new large raster is created.

## Attribution

WorldCover is CC BY 4.0. Retain dataset citation: Zanaga et al., **ESA WorldCover 10 m 2021 v200**, DOI [10.5281/zenodo.7254221](https://doi.org/10.5281/zenodo.7254221). For published maps, retain ESA's requested attribution: © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium. See [ESA data access and terms](https://esa-worldcover.org/en/data-access).
