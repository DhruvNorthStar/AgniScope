"""Bounded raster inspection and original-tile point sampling; no mosaic."""

import csv
from contextlib import ExitStack

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from .config import workspace_path


def raster_report(config: dict, state: gpd.GeoDataFrame, districts: gpd.GeoDataFrame) -> dict:
    paths = config["paths"]
    legend = list(csv.DictReader(workspace_path(paths["worldcover_legend"]).open(encoding="utf-8-sig")))
    expected = {int(row["class_code"]): row["class_name"] for row in legend}
    if len(expected) != 11 or expected.get(10) != "Tree cover":
        raise ValueError("Unexpected WorldCover class legend.")
    raw_paths = sorted(workspace_path(paths["worldcover_raw"]).glob("*.tif"))
    if len(raw_paths) != 5:
        raise ValueError("Expected five original WorldCover tiles.")
    records, bounds = [], []
    with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_CACHEMAX=config["landcover"]["gdal_cache_mb"] * 1024**2):
        for path in raw_paths:
            with rasterio.open(path, "r") as ds:
                tags = ds.tags()
                embedded = {int(line.split(maxsplit=1)[0]): line.split(maxsplit=1)[1].strip() for line in tags.get("legend", "").splitlines() if line.strip()}
                if embedded != expected or ds.crs.to_epsg() != 4326 or ds.count != 1:
                    raise ValueError(f"Unexpected raw WorldCover metadata: {path.name}")
                sample = ds.read(1, out_shape=(64, 64), resampling=Resampling.nearest)
                if not set(np.unique(sample)).issubset(set(expected) | {0}):
                    raise ValueError("Unexpected sampled land-cover code.")
                records.append({"file": path.name, "crs": str(ds.crs), "shape": [ds.height, ds.width], "resolution": list(ds.res), "bounds": list(ds.bounds), "dtype": ds.dtypes[0], "nodata": ds.nodata, "legend_verified": True, "sample_unique_classes": np.unique(sample).tolist(), "tags": tags})
                bounds.append(box(*ds.bounds))
        state_ll = state.to_crs(4326).geometry.union_all()
        districts_ll = districts.to_crs(4326).geometry.union_all()
        coverage = unary_union(bounds)
        with rasterio.open(workspace_path(paths["worldcover_processed_reference"]), "r") as ds:
            size = config["landcover"]["metadata_sample_grid"]
            sample = ds.read(1, out_shape=(size, size), resampling=Resampling.nearest)
            transform = ds.transform * Affine.scale(ds.width / size, ds.height / size)
            outline = state.to_crs(ds.crs).geometry.union_all()
            inside = geometry_mask([mapping(outline)], (size, size), transform, invert=True)
            valid = sample != ds.nodata
            missing = districts.to_crs(ds.crs).geometry.union_all().difference(box(*ds.bounds)).area / 1e6
            processed = {"crs": str(ds.crs), "shape": [ds.height, ds.width], "resolution": list(ds.res), "bounds": list(ds.bounds), "nodata": ds.nodata, "tags": ds.tags(), "diagnostic_grid_shape": [size, size], "valid_sample_pixels": int(valid.sum()), "outside_state_fraction_of_valid_sample": float((valid & ~inside).sum() / valid.sum()) if valid.any() else None, "diagnostic_is_full_resolution_scan": False, "district_footprint_outside_extent_km2": missing}
    return {"legend": expected, "raw_tiles": records, "raw_coverage_contains_state": bool(coverage.covers(state_ll)), "raw_coverage_contains_district_union": bool(coverage.covers(districts_ll)), "processed_reference": processed, "primary_method": "sample original geographic tiles at detection centres; no raster mosaic", "tree_cover_role": "context and sensitivity, not ground truth"}


def sample_landcover(points: gpd.GeoDataFrame, config: dict) -> pd.DataFrame:
    if points.crs is None:
        raise ValueError("Point CRS is required.")
    if points.geometry.isna().any() or points.geometry.is_empty.any() or not points.geom_type.eq("Point").all():
        raise ValueError("Expected nonempty points.")
    geographic = points.to_crs(4326)
    result = pd.DataFrame({"agniscope_worldcover_2021_class": pd.array([pd.NA] * len(points), dtype="Int64"), "agniscope_landcover_status": ["outside_tiles"] * len(points), "agniscope_landcover_tile": [""] * len(points)}, index=points.index)
    with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_CACHEMAX=config["landcover"]["gdal_cache_mb"] * 1024**2):
        for path in sorted(workspace_path(config["paths"]["worldcover_raw"]).glob("*.tif")):
            with rasterio.open(path, "r") as ds:
                if ds.crs is None or ds.crs.to_epsg() != 4326:
                    raise ValueError("Raw tile sampling requires documented EPSG:4326 tiles.")
                # Raster indexing is left-inclusive, right-exclusive, top-inclusive, bottom-exclusive.
                positions = [i for i, p in enumerate(geographic.geometry) if result.iloc[i]["agniscope_landcover_status"] == "outside_tiles" and ds.bounds.left <= p.x < ds.bounds.right and ds.bounds.bottom < p.y <= ds.bounds.top]
                coords = [(geographic.geometry.iloc[i].x, geographic.geometry.iloc[i].y) for i in positions]
                for pos, value in zip(positions, ds.sample(coords, indexes=1, masked=True)):
                    index = result.index[pos]
                    result.at[index, "agniscope_landcover_tile"] = path.name
                    if np.ma.is_masked(value[0]) or int(value[0]) == config["landcover"]["nodata"]:
                        result.at[index, "agniscope_landcover_status"] = "nodata"
                    else:
                        result.at[index, "agniscope_worldcover_2021_class"] = int(value[0])
                        result.at[index, "agniscope_landcover_status"] = "sampled"
    return result
