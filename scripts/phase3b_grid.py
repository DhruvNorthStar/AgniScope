"""Bounded, immutable 2021 district-union grid and district intersection references."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agniscope.provenance import sha256, utc_now
from agniscope.acquisition_store import atomic_json, StoreConflict

TABLE = ROOT / "outputs/tables/phase3b/resume_20261023"
SPATIAL = ROOT / "outputs/spatial/phase3b"


def verify_receipt(directory):
    record = json.loads((directory / "receipt.json").read_text())
    for row in record["files"]:
        assert sha256(ROOT / row["path"]) == row["sha256"]
    return record


def seal(staging, target, specification):
    assert not target.exists()
    files = [{"path": (target / p.relative_to(staging)).relative_to(ROOT).as_posix(),
              "sha256": sha256(p), "size_bytes": p.stat().st_size}
             for p in sorted(staging.rglob("*")) if p.is_file()]
    atomic_json(staging / "receipt.json", {"specification": specification, "files": files,
                "created_utc": utc_now(), "observation_HTTP_calls": 0, "credential_reads": 0, "clustering_refits": 0})
    staging.rename(target)
    return verify_receipt(target)


def geo_metadata(crs):
    return {b"geo": json.dumps({"version": "1.1.0", "primary_column": "geometry", "columns": {
        "geometry": {"encoding": "WKB", "geometry_types": ["Polygon", "MultiPolygon", "GeometryCollection"],
                     "crs": crs.to_json_dict()}}}, sort_keys=True).encode()}


def build(resolution):
    target = SPATIAL / f"grid_{resolution}"
    if target.exists():
        return {"reused": True, "files": len(verify_receipt(target)["files"])}
    assert json.loads((TABLE / "gate_a.json").read_text())["status"] == "passed"
    from phase3b_certificates import require_certificate
    require_certificate(TABLE / "pytest_grid_certified.xml", TABLE / "pytest_grid_certified.certificate.json")
    import tomllib
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shapely
    from agniscope.stage_b_qc import geography
    from agniscope.config import workspace_path
    from agniscope.recurrence import cell_ids
    methods = tomllib.loads((ROOT / "configs/phase3b_methods.toml").read_text())["design"]
    assert resolution in methods["resolutions_m"] and methods["origin_m"] == [0, 0]
    config, districts = geography()
    mask = districts.geometry.union_all()
    shapely.prepare(mask)
    district_geometries = districts.geometry.to_numpy()
    tree = shapely.STRtree(district_geometries)
    extent = methods["extent_m"]
    bounds = mask.bounds
    assert extent[0] <= bounds[0] and extent[1] <= bounds[1] and extent[2] >= bounds[2] and extent[3] >= bounds[3]
    left, bottom, right, top = [int(v // resolution) for v in extent]
    assert all(v % resolution == 0 for v in extent)
    nx, ny = right-left, top-bottom
    staging = target.with_name(target.name + ".staged")
    if staging.exists():
        raise StoreConflict("Uncommitted grid staging is preserved; refusing overwrite.")
    staging.mkdir(parents=True)
    grid_schema = pa.schema([("cell_id", pa.int64()), ("grid_x", pa.int32()), ("grid_y", pa.int32()),
                            ("clipped_area_m2", pa.float64()), ("footprint_fraction", pa.float64()),
                            ("boundary_class", pa.string()), ("geometry", pa.binary())], metadata=geo_metadata(districts.crs))
    part_schema = pa.schema([("cell_id", pa.int64()), ("district_original", pa.string()), ("district_display", pa.string()),
                            ("intersection_area_m2", pa.float64()), ("cell_area_fraction", pa.float64())])
    area_chunks = []
    district_totals = np.zeros(len(districts), dtype=float)
    eligible = full_count = partial_count = boundary_only = 0
    with pq.ParquetWriter(staging / "grid.parquet", grid_schema, compression="zstd") as writer, \
            pq.ParquetWriter(staging / "district_intersections.parquet", part_schema, compression="zstd") as part_writer:
        for start in range(0, nx*ny, 1024):
            ordinal = np.arange(start, min(start+1024, nx*ny), dtype="int64")
            ix, iy = ordinal % nx + left, ordinal // nx + bottom
            boxes = shapely.box(ix*resolution, iy*resolution, (ix+1)*resolution, (iy+1)*resolution)
            intersects = shapely.intersects(mask, boxes)
            if not intersects.any():
                continue
            ix, iy, boxes = ix[intersects], iy[intersects], boxes[intersects]
            full = shapely.covers(mask, boxes)
            clipped = boxes.copy()
            clipped[~full] = shapely.intersection(boxes[~full], mask)
            areas = shapely.area(clipped)
            boundary_only += int((areas == 0).sum())
            positive = areas > 0
            ix, iy, clipped, areas, full = ix[positive], iy[positive], clipped[positive], areas[positive], full[positive]
            if not len(areas):
                continue
            ids = cell_ids(ix, iy, resolution)
            eligible += len(ids)
            full_count += int(full.sum())
            partial_count += int((~full).sum())
            area_chunks.append(math.fsum(areas))
            assert shapely.is_valid(clipped).all() and (areas <= resolution**2+1e-6).all()
            table = pa.Table.from_arrays([pa.array(ids, type=pa.int64()), pa.array(ix, type=pa.int32()), pa.array(iy, type=pa.int32()),
                    pa.array(areas), pa.array(areas/resolution**2), pa.array(np.where(full, "full", "partial")),
                    pa.array(shapely.to_wkb(clipped), type=pa.binary())], schema=grid_schema)
            writer.write_table(table, row_group_size=1024)
            pairs = tree.query(clipped, predicate="intersects")
            counts = np.bincount(pairs[0], minlength=len(clipped))
            part_areas = areas[pairs[0]].copy()
            split = counts[pairs[0]] > 1
            part_areas[split] = shapely.area(shapely.intersection(clipped[pairs[0, split]], district_geometries[pairs[1, split]]))
            keep = part_areas > 0
            point_index, district_index, part_areas = pairs[0, keep], pairs[1, keep], part_areas[keep]
            # Sort deterministically by numeric cell and original district label.
            labels = districts.district_original.to_numpy()[district_index]
            order = np.lexsort((labels, ids[point_index]))
            point_index, district_index, part_areas = point_index[order], district_index[order], part_areas[order]
            accounting = np.bincount(point_index, weights=part_areas, minlength=len(areas))
            assert np.allclose(accounting, areas, rtol=1e-11, atol=.001)
            district_totals += np.bincount(district_index, weights=part_areas, minlength=len(districts))
            parts = pa.Table.from_arrays([pa.array(ids[point_index]),
                pa.array(districts.district_original.to_numpy()[district_index]),
                pa.array(districts.agniscope_display_name.to_numpy()[district_index]),
                pa.array(part_areas), pa.array(part_areas/areas[point_index])], schema=part_schema)
            part_writer.write_table(parts, row_group_size=1024)
    total_area = math.fsum(area_chunks)
    assert math.isclose(total_area, mask.area, rel_tol=1e-11, abs_tol=.001)
    assert np.allclose(district_totals, districts.area.to_numpy(), rtol=1e-11, atol=.001)
    metadata = {"resolution_m": resolution, "origin_m": [0, 0], "extent_m": extent, "CRS": "EPSG:32644",
                "footprint": "2021_district_union", "footprint_file": config["paths"]["analysis_footprint"],
                "footprint_file_sha256": sha256(workspace_path(config["paths"]["analysis_footprint"])),
                "mask_normalized_WKB_sha256": hashlib.sha256(shapely.to_wkb(shapely.normalize(mask))).hexdigest(),
                "bounding_grid_cells": nx*ny, "eligible_positive_area_cells": eligible, "full_cells": full_count,
                "partial_cells": partial_count, "boundary_touch_only_cells_excluded": boundary_only,
                "outside_or_zero_area_cells": nx*ny-eligible, "clipped_area_m2": total_area, "mask_area_m2": mask.area,
                "area_difference_m2": total_area-mask.area, "nominal_full_cell_area_m2": resolution**2,
                "district_areas": [{"district_original": str(row.district_original), "district_display": str(row.agniscope_display_name),
                                    "polygon_area_m2": float(row.geometry.area), "grid_intersection_area_m2": float(district_totals[i])}
                                   for i, row in enumerate(districts.itertuples())],
                "half_open_edges": True, "origin_nested": True,
                "zero_area_owner_policy": methods["boundary_only_owner"],
                "cell_id_encoding": methods["cell_id"], "license": "ODbL-1.0; derived from unchanged geoBoundaries 2021 district assets"}
    atomic_json(staging / "metadata.json", metadata)
    specification = {"stage": "B", "kind": "fixed_grid", "resolution_m": resolution,
                     "methods_sha256": sha256(ROOT / "configs/phase3b_methods.toml"),
                     "kernel_sha256": sha256(ROOT / "src/agniscope/recurrence.py"),
                     "worker_sha256": sha256(Path(__file__)), "grid_tests_sha256": sha256(TABLE / "pytest_grid_certified.xml"),
                     "gate_a_sha256": sha256(TABLE / "gate_a.json")}
    seal(staging, target, specification)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("resolution", type=int, choices=(500, 1000, 2000))
    args = parser.parse_args()
    print(json.dumps(build(args.resolution), indent=2))
