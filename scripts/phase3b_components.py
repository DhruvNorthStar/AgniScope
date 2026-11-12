"""Bounded grid connectivity, clipped component areas and district spatial summaries."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from agniscope.provenance import sha256
from agniscope.acquisition_store import atomic_json, StoreConflict
from phase3b_grid import TABLE, SPATIAL, verify_receipt, seal, geo_metadata


def build(resolution):
    target = SPATIAL/f"components_{resolution}"
    if target.exists(): return {"reused": True, "receipt": verify_receipt(target)}
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shapely
    from pyproj import CRS, Transformer
    from agniscope import recurrence as r
    from agniscope.stage_b_qc import geography
    from phase3b_recurrence import write_frame
    assert resolution in r.RESOLUTIONS
    recurrence_dir = TABLE/"recurrence"/f"r{resolution}"
    verify_receipt(recurrence_dir)
    grid_dir = SPATIAL/f"grid_{resolution}"
    verify_receipt(grid_dir)
    all_rec = pd.read_parquet(recurrence_dir/"cell_recurrence.parquet", filters=[("subset", "=", "all")])
    recurrences = {method: all_rec.loc[all_rec.method.eq(method)].set_index("cell_id", drop=False) for method in r.METHODS}
    minimum_ids = r.threshold_ids(recurrences["detection"], 2)
    cells = pd.read_parquet(grid_dir/"grid.parquet", filters=[("cell_id", "in", sorted(minimum_ids))])
    assert set(cells.cell_id) == minimum_ids
    geometry = dict(zip(cells.cell_id, shapely.from_wkb(cells.geometry.to_numpy())))
    areas = dict(zip(cells.cell_id, cells.clipped_area_m2))
    district_parts = pd.read_parquet(grid_dir/"district_intersections.parquet", filters=[("cell_id", "in", sorted(minimum_ids))])
    district_lookup = defaultdict(list)
    for row in district_parts.itertuples():
        district_lookup[int(row.cell_id)].append((row.district_original, row.district_display, row.intersection_area_m2))
    transform = Transformer.from_crs(32644, 4326, always_xy=True)
    staging = target.with_name(target.name+".staged")
    if staging.exists(): raise StoreConflict("Preserve uncommitted component staging.")
    staging.mkdir(parents=True)
    schema = pa.schema([
        ("component_id", pa.string()), ("resolution_m", pa.int32()), ("subset", pa.string()), ("method", pa.string()),
        ("k", pa.int8()), ("adjacency", pa.int8()), ("minimum_cell_id", pa.int64()), ("cell_count", pa.int32()),
        ("clipped_area_km2", pa.float64()), ("distinct_active_seasons_union", pa.int8()), ("active_years_union", pa.string()),
        ("minimum_cell_active_seasons", pa.int8()), ("maximum_cell_active_seasons", pa.int8()),
        ("x_min_m", pa.float64()), ("y_min_m", pa.float64()), ("x_max_m", pa.float64()), ("y_max_m", pa.float64()),
        ("centroid_x_m", pa.float64()), ("centroid_y_m", pa.float64()), ("centroid_longitude", pa.float64()), ("centroid_latitude", pa.float64()),
        ("districts_original", pa.string()), ("districts_display", pa.string()), ("district_count", pa.int16()),
        ("cross_district", pa.bool_()), ("geometry_type", pa.string()), ("geometry", pa.binary())],
        metadata=geo_metadata(CRS.from_epsg(32644)))
    member_schema = pa.schema([("component_id", pa.string()), ("cell_id", pa.int64()), ("resolution_m", pa.int32()),
                              ("method", pa.string()), ("k", pa.int8()), ("adjacency", pa.int8()), ("season_mask", pa.uint8())])
    csv_fields = [name for name in schema.names if name != "geometry"]
    inventory = []
    with pq.ParquetWriter(staging/"components.geoparquet", schema, compression="zstd") as writer, \
            pq.ParquetWriter(staging/"component_cells.parquet", member_schema, compression="zstd") as member_writer, \
            (staging/"component_summary.csv").open("x", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=csv_fields, lineterminator="\n")
        csv_writer.writeheader()
        for method in r.METHODS:
            rec = recurrences[method]
            for k in (2, 3, 4, 5):
                ids = r.threshold_ids(rec, k)
                selected = cells.loc[cells.cell_id.isin(ids)]
                assert set(selected.cell_id) == ids
                expected_area = math.fsum(areas[c] for c in ids)/1e6
                for adjacency in (4, 8):
                    components = r.connected_cells(selected, adjacency)
                    accounted = set()
                    total = []
                    maximum_cells = 0
                    maximum_area = 0.
                    cross = 0
                    for member_ids in components:
                        assert not accounted.intersection(member_ids)
                        accounted.update(member_ids)
                        component_id = f"R{resolution}_{method}_K{k}_N{adjacency}_C{member_ids[0]}"
                        pieces = [shapely.union_all([geometry[c] for c in member_ids[start:start+256]]) for start in range(0, len(member_ids), 256)]
                        polygon = shapely.union_all(pieces)
                        area = math.fsum(areas[c] for c in member_ids)/1e6
                        assert polygon.is_valid and math.isclose(polygon.area/1e6, area, rel_tol=1e-11, abs_tol=1e-9)
                        mask = 0
                        originals, displays = set(), set()
                        district_area = []
                        members = []
                        for cell in member_ids:
                            value = int(rec.at[cell, "season_mask"])
                            mask |= value
                            for original, display, part_area in district_lookup[cell]:
                                originals.add(original); displays.add(display); district_area.append(part_area)
                            members.append({"component_id": component_id, "cell_id": cell, "resolution_m": resolution,
                                            "method": method, "k": k, "adjacency": adjacency, "season_mask": value})
                        assert math.isclose(math.fsum(district_area)/1e6, area, rel_tol=1e-11, abs_tol=1e-9)
                        centroid = polygon.centroid
                        lon, lat = transform.transform(centroid.x, centroid.y)
                        x0, y0, x1, y1 = polygon.bounds
                        row = {"component_id": component_id, "resolution_m": resolution, "subset": "all", "method": method,
                            "k": k, "adjacency": adjacency, "minimum_cell_id": member_ids[0], "cell_count": len(member_ids),
                            "clipped_area_km2": area, "distinct_active_seasons_union": mask.bit_count(),
                            "active_years_union": ";".join(str(year) for year in r.YEARS if mask & (1 << (year-2019))),
                            "minimum_cell_active_seasons": int(rec.loc[member_ids, "active_seasons"].min()),
                            "maximum_cell_active_seasons": int(rec.loc[member_ids, "active_seasons"].max()),
                            "x_min_m": x0, "y_min_m": y0, "x_max_m": x1, "y_max_m": y1,
                            "centroid_x_m": centroid.x, "centroid_y_m": centroid.y,
                            "centroid_longitude": lon, "centroid_latitude": lat,
                            "districts_original": ";".join(sorted(originals)), "districts_display": ";".join(sorted(displays)),
                            "district_count": len(originals), "cross_district": len(originals) > 1,
                            "geometry_type": polygon.geom_type, "geometry": shapely.to_wkb(polygon)}
                        writer.write_table(pa.Table.from_pylist([row], schema=schema))
                        member_writer.write_table(pa.Table.from_pylist(members, schema=member_schema), row_group_size=4096)
                        csv_writer.writerow({name: row[name] for name in csv_fields})
                        total.append(area)
                        maximum_cells = max(maximum_cells, len(member_ids))
                        maximum_area = max(maximum_area, area)
                        cross += len(originals) > 1
                    assert accounted == ids and math.isclose(math.fsum(total), expected_area, rel_tol=1e-11, abs_tol=1e-9)
                    inventory.append({"resolution_m": resolution, "method": method, "k": k, "adjacency": adjacency,
                        "qualifying_cells": len(ids), "components": len(components), "clipped_area_km2": expected_area,
                        "largest_component_cells": maximum_cells, "largest_component_area_km2": maximum_area,
                        "cross_district_components": cross})
    pd.DataFrame(inventory).to_csv(staging/"component_inventory.csv", index=False, lineterminator="\n")
    # One geometry per district, with wide baseline method/threshold attributes.
    _, districts = geography()
    district_table = pd.read_csv(recurrence_dir/"district_summary.csv")
    district_table = district_table.loc[district_table.subset.eq("all")]
    district_spatial = pd.DataFrame({"district_original": districts.district_original.to_numpy(),
                                    "district_display": districts.agniscope_display_name.to_numpy(),
                                    "eligible_area_km2": districts.area.to_numpy()/1e6,
                                    "geometry": shapely.to_wkb(districts.geometry.to_numpy())})
    for method in r.METHODS:
        for k in (1, 2, 3, 4, 5):
            lookup = district_table.loc[district_table.method.eq(method) & district_table.k.eq(k)].set_index("district_original")
            for field in ("qualifying_cells", "qualifying_area_km2", "district_area_percent"):
                district_spatial[f"{method}_k{k}_{field}"] = district_spatial.district_original.map(lookup[field])
    write_frame(district_spatial, staging/"district_recurrence.geoparquet", geo=True)
    atomic_json(staging/"summary.json", {"resolution_m": resolution, "component_scopes": 24, "population": "all",
        "adjacency": [4, 8], "component_inventory": inventory,
        "interpretation": "Grid-index connectivity; clipped multipolygons may include separated mask pieces. Not burned area or independent incidents.",
        "district_area_allocation": "Exact positive-area district/cell intersections; cross-district cell counts are nonadditive.",
        "geometry_union_batch_cells": 256})
    seal(staging, target, {"kind": "connected_recurrent_cells", "resolution_m": resolution,
        "worker_sha256": sha256(Path(__file__)), "recurrence_receipt_sha256": sha256(recurrence_dir/"receipt.json"),
        "grid_receipt_sha256": sha256(grid_dir/"receipt.json"), "kernel_sha256": sha256(ROOT/"src/agniscope/recurrence.py")})
    return {"resolution_m": resolution, "status": "passed", "component_scopes": len(inventory)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("resolution", type=int, choices=(500, 1000, 2000))
    args = parser.parse_args()
    print(json.dumps(build(args.resolution), indent=2))
