"""Independent recurrence, clipped area, district arithmetic and union-find QA."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from agniscope.provenance import sha256, utc_now
from agniscope.acquisition_store import atomic_json
from preflight_phase3b_recovery import REC, TABLE, SPATIAL
from phase3b_grid import verify_receipt


def polygon_area(geom):
    """Translated shoelace sum, independent of GEOS area implementation."""
    def ring_area(ring):
        points = list(ring.coords)
        ox, oy = points[0]
        return abs(math.fsum((x-ox)*(v-oy)-(u-ox)*(y-oy) for (x, y), (u, v) in zip(points, points[1:])))/2
    if geom.geom_type == "Polygon": return ring_area(geom.exterior)-math.fsum(ring_area(r) for r in geom.interiors)
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"): return math.fsum(polygon_area(g) for g in geom.geoms)
    return 0.


def union_find(cells, adjacency):
    """Separate disjoint-set implementation; production uses breadth-first search."""
    parent = {c: c for c in cells}
    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]; c = parent[c]
        return c
    coordinates = {}
    for cell in cells:
        word = cell % 4294967296
        coordinates[(word % 65536-32768, word//65536-32768)] = cell
    directions = [(1, 0), (0, 1)] + ([(1, 1), (1, -1)] if adjacency == 8 else [])
    for (x, y), cell in coordinates.items():
        for dx, dy in directions:
            other = coordinates.get((x+dx, y+dy))
            if other is not None:
                a, b = find(cell), find(other)
                if a != b: parent[max(a, b)] = min(a, b)
    groups = defaultdict(set)
    for cell in cells: groups[find(cell)].add(cell)
    return {min(ids): ids for ids in groups.values()}


def rows(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-9), (a, b)


def verify(resolution):
    import pyarrow.parquet as pq
    import shapely
    from agniscope.stage_b_qc import geography
    directory = TABLE/"recurrence"/f"r{resolution}"
    grid_dir, comp_dir = SPATIAL/f"grid_{resolution}", SPATIAL/f"components_{resolution}"
    for folder in (directory, grid_dir, comp_dir): verify_receipt(folder)
    _, districts = geography()
    mask = districts.geometry.union_all(); shapely.prepare(mask)
    district_geoms = dict(zip(districts.district_original, districts.geometry))
    for geom in district_geoms.values(): shapely.prepare(geom)
    parts_table = pq.read_table(grid_dir/"district_intersections.parquet", columns=["cell_id", "district_original", "intersection_area_m2"], use_threads=False, pre_buffer=False)
    part_values = parts_table.to_pydict()
    cell_allocations = defaultdict(float)
    allocations_n = defaultdict(int)
    district_sums = defaultdict(float)
    for cell, name, area in zip(part_values["cell_id"], part_values["district_original"], part_values["intersection_area_m2"]):
        assert area > 0 and name in district_geoms
        cell_allocations[cell] += area; allocations_n[cell] += 1; district_sums[name] += area
    areas = {}
    geometadata = json.loads((grid_dir/"metadata.json").read_text())
    cross_cell_geometries = {}
    manual_area_sum = []
    for batch in pq.ParquetFile(grid_dir/"grid.parquet").iter_batches(batch_size=1024):
        data = batch.to_pydict()
        for i, cell in enumerate(data["cell_id"]):
            assert cell not in areas
            x, y = data["grid_x"][i], data["grid_y"][i]
            assert cell == {500: 1, 1000: 2, 2000: 3}[resolution]*4294967296+(y+32768)*65536+x+32768
            geom = shapely.from_wkb(data["geometry"][i]); area = polygon_area(geom)
            assert geom.is_valid and area > 0
            rectangle = shapely.box(x*resolution, y*resolution, (x+1)*resolution, (y+1)*resolution)
            assert rectangle.covers(geom)
            independently_clipped = rectangle if mask.covers(rectangle) else rectangle.intersection(mask)
            assert geom.equals(independently_clipped)
            assert math.isclose(area, data["clipped_area_m2"][i], rel_tol=1e-11, abs_tol=.001)
            assert math.isclose(area, cell_allocations[cell], rel_tol=1e-11, abs_tol=.001)
            areas[cell] = area; manual_area_sum.append(area)
            if allocations_n[cell] > 1: cross_cell_geometries[cell] = geom
    assert len(areas) == geometadata["eligible_positive_area_cells"]
    assert math.isclose(math.fsum(manual_area_sum), polygon_area(mask), rel_tol=1e-11, abs_tol=.001)
    for name, geom in district_geoms.items(): assert math.isclose(district_sums[name], polygon_area(geom), rel_tol=1e-11, abs_tol=.001)
    for cell, name, area in zip(part_values["cell_id"], part_values["district_original"], part_values["intersection_area_m2"]):
        if cell in cross_cell_geometries:
            expected = polygon_area(cross_cell_geometries[cell].intersection(district_geoms[name]))
            assert math.isclose(expected, area, rel_tol=1e-11, abs_tol=.001)
    del cell_allocations, allocations_n, cross_cell_geometries, manual_area_sum
    frequency = rows(directory/"frequency_distribution.csv")
    thresholds = rows(directory/"threshold_comparison.csv")
    district_rows = rows(directory/"district_summary.csv")
    overlaps = rows(directory/"method_overlap.csv")
    baseline = {}
    verified_views = 0
    for subset in ("all", "grid_day", "nominal", "tree"):
        expected_views = {}
        for method in ("detection", "dbscan", "hdbscan"):
            expected = {}
            for year in range(2019, 2026):
                path = TABLE/"activity"/f"CORE_{year}"/"seasonal_activity.parquet"
                values = pq.read_table(path, columns=["cell_id", "detections", "distinct_UTC_dates"],
                    filters=[("resolution_m", "=", resolution), ("subset", "=", subset), ("method", "=", method)], use_threads=False, pre_buffer=False).to_pydict()
                assert len(values["cell_id"]) == len(set(values["cell_id"]))
                for cell, n, ndates in zip(values["cell_id"], values["detections"], values["distinct_UTC_dates"]):
                    value = expected.setdefault(cell, [set(), 0, 0])
                    value[0].add(year); value[1] += n; value[2] += ndates
            expected_views[method] = expected
            actual = pq.read_table(directory/"cell_recurrence.parquet", filters=[("subset", "=", subset), ("method", "=", method)], use_threads=False, pre_buffer=False).to_pylist()
            assert len(actual) == len(expected) and {r["cell_id"] for r in actual} == set(expected)
            for row in actual:
                years, n, ndays = expected[row["cell_id"]]
                assert row["active_seasons"] == len(years) <= 7
                assert row["season_mask"] == sum(2**(y-2019) for y in years)
                assert row["detections"] == n and row["distinct_UTC_dates"] == ndays
                assert row["first_active_year"] == min(years) and row["last_active_year"] == max(years)
                assert row["activity_span_years"] == max(years)-min(years)
                assert row["active_years"] == ";".join(map(str, sorted(years)))
                assert row["observed_active_season_fraction"] == len(years)/7
            prefix = subset+"_"+method+"_"
            columns = ["cell_id"]+[prefix+n for n in ("active_seasons", "season_mask", "detections", "distinct_UTC_dates")]
            wide_count = 0
            for batch in pq.ParquetFile(directory/"eligible_cell_activity.parquet").iter_batches(batch_size=1024, columns=columns):
                for row in batch.to_pylist():
                    years, n, ndays = expected.get(row["cell_id"], (set(), 0, 0))
                    assert row[prefix+"active_seasons"] == len(years)
                    assert row[prefix+"season_mask"] == sum(2**(y-2019) for y in years)
                    assert row[prefix+"detections"] == n and row[prefix+"distinct_UTC_dates"] == ndays
                    wide_count += 1
            assert wide_count == len(areas)
            for count in range(8):
                chosen = ({c for c in areas if c not in expected} if count == 0 else {c for c, v in expected.items() if len(v[0]) == count})
                record = next(v for v in frequency if v["subset"] == subset and v["method"] == method and int(v["active_seasons"]) == count)
                assert int(record["cells"]) == len(chosen)
                close(record["clipped_area_km2"], math.fsum(areas[c] for c in chosen)/1e6)
            for k in (1, 2, 3, 4, 5):
                chosen = {c for c, v in expected.items() if len(v[0]) >= k}
                if k >= 2:
                    row = next(v for v in thresholds if v["subset"] == subset and v["method"] == method and int(v["k"]) == k)
                    assert int(row["recurrent_cells"]) == len(chosen)
                    close(row["recurrent_area_km2"], math.fsum(areas[c] for c in chosen)/1e6)
                sums = defaultdict(float); counts = defaultdict(int)
                for cell, name, area in zip(part_values["cell_id"], part_values["district_original"], part_values["intersection_area_m2"]):
                    if cell in chosen: sums[name] += area; counts[name] += 1
                for row in district_rows:
                    if row["subset"] == subset and row["method"] == method and int(row["k"]) == k:
                        name = row["district_original"]
                        assert int(row["qualifying_cells"]) == counts[name]
                        close(row["qualifying_area_km2"], sums[name]/1e6)
                        close(row["district_area_percent"], sums[name]/district_sums[name]*100)
            verified_views += 1
        for method in ("dbscan", "hdbscan"):
            assert set(expected_views[method]) <= set(expected_views["detection"])
            assert all(v[0] <= expected_views["detection"][c][0] for c, v in expected_views[method].items())
        for record in overlaps:
            if record["subset"] != subset: continue
            k = int(record["k"])
            left = {c for c, v in expected_views[record["left"]].items() if len(v[0]) >= k}
            right = {c for c, v in expected_views[record["right"]].items() if len(v[0]) >= k}
            assert int(record["intersection_cells"]) == len(left & right) and int(record["union_cells"]) == len(left | right)
            if left | right: close(record["cell_jaccard"], len(left & right)/len(left | right))
        if subset == "all": baseline = expected_views
    # Components are checked with independent disjoint-set connectivity and exact
    # membership coverage, then against the original clipped member geometries.
    minimum = {c for c, v in baseline["detection"].items() if len(v[0]) >= 2}
    geoms = pq.read_table(grid_dir/"grid.parquet", columns=["cell_id", "geometry"], filters=[("cell_id", "in", sorted(minimum))], use_threads=False, pre_buffer=False).to_pydict()
    shapes = dict(zip(geoms["cell_id"], shapely.from_wkb(geoms["geometry"])))
    summaries = pq.read_table(comp_dir/"components.geoparquet", use_threads=False, pre_buffer=False).to_pylist()
    member_rows = pq.read_table(comp_dir/"component_cells.parquet", use_threads=False, pre_buffer=False).to_pylist()
    actual_members = defaultdict(set)
    for row in member_rows:
        assert row["cell_id"] not in actual_members[row["component_id"]]
        actual_members[row["component_id"]].add(row["cell_id"])
    summary_map = {row["component_id"]: row for row in summaries}
    assert len(summary_map) == len(summaries) == len(actual_members)
    scopes = 0
    for method in ("detection", "dbscan", "hdbscan"):
        for k in (2, 3, 4, 5):
            chosen = {c for c, v in baseline[method].items() if len(v[0]) >= k}
            for adjacency in (4, 8):
                expected_groups = union_find(chosen, adjacency)
                selected = [row for row in summaries if row["method"] == method and row["k"] == k and row["adjacency"] == adjacency]
                assert len(selected) == len(expected_groups)
                for row in selected:
                    ids = expected_groups[row["minimum_cell_id"]]
                    assert actual_members[row["component_id"]] == ids and row["cell_count"] == len(ids)
                    close(row["clipped_area_km2"], math.fsum(areas[c] for c in ids)/1e6)
                    polygon = shapely.from_wkb(row["geometry"])
                    assert polygon.is_valid
                    reconstructed = shapely.union_all([shapes[c] for c in sorted(ids)])
                    assert polygon.symmetric_difference(reconstructed).area <= .001
                    assert all(math.isclose(a, b, rel_tol=0., abs_tol=1e-7) for a, b in zip(polygon.bounds, reconstructed.bounds))
                    close(polygon_area(polygon)/1e6, row["clipped_area_km2"])
                    years = set().union(*(baseline[method][c][0] for c in ids))
                    assert row["distinct_active_seasons_union"] == len(years)
                    assert row["active_years_union"] == ";".join(map(str, sorted(years)))
                scopes += 1
    assert verified_views == 12 and scopes == 24
    output = REC/"independent"/f"r{resolution}.json"
    assert not output.exists()
    result = {"status": "passed", "verified_utc": utc_now(), "resolution_m": resolution,
        "eligible_cells_verified": len(areas), "independent_shoelace_area_km2": polygon_area(mask)/1e6,
        "recurrence_views": 12, "component_scopes": 24, "components_verified": len(summaries),
        "full_zero_cell_wide_values_verified": True, "cross_district_intersections_independently_verified": True,
        "source_bound_activity_receipts_gate": sha256(REC/"activity_verification_complete.json"),
        "recurrence_receipt_sha256": sha256(directory/"receipt.json"), "component_receipt_sha256": sha256(comp_dir/"receipt.json"),
        "verifier_sha256": sha256(Path(__file__)), "observation_HTTP_calls": 0, "credential_reads": 0}
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("resolution", type=int, choices=(500, 1000, 2000)); args = parser.parse_args()
    verify(args.resolution)
