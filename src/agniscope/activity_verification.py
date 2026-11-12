"""Independent one-scope verification, Arrow columns and bounded Python accumulators.

No recurrence production helpers, GIS geometry, estimator or network imports.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import pyarrow as pa
import pyarrow.parquet as pq

ID = "agniscope_observation_id"
UTC = "agniscope_acquired_utc"
X, Y = "agniscope_x_m", "agniscope_y_m"
DAY_NS = 86400*10**9


def independent_cell(x, y, resolution):
    code = {500: 1, 1000: 2, 2000: 3}[resolution]
    col, row = math.floor(float(x)/resolution), math.floor(float(y)/resolution)
    if not (-32768 <= col <= 32767 and -32768 <= row <= 32767):
        raise ValueError("Cell indices exceed the declared signed range.")
    return (code << 32) | ((row+32768) << 16) | (col+32768)


def sampling_indices(values):
    best = {}
    for i, (x, y, ticks, raw, source) in enumerate(zip(values[X], values[Y], values[UTC], values["agniscope_raw_sha256"], values["agniscope_source_record"])):
        key = (math.floor(x/1000), math.floor(y/1000), ticks//DAY_NS)
        candidate = (ticks, raw, int(source))
        if key not in best or candidate < best[key][0]: best[key] = (candidate, i)
    return {value[1] for value in best.values()}


def load_population(master, specification):
    year, subset = specification["year"], specification["subset"]
    if year not in range(2019, 2026) or subset not in ("all", "grid_day", "nominal", "tree"):
        raise ValueError("Scope lies outside the declared population/year contract.")
    columns = [ID, UTC, X, Y]
    filters = [(UTC, ">=", datetime(year, 2, 15, tzinfo=timezone.utc)),
               (UTC, "<", datetime(year, 6, 16, tzinfo=timezone.utc))]
    predicate_columns = []
    if subset == "nominal": filters.append(("agniscope_confidence", "=", "nominal")); predicate_columns.append("agniscope_confidence")
    if subset == "tree": filters.append(("agniscope_worldcover_2021_class", "=", 10)); predicate_columns.append("agniscope_worldcover_2021_class")
    if subset == "grid_day": columns += ["agniscope_raw_sha256", "agniscope_source_record"]
    table = pq.read_table(master, columns=columns, filters=filters, use_threads=False, pre_buffer=False)
    if str(table[UTC].type) != "timestamp[ns, tz=UTC]": raise ValueError("Explicit original UTC nanosecond timestamps required.")
    values = {name: table[name].to_pylist() for name in columns if name != UTC}
    values[UTC] = table[UTC].cast(pa.int64()).to_pylist()
    full_rows, table_bytes = table.num_rows, table.nbytes
    del table
    if subset == "grid_day":
        selected = sampling_indices(values)
        values = {name: [v for i, v in enumerate(column) if i in selected] for name, column in values.items()}
    if len(values[ID]) != len(set(values[ID])): raise ValueError("Duplicate source identities in independently selected population.")
    pf = pq.ParquetFile(master)
    utc_index = pf.schema_arrow.names.index(UTC)
    selected_groups = 0
    projected_bytes = 0
    start, end = filters[0][2], filters[1][2]
    for group in range(pf.num_row_groups):
        row_group = pf.metadata.row_group(group)
        stats = row_group.column(utc_index).statistics
        if stats is None or (stats.max >= start and stats.min < end):
            selected_groups += 1
            projected_bytes += sum(row_group.column(pf.schema_arrow.names.index(name)).total_uncompressed_size
                                   for name in set(columns+predicate_columns))
    profile = {"master_total_columns": len(pf.schema_arrow), "requested_columns": columns,
               "predicate_columns": predicate_columns, "returned_Arrow_bytes": table_bytes,
               "returned_rows_before_temporal_sampling": full_rows, "population_rows": len(values[ID]),
               "master_row_groups": pf.num_row_groups, "time_overlapping_row_groups": selected_groups,
               "projected_uncompressed_row_group_bytes": projected_bytes,
               "use_threads": False, "pre_buffer": False, "full_grid_geometries_read": 0}
    return values, profile


def expectation(values, resolution, member_labels=None, strengths=None):
    """Independent date sets/counts and compensated sums, one scope at a time."""
    cells = {}
    supporting = 0
    for identity, x, y, ticks in zip(values[ID], values[X], values[Y], values[UTC]):
        if member_labels is not None and member_labels[identity] < 0: continue
        if not math.isfinite(x) or not math.isfinite(y): raise ValueError("Nonfinite authentic point.")
        cell = independent_cell(x, y, resolution)
        state = cells.setdefault(cell, [0, set(), 0, 0., 0., None, None])
        state[0] += 1; state[1].add(ticks//DAY_NS); supporting += 1
        if strengths is not None:
            value = strengths[identity]
            if value is None or not math.isfinite(value) or not 0 <= value <= 1: raise ValueError("Invalid member strength.")
            state[2] += 1
            total = state[3]+value
            state[4] += ((state[3]-total)+value if abs(state[3]) >= abs(value) else (value-total)+state[3])
            state[3] = total
            state[5] = value if state[5] is None else min(state[5], value)
            state[6] = value if state[6] is None else max(state[6], value)
    return {cell: {"detections": s[0], "distinct_UTC_dates": len(s[1]), "strength_available": s[2],
                   "strength_sum": s[3]+s[4], "strength_min": s[5], "strength_max": s[6]}
            for cell, s in cells.items()}, supporting


def fingerprint(expected):
    return hashlib.sha256(json.dumps(sorted(expected.items()), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compare_saved(path, specification, expected):
    columns = ["cell_id", "year", "activity", "detections", "distinct_UTC_dates",
               "strength_available", "strength_sum", "strength_min", "strength_max"]
    filters = [("resolution_m", "=", specification["resolution_m"]), ("subset", "=", specification["subset"]),
               ("method", "=", specification["method"])]
    table = pq.read_table(path, columns=columns, filters=filters, use_threads=False, pre_buffer=False)
    seen = set()
    for batch in table.to_batches(max_chunksize=1024):
        values = batch.to_pydict()
        for i, cell in enumerate(values["cell_id"]):
            if cell in seen or cell not in expected: raise ValueError("Unexpected or duplicate activity cell.")
            seen.add(cell)
            assert values["year"][i] == specification["year"] and values["activity"][i] == 1
            reference = expected[cell]
            for name in ("detections", "distinct_UTC_dates", "strength_available"):
                assert values[name][i] == reference[name]
            if specification["method"] == "hdbscan":
                assert math.isclose(values["strength_sum"][i], reference["strength_sum"], rel_tol=1e-12, abs_tol=1e-12)
                assert values["strength_min"][i] == reference["strength_min"] and values["strength_max"][i] == reference["strength_max"]
            else:
                assert values["strength_min"][i] is None and values["strength_max"][i] is None
    assert seen == set(expected)
    return {"active_cells": len(seen), "saved_Arrow_bytes": table.nbytes, "saved_columns_read": columns,
            "source_supporting_observations": sum(v["detections"] for v in expected.values())}
