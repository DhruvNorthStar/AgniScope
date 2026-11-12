"""A fresh guarded process verifies one pending scope and commits one receipt."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from agniscope.provenance import sha256, utc_now
from agniscope.acquisition_store import atomic_json
from agniscope.resources import resource_snapshot
from preflight_phase3b_recovery import REC, MASTER


def verify(scope_id):
    preflight = json.loads((REC/"preflight.json").read_text())
    plan_path = REC/"scope_plan.json"
    assert sha256(plan_path) == preflight["scope_plan_sha256"]
    specification = next(p for p in json.loads(plan_path.read_text()) if p["scope_id"] == scope_id)
    assert not specification["prior_verified"], "Historical verified scopes must be reused, not recomputed."
    target = REC/"scopes"/(scope_id[:26]+".json")
    if target.exists():
        previous = json.loads(target.read_text()); assert previous["status"] == "passed" and previous["specification"] == specification
        return {"status": "passed", "reused": True, "scope_id": scope_id}
    checks = [(MASTER, specification["master_sha256"]), (ROOT/specification["activity_path"], specification["activity_sha256"]),
              (ROOT/specification["population_path"], specification["population_sha256"])]
    if specification["assignment_path"]: checks.append((ROOT/specification["assignment_path"], specification["assignment_sha256"]))
    for path, digest in checks: assert sha256(path) == digest
    stages = [{"stage": "before_numeric_import", **resource_snapshot()}]
    import pyarrow.parquet as pq
    from agniscope.activity_verification import ID, load_population, expectation, compare_saved, fingerprint
    values, profile = load_population(MASTER, specification)
    stages.append({"stage": "master_population_loaded", **resource_snapshot()})
    population_ids = pq.read_table(ROOT/specification["population_path"], columns=[ID], use_threads=False, pre_buffer=False)[ID].to_pylist()
    assert len(population_ids) == len(set(population_ids)) == len(values[ID]) and set(population_ids) == set(values[ID])
    del population_ids
    labels = strengths = None
    if specification["method"] != "detection":
        names = [ID, "cluster_label"] + (["membership_strength"] if specification["method"] == "hdbscan" else [])
        assignment = pq.read_table(ROOT/specification["assignment_path"], columns=names, use_threads=False, pre_buffer=False).to_pydict()
        assert len(assignment[ID]) == len(set(assignment[ID])) and set(assignment[ID]) == set(values[ID])
        labels = dict(zip(assignment[ID], assignment["cluster_label"]))
        assert all(isinstance(label, int) and label >= 0 or label in (-1, -2, -3) for label in labels.values())
        if specification["method"] == "hdbscan": strengths = dict(zip(assignment[ID], assignment["membership_strength"]))
        del assignment
    stages.append({"stage": "source_id_join_complete", **resource_snapshot()})
    expected, supporting = expectation(values, specification["resolution_m"], labels, strengths)
    stages.append({"stage": "one_scope_expectation_complete", **resource_snapshot()})
    result = compare_saved(ROOT/specification["activity_path"], specification, expected)
    assert supporting == result["source_supporting_observations"]
    for path, digest in checks: assert sha256(path) == digest
    result.update(status="passed", verified_utc=utc_now(), specification=specification, reused_historical_verification=False,
                  independent_reconstruction=True, original_source_identity_join_exact=True,
                  expected_cell_payload_sha256=fingerprint(expected), read_profile=profile, resource_stages=stages,
                  verifier_sha256=sha256(Path(__file__)), kernel_sha256=sha256(ROOT/"src/agniscope/activity_verification.py"),
                  observation_HTTP_calls=0, credential_reads=0, clustering_refits=0)
    atomic_json(target, result)
    return {"status": "passed", "scope_id": scope_id, "active_cells": result["active_cells"]}


if __name__ == "__main__": print(json.dumps(verify(sys.argv[1])))
