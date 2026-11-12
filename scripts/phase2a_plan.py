"""Generate offline Phase 2A planning tables; never download observations."""

import csv
from collections import Counter
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agniscope.config import load_config, workspace_path
from agniscope.planning import build_request_plan, build_season_plan, coverage_state, dates, resource_estimates
from agniscope.provenance import sha256, utc_now, verify_sources, write_json
from agniscope.resources import resource_snapshot


def read_csv(path):
    with workspace_path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(directory, name, records):
    target = directory / name
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    settings = load_config(workspace_path("configs/historical_protocol.toml"))
    plan = build_request_plan(settings)
    output = workspace_path(settings["outputs"]["directory"], writable=True)
    # Fixed Phase 2A directory, never a caller-supplied data/output destination.
    if output != workspace_path("outputs/tables/phase2a"):
        raise ValueError("Planning output must stay in the separate Phase 2A directory.")
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    for item in preflight["protected_files"]:
        if sha256(workspace_path(item["path"])) != item["sha256"]:
            raise ValueError("Existing project evidence changed; planning stopped.")
    if not verify_sources(load_config())["all_unchanged"]:
        raise ValueError("Copied original GIS integrity failed.")
    reuse = settings["reuse"]
    raw_path = workspace_path(reuse["raw_csv"])
    if sha256(raw_path) != reuse["raw_sha256"]:
        raise ValueError("Authentic pilot integrity failed.")
    raw, accepted = read_csv(reuse["raw_csv"]), read_csv(reuse["qc_csv"])
    if (len(raw), len(accepted)) != (reuse["raw_rows"], reuse["accepted_rows"]):
        raise ValueError("Saved pilot population contract failed.")
    raw_daily, accepted_daily = Counter(r["acq_date"] for r in raw), Counter(r["acq_date"] for r in accepted)
    metadata = json.loads((output / "metadata_evidence.json").read_text(encoding="utf-8"))
    available = metadata.get("availability", {}).get("availability", {})
    issues = read_csv(settings["outputs"]["upstream_issues"])
    daily = []
    for request in plan:
        start, end = (date.fromisoformat(request[k]) for k in ("start_utc_date", "end_utc_date"))
        acquired = request["status"] == "reuse_verified_pilot"
        for day in dates(start, end):
            day_text = str(day)
            # Open-ended notices mark their onset, not a fabricated continuous outage.
            concerns = [issue["issue_id"] for issue in issues
                        if issue["start_utc"][:10] <= day_text <= (issue["end_utc"] or issue["start_utc"])[:10]]
            daily.append({
                "expected_utc_date": day_text, "request_id": request["request_id"],
                "expected_in_study": True, "planned_source": settings["study"]["source"],
                "requested": acquired, "api_http_status": 200 if acquired else None,
                "response_validated": acquired, "retrieved_observations": raw_daily[day_text] if acquired else None,
                "accepted_observations": accepted_daily[day_text] if acquired else None,
                "outside_footprint": raw_daily[day_text] - accepted_daily[day_text] if acquired else None,
                "acquisition_status": coverage_state(requested=acquired, validated=acquired, raw_rows=raw_daily[day_text] if acquired else None),
                "failure_code": None, "actual_source": settings["study"]["source"] if acquired else None,
                "actual_collection_values": "2" if acquired else None,
                "qc_status": "existing_pilot_pass" if acquired else "not_run",
                "within_source_date_range": available.get("min_date", "9999") <= day_text <= available.get("max_date", "0000"),
                "upstream_concern_ids": ";".join(concerns),
                "upstream_calendar_status": "incomplete_nonexhaustive_register",
                "local_observation_opportunity": "unverified",
                "raw_sha256": reuse["raw_sha256"] if acquired else None,
            })
    annual = []
    for year in range(2019, 2026):
        requests = [r for r in plan if r["year"] == year]
        rows = [r for r in daily if r["expected_utc_date"].startswith(str(year))]
        annual.append({"year": year, "expected_days": len(rows), "logical_intervals": len(requests),
                       "future_observation_requests": sum(r["status"] == "planned_not_requested" for r in requests),
                       "already_queried_days": sum(r["requested"] for r in rows),
                       "source_range_covers_all_dates": all(r["within_source_date_range"] for r in rows),
                       "daily_satellite_completeness_verified": False,
                       "known_issue_register_nonexhaustive": True})
    write_csv(output, "request_plan.csv", plan)
    write_csv(output, "daily_coverage_plan.csv", daily)
    write_csv(output, "annual_plan.csv", annual)
    write_csv(output, "season_calendar_plan.csv", build_season_plan(settings))
    summary = {"generated_utc": utc_now(), "phase": "2A_planning_only", "observation_requests_this_phase": 0,
               "execution_authorized": False, "config_sha256": sha256(workspace_path("configs/historical_protocol.toml")),
               "protected_existing_files_verified": len(preflight["protected_files"]),
               "resources": resource_snapshot(), "estimates": resource_estimates(settings, plan, raw_path.stat().st_size),
               "stage_future_requests": dict(Counter(r["proposed_stage"] for r in plan if r["status"] == "planned_not_requested")),
               "daily_status_counts": dict(Counter(r["acquisition_status"] for r in daily)),
               "pilot_raw_rows": len(raw), "pilot_retained_rows": len(accepted),
               "curated_upstream_issue_entries": len(issues),
               "notes": ["No authenticated URL or key is included.",
                         "Empty CSV fields for unrequested dates mean unknown, never zero.",
                         "Season windows in study range do not establish complete satellite coverage.",
                         "Issue register is nonexhaustive; open-ended notices flag onset only and require follow-up."]}
    write_json(output / "plan_summary.json", summary)
    print(json.dumps({"intervals": len(plan), "days": len(daily), "future_observation_requests": summary["estimates"]["future_observation_requests"],
                      "stage_future_requests": summary["stage_future_requests"], "observation_requests_executed": 0}))


if __name__ == "__main__":
    main()
