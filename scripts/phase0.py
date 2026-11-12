"""Bounded Phase 0 commands. No ML fitting or dashboard code."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agniscope.acquisition import AcquisitionError, acquire_pilot
from agniscope.config import load_config, workspace_path
from agniscope.firms import SchemaError, save_validation, validate_csv
from agniscope.landcover import raster_report, sample_landcover
from agniscope.provenance import utc_now, verify_sources, write_json
from agniscope.resources import resource_snapshot
from agniscope.spatial import boundary_report, load_boundaries, save_footprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("assets", help="Validate GIS sources and create/reuse the separate analytical footprint")
    acquire = sub.add_parser("firms", help="Check credential and availability; download at most the configured five-day pilot")
    acquire.add_argument("--download-pilot", action="store_true")
    validate = sub.add_parser("validate", help="QC a CSV already inside the workspace")
    validate.add_argument("input_csv")
    validate.add_argument("--sample-landcover", action="store_true")
    sub.add_parser("integrity", help="Verify that all fourteen source files remain unchanged")
    args = parser.parse_args()
    config = load_config()
    started = time.perf_counter()
    before = resource_snapshot()
    try:
        if args.command == "integrity":
            report = verify_sources(config)
            write_json("outputs/tables/source_integrity.json", report)
            print(json.dumps({"all_sources_unchanged": report["all_unchanged"], "files": len(report["files"])}))
            return 0 if report["all_unchanged"] else 1
        if args.command == "assets":
            integrity = verify_sources(config)
            if not integrity["all_unchanged"]:
                raise ValueError("Immutable-source inventory changed; stopping.")
            state, districts = load_boundaries(config)
            spatial = boundary_report(state, districts)
            save_footprint(districts, config)
            landcover = raster_report(config, state, districts)
            after_integrity = verify_sources(config)
            if not after_integrity["all_unchanged"]:
                raise ValueError("Source integrity failed after GIS operations.")
            write_json("outputs/tables/spatial_validation.json", spatial)
            write_json("outputs/tables/landcover_validation.json", landcover)
            write_json("outputs/tables/source_integrity.json", after_integrity)
            write_json("outputs/tables/asset_run_resources.json", {"generated_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started, "before": before, "after": resource_snapshot(), "full_resolution_raster_scan": False, "raster_mosaic_created": False})
            print(json.dumps({"symmetric_difference_km2": spatial["symmetric_difference_km2"], "district_union_area_km2": spatial["district_union_area_km2"], "raw_tiles_cover_footprint": landcover["raw_coverage_contains_district_union"], "sources_unchanged": after_integrity["all_unchanged"]}, indent=2))
        elif args.command == "firms":
            report = acquire_pilot(config, download=args.download_pilot)
            write_json("outputs/tables/firms_acquisition_status.json", report)
            print(json.dumps(report, indent=2))
            if report["authentic_sample_acquired"]:
                _, districts = load_boundaries(config)
                result = validate_csv(report["raw_file"], districts, config, dataset_kind="nasa_firms_api")
                identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                target = save_validation(result, f"data/interim/firms_qc/{identifier}")
                write_json("outputs/tables/pilot_qc_summary.json", result.summary)
                print("Pilot QC saved inside data/interim/firms_qc/" + identifier)
        elif args.command == "validate":
            _, districts = load_boundaries(config)
            result = validate_csv(args.input_csv, districts, config)
            if args.sample_landcover:
                import geopandas as gpd
                accepted = result.accepted
                geo = gpd.GeoDataFrame(index=accepted.index, geometry=gpd.points_from_xy(accepted["agniscope_longitude"], accepted["agniscope_latitude"]), crs="EPSG:4326")
                sampled = sample_landcover(geo, config)
                result.records = result.records.join(sampled)
                result.summary["landcover_sampling"] = {"reference_year": 2021, "status_counts": {str(k): int(v) for k, v in sampled["agniscope_landcover_status"].value_counts().items()}, "tree_cover_filter_applied": False}
            identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target = save_validation(result, f"data/interim/firms_qc/{identifier}")
            print(json.dumps({"output": target.relative_to(ROOT).as_posix(), "input_rows": result.summary["input_rows"], "retained_rows": result.summary["retained_rows"]}))
        return 0
    except (AcquisitionError, SchemaError, ValueError) as exc:
        # AcquisitionError messages are sanitized at the network boundary.
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
