"""Small metadata records and immutable-source verification."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .config import workspace_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    with workspace_path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: str | Path, value: dict) -> Path:
    target = workspace_path(path, writable=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return target


def verify_sources(config: dict) -> dict:
    inventory = json.loads(workspace_path(config["paths"]["source_inventory"]).read_text(encoding="utf-8"))
    results = []
    for item in inventory["files"]:
        path = workspace_path(item["path"])
        exists = path.is_file()
        match = exists and path.stat().st_size == item["size_bytes"] and sha256(path) == item["sha256"]
        results.append({"path": item["path"], "exists": exists, "unchanged": match})
    return {"verified_utc": utc_now(), "all_unchanged": all(x["unchanged"] for x in results), "files": results}
