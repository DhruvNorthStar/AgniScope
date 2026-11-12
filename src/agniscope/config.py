"""Central configuration and workspace confinement."""

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    target = workspace_path(path or ROOT / "configs/project.toml")
    with target.open("rb") as handle:
        return tomllib.load(handle)


def workspace_path(path: str | Path, *, writable: bool = False) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    target = target.resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Path must remain inside the AgniScope workspace.")
    if writable and target.is_relative_to(ROOT / "data/external"):
        raise ValueError("The GIS source folder is immutable.")
    return target
