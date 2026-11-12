"""Durable JSON snapshots, exclusive process locks and immutable file promotion."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid

from .config import workspace_path


class StoreConflict(RuntimeError):
    pass


def staged_bytes(path, body: bytes):
    target = workspace_path(path, writable=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def atomic_json(path, value):
    """A crash exposes either the old complete snapshot or the new complete snapshot."""
    target = workspace_path(path, writable=True)
    temporary = target.with_name(target.name + ".tmp_" + uuid.uuid4().hex)
    staged_bytes(temporary, json_bytes(value))
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def promote_immutable(staged, final):
    """Hard-link creation is atomic and never replaces a destination on Windows/NTFS."""
    staged, final = workspace_path(staged, writable=True), workspace_path(final, writable=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(staged, final)
    except FileExistsError:
        raise StoreConflict("Destination already exists; immutable data was preserved.") from None
    staged.unlink()


@contextmanager
def store_lock(directory):
    directory = workspace_path(directory, writable=True)
    directory.mkdir(parents=True, exist_ok=True)
    # OS-released advisory locking survives a crashed owner without a stale PID lock.
    handle = (directory / "executor.lock").open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise StoreConflict("Another executor owns this interval; no request attempted.") from None
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
