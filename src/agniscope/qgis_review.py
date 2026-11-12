"""Small independent map-review contracts, without GIS/numeric imports."""
import math
import struct
from pathlib import Path


def cell_identity(x, y, resolution):
    code = {500:1, 1000:2, 2000:3}[resolution]
    ix, iy = math.floor(x/resolution), math.floor(y/resolution)
    return code*4294967296+(iy+32768)*65536+ix+32768


def decode(identity):
    word = identity % 4294967296
    return word % 65536-32768, word//65536-32768


def parent_identity(identity, fine, coarse):
    assert coarse % fine == 0 and coarse > fine
    x, y = decode(identity)
    return cell_identity(x*fine, y*fine, coarse)


def merge_year_votes(mapping, fine, coarse):
    result = {}
    for identity, mask in mapping.items():
        parent = parent_identity(identity, fine, coarse)
        result[parent] = result.get(parent, 0) | mask
    return result


def gpkg_wkb(blob):
    assert blob[:2] == b"GP" and blob[2] == 0
    envelope = (blob[3] >> 1) & 7
    assert envelope in (0, 1, 2, 3, 4)
    offset = 8+{0:0,1:32,2:48,3:48,4:64}[envelope]
    assert len(blob) > offset
    return blob[offset:]


def local_datasource(uri, project_dir, workspace):
    filename = uri.split("|",1)[0]
    assert not "://" in filename, "No network layers are permitted in this project."
    target = (Path(project_dir)/filename).resolve()
    assert target.is_relative_to(Path(workspace).resolve()), "Datasource leaves AgniScope."
    assert target.is_file(), "Datasource is missing."
    return target
