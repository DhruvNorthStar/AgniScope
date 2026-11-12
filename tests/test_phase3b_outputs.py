"""Independent spatial QA kernels: holes, translated coordinates and diagonals."""
import sys
from pathlib import Path
import pytest
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from verify_phase3b_results import polygon_area, union_find


def cell(x, y, resolution_code=2):
    return resolution_code*4294967296+(y+32768)*65536+x+32768


@pytest.mark.parametrize("offset", [0, 3200000])
def test_manual_area_subtracts_holes_at_projected_coordinates(offset):
    outer = [(offset, offset), (offset+10, offset), (offset+10, offset+10), (offset, offset+10)]
    hole = [(offset+2, offset+2), (offset+4, offset+2), (offset+4, offset+5), (offset+2, offset+5)]
    assert polygon_area(Polygon(outer, [hole])) == 94


def test_manual_area_multipart_ignores_nonarea_geometry():
    a = Polygon([(0, 0), (2, 0), (2, 3), (0, 3)])
    b = Polygon([(10, 0), (11, 0), (11, 1), (10, 1)])
    assert polygon_area(MultiPolygon([a, b])) == 7
    assert polygon_area(GeometryCollection([a, LineString([(0, 0), (3, 3)])])) == 6


@pytest.mark.parametrize("adjacency,groups", [(4, 3), (8, 2)])
def test_union_find_diagonal_and_isolated_cells(adjacency, groups):
    cells = {cell(-1, -1), cell(0, 0), cell(1, 0), cell(100, 100)}
    result = union_find(cells, adjacency)
    assert len(result) == groups
    assert set().union(*result.values()) == cells
    assert sum(map(len, result.values())) == len(cells)
    assert all(identity == min(members) for identity, members in result.items())
    assert result == union_find(set(reversed(sorted(cells))), adjacency)
