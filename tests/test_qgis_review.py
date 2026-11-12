"""Map-review boundary/support contracts and the Windows display-schema regression."""
from pathlib import Path
from contextlib import closing
import sqlite3
import struct
import pytest
from agniscope.qgis_review import cell_identity,decode,parent_identity,merge_year_votes,gpkg_wkb,local_datasource


@pytest.mark.parametrize("resolution",[500,1000,2000])
def test_half_open_boundary_ownership(resolution):
    assert decode(cell_identity(0,0,resolution))==(0,0)
    assert decode(cell_identity(-.001,-.001,resolution))==(-1,-1)
    assert decode(cell_identity(resolution,2*resolution,resolution))==(1,2)


@pytest.mark.parametrize("fine,coarse",[(500,1000),(500,2000),(1000,2000)])
def test_coarse_votes_are_year_union_not_sum_or_max(fine,coarse):
    one=cell_identity(10,10,fine);two=cell_identity(fine+10,10,fine)
    children={one:0b0000011,two:0b0001100}
    parent=cell_identity(10,10,coarse)
    assert merge_year_votes(children,fine,coarse)=={parent:0b0001111}
    assert all(mask.bit_count()<4 for mask in children.values())
    assert merge_year_votes(children,fine,coarse)[parent].bit_count()==4


@pytest.mark.parametrize("fine,coarse",[(500,1000),(500,2000),(1000,2000)])
def test_signed_parent_assignment_uses_floor(fine,coarse):
    child=cell_identity(-.1,-fine-.1,fine)
    assert parent_identity(child,fine,coarse)==cell_identity(-.1,-fine-.1,coarse)


@pytest.mark.parametrize("envelope,length",[(0,0),(1,32),(2,48),(3,48),(4,64)])
def test_geopackage_header_envelope_is_not_geometry(envelope,length):
    wkb=struct.pack("<BIdd",1,1,300000.,3300000.)
    blob=b"GP"+bytes([0,1+(envelope<<1)])+struct.pack("<i",32644)+bytes(length)+wkb
    assert gpkg_wkb(blob)==wkb


def test_relative_paths_remain_inside_project(tmp_path):
    project=tmp_path/"qgis";project.mkdir();file=tmp_path/"layer.gpkg";file.write_bytes(b"fixture")
    assert local_datasource("../layer.gpkg|layername=points",project,tmp_path)==file
    with pytest.raises(AssertionError):local_datasource("../../outside.gpkg",project,tmp_path)
    with pytest.raises(AssertionError):local_datasource("https://example.invalid/layer",project,tmp_path)
    with pytest.raises(AssertionError):local_datasource("missing.gpkg",project,tmp_path)


def test_null_first_batch_strength_stays_real_and_file_handle_closes(tmp_path):
    import pandas as pd
    import geopandas as gpd
    import pyogrio
    from shapely.geometry import Point
    stage=tmp_path/"display.staged.gpkg";target=tmp_path/"display.gpkg"
    for index,values in enumerate(([None,None],[0.6777537778349222,0.])):
        frame=gpd.GeoDataFrame({"strength":pd.array(values,dtype="Float64")},geometry=[Point(0,0),Point(1,1)],crs=32644)
        pyogrio.write_dataframe(frame,stage,layer="points",driver="GPKG",append=bool(index))
    with closing(sqlite3.connect(stage)) as connection:
        column=next(row for row in connection.execute("PRAGMA table_info(points)") if row[1]=="strength")
        assert column[2]=="REAL"
        assert connection.execute("SELECT strength FROM points ORDER BY fid").fetchall()==[(None,),(None,),(0.6777537778349222,),(0.,)]
    stage.rename(target)
    assert target.is_file() and not stage.exists()
