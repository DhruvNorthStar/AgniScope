import copy

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from agniscope.landcover import sample_landcover
from agniscope.spatial import load_boundaries


def test_categorical_sampling_and_nodata(tmp_path, config):
    folder = tmp_path / "synthetic_tiles"
    folder.mkdir()
    tile = folder / "synthetic_worldcover.tif"
    with rasterio.open(tile, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8", crs=4326, transform=from_origin(78, 31, 0.5, 0.5), nodata=0) as ds:
        ds.write(np.array([[10, 40], [0, 30]], dtype=np.uint8), 1)
    config["paths"]["worldcover_raw"] = str(folder)
    points = gpd.GeoDataFrame(geometry=[Point(78.25, 30.75), Point(78.75, 30.75), Point(78.25, 30.25), Point(80, 30)], crs=4326)
    result = sample_landcover(points, config)
    assert result.iloc[0]["agniscope_worldcover_2021_class"] == 10
    assert result.iloc[1]["agniscope_worldcover_2021_class"] == 40
    assert result["agniscope_landcover_status"].tolist() == ["sampled", "sampled", "nodata", "outside_tiles"]


def test_original_tiles_sample_actual_district_representative_points(config):
    _, districts = load_boundaries(config)
    points = gpd.GeoDataFrame(geometry=districts.geometry.representative_point(), crs=districts.crs)
    result = sample_landcover(points, config)
    assert len(result) == 13
    assert result["agniscope_landcover_status"].eq("sampled").all()
    assert set(result["agniscope_worldcover_2021_class"]).issubset({10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100})
