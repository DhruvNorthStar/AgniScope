"""All generated fixtures are synthetic and remain in the workspace test cache."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from agniscope.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def districts():
    return gpd.GeoDataFrame({"district": ["A", "B"], "district_original": ["A", "B"], "agniscope_display_name": ["District A", "District B"]}, geometry=[box(500000, 3300000, 500100, 3300100), box(500100, 3300000, 500200, 3300100)], crs=32644)


@pytest.fixture
def valid_record(districts):
    point = districts.geometry.iloc[0].representative_point()
    geographic = gpd.GeoSeries([point], crs=districts.crs).to_crs(4326).iloc[0]
    return {"latitude": str(geographic.y), "longitude": str(geographic.x), "acq_date": "2024-04-01", "acq_time": "3", "satellite": "N", "confidence": "n", "version": "2.0", "bright_ti4": "320", "bright_ti5": "290", "scan": "0.38", "track": "0.59", "frp": "2", "daynight": "N", "instrument": "VIIRS", "custom_source_field": "preserve me"}
