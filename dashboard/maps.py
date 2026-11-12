"""Small offline deck.gl layers; geographic conversion is explicit and tested."""
import json
from functools import lru_cache
import streamlit as st
from dashboard import data
from dashboard.content import FREQUENCY_COLORS

class MapDisplayError(RuntimeError):pass

@lru_cache(maxsize=1)
def _transformer():
    from pyproj import Transformer
    return Transformer.from_crs(32644,4326,always_xy=True)

def geographic_xy(x,y):
    longitude,latitude=_transformer().transform(x,y)
    return longitude,latitude

def check_geoparquet_crs(path):
    import pyarrow.parquet as pq
    with pq.ParquetFile(path) as file:
        metadata=file.schema_arrow.metadata or {}
    try:
        geo=json.loads(metadata[b"geo"]);column=geo["columns"][geo["primary_column"]]
        crs=column["crs"]
        from pyproj import CRS
        if CRS.from_user_input(crs).to_epsg()!=32644:raise ValueError("Unexpected projected CRS")
    except Exception as error:raise data.DataIntegrityError("A saved map layer does not declare the required EPSG:32644 geometry.") from error

def geometry_geojson(wkb,tolerance_m=0):
    from shapely import from_wkb,transform
    from shapely.geometry import mapping
    geometry=from_wkb(wkb)
    if geometry.is_empty or not geometry.is_valid:raise data.DataIntegrityError("A selected saved map geometry is empty or invalid.")
    if tolerance_m:geometry=geometry.simplify(tolerance_m,preserve_topology=True)
    return mapping(transform(geometry,_transformer().transform,interleaved=False))

@st.cache_data(show_spinner=False,max_entries=1)
def _district_geojson(identity,tolerance):
    import pyarrow.parquet as pq
    check_geoparquet_crs(identity[0])
    rows=pq.read_table(identity[0],columns=["district_display","geometry"]).to_pylist()
    return {"type":"FeatureCollection","features":[{"type":"Feature","geometry":geometry_geojson(row["geometry"],tolerance),
        "properties":{"District":row["district_display"]},"tooltip":row["district_display"]} for row in rows]}

def district_geojson():
    return _district_geojson(data.token("district_boundaries"),data.config()["application"]["boundary_display_tolerance_m"])

def boundary_layer(selected=None):
    import pydeck as pdk
    collection=district_geojson()
    if selected:collection={**collection,"features":[feature for feature in collection["features"] if feature["properties"]["District"]==selected]}
    return pdk.Layer("GeoJsonLayer",collection,id="district-boundaries",pickable=True,filled=False,stroked=True,
        get_line_color=[84,111,130,235],get_line_width=1.2,line_width_units="'pixels'",
        line_width_min_pixels=1,line_width_max_pixels=1.8)

def deck(layers,*,district=None,height=440,compact=False):
    import pydeck as pdk
    focus=[79.32,30.12];zoom=6.0 if compact else 6.65
    if district:
        from shapely.geometry import shape
        match=[f for f in district_geojson()["features"] if f["properties"]["District"]==district]
        if match:
            centre=shape(match[0]["geometry"]).centroid;focus=[centre.x,centre.y];zoom=8.0
    # A null style makes Streamlit substitute its remote CARTO default. Supply a
    # complete source-free style instead; no tiles, glyphs, sprites or token.
    blank_style={"version":8,"sources":{},"layers":[{"id":"background","type":"background",
        "paint":{"background-color":"#f6f9fb"}}]}
    # This Streamlit frontend expects a style URL, even though pydeck accepts a
    # dict. A self-contained data URI keeps that interface entirely offline.
    from urllib.parse import quote
    style_uri="data:application/json,"+quote(json.dumps(blank_style,separators=(",",":")),safe="")
    return pdk.Deck(layers=layers,initial_view_state=pdk.ViewState(longitude=focus[0],latitude=focus[1],zoom=zoom,pitch=0),
        map_style=style_uri,map_provider="mapbox",tooltip={"html":"{tooltip}","style":{"backgroundColor":"#16324a","color":"white","fontSize":"13px"}},
        height=height)

def intro_map():
    """Cartographic district display, with no invented elevation or activity."""
    import pydeck as pdk
    from shapely.geometry import shape
    collection=district_geojson()
    palette=([204,230,222,255],[221,234,211,255],[194,221,226,255],[232,227,210,255])
    features=[];labels=[]
    for index,feature in enumerate(sorted(collection['features'],key=lambda f:f['properties']['District'])):
        name=feature['properties']['District']
        features.append({**feature,'properties':{**feature['properties'],'color':palette[index%len(palette)]}})
        anchor=shape(feature['geometry']).representative_point()
        label=name.replace(' Garhwal','\nGarhwal').replace('Udham Singh Nagar','Udham Singh\nNagar')
        labels.append(dict(district=name,name=label,position=[anchor.x,anchor.y]))
    fill=pdk.Layer('GeoJsonLayer',{'type':'FeatureCollection','features':features},id='study-district-fill',
        filled=True,stroked=False,pickable=True,get_fill_color='properties.color')
    text=pdk.Layer('TextLayer',labels,id='study-district-labels',get_position='position',get_text='name',get_size=14,
        get_color=[23,50,77,255],font_family="'Segoe UI'",font_weight=600,get_text_anchor="'middle'",
        get_alignment_baseline="'center'",font_settings={'sdf':True},outline_width=.1,outline_color=[255,255,255,230],pickable=False)
    return deck([fill,boundary_layer(),text])

def point_map(frame,*,algorithm=None,district=None,limit=None):
    import pydeck as pdk
    import pandas as pd
    limit=limit or data.config()["application"]["point_display_limit"]
    displayed=data.deterministic_sample(frame,limit,"agniscope_observation_id")
    if "agniscope_longitude" in displayed:
        longitude=displayed["agniscope_longitude"].to_numpy();latitude=displayed["agniscope_latitude"].to_numpy()
    else:longitude,latitude=geographic_xy(displayed["agniscope_x_m"].to_numpy(),displayed["agniscope_y_m"].to_numpy())
    if len(displayed) and not (pd.Series(longitude).between(77,82).all() and pd.Series(latitude).between(28,32).all()):
        raise data.DataIntegrityError("Map coordinates are outside the verified geographic study extent.")
    records=[]
    from html import escape
    for (index,row),lon,lat in zip(displayed.iterrows(),longitude,latitude):
        confidence=str(row["agniscope_confidence"])
        if algorithm:
            label=int(row["cluster_label"])
            color=([50,118,150,190] if algorithm=="dbscan" else [200,119,71,190]) if label>=0 else [135,145,152,120]
            category=("cluster "+str(label)) if label>=0 else ("noise (-1)" if label==-1 else "excluded input "+str(label))
        else:color={"low":[217,119,69,155],"nominal":[30,122,115,155],"high":[123,80,132,185]}.get(confidence,[130,140,150,130]);category=confidence+" confidence"
        district_name=row["agniscope_district_display"]
        district_name=str(district_name) if pd.notna(district_name) else "Unassigned / ambiguous"
        text=f"<b>{escape(district_name)}</b><br>{escape(category)}<br>UTC: {escape(str(row['agniscope_acquired_utc']))}<br>IST date: {escape(str(row['agniscope_ist_date']))}<br>FRP: {escape(str(row['frp']))} MW"
        records.append({"position":[float(lon),float(lat)],"color":color,"tooltip":text})
    layer=pdk.Layer("ScatterplotLayer",records,id="thermal-observations",get_position="position",get_fill_color="color",
        get_radius=90,radius_min_pixels=2,radius_max_pixels=5,pickable=True,opacity=.8,stroked=False)
    return deck([layer,boundary_layer(district)],district=district,compact=bool(algorithm)),len(displayed)

def recurrence_map(frame,resolution,*,district=None):
    import pydeck as pdk
    check_geoparquet_crs(data.source_file(f"cells_{resolution}"))
    displayed=data.deterministic_sample(frame,data.config()["application"]["cell_display_limit"],"cell_id")
    features=[]
    for row in displayed.itertuples(index=False):
        years=", ".join(map(str,data.years_from_mask(row.season_mask)))
        frequency=int(row.active_seasons)
        tooltip=f"<b>{frequency} of 7 active primary seasons</b><br>Years: {years}<br>Cell: {row.cell_id}<br>Saved clipped support: {row.clipped_area_m2/1e6:.4f} km²"
        features.append({"type":"Feature","geometry":geometry_geojson(row.geometry),"tooltip":tooltip,"properties":{
            "cell_id":str(row.cell_id),"active_seasons":frequency,"years":years,"color":FREQUENCY_COLORS[frequency],
            "tooltip":tooltip}})
    layer=pdk.Layer("GeoJsonLayer",{"type":"FeatureCollection","features":features},id="recurring-grid-support",pickable=True,
        get_fill_color="properties.color",get_line_color="properties.color",get_line_width=.7,line_width_units="'pixels'",
        line_width_min_pixels=.7,line_width_max_pixels=1.2,stroked=True,filled=True,
        auto_highlight=True,highlight_color=[217,119,69,230])
    return deck([layer,boundary_layer(district)],district=district),len(displayed)
