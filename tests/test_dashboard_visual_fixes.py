"""Visual changes preserve real geometry, frequency meanings and branding."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
from dashboard import maps,data,ui,ROOT
from dashboard.content import FREQUENCY_COLORS

def contrast(first,second):
    def luminance(rgb):
        values=[v/255 for v in rgb[:3]]
        linear=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in values]
        return sum(v*w for v,w in zip(linear,(.2126,.7152,.0722)))
    a,b=sorted((luminance(first),luminance(second)))
    return (b+.05)/(a+.05)

def test_study_area_fill_and_labels_use_only_verified_district_geometries():
    from shapely.geometry import shape,Point
    value=json.loads(maps.intro_map().to_json())
    districts={f['properties']['District']:f['geometry'] for f in json.loads(json.dumps(maps.district_geojson()))['features']}
    layers={layer['id']:layer for layer in value['layers']}
    assert layers['study-district-labels']['fontFamily']=='Segoe UI'
    assert layers['study-district-labels']['fontWeight']==600
    assert layers['study-district-labels']['fontSettings']['sdf'] is True
    assert 0<layers['study-district-labels']['outlineWidth']<.25
    filled=layers['study-district-fill']['data']['features'];labels=layers['study-district-labels']['data']
    assert len(filled)==len(labels)==len(districts)==13
    assert {f['properties']['District']:f['geometry'] for f in filled}==districts
    assert {row['district'] for row in labels}==set(districts)
    for row in labels:assert shape(districts[row['district']]).covers(Point(row['position']))
    assert layers['study-district-fill']['filled'] and not layers['district-boundaries']['filled']

def test_recurrence_contrast_improves_without_changing_cells_or_frequency():
    frame=data.recurrence_cells().head(3);before=frame.copy(deep=True)
    value,shown=maps.recurrence_map(frame,1000)
    layer=next(row for row in json.loads(value.to_json())['layers'] if row['id']=='recurring-grid-support')
    assert shown==len(frame)==3
    for feature,row in zip(layer['data']['features'],frame.itertuples(index=False)):
        assert feature['geometry']==json.loads(json.dumps(maps.geometry_geojson(row.geometry)))
        assert feature['properties']['cell_id']==str(row.cell_id)
        assert feature['properties']['active_seasons']==row.active_seasons
    for count in range(4,8):
        assert FREQUENCY_COLORS[count][3]==255
        assert contrast(FREQUENCY_COLORS[count],[246,249,251])>=3
    assert layer['autoHighlight'] and layer['getLineColor']=='@@=properties.color'
    import pandas as pd
    pd.testing.assert_frame_equal(frame,before)

def test_visible_branding_and_formal_footer(monkeypatch):
    from dashboard.content import APP_NAME,FORMAL_TITLE
    assert APP_NAME=='AgniScope' and FORMAL_TITLE.startswith(APP_NAME+': ')
    logo=ET.parse(ROOT/'dashboard/assets/wordmark.svg').getroot()
    assert ''.join(logo.itertext()).strip()=='AgniScope'
    captions=[];monkeypatch.setattr(ui.st,'caption',captions.append);monkeypatch.setattr(ui.st,'divider',lambda:None)
    ui.footer()
    assert any(FORMAL_TITLE in value for value in captions)
    assert all('AgniScope AI' not in value for value in captions)

def test_frequency_legend_matches_new_cell_colors(monkeypatch):
    captured=[];monkeypatch.setattr(ui.st,'markdown',lambda value,**kwargs:captured.append(value))
    ui.frequency_legend(4)
    for frequency in range(4,8):
        colour='#'+''.join(f'{v:02x}' for v in FREQUENCY_COLORS[frequency][:3])
        assert colour in captured[0] and f'{frequency}/7' in captured[0]
