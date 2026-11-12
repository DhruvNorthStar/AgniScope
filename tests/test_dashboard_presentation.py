"""Presentation contracts that protect data completeness and readable map layout."""
import json
import pandas as pd
import pytest
from dashboard import ui

def test_horizontal_chart_keeps_every_district_and_population():
    frame=pd.DataFrame({'district':['Tehri Garhwal','Udham Singh Nagar','Almora'],'observations':[12,7,0]})
    before=frame.copy(deep=True)
    result=ui.build_chart(frame,'district:N','observations:Q',horizontal=True,x_title='District',y_title='Observations')
    spec=result.to_dict()
    assert spec['encoding']['y']['field']=='district'
    assert spec['encoding']['y']['type']=='nominal'
    assert spec['encoding']['x']['field']=='observations'
    assert spec['encoding']['x']['type']=='quantitative'
    assert sum(r['observations'] for values in spec['datasets'].values() for r in values)==19
    assert sum(len(values) for values in spec['datasets'].values())==3
    pd.testing.assert_frame_equal(frame,before)

def test_grouped_chart_offsets_follow_orientation():
    frame=pd.DataFrame({'year':['2023','2024']*2,'n':[7,21,13,35],'population':['Retained','Retained','Raw','Raw']})
    for horizontal,offset in ((False,'xOffset'),(True,'yOffset')):
        spec=ui.build_chart(frame,'year:O','n:Q',color='population:N',horizontal=horizontal).to_dict()
        assert spec['encoding'][offset]['field']=='population'
        assert spec['encoding']['color']['field']=='population'
        assert len(next(iter(spec['datasets'].values())))==4

def test_map_height_is_forwarded_without_altering_deck(monkeypatch):
    captured=[]
    monkeypatch.setattr(ui.st,'pydeck_chart',lambda value,**kwargs:captured.append((value,kwargs)))
    monkeypatch.setattr(ui.st,'caption',lambda *args,**kwargs:None)
    deck=object()
    ui.render_map(deck,'reference_map',height=650)
    assert captured==[(deck,dict(key='reference_map',width='stretch',height=650,on_select='ignore'))]

def test_line_chart_keeps_time_axis_and_no_category_offset():
    frame=pd.DataFrame({'date':['2024-04-01','2024-04-02'],'n':[8,3]})
    spec=ui.build_chart(frame,'date:T','n:Q',kind='line').to_dict()
    assert spec['encoding']['x']['type']=='temporal'
    assert spec['mark']['type']=='line'
    assert 'xOffset' not in spec['encoding'] and 'yOffset' not in spec['encoding']
