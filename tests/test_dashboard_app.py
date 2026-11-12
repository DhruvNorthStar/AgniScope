"""Actual Streamlit runtime tests; these do not claim browser/WebGL visual QA."""
from pathlib import Path
import builtins,json,socket
import pytest
from streamlit.testing.v1 import AppTest
from dashboard import data,ROOT
from dashboard.content import PAGES

def app(page="overview"):
    result=AppTest.from_file(str(ROOT/"dashboard/app.py"),default_timeout=60).run()
    if page!="overview":result.switch_page("pages/"+page+".py").run(timeout=60)
    assert not result.exception and not result.error
    return result

def metric(result,label):return next(item.value for item in result.metric if item.label==label)

@pytest.fixture(autouse=True)
def clear_large_caches():
    yield
    data._observations.clear();data._cells.clear();data._parquet.clear()

@pytest.mark.parametrize("page",[row[2] for row in PAGES])
def test_every_page_starts_without_exceptions(page):
    result=app(page)
    assert len(result.title)==1
    assert not any("accuracy" in item.label.lower() or "prediction" in item.label.lower() for item in result.metric)

def test_overview_totals_and_default_recurrence():
    result=app()
    assert metric(result,"Raw satellite observations")=="158,959"
    assert metric(result,"Validated observations")=="95,052"
    assert metric(result,"Recurring 1 km cells")=="1,577"

def test_explorer_year_period_and_district_controls():
    result=app("explorer")
    assert metric(result,"Selected observations")=="21,728"
    result.selectbox(key="explorer_year").select(2023).run()
    assert metric(result,"Selected observations")=="7,067"
    result.selectbox(key="explorer_period").select("Primary season · Feb 15–Jun 15").run()
    assert metric(result,"Selected observations")=="4,422"
    result.selectbox(key="explorer_district").select("Almora").run()
    expected=len(data.observations(data.ObservationFilter(year=2023,primary=True,district="Almora")))
    assert metric(result,"Selected observations")==f"{expected:,}" and not result.exception

def test_explorer_empty_multiselect_is_explicit():
    result=app("explorer")
    result.multiselect(key="explorer_confidence").set_value([]).run()
    assert metric(result,"Selected observations")=="0"
    assert any("No observations match" in element.value for element in result.info)
    assert not result.exception and not result.error

def test_explorer_confidence_daynight_and_landcover_controls():
    result=app("explorer")
    result.multiselect(key="explorer_confidence").set_value(["nominal"]).run()
    result.multiselect(key="explorer_daynight").set_value(["N"]).run()
    result.multiselect(key="explorer_landcover").set_value([10]).run()
    expected=len(data.observations(data.ObservationFilter(confidence=("nominal",),daynight=("N",),landcover=(10,))))
    assert metric(result,"Selected observations")==f"{expected:,}" and not result.exception

def test_recurrence_controls_use_correct_saved_support_and_resolution():
    result=app("recurrence")
    assert metric(result,"Recurring cells")=="1,577" and metric(result,"DBSCAN / HDBSCAN shared")=="390"
    result.selectbox(key="recurrence_resolution").select(500).run()
    assert metric(result,"Recurring cells")=="325"
    result.selectbox(key="recurrence_method").select("dbscan").run()
    assert metric(result,"Recurring cells")=="192"
    result.selectbox(key="recurrence_k").select(5).run()
    expected=data.recurrence_summary(500,"dbscan",5)
    assert metric(result,"Recurring cells")==f'{expected["cells"]:,}' and not result.exception

def test_recurrence_district_area_and_scope_notice():
    result=app("recurrence")
    result.selectbox(key="recurrence_district").select("Tehri Garhwal").run()
    expected=data.recurrence_summary(1000,"detection",4,"Tehri Garhwal")
    assert metric(result,"Clipped recurring support")==f'{expected["area_km2"]:,.2f} km²'
    assert metric(result,"Connected areas · statewide")=="939"
    assert any("Boundary-straddling" in item.value for item in result.info)

def test_clustering_changes_saved_season_without_fitting(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith("sklearn") or name in ("agniscope.seasonal_clustering","agniscope.recurrence"):
            raise AssertionError("Model/recurrence implementation must not load in the dashboard.")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    result=app("clustering")
    result.selectbox(key="cluster_season").select("CORE_2020").run()
    assert [item.value for item in result.metric if item.label=="Saved spatial groups"]==["58","51"]
    result.radio(key="cluster_view").set_value("HDBSCAN").run()
    assert not result.exception and not result.error
    assert any("47.80 km" in item.value for item in result.warning)

def test_explicit_extended_reference_has_no_invented_parameter_grid():
    result=app("clustering")
    result.selectbox(key="cluster_season").select("EXT_2019_2020").run()
    assert [item.value for item in result.metric if item.label=="Saved spatial groups"]==["154","92"]
    assert any("parameter grid was not executed" in item.value for item in result.info)

def test_cache_returns_isolated_copies_and_ui_reruns_stable():
    selection=data.ObservationFilter(year=2024,month=4)
    first=data.observations(selection);count=len(first);first.drop(first.index,inplace=True)
    second=data.observations(selection)
    assert len(second)==count>0
    result=app();before=[item.value for item in result.metric];result.run()
    assert [item.value for item in result.metric]==before and not result.exception

def test_startup_does_not_read_env_or_use_network(monkeypatch):
    original=Path.open
    def guarded(path,*args,**kwargs):
        if path.name==".env":raise AssertionError("Dashboard attempted to read a NASA credential file.")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    def denied(*args,**kwargs):raise AssertionError("Dashboard attempted a network connection.")
    monkeypatch.setattr(socket.socket,"connect",denied)
    monkeypatch.setenv("FIRMS_MAP_KEY","TEST_CREDENTIAL_MUST_NOT_APPEAR")
    result=app()
    text=" ".join(item.value for item in result.markdown)
    assert "TEST_CREDENTIAL_MUST_NOT_APPEAR" not in text
    assert metric(result,"Validated observations")=="95,052"

def test_missing_optional_figure_is_honest_message(monkeypatch):
    original=data.catalog
    modified=dict(original());modified["sources"]=dict(modified["sources"])
    modified["sources"]["recurrence_figure"]={**modified["sources"]["recurrence_figure"],"path":"outputs/figures/phase4b_optional_missing.png"}
    monkeypatch.setattr(data,"catalog",lambda:modified)
    result=app("research")
    assert any("optional static figure is unavailable" in item.value for item in result.info)
    assert not result.exception

def test_integrity_error_is_not_hidden_or_replaced(monkeypatch):
    def fail(*args,**kwargs):raise data.DataIntegrityError("fixture checksum mismatch")
    monkeypatch.setattr(data,"annual",fail)
    result=AppTest.from_file(str(ROOT/"dashboard/app.py"),default_timeout=60).run()
    assert any("Scientific source integrity check failed" in item.value for item in result.error)
    assert not result.metric
