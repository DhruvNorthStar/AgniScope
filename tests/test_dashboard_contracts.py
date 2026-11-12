"""Authentic source contracts, filtering, map support and scientific accounting."""
from pathlib import Path
import ast,hashlib,json
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from dashboard import data,maps,ROOT
from dashboard.content import PAGES,RECURRENCE_LABEL,LIMITATION

@pytest.fixture(autouse=True)
def bounded_caches():
    yield
    data._observations.clear();data._cells.clear();data._parquet.clear()

def test_exactly_six_existing_page_files_and_no_model_calls():
    assert len(PAGES)==6 and len({row[2] for row in PAGES})==6
    for _,file,_ in PAGES:assert (ROOT/"dashboard/pages"/file).is_file()
    for file in (ROOT/"dashboard").rglob("*.py"):
        tree=ast.parse(file.read_text(encoding="utf-8"))
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ("fit","fit_predict","read_env","load_dotenv") for n in ast.walk(tree))

def test_catalog_is_bound_to_prior_manifests_and_config():
    catalog=data.catalog();assert len(catalog["references"])==26
    for name,digest in catalog["parent_manifests"].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert catalog["no_scientific_data_copied"]
    for row in catalog["sources"].values():
        assert (ROOT/row["path"]).resolve().is_relative_to(ROOT)
        manifest=json.loads((ROOT/row["verified_source_manifest"]).read_text(encoding="utf-8"))
        assert any(r["path"]==row["path"] and r["sha256"]==row["sha256"] for r in manifest["files"])

def test_master_metadata_and_seven_year_totals():
    with pq.ParquetFile(data.source_file("master")) as file:
        assert file.metadata.num_rows==95052 and len(file.schema_arrow)==49
    annual=data.annual();assert annual.raw.sum()==158959 and annual.retained.sum()==95052
    assert annual.geographic_exclusions.sum()==63907 and annual.other_exclusions.sum()==0
    saved=data.table("annual").sort_values("utc_year").reset_index(drop=True)
    assert set(annual.columns)==set(saved.columns)
    pd.testing.assert_frame_equal(annual.sort_values("utc_year").reset_index(drop=True)[saved.columns],saved,check_dtype=False)

@pytest.mark.parametrize("year,expected",[(2019,15829),(2020,5885),(2021,22812),(2022,15112),(2023,7067),(2024,21728),(2025,6619)])
def test_year_filtered_population_matches_saved_annual(year,expected):
    frame=data.observations(data.ObservationFilter(year=year))
    assert len(frame)==expected and set(frame.agniscope_calendar_year)=={year}
    assert set(frame.columns)==set(data.OBS_COLUMNS) and len(frame.columns)<49
    assert frame.agniscope_observation_id.is_unique

@pytest.mark.parametrize("year,expected",[(2019,14388),(2020,903),(2021,20251),(2022,13782),(2023,4422),(2024,17936),(2025,3215)])
def test_primary_period_inclusive_utc_boundaries(year,expected):
    frame=data.observations(data.ObservationFilter(year=year,primary=True))
    md=pd.to_datetime(frame.agniscope_acquired_utc,utc=True).dt.strftime("%m-%d")
    assert len(frame)==expected and md.between("02-15","06-15").all()

@pytest.mark.parametrize("month",[1,4,12])
def test_month_uses_utc_not_ist(month):
    frame=data.observations(data.ObservationFilter(month=month))
    assert (pd.to_datetime(frame.agniscope_acquired_utc,utc=True).dt.month==month).all()
    expected=int(data.table("monthly").set_index("month").loc[month,"retained_2024"])
    assert len(frame)==expected

def test_utc_and_ist_values_match_actual_conversion():
    frame=data.observations(data.ObservationFilter(year=2024))
    utc=pd.to_datetime(frame.agniscope_acquired_utc,utc=True);ist=utc.dt.tz_convert("Asia/Kolkata")
    assert (ist.dt.strftime("%Y-%m-%d")==frame.agniscope_ist_date).all()
    assert (utc.dt.strftime("%Y-%m-%d")==frame.agniscope_utc_date).all()
    assert (ist==pd.to_datetime(frame.agniscope_acquired_ist)).all()
    assert (frame.agniscope_ist_date!=frame.agniscope_utc_date).any()

@pytest.mark.parametrize("district",["Almora","Pauri Garhwal","Haridwar"])
def test_district_alias_filter_and_saved_totals(district):
    frame=data.observations(data.ObservationFilter(district=district))
    expected=data.single(data.table("district_observations"),display_name=district)["retained_2024"]
    assert len(frame)==expected and set(frame.agniscope_district_display)=={district}

@pytest.mark.parametrize("confidence",["low","nominal","high"])
def test_confidence_filter_preserves_full_default_population(confidence):
    full=data.observations(data.ObservationFilter())
    selected=data.observations(data.ObservationFilter(confidence=(confidence,)))
    assert len(selected)==int((full.agniscope_confidence==confidence).sum())
    assert set(selected.agniscope_confidence)<=set([confidence])

@pytest.mark.parametrize("daynight",["D","N"])
def test_daynight_filter(daynight):
    full=data.observations(data.ObservationFilter())
    selected=data.observations(data.ObservationFilter(daynight=(daynight,)))
    assert len(selected)==int((full.daynight==daynight).sum())

@pytest.mark.parametrize("classes",[(10,),(40,50),(-1,)])
def test_tree_other_and_nodata_filter(classes):
    full=data.observations(data.ObservationFilter())
    values=full.agniscope_worldcover_2021_class
    expected=values.isin(classes)
    if -1 in classes:expected=expected|values.isna()|(values==0)
    selected=data.observations(data.ObservationFilter(landcover=classes))
    assert len(selected)==int(expected.sum())

@pytest.mark.parametrize("selection",[data.ObservationFilter(confidence=()),data.ObservationFilter(daynight=()),data.ObservationFilter(landcover=()),data.ObservationFilter(district="Unassigned / ambiguous")])
def test_explicit_empty_states(selection):
    frame=data.observations(selection);assert frame.empty
    assert data.frp_summary(frame)["median_MW"] is None

@pytest.mark.parametrize("selection",[data.ObservationFilter(year=2018),data.ObservationFilter(year=2026),data.ObservationFilter(month=13)])
def test_invalid_filters_fail(selection):
    with pytest.raises(ValueError):data.observations(selection)

def test_frp_available_denominator_uses_full_population():
    frame=data.observations(data.ObservationFilter());summary=data.frp_summary(frame)
    values=pd.to_numeric(frame.frp,errors="coerce").dropna()
    assert summary["available"]+summary["missing"]==len(frame)
    assert summary["median_MW"]==values.median() and summary["mean_MW"]==values.mean()

def test_display_sample_is_stable_bounded_and_does_not_mutate_metrics():
    frame=data.observations(data.ObservationFilter());before=len(frame)
    one=data.deterministic_sample(frame,4000,"agniscope_observation_id")
    two=data.deterministic_sample(frame.iloc[::-1],4000,"agniscope_observation_id")
    assert len(one)==4000 and len(frame)==before==21728
    assert one.agniscope_observation_id.tolist()==two.agniscope_observation_id.tolist()

@pytest.mark.parametrize("method",["dbscan","hdbscan"])
def test_saved_reference_identity_and_label_accounting(method):
    ref=data.reference("CORE_2024",method);stats=data.model_stats("CORE_2024",method)
    points=data.assignments("CORE_2024",method);clusters=data.cluster_table("CORE_2024",method)
    assert len(points)==ref["n"]==17936
    assert int((points.cluster_label>=0).sum())==stats["n_clustered"]
    assert int((points.cluster_label==-1).sum())==stats["n_noise_minus1"]
    assert int((points.cluster_label<0).sum())==stats["n_nonmember"]
    assert len(clusters)==stats["n_clusters"] and clusters.n_observations.sum()==stats["n_clustered"]
    sizes=points.loc[points.cluster_label>=0].groupby("cluster_label").size().sort_index()
    assert sizes.tolist()==clusters.set_index("cluster_label").n_observations.sort_index().tolist()
    if method=="dbscan":assert ref["parameters"]==dict(algorithm="dbscan",eps_km=1.,min_samples=3)
    else:assert ref["parameters"]==dict(algorithm="hdbscan",min_cluster_size=5,min_samples=5)

def test_default_recurrence_metrics_and_overlap():
    assert [data.recurrence_summary(1000,m,4)["cells"] for m in ("detection","dbscan","hdbscan")]==[1577,699,587]
    assert data.recurrence_summary()["area_km2"]==pytest.approx(1576.456896260,abs=1e-8)
    row=data.single(data.table("overlap"),resolution_m=1000,subset="all",k=4,left="dbscan",right="hdbscan")
    assert row.intersection_cells==390 and row.cell_jaccard==pytest.approx(390/896)

@pytest.mark.parametrize("resolution,expected,area",[(500,325,81.25),(1000,1577,1576.456896260),(2000,2684,10684.761236939)])
def test_resolution_identity_and_area_units(resolution,expected,area):
    frame=data.recurrence_cells(resolution)
    assert len(frame)==expected and frame.cell_id.is_unique
    assert frame.display_scope_area_m2.sum()/1e6==pytest.approx(area,abs=1e-6)
    maps.check_geoparquet_crs(data.source_file(f"cells_{resolution}"))

@pytest.mark.parametrize("k",[2,3,4,5])
def test_k_threshold_filtering_matches_saved_summary(k):
    frame=data.recurrence_cells(1000,"hdbscan",k)
    assert (frame.active_seasons>=k).all() and frame.active_seasons.max()<=7
    assert len(frame)==data.recurrence_summary(1000,"hdbscan",k)["cells"]

def test_district_recurrence_area_is_intersection_not_full_cell_area():
    frame=data.recurrence_cells(1000,"detection",4,"Tehri Garhwal")
    saved=data.recurrence_summary(1000,"detection",4,"Tehri Garhwal")
    assert len(frame)==saved["cells"]
    assert frame.intersection_area_m2.sum()/1e6==pytest.approx(saved["area_km2"],abs=1e-8)
    assert frame.display_scope_area_m2.sum()<=frame.clipped_area_m2.sum()

@pytest.mark.parametrize("values",[(250,"detection",4),(1000,"risk",4),(1000,"detection",1)])
def test_unsupported_recurrence_selection(values):
    with pytest.raises(ValueError):data.recurrence_summary(*values)

def test_geographic_map_coordinates_match_original_source():
    frame=data.observations(data.ObservationFilter()).head(50)
    longitude,latitude=maps.geographic_xy(frame.agniscope_x_m.to_numpy(),frame.agniscope_y_m.to_numpy())
    assert max(abs(longitude-frame.agniscope_longitude))<1e-8
    assert max(abs(latitude-frame.agniscope_latitude))<1e-8
    assert all(77<float(x)<82 for x in longitude) and all(28<float(y)<32 for y in latitude)

def test_saved_cell_geographic_geometry_and_season_masks():
    frame=data.recurrence_cells().head(1);geometry=maps.geometry_geojson(frame.iloc[0].geometry)
    from shapely.geometry import shape
    bounds=shape(geometry).bounds
    assert 77<bounds[0]<82 and 28<bounds[1]<32 and bounds[2]<82 and bounds[3]<32
    assert data.years_from_mask(0)==[] and data.years_from_mask(127)==list(range(2019,2026))
    assert data.years_from_mask(0b1000001)==[2019,2025]


def test_map_style_is_source_free_and_boundary_width_is_a_literal():
    from urllib.parse import unquote
    value=json.loads(maps.intro_map().to_json())
    assert value["mapStyle"].startswith("data:application/json,")
    style=json.loads(unquote(value["mapStyle"].split(",",1)[1]))
    assert style["version"]==8 and style["sources"]=={}
    assert all(layer["type"]=="background" for layer in style["layers"])
    assert not any(key in style for key in ("sprite","glyphs","imports"))
    boundary=next(layer for layer in value["layers"] if layer["id"]=="district-boundaries")
    assert boundary["filled"] is False and boundary["lineWidthUnits"]=="pixels"
    assert boundary["lineWidthMaxPixels"]==1.8
    assert value["initialViewState"]["zoom"]==6.65 # whole state in the full-width overview map


@pytest.mark.parametrize("encoding",[b"",b"gzip"])
def test_static_media_response_emits_headers_before_body(encoding):
    """Exercise the real Streamlit/Starlette middleware without an HTTP socket."""
    import asyncio
    from starlette.responses import Response
    from streamlit.web.server.starlette.starlette_gzip_middleware import MediaAwareGZipMiddleware
    messages=[]
    async def application(scope,receive,send):
        await Response(b"\x89PNG"+b"fixture"*150,media_type="image/png")(scope,receive,send)
    async def send(message):messages.append(message)
    async def receive():return {"type":"http.request","body":b"","more_body":False}
    scope={"type":"http","method":"GET","path":"/media/test.png","headers":[(b"accept-encoding",encoding)]}
    asyncio.run(MediaAwareGZipMiddleware(application)(scope,receive,send))
    assert messages[0]["type"]=="http.response.start" and messages[0]["status"]==200
    assert messages[1]["type"]=="http.response.body" and messages[1]["body"].startswith(b"\x89PNG")

def test_source_corruption_missing_file_and_invalid_schema_fail_explicitly(tmp_path,monkeypatch):
    file=tmp_path/"fixture.csv";file.write_text("x\n1\n",encoding="utf-8")
    good=hashlib.sha256(file.read_bytes()).hexdigest()
    row={"path":"fixture.csv","sha256":good,"size_bytes":file.stat().st_size}
    monkeypatch.setattr(data,"ROOT",tmp_path);monkeypatch.setattr(data,"catalog",lambda:{"sources":{"fixture":row}})
    assert data.source_file("fixture")==file
    file.write_text("x\n2\n",encoding="utf-8");data._hash_matches.cache_clear()
    with pytest.raises(data.DataIntegrityError,match="checksum"):data.source_file("fixture")
    file.unlink()
    with pytest.raises(data.DataIntegrityError,match="missing"):data.source_file("fixture")
    with pytest.raises(data.DataIntegrityError,match="schema"):data.require(pd.DataFrame({"wrong":[]}),["needed"],"fixture")

def test_scientific_labels_are_explicit_and_supervised_scores_absent():
    assert "Observed satellite thermal-detection recurrence" in RECURRENCE_LABEL
    assert "not independent confirmed fire incidents" in LIMITATION
    source="\n".join(path.read_text(encoding="utf-8") for path in (ROOT/"dashboard/pages").glob("*.py"))
    assert "WorldCover 2021" in source and "modifiable areal unit" in source
    assert 'st.metric("Accuracy"' not in source and 'st.metric("Fire risk"' not in source

def test_all_protected_phase4a_delivery_hashes_unchanged():
    seal=ROOT/"outputs/tables/phase4a/file_integrity_manifest.json"
    assert hashlib.sha256(seal.read_bytes()).hexdigest()==data.catalog()["parent_manifests"][seal.relative_to(ROOT).as_posix()]
    for row in json.loads(seal.read_text(encoding="utf-8"))["files"]:
        path=ROOT/row["path"]
        with path.open("rb") as handle:digest=hashlib.file_digest(handle,"sha256").hexdigest()
        assert path.stat().st_size==row["size_bytes"] and digest==row["sha256"]
