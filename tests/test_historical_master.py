"""Synthetic integration and season fixtures; no network, fitting or source writes."""
from datetime import datetime,timezone,date
from pathlib import Path
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from agniscope import historical_master as m
from agniscope.seasonal_inputs import season_bounds,iter_season
from agniscope.config import workspace_path,load_config
from agniscope.planning import build_request_plan
from agniscope.acquisition_store import StoreConflict

@pytest.fixture
def annuals(tmp_path):
    schema=pq.read_schema(workspace_path('data/processed/firms_2020/validated_observations_2020.parquet'))
    sources={}
    for year in m.YEARS:
        own=pa.schema([pa.field(f.name.replace('_core_2020',f'_core_{year}'),f.type) for f in schema]);rows=[]
        for n,(month,day) in enumerate([(11,1),(2,15),(2,14)]):
            row={}
            for f in own:
                if pa.types.is_timestamp(f.type):value=datetime(year,month,day,tzinfo=timezone.utc)
                elif pa.types.is_boolean(f.type):value=False
                elif pa.types.is_integer(f.type):value=0
                elif pa.types.is_floating(f.type):value=0.0
                else:value='SYNTHETIC_TEST_ONLY'
                row[f.name]=value
            row.update(agniscope_calendar_year=year,agniscope_utc_date=f'{year}-{month:02d}-{day:02d}',
              agniscope_observation_id=f'SYNTHETIC_ID_{year}_{n}',agniscope_source_payload_sha256=f'SYNTHETIC_PAYLOAD_{year}_{n}',
              agniscope_in_footprint=True,agniscope_disposition='retained')
            rows.append(row)
        path=tmp_path/f'{year}.parquet';pq.write_table(pa.Table.from_pylist(rows,schema=own),path);sources[year]=path
    return sources

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def test_deterministic_49_fields_sorted_roundtrip_and_immutability(annuals,tmp_path):
    before={y:digest(p) for y,p in annuals.items()}
    a=tmp_path/'a.parquet';b=tmp_path/'b.parquet'
    first=m.build_sorted_parquet(annuals,a,tmp_path/'a.sqlite')
    second=m.build_sorted_parquet(dict(reversed(list(annuals.items()))),b,tmp_path/'b.sqlite')
    assert first==second and digest(a)==digest(b) and first['field_count']==49 and first['retained_rows']==21
    table=pq.read_table(a);times=table['agniscope_acquired_utc'].cast(pa.int64()).to_pylist()
    assert times==sorted(times) and table.schema.field('agniscope_acquired_ist').type.tz=='Asia/Kolkata'
    assert m.CORE in table.column_names and all(f'agniscope_core_{y}' not in table.column_names for y in m.YEARS)
    assert before=={y:digest(p) for y,p in annuals.items()}

@pytest.mark.parametrize('field',['agniscope_observation_id','agniscope_source_payload_sha256'])
def test_cross_year_duplicates_stop_without_silent_removal(annuals,tmp_path,field):
    table=pq.read_table(annuals[2020]);rows=table.to_pylist();rows[0][field]=pq.read_table(annuals[2019]).to_pylist()[0][field]
    pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),annuals[2020])
    with pytest.raises(StoreConflict,match='Duplicate'):m.build_sorted_parquet(annuals,tmp_path/'out.parquet',tmp_path/'idx.sqlite')

@pytest.mark.parametrize('defect',['missing_year','schema','utc_year','qc'])
def test_incompatible_integration_inputs_rejected(annuals,tmp_path,defect):
    if defect=='missing_year':annuals.pop(2019)
    elif defect=='schema':
        t=pq.read_table(annuals[2020]);pq.write_table(t.drop(['frp']),annuals[2020])
    else:
        t=pq.read_table(annuals[2020]);rows=t.to_pylist()
        if defect=='utc_year':rows[0]['agniscope_utc_date']='2021-11-01'
        else:rows[0]['agniscope_in_footprint']=False
        pq.write_table(pa.Table.from_pylist(rows,schema=t.schema),annuals[2020])
    with pytest.raises(StoreConflict):m.build_sorted_parquet(annuals,tmp_path/'bad.parquet',tmp_path/'bad.sqlite')

def test_master_dates_plan_leaps_and_seven_year_accounting():
    dates=m.independent_calendar();plan=build_request_plan(load_config(workspace_path('configs/historical_protocol.toml')))
    assert len(dates)==len(set(dates))==2557 and dates[0]=='2019-01-01' and dates[-1]=='2025-12-31'
    assert '2020-02-29' in dates and '2024-02-29' in dates and len(plan)==513 and sum(p['days'] for p in plan)==2557
    counts=[{'raw':10,'retained':7,'geographic_exclusions':2,'other_exclusions':1} for _ in m.YEARS]
    assert m.reconcile_counts(counts)=={'raw':70,'retained':49,'geographic_exclusions':14,'other_exclusions':7}
    counts[-1]['raw']=11
    with pytest.raises(StoreConflict):m.reconcile_counts(counts)

@pytest.mark.parametrize('year,days',[(2019,121),(2020,122),(2021,121),(2022,121),(2023,121),(2024,122),(2025,121)])
def test_primary_utc_inclusive_boundaries(year,days):
    b=season_bounds('core',year)
    assert (b['end_exclusive_utc']-b['start_utc']).days==days and b['calendar_supported']
    assert (b['start_utc'].month,b['start_utc'].day)==(2,15) and (b['end_exclusive_utc'].month,b['end_exclusive_utc'].day)==(6,16)

@pytest.mark.parametrize('year,days',[(2020,243),(2021,242),(2022,242),(2023,242),(2024,243),(2025,242)])
def test_six_complete_extended_windows(year,days):
    b=season_bounds('extended',year)
    assert b['calendar_supported'] and not b['left_truncated'] and (b['end_exclusive_utc']-b['start_utc']).days==days

def test_left_truncated_2018_2019_and_ist_boundary():
    b=season_bounds('extended',2019)
    assert b['left_truncated'] and not b['calendar_supported'] and (datetime(2019,1,1,tzinfo=timezone.utc)-b['start_utc']).days==61
    assert season_bounds('extended',2026)['right_truncated']
    local=season_bounds('core',2019,basis='IST');utc=season_bounds('core',2019)
    assert (utc['start_utc']-local['start_utc']).total_seconds()==19800
    assert season_bounds('calendar',2019,basis='IST')['left_truncated']

def test_season_interface_is_deterministic_and_requires_partial_optin(annuals,tmp_path):
    master=tmp_path/'master.parquet';m.build_sorted_parquet(annuals,master,tmp_path/'sort.sqlite')
    rows=[r for table in iter_season(master,'core',2019,batch_size=2) for r in table.to_pylist()]
    assert len(rows)==1 and rows[0]['agniscope_utc_date']=='2019-02-15' and rows[0]['ml_season_id']=='CORE_2019'
    extended=[r for t in iter_season(master,'extended',2020) for r in t.to_pylist()]
    assert [r['agniscope_utc_date'] for r in extended]==['2019-11-01','2020-02-14','2020-02-15']
    assert all(r['ml_calendar_supported'] for r in extended)
    with pytest.raises(ValueError):list(iter_season(master,'extended',2019))
    partial=[r for t in iter_season(master,'extended',2019,allow_partial=True) for r in t.to_pylist()]
    assert len(partial)==2 and all(not r['ml_calendar_supported'] for r in partial)

def test_master_refuses_existing_destination(annuals,tmp_path):
    path=tmp_path/'out.parquet';path.write_bytes(b'SYNTHETIC_IMMUTABLE')
    with pytest.raises(StoreConflict):m.build_sorted_parquet(annuals,path,tmp_path/'idx.sqlite')
    assert path.read_bytes()==b'SYNTHETIC_IMMUTABLE'
