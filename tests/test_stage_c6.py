"""C6 network fixtures are synthetic; all writes are confined to the test cache."""

from copy import deepcopy
from datetime import date, timedelta
import json
import pandas as pd
import pyarrow.parquet as pq
import pytest

from agniscope import acquisition as a, acquisition_store as store, stage_b as b, stage_b_qc as q
from agniscope.config import load_config, workspace_path
from agniscope.provenance import sha256
from test_stage_b import Session, SYNTHETIC_KEY


@pytest.fixture
def c6(tmp_path, monkeypatch):
    settings = load_config(workspace_path("configs/phase2b_stage_c6_2019.toml"))
    original = workspace_path
    def resolver(path, **kwargs):
        text = str(path).replace("\\", "/")
        for prefix, target in [(settings['paths']['output'], tmp_path/'o'), (settings['paths']['derived'], tmp_path/'d'), (settings['paths']['annual'], tmp_path/'a')]:
            if text == prefix or text.startswith(prefix+'/'):
                return target/text[len(prefix):].lstrip('/')
        if text.startswith('data/raw/firms/VIIRS_SNPP_SP_2019-'):
            return tmp_path/'r'/text.rsplit('/',1)[1]
        return original(path, **kwargs)
    monkeypatch.setattr(b,'workspace_path',resolver)
    monkeypatch.setattr(q,'workspace_path',resolver)
    checker=b.protected_integrity
    monkeypatch.setattr(b,'protected_integrity',lambda s: {'synthetic':True})
    monkeypatch.setattr(b,'resource_snapshot',lambda: {'disk_free_bytes':32*1024**3,'available_physical_bytes':2*1024**3})
    monkeypatch.setattr(a,'local_map_key',lambda:SYNTHETIC_KEY)
    def stub(settings,request,entry):
        path=tmp_path/('q_'+request['request_id'][6:22]+'.json')
        result={'qc_passed':True,'retained_rows':entry['raw_observation_count'],'qc_summary_file':path.relative_to(original('.')).as_posix()}
        store.atomic_json(path,result);return result
    return settings,resolver,stub,checker


def batch(c6, session=None, size=1, qc=None):
    settings,_,stub,_=c6
    return b.run_batch(settings,execute=True,batch_size=size,session=session or Session(),sleeper=lambda _:None,qc_function=qc or stub)


def test_365_days_73_exact_boundaries_and_default_dry_run(c6):
    s,resolver,_,_=c6;plan=b.plan_2019(s)
    assert len(plan)==73 and all(r['days']==5 for r in plan)
    expected=[str(date(2019,1,1)+timedelta(days=n)) for n in range(365)]
    actual=[str(date.fromisoformat(r['start_utc_date'])+timedelta(days=n)) for r in plan for n in range(5)]
    assert actual==expected and plan[-1]['start_utc_date']=='2019-12-27'
    session=Session();result=b.run_batch(s,session=session)
    assert result['calendar_days']==365 and result['logical_intervals']==73 and not result['pilot_reused']
    assert not session.observation_dates and session.metadata_calls==0 and not resolver(s['paths']['output']).exists()


@pytest.mark.parametrize('year',[2018,2020,2021,2022,2023,2024,2025,2026])
def test_unauthorized_year_rejected_before_any_network(c6,year):
    s,_,_,_=c6;s['study']['start']=f'{year}-01-01';s['study']['end']=f'{year}-12-31'
    with pytest.raises(a.AcquisitionError):batch(c6)


@pytest.mark.parametrize('section,key,value',[('study','stage','C4'),('study','maximum_logical_intervals',74),('study','maximum_new_intervals',74),('batch','concurrency',2),('batch','maximum_new_intervals',11),('scope','other_years_authorized',True),('study','source','VIIRS_NOAA20_SP'),('study','maximum_response_bytes',10485761)])
def test_c6_scope_limits(c6,section,key,value):
    s,_,_,_=c6;s[section][key]=value
    with pytest.raises(a.AcquisitionError):batch(c6)


def test_wrong_wrapper_and_tampered_boundary_rejected(c6):
    s,_,_,_=c6;r=b.plan_2019(s)[0]
    with pytest.raises(a.AcquisitionError):b.acquire_stage_b_interval(s,r,execute=True)
    r['end_utc_date']='2019-01-06'
    with pytest.raises(a.AcquisitionError):b.acquire_stage_c6_interval(s,r,execute=True)


def test_retry_resume_idempotency_and_secret_redaction(c6):
    s,_,_,_=c6;session=Session(failures=[500,200])
    first=batch(c6,session,size=2)
    assert first['unique_new_intervals_attempted']==2 and first['total_observation_attempt_slots']==3
    assert session.observation_dates==['2019-01-01','2019-01-01','2019-01-06']
    replay=Session();r=b.acquire_stage_c6_interval(s,b.plan_2019(s)[0],execute=True,session=replay)
    assert r['qc_status']=='passed' and replay.metadata_calls==0 and not replay.observation_dates
    catalog=b.load_catalog(s);assert SYNTHETIC_KEY not in json.dumps(catalog)
    assert all(not e['reused_existing_pilot'] for e in catalog['intervals'].values())
    resumed=Session();batch(c6,resumed);assert resumed.observation_dates==['2019-01-11']


def test_interruption_after_raw_placement_recovers_without_http(c6,monkeypatch):
    original=store.promote_immutable
    def crash(*args):
        original(*args);raise RuntimeError('synthetic interruption')
    monkeypatch.setattr(store,'promote_immutable',crash)
    with pytest.raises(RuntimeError):batch(c6)
    monkeypatch.setattr(store,'promote_immutable',original)
    s,_,_,_=c6;catalog=b.load_catalog(s);session=Session()
    result=b.acquire_stage_c6_interval(s,b.plan_2019(s)[0],execute=True,session=session)
    assert result['acquisition_status']=='acquired_positive' and not session.observation_dates and session.metadata_calls==0


@pytest.mark.parametrize('target',['raw','manifest','protected'])
def test_integrity_failures_stop_before_http(c6,tmp_path,target):
    s,resolver,_,checker=c6;batch(c6);r=b.plan_2019(s)[0];entry=b.load_catalog(s)['intervals'][r['request_id']]
    if target=='raw':workspace_path(entry['raw_file']).write_bytes(b'synthetic tampering')
    elif target=='manifest':
        catalog=b.load_catalog(s);catalog['intervals'][r['request_id']]['collection_values']=['2NRT']
        store.atomic_json(resolver(s['paths']['output'])/'acquisition_manifest.json',catalog)
    else:
        asset=tmp_path/'protected';asset.write_bytes(b'original')
        store.atomic_json(resolver(s['paths']['preflight']),{'protected_files':[{'path':str(asset),'sha256':sha256(asset)}]})
        asset.write_bytes(b'changed')
        with pytest.raises(store.StoreConflict):checker(s)
        return
    session=Session()
    with pytest.raises(store.StoreConflict):batch(c6,session)
    assert not session.observation_dates and session.metadata_calls==0


@pytest.mark.parametrize('statuses,count',[([403],1),([503,503,503],3)])
def test_auth_and_sustained_failure_halt(c6,statuses,count):
    session=Session(failures=statuses);result=batch(c6,session,size=10)
    assert result['status']=='stopped' and len(session.observation_dates)==count and len(set(session.observation_dates))==1
    with pytest.raises(store.StoreConflict):batch(c6)


def test_pending_failed_and_verified_zero_daily_states_are_distinct(c6):
    s,_,_,_=c6;plan=b.plan_2019(s);catalog=b.load_catalog(s,create=True)
    pending=q.coverage_rows(s,catalog,plan)
    assert len(pending)==365 and pending[0]['raw_observations'] is None
    catalog['intervals'][plan[0]['request_id']].update(acquisition_status='failed',failure_category='synthetic_failure')
    failed=q.coverage_rows(s,catalog,plan)
    assert failed[0]['retrieval_state']=='failed' and failed[0]['raw_observations'] is None
    batch(c6,Session(zero_dates={'2019-01-01'}),qc=q.ensure_interval_qc)
    actual=q.coverage_rows(s,b.load_catalog(s),plan)
    assert actual[0]['raw_observations']==0 and actual[0]['retrieval_state']=='zero_returned_detections'
    assert all(r['satellite_coverage']=='unknown' for r in actual)


def test_utc_ist_boundary_identity_and_cross_year_schema(c6):
    s,_,_,_=c6;request=b.plan_2019(s)[-1]
    entry=b.acquire_stage_c6_interval(s,request,execute=True,session=Session(row_date='2019-12-31'),sleeper=lambda _:None)
    summary=q.ensure_interval_qc(s,request,entry)
    frame=pd.read_parquet(q.bundle_path(s,request)/'retained.parquet')
    assert frame.agniscope_utc_date.tolist()==['2019-12-31'] and frame.agniscope_ist_date.tolist()==['2020-01-01']
    assert frame.agniscope_observation_id.tolist()==[entry['sha256']+':1'] and frame.version.tolist()==['2']
    assert summary['qc_passed'] and frame.frp.tolist()==['0']
    schema25=pq.read_schema(q.bundle_path(s,request)/'retained.parquet')
    schema24=pq.read_schema(workspace_path('data/processed/firms_2024/validated_observations_2024.parquet'))
    normalized=lambda schema:{f.name.replace('_core_2024','_core_YEAR').replace('_core_2019','_core_YEAR'):str(f.type) for f in schema}
    assert normalized(schema24)==normalized(schema25)
    membership=q.season_membership(['2019-02-14','2019-02-15','2019-06-15','2019-06-16','2019-11-01'],2019)
    assert membership.agniscope_core_2019.tolist()==[False,True,True,False,False]
    assert membership.agniscope_extended_season_id.iloc[-1]=='EXT_2019_2020'


def test_complete_mock_2019_accounting_duplicates_collection_and_replay(c6):
    s,resolver,_,_=c6;session=Session(zero_dates={'2019-01-01'})
    for _ in range(8):assert batch(c6,session,size=10,qc=q.ensure_interval_qc)['status']=='completed'
    assert len(session.observation_dates)==73 and all(d.startswith('2019-') for d in session.observation_dates)
    catalog=b.load_catalog(s);plan=b.plan_2019(s)
    summary,reused=q.integrate_year(s,catalog,plan)
    assert not reused and summary['counts']['raw_rows']==72 and summary['counts']['retained_rows']==72
    assert summary['counts']['cross_interval_duplicate_occurrences']==0 and summary['literal_version_counts']=={'2':72}
    assert summary['coverage']['validated_dates']==365 and summary['coverage']['raw_zero_dates']==293
    frame=pd.read_parquet(resolver(s['paths']['annual'])/'validated_observations_2019.parquet')
    assert frame.agniscope_observation_id.is_unique and frame.agniscope_calendar_year.eq(2019).all()
    again,reused=q.integrate_year(s,catalog,plan);assert reused and again==summary
    replay=Session();result=batch(c6,replay,size=10,qc=q.ensure_interval_qc)
    assert result['new_intervals_this_batch']==0 and not replay.observation_dates and replay.metadata_calls==0


def test_response_cap_is_enforced_before_acceptance(c6):
    s,_,_,_=c6;s['study']['maximum_response_bytes']=10
    result=batch(c6)
    assert result['status']=='stopped' and result['halt_reason']=='response_size_limit'


def test_unexpected_collection_stops_without_relabelling(c6):
    from test_stage_b import Response,HEADER
    class Incompatible(Session):
        def get(self,url,**kwargs):
            if '/area/' not in url:return super().get(url,**kwargs)
            self.observation_dates.append(url.rsplit('/',1)[1])
            return Response(HEADER+b'30,79,320,0.38,0.59,2019-01-01,2330,N,VIIRS,n,2NRT,290,1,N,0\n')
    result=batch(c6,Incompatible(),size=10);assert result['status']=='stopped'
    s,_,_,_=c6;entry=b.load_catalog(s)['intervals'][b.plan_2019(s)[0]['request_id']]
    assert entry['collection_values']==['2NRT'] and entry['acquisition_status']=='retrieved_qc_failed'


def test_exact_duplicates_remain_in_raw_and_ledger(c6):
    from test_stage_b import Response,HEADER
    class Duplicate(Session):
        def get(self,url,**kwargs):
            if '/area/' not in url:return super().get(url,**kwargs)
            row=b'30,79,320,0.38,0.59,2019-01-01,2330,N,VIIRS,n,2,290,1,N,0\n'
            return Response(HEADER+row+row)
    s,_,_,_=c6;r=b.plan_2019(s)[0]
    entry=b.acquire_stage_c6_interval(s,r,execute=True,session=Duplicate(),sleeper=lambda _:None)
    summary=q.ensure_interval_qc(s,r,entry)
    assert summary['raw_rows']==2 and summary['retained_rows']==1 and summary['qc_contract']['duplicate_rows_removed']==1
    assert len(pd.read_parquet(q.bundle_path(s,r)/'row_ledger.parquet'))==2


def test_c6_review_register_is_additive_and_does_not_change_prior_qc(c6):
    settings,_,_,_=c6
    original=q.issue_register();review=q.issue_register(settings)
    assert len(original)==15 and len(review)==34 and review[:15]==original
    assert 'snpp_2019_sbc_review_range' in q.issue_ids('2019-01-22T01:00:00Z',review)
    assert 'snpp_2019_sbc_review_range' not in q.issue_ids('2019-01-22T01:25:00Z',review)
    assert len(q.issue_register(load_config(workspace_path('configs/phase2b_stage_c3_2022.toml'))))==31
