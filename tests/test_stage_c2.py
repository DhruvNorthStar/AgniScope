"""C2 network fixtures are synthetic; all writes are confined to the test cache."""

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
def c2(tmp_path, monkeypatch):
    settings = load_config(workspace_path("configs/phase2b_stage_c2_2023.toml"))
    original = workspace_path
    def resolver(path, **kwargs):
        text = str(path).replace("\\", "/")
        for prefix, target in [(settings['paths']['output'], tmp_path/'o'), (settings['paths']['derived'], tmp_path/'d'), (settings['paths']['annual'], tmp_path/'a')]:
            if text == prefix or text.startswith(prefix+'/'):
                return target/text[len(prefix):].lstrip('/')
        if text.startswith('data/raw/firms/VIIRS_SNPP_SP_2023-') and '2023-04-01_5days' not in text:
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


def batch(c2, session=None, size=1, qc=None):
    settings,_,stub,_=c2
    return b.run_batch(settings,execute=True,batch_size=size,session=session or Session(),sleeper=lambda _:None,qc_function=qc or stub)


def test_365_days_73_exact_boundaries_and_default_dry_run(c2):
    s,resolver,_,_=c2;plan=b.plan_2023(s)
    assert len(plan)==73 and all(r['days']==5 for r in plan)
    expected=[str(date(2023,1,1)+timedelta(days=n)) for n in range(365)]
    actual=[str(date.fromisoformat(r['start_utc_date'])+timedelta(days=n)) for r in plan for n in range(5)]
    assert actual==expected and plan[-1]['start_utc_date']=='2023-12-27'
    session=Session();result=b.run_batch(s,session=session)
    assert result['calendar_days']==365 and result['logical_intervals']==73 and result['pilot_reused'] and result['new_intervals']==72
    assert not session.observation_dates and session.metadata_calls==0 and not resolver(s['paths']['output']).exists()


@pytest.mark.parametrize('year',[2019,2020,2021,2022,2024,2025,2026])
def test_unauthorized_year_rejected_before_any_network(c2,year):
    s,_,_,_=c2;s['study']['start']=f'{year}-01-01';s['study']['end']=f'{year}-12-31'
    with pytest.raises(a.AcquisitionError):batch(c2)


@pytest.mark.parametrize('section,key,value',[('study','stage','C3'),('study','maximum_logical_intervals',74),('study','maximum_new_intervals',73),('batch','concurrency',2),('batch','maximum_new_intervals',11),('scope','other_years_authorized',True),('study','source','VIIRS_NOAA20_SP'),('study','maximum_response_bytes',10485761)])
def test_c2_scope_limits(c2,section,key,value):
    s,_,_,_=c2;s[section][key]=value
    with pytest.raises(a.AcquisitionError):batch(c2)


def test_wrong_wrapper_and_tampered_boundary_rejected(c2):
    s,_,_,_=c2;r=b.plan_2023(s)[0]
    with pytest.raises(a.AcquisitionError):b.acquire_stage_b_interval(s,r,execute=True)
    r['end_utc_date']='2023-01-06'
    with pytest.raises(a.AcquisitionError):b.acquire_stage_c2_interval(s,r,execute=True)


def test_retry_resume_idempotency_and_secret_redaction(c2):
    s,_,_,_=c2;session=Session(failures=[500,200])
    first=batch(c2,session,size=2)
    assert first['unique_new_intervals_attempted']==2 and first['total_observation_attempt_slots']==3
    assert session.observation_dates==['2023-01-01','2023-01-01','2023-01-06']
    replay=Session();r=b.acquire_stage_c2_interval(s,b.plan_2023(s)[0],execute=True,session=replay)
    assert r['qc_status']=='passed' and replay.metadata_calls==0 and not replay.observation_dates
    catalog=b.load_catalog(s);assert SYNTHETIC_KEY not in json.dumps(catalog)
    assert sum(e['reused_existing_pilot'] for e in catalog['intervals'].values())==1
    resumed=Session();batch(c2,resumed);assert resumed.observation_dates==['2023-01-11']


def test_interruption_after_raw_placement_recovers_without_http(c2,monkeypatch):
    original=store.promote_immutable
    def crash(*args):
        original(*args);raise RuntimeError('synthetic interruption')
    monkeypatch.setattr(store,'promote_immutable',crash)
    with pytest.raises(RuntimeError):batch(c2)
    monkeypatch.setattr(store,'promote_immutable',original)
    s,_,_,_=c2;catalog=b.load_catalog(s);session=Session()
    result=b.acquire_stage_c2_interval(s,b.plan_2023(s)[0],execute=True,session=session)
    assert result['acquisition_status']=='acquired_positive' and not session.observation_dates and session.metadata_calls==0


@pytest.mark.parametrize('target',['raw','manifest','protected'])
def test_integrity_failures_stop_before_http(c2,tmp_path,target):
    s,resolver,_,checker=c2;batch(c2);r=b.plan_2023(s)[0];entry=b.load_catalog(s)['intervals'][r['request_id']]
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
    with pytest.raises(store.StoreConflict):batch(c2,session)
    assert not session.observation_dates and session.metadata_calls==0


@pytest.mark.parametrize('statuses,count',[([403],1),([503,503,503],3)])
def test_auth_and_sustained_failure_halt(c2,statuses,count):
    session=Session(failures=statuses);result=batch(c2,session,size=10)
    assert result['status']=='stopped' and len(session.observation_dates)==count and len(set(session.observation_dates))==1
    with pytest.raises(store.StoreConflict):batch(c2)


def test_pending_failed_and_verified_zero_daily_states_are_distinct(c2):
    s,_,_,_=c2;plan=b.plan_2023(s);catalog=b.load_catalog(s,create=True)
    pending=q.coverage_rows(s,catalog,plan)
    assert len(pending)==365 and pending[0]['raw_observations'] is None
    catalog['intervals'][plan[0]['request_id']].update(acquisition_status='failed',failure_category='synthetic_failure')
    failed=q.coverage_rows(s,catalog,plan)
    assert failed[0]['retrieval_state']=='failed' and failed[0]['raw_observations'] is None
    batch(c2,Session(zero_dates={'2023-01-01'}),qc=q.ensure_interval_qc)
    actual=q.coverage_rows(s,b.load_catalog(s),plan)
    assert actual[0]['raw_observations']==0 and actual[0]['retrieval_state']=='zero_returned_detections'
    assert all(r['satellite_coverage']=='unknown' for r in actual)


def test_utc_ist_boundary_identity_and_cross_year_schema(c2):
    s,_,_,_=c2;request=b.plan_2023(s)[-1]
    entry=b.acquire_stage_c2_interval(s,request,execute=True,session=Session(row_date='2023-12-31'),sleeper=lambda _:None)
    summary=q.ensure_interval_qc(s,request,entry)
    frame=pd.read_parquet(q.bundle_path(s,request)/'retained.parquet')
    assert frame.agniscope_utc_date.tolist()==['2023-12-31'] and frame.agniscope_ist_date.tolist()==['2024-01-01']
    assert frame.agniscope_observation_id.tolist()==[entry['sha256']+':1'] and frame.version.tolist()==['2']
    assert summary['qc_passed'] and frame.frp.tolist()==['0']
    schema23=pq.read_schema(q.bundle_path(s,request)/'retained.parquet')
    schema24=pq.read_schema(workspace_path('data/processed/firms_2024/validated_observations_2024.parquet'))
    normalized=lambda schema:{f.name.replace('_core_2024','_core_YEAR').replace('_core_2023','_core_YEAR'):str(f.type) for f in schema}
    assert normalized(schema24)==normalized(schema23)
    membership=q.season_membership(['2023-02-14','2023-02-15','2023-06-15','2023-06-16','2023-11-01'],2023)
    assert membership.agniscope_core_2023.tolist()==[False,True,True,False,False]
    assert membership.agniscope_extended_season_id.iloc[-1]=='EXT_2023_2024'


def test_complete_mock_2023_accounting_duplicates_collection_and_replay(c2):
    s,resolver,_,_=c2;session=Session(zero_dates={'2023-01-01'})
    for _ in range(8):assert batch(c2,session,size=10,qc=q.ensure_interval_qc)['status']=='completed'
    assert len(session.observation_dates)==72 and '2023-04-01' not in session.observation_dates and all(d.startswith('2023-') for d in session.observation_dates)
    catalog=b.load_catalog(s);plan=b.plan_2023(s)
    summary,reused=q.integrate_year(s,catalog,plan)
    assert not reused and summary['counts']['raw_rows']==106 and summary['counts']['retained_rows']==87
    assert summary['counts']['cross_interval_duplicate_occurrences']==0 and summary['literal_version_counts']=={'2':106}
    assert summary['coverage']['validated_dates']==365 and summary['coverage']['raw_zero_dates']==365-len(set(session.observation_dates[1:]) | set(pd.read_csv(workspace_path('data/raw/firms/VIIRS_SNPP_SP_2023-04-01_5days_uttarakhand_bbox.csv')).acq_date))
    frame=pd.read_parquet(resolver(s['paths']['annual'])/'validated_observations_2023.parquet')
    assert frame.agniscope_observation_id.is_unique and frame.agniscope_calendar_year.eq(2023).all()
    again,reused=q.integrate_year(s,catalog,plan);assert reused and again==summary
    replay=Session();result=batch(c2,replay,size=10,qc=q.ensure_interval_qc)
    assert result['new_intervals_this_batch']==0 and not replay.observation_dates and replay.metadata_calls==0


def test_stage_a_reference_cannot_be_requested_or_reserved(c2):
    s,resolver,_,_=c2;request=next(r for r in b.plan_2023(s) if r['proposed_stage']=='A')
    session=Session()
    with pytest.raises(a.AcquisitionError,match='reused'):
        b.acquire_stage_c2_interval(s,request,execute=True,session=session)
    assert not session.observation_dates and session.metadata_calls==0
    catalog=b.load_catalog(s,create=True);entry=catalog['intervals'][request['request_id']]
    assert entry['raw_observation_count']==35 and entry['reused_existing_stage_a'] and entry['attempt_count_stage_b']==0
    catalog['attempt_reservations'].append({'request_id':request['request_id'],'number':1})
    store.atomic_json(resolver(s['paths']['output'])/'acquisition_manifest.json',catalog)
    with pytest.raises(store.StoreConflict):b.load_catalog(s)


def test_reused_stage_a_keeps_original_observation_ids_and_all_source_fields(c2):
    s,_,_,_=c2;request=next(r for r in b.plan_2023(s) if r['proposed_stage']=='A')
    entry=b.pilot_entry(s,request);summary=q.ensure_interval_qc(s,request,entry)
    fresh=pd.read_parquet(q.bundle_path(s,request)/'retained.parquet')
    original=pd.read_parquet(workspace_path('data/interim/phase2b_stage_a/retained_with_landcover.parquet'))
    assert summary['raw_rows']==35 and summary['retained_rows']==16 and summary['qc_passed']
    assert fresh.agniscope_observation_id.tolist()==original.agniscope_observation_id.tolist()
    assert fresh.agniscope_source_record.tolist()==original.agniscope_source_record.tolist()
    assert fresh.version.tolist()==original.version.tolist()


@pytest.mark.parametrize('field',['source_manifest_file','source_manifest_sha256','collection_values'])
def test_conflicting_reused_evidence_is_rejected(c2,field):
    s,resolver,_,_=c2;catalog=b.load_catalog(s,create=True)
    entry=next(e for e in catalog['intervals'].values() if e['reused_existing_pilot'])
    entry[field]='synthetic_conflict'
    store.atomic_json(resolver(s['paths']['output'])/'acquisition_manifest.json',catalog)
    with pytest.raises(store.StoreConflict):b.load_catalog(s)


def test_incompatible_collection_is_preserved_as_failed_qc_without_acceptance(c2):
    from test_stage_b import Response,HEADER
    class WrongCollection(Session):
        def get(self,url,**kwargs):
            if '/area/' not in url:return super().get(url,**kwargs)
            self.observation_dates.append(url.rsplit('/',1)[1])
            return Response(HEADER+b'30,79,320,0.38,0.59,2023-01-01,2330,N,VIIRS,n,2NRT,290,1,N,0\n')
    result=batch(c2,WrongCollection(),size=10)
    assert result['status']=='stopped' and result['qc_passed_intervals']==0
    s,_,_,_=c2;entry=b.load_catalog(s)['intervals'][b.plan_2023(s)[0]['request_id']]
    assert entry['acquisition_status']=='retrieved_qc_failed' and entry['collection_values']==['2NRT']


def test_duplicate_source_rows_remain_raw_and_have_one_canonical_retained_id(c2):
    from test_stage_b import Response,HEADER
    class Duplicated(Session):
        def get(self,url,**kwargs):
            if '/area/' not in url:return super().get(url,**kwargs)
            row=b'30,79,320,0.38,0.59,2023-01-01,2330,N,VIIRS,n,2,290,1,N,0\n'
            self.observation_dates.append('2023-01-01');return Response(HEADER+row+row)
    s,_,_,_=c2;request=b.plan_2023(s)[0]
    entry=b.acquire_stage_c2_interval(s,request,execute=True,session=Duplicated(),sleeper=lambda _:None)
    summary=q.ensure_interval_qc(s,request,entry)
    assert summary['raw_rows']==2 and summary['retained_rows']==1 and summary['qc_contract']['duplicate_rows_removed']==1
    ledger=pd.read_parquet(q.bundle_path(s,request)/'row_ledger.parquet')
    assert len(ledger)==2 and ledger.agniscope_observation_id.is_unique
