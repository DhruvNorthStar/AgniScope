"""C4 network fixtures are synthetic; all writes are confined to the test cache."""

from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import json
import pandas as pd
import pyarrow.parquet as pq
import pytest

from agniscope import acquisition as a, acquisition_store as store, stage_b as b, stage_b_qc as q
from agniscope.config import load_config, workspace_path
from agniscope.provenance import sha256
from agniscope import paired_acquisition as p
from test_stage_b import Session, SYNTHETIC_KEY


@pytest.fixture
def c4(tmp_path, monkeypatch):
    monkeypatch.setattr(p,'require_authorization',lambda settings: {'synthetic':True})
    settings = load_config(workspace_path("configs/phase2b_stage_c4_2021.toml"))
    original = workspace_path
    def resolver(path, **kwargs):
        text = str(path).replace("\\", "/")
        for prefix, target in [(settings['paths']['output'], tmp_path/'o'), (settings['paths']['derived'], tmp_path/'d'), (settings['paths']['annual'], tmp_path/'a')]:
            if text == prefix or text.startswith(prefix+'/'):
                return target/text[len(prefix):].lstrip('/')
        if text.startswith(('data/raw/firms/VIIRS_SNPP_SP_2021-','data/raw/firms/VIIRS_SNPP_SP_2020-')):
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


def batch(c4, session=None, size=1, qc=None):
    settings,_,stub,_=c4
    return b.run_batch(settings,execute=True,batch_size=size,session=session or Session(),sleeper=lambda _:None,qc_function=qc or stub)


def test_365_days_73_exact_boundaries_and_default_dry_run(c4):
    s,resolver,_,_=c4;plan=b.plan_c4_c5(s)
    assert len(plan)==73 and all(r['days']==5 for r in plan)
    expected=[str(date(2021,1,1)+timedelta(days=n)) for n in range(365)]
    actual=[str(date.fromisoformat(r['start_utc_date'])+timedelta(days=n)) for r in plan for n in range(5)]
    assert actual==expected and plan[-1]['start_utc_date']=='2021-12-27'
    session=Session();result=b.run_batch(s,session=session)
    assert result['calendar_days']==365 and result['logical_intervals']==73 and not result['pilot_reused']
    assert not session.observation_dates and session.metadata_calls==0 and not resolver(s['paths']['output']).exists()


@pytest.mark.parametrize('year',[2019,2020,2022,2023,2024,2025,2026])
def test_unauthorized_year_rejected_before_any_network(c4,year):
    s,_,_,_=c4;s['study']['start']=f'{year}-01-01';s['study']['end']=f'{year}-12-31'
    with pytest.raises(a.AcquisitionError):batch(c4)


@pytest.mark.parametrize('section,key,value',[('study','stage','C6'),('study','maximum_logical_intervals',74),('study','maximum_new_intervals',74),('batch','concurrency',2),('batch','maximum_new_intervals',11),('scope','other_years_authorized',True),('study','source','VIIRS_NOAA20_SP'),('study','maximum_response_bytes',10485761)])
def test_c4_scope_limits(c4,section,key,value):
    s,_,_,_=c4;s[section][key]=value
    with pytest.raises(a.AcquisitionError):batch(c4)


def test_wrong_wrapper_and_tampered_boundary_rejected(c4):
    s,_,_,_=c4;r=b.plan_c4_c5(s)[0]
    with pytest.raises(a.AcquisitionError):b.acquire_stage_b_interval(s,r,execute=True)
    r['end_utc_date']='2021-01-06'
    with pytest.raises(a.AcquisitionError):b.acquire_c4_c5_interval(s,r,execute=True)


def test_retry_resume_idempotency_and_secret_redaction(c4):
    s,_,_,_=c4;session=Session(failures=[500,200])
    first=batch(c4,session,size=2)
    assert first['unique_new_intervals_attempted']==2 and first['total_observation_attempt_slots']==3
    assert session.observation_dates==['2021-01-01','2021-01-01','2021-01-06']
    replay=Session();r=b.acquire_c4_c5_interval(s,b.plan_c4_c5(s)[0],execute=True,session=replay)
    assert r['qc_status']=='passed' and replay.metadata_calls==0 and not replay.observation_dates
    catalog=b.load_catalog(s);assert SYNTHETIC_KEY not in json.dumps(catalog)
    assert all(not e['reused_existing_pilot'] for e in catalog['intervals'].values())
    resumed=Session();batch(c4,resumed);assert resumed.observation_dates==['2021-01-11']


def test_interruption_after_raw_placement_recovers_without_http(c4,monkeypatch):
    original=store.promote_immutable
    def crash(*args):
        original(*args);raise RuntimeError('synthetic interruption')
    monkeypatch.setattr(store,'promote_immutable',crash)
    with pytest.raises(RuntimeError):batch(c4)
    monkeypatch.setattr(store,'promote_immutable',original)
    s,_,_,_=c4;catalog=b.load_catalog(s);session=Session()
    result=b.acquire_c4_c5_interval(s,b.plan_c4_c5(s)[0],execute=True,session=session)
    assert result['acquisition_status']=='acquired_positive' and not session.observation_dates and session.metadata_calls==0


@pytest.mark.parametrize('target',['raw','manifest','protected'])
def test_integrity_failures_stop_before_http(c4,tmp_path,target):
    s,resolver,_,checker=c4;batch(c4);r=b.plan_c4_c5(s)[0];entry=b.load_catalog(s)['intervals'][r['request_id']]
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
    with pytest.raises(store.StoreConflict):batch(c4,session)
    assert not session.observation_dates and session.metadata_calls==0


@pytest.mark.parametrize('statuses,count',[([403],1),([503,503,503],3)])
def test_auth_and_sustained_failure_halt(c4,statuses,count):
    session=Session(failures=statuses);result=batch(c4,session,size=10)
    assert result['status']=='stopped' and len(session.observation_dates)==count and len(set(session.observation_dates))==1
    with pytest.raises(store.StoreConflict):batch(c4)


def test_pending_failed_and_verified_zero_daily_states_are_distinct(c4):
    s,_,_,_=c4;plan=b.plan_c4_c5(s);catalog=b.load_catalog(s,create=True)
    pending=q.coverage_rows(s,catalog,plan)
    assert len(pending)==365 and pending[0]['raw_observations'] is None
    catalog['intervals'][plan[0]['request_id']].update(acquisition_status='failed',failure_category='synthetic_failure')
    failed=q.coverage_rows(s,catalog,plan)
    assert failed[0]['retrieval_state']=='failed' and failed[0]['raw_observations'] is None
    batch(c4,Session(zero_dates={'2021-01-01'}),qc=q.ensure_interval_qc)
    actual=q.coverage_rows(s,b.load_catalog(s),plan)
    assert actual[0]['raw_observations']==0 and actual[0]['retrieval_state']=='zero_returned_detections'
    assert all(r['satellite_coverage']=='unknown' for r in actual)


def test_utc_ist_boundary_identity_and_cross_year_schema(c4):
    s,_,_,_=c4;request=b.plan_c4_c5(s)[-1]
    entry=b.acquire_c4_c5_interval(s,request,execute=True,session=Session(row_date='2021-12-31'),sleeper=lambda _:None)
    summary=q.ensure_interval_qc(s,request,entry)
    frame=pd.read_parquet(q.bundle_path(s,request)/'retained.parquet')
    assert frame.agniscope_utc_date.tolist()==['2021-12-31'] and frame.agniscope_ist_date.tolist()==['2022-01-01']
    assert frame.agniscope_observation_id.tolist()==[entry['sha256']+':1'] and frame.version.tolist()==['2']
    assert summary['qc_passed'] and frame.frp.tolist()==['0']
    schema25=pq.read_schema(q.bundle_path(s,request)/'retained.parquet')
    schema24=pq.read_schema(workspace_path('data/processed/firms_2024/validated_observations_2024.parquet'))
    normalized=lambda schema:{f.name.replace('_core_2024','_core_YEAR').replace('_core_2021','_core_YEAR'):str(f.type) for f in schema}
    assert normalized(schema24)==normalized(schema25)
    membership=q.season_membership(['2021-02-14','2021-02-15','2021-06-15','2021-06-16','2021-11-01'],2021)
    assert membership.agniscope_core_2021.tolist()==[False,True,True,False,False]
    assert membership.agniscope_extended_season_id.iloc[-1]=='EXT_2021_2022'


def test_complete_mock_2021_accounting_duplicates_collection_and_replay(c4):
    s,resolver,_,_=c4;session=Session(zero_dates={'2021-01-01'})
    for _ in range(8):assert batch(c4,session,size=10,qc=q.ensure_interval_qc)['status']=='completed'
    assert len(session.observation_dates)==73 and all(d.startswith('2021-') for d in session.observation_dates)
    catalog=b.load_catalog(s);plan=b.plan_c4_c5(s)
    summary,reused=q.integrate_year(s,catalog,plan)
    assert not reused and summary['counts']['raw_rows']==72 and summary['counts']['retained_rows']==72
    assert summary['counts']['cross_interval_duplicate_occurrences']==0 and summary['literal_version_counts']=={'2':72}
    assert summary['coverage']['validated_dates']==365 and summary['coverage']['raw_zero_dates']==293
    frame=pd.read_parquet(resolver(s['paths']['annual'])/'validated_observations_2021.parquet')
    assert frame.agniscope_observation_id.is_unique and frame.agniscope_calendar_year.eq(2021).all()
    again,reused=q.integrate_year(s,catalog,plan);assert reused and again==summary
    replay=Session();result=batch(c4,replay,size=10,qc=q.ensure_interval_qc)
    assert result['new_intervals_this_batch']==0 and not replay.observation_dates and replay.metadata_calls==0


def test_response_cap_is_enforced_before_acceptance(c4):
    s,_,_,_=c4;s['study']['maximum_response_bytes']=10
    result=batch(c4)
    assert result['status']=='stopped' and result['halt_reason']=='response_size_limit'


def test_unexpected_collection_stops_without_relabelling(c4):
    from test_stage_b import Response,HEADER
    class Incompatible(Session):
        def get(self,url,**kwargs):
            if '/area/' not in url:return super().get(url,**kwargs)
            self.observation_dates.append(url.rsplit('/',1)[1])
            return Response(HEADER+b'30,79,320,0.38,0.59,2021-01-01,2330,N,VIIRS,n,2NRT,290,1,N,0\n')
    result=batch(c4,Incompatible(),size=10);assert result['status']=='stopped'
    s,_,_,_=c4;entry=b.load_catalog(s)['intervals'][b.plan_c4_c5(s)[0]['request_id']]
    assert entry['collection_values']==['2NRT'] and entry['acquisition_status']=='retrieved_qc_failed'


def test_exact_duplicates_remain_in_raw_and_ledger(c4):
    from test_stage_b import Response,HEADER
    class Duplicate(Session):
        def get(self,url,**kwargs):
            if '/area/' not in url:return super().get(url,**kwargs)
            row=b'30,79,320,0.38,0.59,2021-01-01,2330,N,VIIRS,n,2,290,1,N,0\n'
            return Response(HEADER+row+row)
    s,_,_,_=c4;r=b.plan_c4_c5(s)[0]
    entry=b.acquire_c4_c5_interval(s,r,execute=True,session=Duplicate(),sleeper=lambda _:None)
    summary=q.ensure_interval_qc(s,r,entry)
    assert summary['raw_rows']==2 and summary['retained_rows']==1 and summary['qc_contract']['duplicate_rows_removed']==1
    assert len(pd.read_parquet(q.bundle_path(s,r)/'row_ledger.parquet'))==2


@pytest.mark.parametrize('year',[2019,2022,2023,2024,2025,2026])
def test_joint_mode_rejects_other_years(year):
    with pytest.raises(store.StoreConflict):p.configuration(year)


def test_joint_731_dates_147_intervals_and_leap_day():
    plans=p.combined_plan();assert {y:len(v) for y,v in plans.items()}=={2021:73,2020:74}
    assert sum(r['days'] for v in plans.values() for r in v)==731
    leap=[r for r in plans[2020] if r['start_utc_date']<='2020-02-29'<=r['end_utc_date']]
    assert len(leap)==1 and leap[0]['end_utc_date']=='2020-02-29'
    assert plans[2020][-1]['start_utc_date']==plans[2020][-1]['end_utc_date']=='2020-12-31'
    assert plans[2020][-1]['days']==1 and all(r['days']==5 for r in plans[2020][:-1])


def test_last_partial_request_and_six_year_schema(c4):
    import re
    s,_,_,_=c4;s.clear();s.update(p.configuration(2020));request=b.plan_c4_c5(s)[-1]
    result=b.acquire_c4_c5_interval(s,request,execute=True,session=Session(row_date='2020-12-31'),sleeper=lambda _:None)
    assert result['request']['days']==1
    q.ensure_interval_qc(s,request,result);frame=pd.read_parquet(q.bundle_path(s,request)/'retained.parquet')
    assert frame.agniscope_utc_date.tolist()==['2020-12-31'] and frame.agniscope_ist_date.tolist()==['2021-01-01']
    normal=lambda schema:{re.sub(r'_core_202[0-5]$','_core_YEAR',f.name):str(f.type) for f in schema}
    expected=normal(pq.read_schema(q.bundle_path(s,request)/'retained.parquet'))
    for year in (2022,2023,2024,2025):assert normal(pq.read_schema(workspace_path(f'data/processed/firms_{year}/validated_observations_{year}.parquet')))==expected
    assert q.season_membership(['2020-02-29'],2020).agniscope_core_2020.tolist()==[True]


@pytest.fixture
def gates(tmp_path,monkeypatch):
    original=workspace_path
    def resolve(name,**kwargs):
        text=str(name).replace('\\','/')
        for prefix,target in [(p.JOINT,tmp_path/'j'),('outputs/tables/phase2b_stage_c4_2021',tmp_path/'o21'),('outputs/tables/phase2b_stage_c5_2020',tmp_path/'o20')]:
            if text==prefix or text.startswith(prefix+'/'):return target/text[len(prefix):].lstrip('/')
        return original(name,**kwargs)
    monkeypatch.setattr(p,'workspace_path',resolve)
    anchor=tmp_path/'anchor.txt';anchor.write_text('bound evidence')
    files=p.file_records([anchor]);base={'status':'passed','files':files}
    store.atomic_json(resolve(p.JOINT+'/gate0.json'),base)
    return resolve,base,anchor


def test_c5_denied_before_c4_gate_without_http(gates):
    resolve,base,anchor=gates;session=Session();s=p.configuration(2020)
    with pytest.raises(store.StoreConflict,match='2021 first'):b.acquire_c4_c5_interval(s,b.plan_c4_c5(s)[0],execute=True,session=session)
    assert not session.observation_dates and not session.metadata_calls


@pytest.mark.parametrize('defect',['status','partial','wrong_year','hash'])
def test_c4_gate_failure_hard_stops_2020(gates,defect):
    resolve,base,anchor=gates;record={**base,'c4_status':'COMPLETE','authorized_next_year':2020}
    if defect=='status':record['status']='failed'
    elif defect=='partial':record['c4_status']='PARTIALLY COMPLETE'
    elif defect=='wrong_year':record['authorized_next_year']=2019
    else:anchor.write_text('changed evidence')
    store.atomic_json(resolve(p.JOINT+'/gate_c4_to_c5.json'),record)
    with pytest.raises(store.StoreConflict):p.require_authorization(p.configuration(2020))


def test_passing_gate_permits_automatic_2020_continuation(gates):
    resolve,base,anchor=gates
    store.atomic_json(resolve(p.JOINT+'/gate_c4_to_c5.json'),{**base,'c4_status':'COMPLETE','authorized_next_year':2020})
    assert p.require_authorization(p.configuration(2020))['authorized_year']==2020


@pytest.mark.parametrize('defect',['duplicate_attempt','unauthorized_identity','fourth_retry','wrong_stage'])
def test_independent_annual_and_combined_request_budgets(gates,defect):
    resolve,base,anchor=gates;plans=p.combined_plan()
    store.atomic_json(resolve(p.JOINT+'/gate_c4_to_c5.json'),{**base,'c4_status':'COMPLETE','authorized_next_year':2020})
    for year in (2021,2020):
        slots=[{'request_id':r['request_id'],'number':1} for r in plans[year]]
        catalog={'stage':'C4' if year==2021 else 'C5','intervals':{r['request_id']:{} for r in plans[year]},'attempt_reservations':slots}
        store.atomic_json(resolve(p.configuration(year)['paths']['output'])/'acquisition_manifest.json',catalog)
    assert p.require_authorization(p.configuration(2020))['combined_distinct_intervals_so_far']==147
    path=resolve(p.configuration(2020)['paths']['output'])/'acquisition_manifest.json';catalog=json.loads(path.read_text())
    if defect=='duplicate_attempt':catalog['attempt_reservations'].append(catalog['attempt_reservations'][0])
    elif defect=='unauthorized_identity':catalog['attempt_reservations'].append({'request_id':'not_authorized','number':1})
    elif defect=='fourth_retry':catalog['attempt_reservations'][0]['number']=4
    else:catalog['stage']='C6'
    store.atomic_json(path,catalog)
    with pytest.raises(store.StoreConflict):p.require_authorization(p.configuration(2020))


@pytest.mark.parametrize('ending',[2021,2022])
def test_new_extended_windows_exact_boundaries_and_unknown_opportunity(ending):
    from agniscope.season_audit import utc_extended_mask,audit_extended_daily
    first=f'{ending-1}-11-01';last=f'{ending}-06-30'
    days=pd.date_range(first,last).strftime('%Y-%m-%d')
    frame=pd.DataFrame({'utc_date':days,'raw_observations':2,'retained_observations':1,'retrieval_state':'positive_detections','qc_status':'passed','provenance_status':'hash_verified','satellite_coverage':'unknown'})
    audit=audit_extended_daily(frame,ending);assert audit['validated_dates']==242 and audit['retained_observations']==242
    assert audit['satellite_opportunity_unknown_dates']==242 and audit['calendar_retrieval_eligible']
    assert utc_extended_mask([first+'T00:00:00Z',last+'T23:59:59Z',f'{ending}-07-01T00:00:00Z'],ending).tolist()==[True,True,False]


def test_new_issue_register_preserves_prior_register(c4):
    settings,_,_,_=c4;prior=q.issue_register();combined=q.issue_register(settings)
    assert len(prior)==15 and len(combined)==35 and combined[:15]==prior
    assert 'snpp_2021_daytime_img_limitation' in q.issue_ids('2021-08-05T07:00:00Z',combined)
    assert 'snpp_2020_jan08a' in q.issue_ids('2020-01-08T16:50:00Z',combined)


@pytest.mark.parametrize('key,value',[('year_order',[2020,2021]),('maximum_distinct_new_intervals',148),('calendar_days',732)])
def test_joint_authorization_cannot_expand_or_reverse(monkeypatch,key,value):
    original=p.load_config
    def changed(path):
        result=original(path)
        if Path(path).name=='phase2b_c4_c5.toml':result['authorization'][key]=value
        return result
    monkeypatch.setattr(p,'load_config',changed)
    with pytest.raises(store.StoreConflict):p.combined_plan()


@pytest.mark.parametrize('section,key,value',[('study','maximum_new_intervals',75),('study','maximum_logical_intervals',75),('batch','concurrency',2),('study','maximum_response_bytes',10485761)])
def test_c5_independent_limits(section,key,value):
    settings=p.configuration(2020);settings[section][key]=value
    with pytest.raises(a.AcquisitionError):b.plan_c4_c5(settings)


def test_gate0_is_required_even_for_2021(gates):
    resolve,_,_=gates;resolve(p.JOINT+'/gate0.json').unlink()
    with pytest.raises(store.StoreConflict,match='Gate 0'):p.require_authorization(p.configuration(2021))


def test_transition_gate_rejects_incomplete_2021(gates,monkeypatch):
    monkeypatch.setattr(p,'test_gate',lambda *args,**kwargs:None)
    monkeypatch.setattr(b,'protected_integrity',lambda *args:None)
    monkeypatch.setattr(b,'load_catalog',lambda *args: {'halt_reason':'synthetic_qc_failure'})
    with pytest.raises(store.StoreConflict,match='All C4 intervals'):p.open_c4_gate()
