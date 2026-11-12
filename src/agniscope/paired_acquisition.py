"""Prompt 12 authorization and ordered gates around the existing annual executor."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from .config import ROOT,load_config,workspace_path
from .provenance import sha256,utc_now
from .acquisition_store import StoreConflict,atomic_json
from .resources import resource_snapshot

JOINT='outputs/tables/phase2b_c4_c5'
CONFIG='configs/phase2b_c4_c5.toml'
CRITICAL=['src/agniscope/acquisition.py','src/agniscope/acquisition_store.py','src/agniscope/stage_b.py','src/agniscope/stage_b_qc.py',
 'src/agniscope/paired_acquisition.py','src/agniscope/season_audit.py','scripts/phase2b_c4_c5.py','scripts/verify_phase2b_stage_b.py',
 'scripts/verify_c4_c5.py','configs/phase2b_c4_c5.toml','configs/phase2b_c4_c5_upstream_issues.csv',
 'configs/phase2b_stage_c4_2021.toml','configs/phase2b_stage_c5_2020.toml','tests/test_c4_c5.py']

def configuration(year):
    if type(year) is not int or year not in (2021,2020):raise StoreConflict('Only C4/2021 and C5/2020 are authorized.')
    stage='c4' if year==2021 else 'c5'
    return load_config(workspace_path(f'configs/phase2b_stage_{stage}_{year}.toml'))

def combined_plan():
    from .stage_b import approved_plan
    value=load_config(workspace_path(CONFIG))
    expected={'source':'VIIRS_SNPP_SP','bbox':[77.5,28.6,81.2,31.6],'year_order':[2021,2020],
      'maximum_distinct_new_intervals':147,'calendar_days':731,'require_c4_gate_before_c5':True}
    if value['authorization']!=expected or any(value['scope'].values()):raise StoreConflict('Joint authorization differs from Prompt 12.')
    plans={year:approved_plan(configuration(year)) for year in (2021,2020)}
    if sum(len(p) for p in plans.values())!=147 or sum(r['days'] for p in plans.values() for r in p)!=731:raise StoreConflict('Joint calendar budget differs.')
    return plans

def file_records(paths):
    result=[]
    for name in sorted(set(map(str,paths))):
        p=workspace_path(name);result.append({'path':p.relative_to(ROOT).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha256(p)})
    return result

def verify_evidence(record):
    if record.get('status')!='passed' or not record.get('files'):raise StoreConflict('A passing, bound gate record is required.')
    for row in record['files']:
        p=workspace_path(row['path'])
        if not p.is_file() or p.stat().st_size!=row['size_bytes'] or sha256(p)!=row['sha256']:raise StoreConflict('Gate-bound evidence changed.')
    return record

def verify_gate0():
    p=workspace_path(f'{JOINT}/gate0.json')
    if not p.is_file():raise StoreConflict('Gate 0 must pass before observation acquisition.')
    return verify_evidence(json.loads(p.read_text(encoding='utf-8')))

def verify_c4_gate():
    p=workspace_path(f'{JOINT}/gate_c4_to_c5.json')
    if not p.is_file():raise StoreConflict('2020 is not started: complete and verify 2021 first.')
    record=verify_evidence(json.loads(p.read_text(encoding='utf-8')))
    if record.get('c4_status')!='COMPLETE' or record.get('authorized_next_year')!=2020:raise StoreConflict('C4 completion gate does not authorize 2020.')
    return record

def require_authorization(settings):
    from .stage_b import stage_profile
    profile=stage_profile(settings)
    if (profile['stage'],profile['year']) not in (('C4',2021),('C5',2020)):raise StoreConflict('This mode permits only C4 and C5.')
    plans=combined_plan();verify_gate0()
    total=0
    for year,plan in plans.items():
        path=workspace_path(configuration(year)['paths']['output'])/'acquisition_manifest.json'
        if not path.exists():continue
        catalog=json.loads(path.read_text(encoding='utf-8'))
        allowed={r['request_id'] for r in plan};slots=catalog['attempt_reservations']
        if catalog['stage']!=('C4' if year==2021 else 'C5') or set(catalog['intervals'])!=allowed:raise StoreConflict('Joint annual manifest scope conflicts.')
        if any(r['request_id'] not in allowed or r['number'] not in (1,2,3) for r in slots):raise StoreConflict('Unauthorized attempt reservation.')
        identities={(r['request_id'],r['number']) for r in slots}
        unique={r['request_id'] for r in slots}
        if len(identities)!=len(slots) or len(unique)>len(plan) or len(slots)>3*len(plan):raise StoreConflict('Annual distinct interval or retry budget exceeded.')
        total+=len(unique)
    if total>147:raise StoreConflict('Combined distinct interval budget exceeded.')
    if profile['year']==2020:verify_c4_gate()
    return {'authorized_year':profile['year'],'combined_distinct_intervals_so_far':total,'maximum_distinct_intervals':147}

def test_gate(year,certify=False):
    settings=configuration(year);out=workspace_path(settings['paths']['output']);xml=out/'pytest_regression.xml';path=out/'executor_test_gate.json'
    if certify:
        suite=ET.parse(xml).getroot().find('testsuite')
        if suite is None or int(suite.attrib['tests'])<343 or any(int(suite.attrib[k]) for k in ('failures','errors','skipped')):raise StoreConflict('A full passing regression suite is required.')
        atomic_json(path,{'status':'passed','certified_utc':utc_now(),'tests':int(suite.attrib['tests']),'xml_sha256':sha256(xml),'files':file_records(CRITICAL)})
    value=verify_evidence(json.loads(path.read_text(encoding='utf-8')))
    if sha256(xml)!=value['xml_sha256']:raise StoreConflict('Passing test XML changed; recertification required.')
    return value

def open_gate0():
    from .stage_b import protected_integrity
    plans=combined_plan();settings=configuration(2021);out=workspace_path(settings['paths']['output'])
    protected_integrity(settings);test_gate(2021)
    pre=json.loads(workspace_path(f'{JOINT}/preflight.json').read_text(encoding='utf-8'))
    baseline=ET.parse(workspace_path(f'{JOINT}/pytest_baseline.xml')).getroot().find('testsuite')
    if int(baseline.attrib['tests'])!=343 or any(int(baseline.attrib[k]) for k in ('errors','failures','skipped')):raise StoreConflict('The original 343-test baseline did not pass.')
    metadata=json.loads((out/'upstream_metadata_evidence.json').read_text(encoding='utf-8'))
    if metadata['source']!='VIIRS_SNPP_SP' or metadata['availability']['min_date']>'2020-01-01' or metadata['availability']['max_date']<'2021-12-31':raise StoreConflict('Historical availability preflight failed.')
    paths=[*CRITICAL,f'{JOINT}/preflight.json',f'{JOINT}/pytest_baseline.xml',f"{settings['paths']['output']}/pytest_before_download.xml",
      f"{settings['paths']['output']}/executor_test_gate_at_start.json",f"{settings['paths']['output']}/upstream_metadata_evidence.json"]
    record={'status':'passed','checked_utc':utc_now(),'calendar_days':731,'intervals':147,'year_order':[2021,2020],
      'original_gis_unchanged':14,'protected_files':len(pre['protected_files']),'files':file_records(paths)}
    atomic_json(workspace_path(f'{JOINT}/gate0.json'),record);return record

def open_c4_gate():
    from . import stage_b as b,stage_b_qc as q
    import pyarrow.parquet as pq
    import re
    verify_gate0();settings=configuration(2021);out=workspace_path(settings['paths']['output']);b.protected_integrity(settings)
    test_gate(2021);catalog=b.load_catalog(settings)
    if catalog is None or catalog['halt_reason'] or any(e['qc_status']!='passed' or e['acquisition_status'] not in b.SUCCESS for e in catalog['intervals'].values()):raise StoreConflict('All C4 intervals must pass before C5.')
    summary,reused=q.integrate_year(settings,catalog,b.approved_plan(settings))
    if not reused or summary['stage_c4_status']!='COMPLETE' or summary['coverage']['validated_dates']!=365 or summary['coverage']['failed_dates'] or summary['counts']['cross_interval_duplicate_occurrences']:raise StoreConflict('C4 annual verification failed.')
    verified=json.loads((out/'final_verification.json').read_text());replay=json.loads((out/'idempotency_verification.json').read_text())
    if verified['status']!='passed' or replay['status']!='passed' or replay['http_calls'] or replay['credential_accesses']:raise StoreConflict('Independent C4 verification and replay must pass.')
    normalize=lambda schema:{re.sub(r'_core_202[0-5]$','_core_YEAR',f.name):str(f.type) for f in schema}
    schema=normalize(pq.read_schema(workspace_path('data/processed/firms_2021/validated_observations_2021.parquet')))
    for year in (2022,2023,2024,2025):
        if schema!=normalize(pq.read_schema(workspace_path(f'data/processed/firms_{year}/validated_observations_{year}.parquet'))):raise StoreConflict('C4 source schema differs from previous years.')
    resources=resource_snapshot()
    if resources['disk_free_bytes']<15*1024**3 or resources['available_physical_bytes']<512*1024**2:raise StoreConflict('Gate 2 resource reserve failed.')
    latest=[json.loads(workspace_path(e['manifest_file']).read_text()) for e in catalog['intervals'].values()]
    fresh=json.loads((workspace_path(configuration(2020)['paths']['output'])/'upstream_metadata_evidence.json').read_text())
    if fresh['source']!='VIIRS_SNPP_SP' or fresh['availability']['min_date']>'2020-01-01' or fresh['availability']['max_date']<'2020-12-31':raise StoreConflict('Fresh C5 availability evidence required.')
    quota=fresh['quota']
    if quota['transaction_limit']-quota['current_transactions']<max(1000,quota['transaction_limit']*.2):raise StoreConflict('Gate 2 transaction reserve failed.')
    annual=workspace_path(settings['paths']['annual']);receipt=json.loads((annual/'receipt.json').read_text())
    paths=[*CRITICAL,*[r['path'] for r in receipt['files']],str(annual/'receipt.json'),str(out/'acquisition_manifest.json'),str(out/'final_verification.json'),
      str(out/'idempotency_verification.json'),str(out/'pytest_regression.xml'),str(out/'file_integrity_manifest.json'),'docs/phase2b_stage_c4_2021_report.md',
      str(workspace_path(configuration(2020)['paths']['output'])/'upstream_metadata_evidence.json')]
    record={'status':'passed','checked_utc':utc_now(),'gate1':'passed','gate2':'passed','c4_status':'COMPLETE','authorized_next_year':2020,
      'c4_counts':summary['counts'],'c4_dates':365,'c4_intervals':73,'schemas_compatible':True,'resources':resources,'last_c4_quota':quota,'files':file_records(paths)}
    atomic_json(workspace_path(f'{JOINT}/gate_c4_to_c5.json'),record)
    # Freeze the completed C4 evidence into the next year's protection snapshot.
    protected=[]
    for folder in ('configs','src','scripts','tests','docs','data','outputs'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.lock' and not p.is_relative_to(workspace_path(configuration(2020)['paths']['output'])):
                protected.append({'path':p.relative_to(ROOT).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha256(p)})
    atomic_json(workspace_path(configuration(2020)['paths']['preflight']),{'checked_utc':utc_now(),'protected_files':protected,'gate_c4_to_c5_sha256':sha256(workspace_path(f'{JOINT}/gate_c4_to_c5.json')),
      'approved_source_extensions':[],'original_gis_unchanged':14,'resources':resources,'prior_c4_status':'COMPLETE'})
    return record
