"""Explicit offline Phase 3A plan, bounded population preparation and serial workers."""
import argparse,hashlib,json,os,subprocess,sys,time,ctypes,ast
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agniscope.config import ROOT,load_config,workspace_path
from agniscope.provenance import sha256,utc_now
from agniscope.resources import resource_snapshot
from agniscope.acquisition_store import atomic_json,StoreConflict,store_lock

OUT=ROOT/'outputs/tables/phase3a';POP=ROOT/'data/interim/phase3a/populations'
CONFIG=ROOT/'configs/phase3a.toml'
MASTER=ROOT/'data/processed/firms_2019_2025/validated_observations_2019_2025.parquet'
EXPECTED='e1e1261bfd63b37664ccec1100bd70752231c8e9ebd615b5f3286858057f3966'
DB=[{'algorithm':'dbscan','eps_km':eps,'min_samples':m} for eps in (.5,1.,2.) for m in (3,5,10)]
HDB=[{'algorithm':'hdbscan','min_cluster_size':size,'min_samples':m} for size in (3,5,10,15) for m in (3,5,10)]
REFS=[{'algorithm':'dbscan','eps_km':1.,'min_samples':3},{'algorithm':'hdbscan','min_cluster_size':5,'min_samples':5}]
TRACKED=['configs/phase3a.toml','src/agniscope/seasonal_clustering.py','scripts/phase3a.py','scripts/phase3a_worker.py','tests/test_seasonal_clustering.py']

def protected():
    before=json.loads((OUT/'preflight.json').read_text())
    changed=[r['path'] for r in before['protected_files'] if sha256(ROOT/r['path'])!=r['sha256']]
    if changed:raise StoreConflict('Previously protected input, code or scientific evidence changed.')
    if sha256(MASTER)!=EXPECTED:raise StoreConflict('Master checksum changed.')
    return len(before['protected_files'])

def config():
    settings=load_config(CONFIG);e=settings['experiment'];r=settings['resources']
    if e['master_sha256']!=EXPECTED or e['primary_years']!=list(range(2019,2026)) or e['extended_ending_years']!=list(range(2020,2026)) or e['dbscan_eps_km']!=[.5,1.,2.] or e['dbscan_min_samples']!=[3,5,10] or e['hdbscan_min_cluster_size']!=[3,5,10,15] or e['hdbscan_min_samples']!=[3,5,10]:raise StoreConflict('Declared grid/season scope changed.')
    if r['concurrency']!=1 or e['analysis_crs']!='EPSG:32644':raise StoreConflict('Metric CRS and sequential execution required.')
    # The controller must remain small while a numeric/GIS worker is active.
    # Read the literal contract without importing numpy/pandas/sklearn here.
    tree=ast.parse((ROOT/'src/agniscope/seasonal_clustering.py').read_text(encoding='utf-8'))
    HDB_FIXED=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HDB_FIXED' for t in node.targets))
    if settings['hdbscan']!=HDB_FIXED or e['temporal_cell_m']!=1000. or e['temporal_origin']!=[0.,0.] or e['broad_cluster_diameter_m']!=10000.:raise StoreConflict('Scientific definitions differ from declared implementation.')
    return settings

def validate_plan(plan):
    seen=set();counts={}
    for spec in plan:
        body={k:v for k,v in spec.items() if k!='run_id'}
        identity='fit_'+hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if spec['run_id']!=identity or identity in seen:raise StoreConflict('Conflicting or repeated experiment identity.')
        seen.add(identity);role=spec['role'];counts[role]=counts.get(role,0)+1
        season=spec['season_id'];params=spec['parameters'];subset=spec['subset']
        core=season in [f'CORE_{y}' for y in range(2019,2026)]
        if role=='primary_grid':valid=core and subset=='all' and params in DB+HDB
        elif role=='primary_sensitivity':valid=core and subset in ('grid_day','nominal','tree') and params in REFS
        elif role=='extended_reference':valid=season in [f'EXT_{y-1}_{y}' for y in range(2020,2026)] and subset=='all' and params in REFS
        elif role=='high_volume_probe':valid=season=='CORE_2021' and subset in ('systematic_probe','dense_probe') and params in REFS+[DB[6]] and spec['n_observations']<=5000
        elif role=='extended_probe':valid=season=='EXT_2020_2021' and subset=='dense_probe' and params in REFS and spec['n_observations']<=5000
        else:valid=False
        if not valid or spec['reference']!=(params in REFS):raise StoreConflict('Experiment outside predeclared scope.')
    if counts!={'primary_grid':147,'primary_sensitivity':42,'high_volume_probe':6,'extended_reference':12,'extended_probe':2}:raise StoreConflict('Incomplete or expanded experiment plan.')
    return plan

def ledger(plan):
    rows=[]
    for spec in plan:
        directory=OUT/'runs'/spec['run_id'][:20]
        status='completed' if (directory/'receipt.json').exists() else ('failed' if (directory.with_name(directory.name+'.staged')/'failure.json').exists() else 'not_started')
        rows.append({'run_id':spec['run_id'],'season_id':spec['season_id'],'subset':spec['subset'],'role':spec['role'],'parameters':spec['parameters'],'status':status,'skipped':False})
    return rows

def test_gate(certify=False):
    path=OUT/'executor_test_gate.json';xml=OUT/'pytest_regression.xml'
    if certify:
        suite=ET.parse(xml).getroot().find('testsuite')
        if suite is None or int(suite.attrib['tests'])<459 or any(int(suite.attrib[k]) for k in ('errors','failures','skipped')):raise StoreConflict('Full passing regression required.')
        validate_plan(json.loads((OUT/'experiment_plan.json').read_text()))
        atomic_json(path,{'tests':int(suite.attrib['tests']),'xml_sha256':sha256(xml),'plan_sha256':sha256(OUT/'experiment_plan.json'),'files':{name:sha256(ROOT/name) for name in TRACKED},'certified_utc':utc_now()})
    gate=json.loads(path.read_text())
    if gate['xml_sha256']!=sha256(xml) or gate['plan_sha256']!=sha256(OUT/'experiment_plan.json') or any(sha256(ROOT/name)!=value for name,value in gate['files'].items()):raise StoreConflict('Implementation differs from passing test gate.')
    return gate

def prepare():
    import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
    from agniscope.seasonal_inputs import iter_season
    from agniscope.seasonal_clustering import populations,canonical_points,feature_identity,neighbour_cost,XY
    protected();settings=config();records=[];costs=[]
    columns=['agniscope_observation_id','agniscope_raw_sha256','agniscope_source_record','latitude','longitude','agniscope_x_m','agniscope_y_m','agniscope_acquired_utc','agniscope_utc_date','agniscope_ist_date','agniscope_calendar_year','agniscope_confidence','agniscope_worldcover_2021_class','agniscope_district_original','agniscope_district_display','daynight','frp','agniscope_request_id','agniscope_all_flags','agniscope_upstream_issue_ids','agniscope_satellite_opportunity']
    def persist(points,season,kind,subset):
        points=canonical_points(points);directory=POP/season;directory.mkdir(parents=True,exist_ok=True);path=directory/(subset+'.parquet')
        table=pa.Table.from_pandas(points,preserve_index=False).replace_schema_metadata(None)
        if path.exists():
            old=pq.read_table(path);assert old.equals(table)
        else:pq.write_table(table,path,compression='zstd',row_group_size=4096)
        records.append({'season_id':season,'kind':kind,'subset':subset,'observations':len(points),'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path),'feature_identity':feature_identity(points),'master_sha256':EXPECTED})
        return points
    for kind,years in [('core',range(2019,2026)),('extended',range(2020,2026))]:
        for year in years:
            season=f'CORE_{year}' if kind=='core' else f'EXT_{year-1}_{year}'
            parts=[table.select(columns).to_pandas() for table in iter_season(MASTER,kind,year)]
            points=canonical_points(pd.concat(parts,ignore_index=True));del parts
            choices,ledger=populations(points) if kind=='core' else ({'all':points},None)
            for subset,frame in choices.items():persist(frame,season,kind,subset)
            if ledger is not None:
                path=POP/season/'temporal_sampling_ledger.parquet'
                if not path.exists():ledger.to_parquet(path,index=False,compression='zstd')
            costs.extend({'season_id':season,**row} for row in neighbour_cost(points[XY].to_numpy(float)))
            if season in ('CORE_2021','EXT_2020_2021'):
                n=min(5000,len(points));indices=np.linspace(0,len(points)-1,n,dtype=int)
                persist(points.iloc[indices],season,'probe','systematic_probe')
                from scipy.spatial import cKDTree
                counts=cKDTree(points[XY]).query_ball_point(points[XY],2000.,return_length=True,workers=1)
                indices=np.sort(np.argsort(-counts,kind='stable')[:n]);persist(points.iloc[indices],season,'probe','dense_probe')
    atomic_json(OUT/'population_inventory.json',records);pd.DataFrame(costs).to_csv(OUT/'neighbour_cost_estimates.csv',index=False)
    specs=[]
    def add(pop,parameters,role):
        body={'season_id':pop['season_id'],'kind':pop['kind'],'subset':pop['subset'],'input_path':pop['path'],'input_sha256':pop['sha256'],'input_feature_identity':pop['feature_identity'],'n_observations':pop['observations'],'parameters':parameters,'role':role,'reference':parameters in REFS}
        identity=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest();body['run_id']='fit_'+identity;specs.append(body)
    for pop in records:
        if pop['kind']=='core':
            for params in (DB+HDB if pop['subset']=='all' else REFS):add(pop,params,'primary_grid' if pop['subset']=='all' else 'primary_sensitivity')
        elif pop['kind']=='extended':
            for params in REFS:add(pop,params,'extended_reference')
        elif pop['season_id']=='CORE_2021':
            for params in [*REFS,DB[6]]:add(pop,params,'high_volume_probe')
        elif pop['subset']=='dense_probe':
            for params in REFS:add(pop,params,'extended_probe')
    assert len([s for s in specs if s['role']!='high_volume_probe' and s['role']!='extended_probe'])==201 and len(specs)==209
    atomic_json(OUT/'experiment_plan.json',specs)
    print(json.dumps({'primary_scientific_runs':189,'extended_scientific_runs':12,'resource_probe_runs':8,'population_files':len(records),'planned_total':len(specs)},indent=2))

def worker_memory(pid):
    if sys.platform!='win32':return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ('peak','working','pp','p','npp','np','page','peakpage')]
    kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=wintypes.HANDLE
    handle=kernel.OpenProcess(0x0400|0x0010,False,pid)
    if not handle:return None
    try:
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        if fn(handle,ctypes.byref(counters),counters.cb):return {'working':counters.working,'peak':counters.peak}
    finally:
        kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle(handle)
    return None

def execute(spec,*,repeat=False,verify_before_commit=False):
    settings=config();r=settings['resources'];directory=OUT/'runs'/spec['run_id'][:20];receipt=directory/'receipt.json'
    if receipt.exists() and not repeat:
        record=json.loads(receipt.read_text());assert record['spec']==spec
        for row in record['files']:assert sha256(ROOT/row['path'])==row['sha256']
        return {'run_id':spec['run_id'],'status':'completed','reused':True,'receipt_sha256':sha256(receipt)}
    snapshot=resource_snapshot()
    if snapshot['available_physical_bytes']<r['minimum_available_ram_bytes'] or snapshot['disk_free_bytes']<r['minimum_disk_free_bytes']:raise StoreConflict('Unsafe RAM/disk reserve before fit.')
    cap=min(r['maximum_worker_working_set_bytes'],int(snapshot['available_physical_bytes']*.5))
    staging=directory.with_name(directory.name+('.repeat' if repeat else '.staged'))
    if staging.exists():raise StoreConflict('Uncommitted fit staging requires reconciliation; not overwritten.')
    staging.mkdir(parents=True);atomic_json(staging/'spec.json',spec)
    command=[sys.executable,'-B',str(ROOT/'scripts/phase3a_worker.py'),str(staging.relative_to(ROOT))]
    if verify_before_commit:command.append('--verify-before-commit')
    start=time.perf_counter();process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    peak=0;minimum=snapshot['available_physical_bytes'];failure=None
    while process.poll() is None:
        now=resource_snapshot();minimum=min(minimum,now['available_physical_bytes']);memory=worker_memory(process.pid)
        if memory:peak=max(peak,memory['peak'])
        if now['available_physical_bytes']<r['minimum_available_ram_bytes']:failure='available_RAM_below_reserve'
        elif peak>cap:failure='worker_working_set_cap'
        elif time.perf_counter()-start>r['maximum_fit_seconds']:failure='fit_wall_time_cap'
        if failure:process.terminate();break
        time.sleep(r['monitor_interval_seconds'])
    stdout,stderr=process.communicate();elapsed=time.perf_counter()-start
    resource={'elapsed_seconds':elapsed,'peak_worker_working_set_bytes':peak,'minimum_available_ram_bytes':minimum,'worker_cap_bytes':cap,'resources_before':snapshot,
      'resources_after_worker_exit':resource_snapshot(),'worker_process_reaped':process.poll() is not None,'controller_numeric_modules_loaded':any(name in sys.modules for name in ('numpy','pandas','sklearn','geopandas'))}
    if process.returncode or failure:
        record={'run_id':spec['run_id'],'status':'failed','failure':failure or 'worker_exception','resources':resource}
        # Worker emits only a safe JSON error class, never network exceptions.
        try:record['worker_diagnostic']=json.loads(stdout.decode())
        except Exception:record['worker_diagnostic']='No safe worker diagnostic.'
        atomic_json(staging/'failure.json',record);return record
    if verify_before_commit:
        proof=json.loads((staging/'precommit_verification.json').read_text())
        if proof['status']!='passed' or proof['run_id']!=spec['run_id']:raise StoreConflict('Independent precommit verification required.')
    if repeat:
        record=json.loads(receipt.read_text());checks=[]
        for row in record['files']:
            name=Path(row['path']).name
            if name in ('spec.json','timing.json'):continue
            checks.append({'file':name,'match':sha256(staging/name)==row['sha256']})
        if not all(c['match'] for c in checks):raise StoreConflict('Reference scientific artifact rerun differs.')
        atomic_json(staging/'repeat_verification.json',{'status':'passed','run_id':spec['run_id'],'files':checks,'resources':resource})
        return {'run_id':spec['run_id'],'status':'completed','repeat_verified':True,'resources':resource}
    directory.parent.mkdir(parents=True,exist_ok=True)
    files=[{'path':(directory/p.name).relative_to(ROOT).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(staging.iterdir()) if p.is_file()]
    atomic_json(staging/'receipt.json',{'spec':spec,'files':files,'resources':resource});staging.rename(directory)
    return {'run_id':spec['run_id'],'status':'completed','reused':False,'resources':resource,'receipt_sha256':sha256(directory/'receipt.json')}

def run_group(mode):
    protected();test_gate();plan=validate_plan(json.loads((OUT/'experiment_plan.json').read_text()));results=[]
    if mode=='feasibility':
        selected=[s for s in plan if (s['season_id']=='CORE_2020' and s['subset']=='all' and s['reference']) or s['role']=='high_volume_probe']
    elif mode=='primary':
        gate=json.loads((OUT/'feasibility_gate.json').read_text());assert gate['status']=='PASS'
        assert json.loads((OUT/'feasibility_verification.json').read_text())['status']=='passed'
        assert json.loads((OUT/'historical_equivalence.json').read_text())['status']=='passed'
        selected=[s for s in plan if s['role'] in ('primary_grid','primary_sensitivity')]
    elif mode=='extended':
        gate=json.loads((OUT/'primary_verification.json').read_text());assert gate['status']=='passed'
        selected=[s for s in plan if s['role'] in ('extended_probe','extended_reference')]
        selected.sort(key=lambda s:s['role']!='extended_probe')
    elif mode=='repeat':selected=[s for s in plan if s['role'] in ('primary_grid','primary_sensitivity','extended_reference') and s['reference']]
    else:raise ValueError('Unknown experiment group.')
    current=None
    for spec in selected:
        if spec['season_id']!=current:protected();current=spec['season_id']
        result=execute(spec,repeat=mode=='repeat');results.append(result)
        atomic_json(OUT/f'{mode}_execution.json',{'mode':mode,'runs':results,'updated_utc':utc_now()})
        atomic_json(OUT/'experimental_run_ledger.json',ledger(plan))
        print(json.dumps({'mode':mode,'completed':len(results),'planned':len(selected),'season':spec['season_id'],'subset':spec['subset'],'parameters':spec['parameters'],'status':result['status'],'reused':result.get('reused',False)}),flush=True)
        if result['status']!='completed':raise StoreConflict('Execution stopped on a failed resource or scientific gate.')
    protected()
    if mode=='feasibility':
        atomic_json(OUT/'feasibility_gate.json',{'status':'PASS','checked_utc':utc_now(),'runs':results,'initial_full_season':'CORE_2020','high_volume_season':'CORE_2021','stress_subsets':'5000 systematic source-ordered and 5000 densest 2km-neighbour-count points; diagnostic only','full_populations_not_subsampled':True})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--certify-tests',action='store_true');parser.add_argument('--run',choices=('feasibility','primary','extended','repeat'));args=parser.parse_args()
    config()
    if args.prepare:prepare()
    elif args.certify_tests:print(json.dumps(test_gate(True)))
    elif args.run:
        with store_lock(OUT):run_group(args.run)
    else:print(json.dumps({'mode':'dry_run','primary_grid':147,'primary_sensitivities':42,'extended_references':12,'resource_probes':8,'execution':False,'observation_requests':0}))
if __name__=='__main__':main()
