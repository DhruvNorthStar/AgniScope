"""Thin Prompt 15 recovery controls around the established serial executor."""
from pathlib import Path
import argparse,json,csv,sys,time,tomllib,xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agniscope.provenance import sha256,utc_now
from agniscope.acquisition_store import atomic_json,StoreConflict,store_lock
from agniscope.resources import resource_snapshot
import phase3a as original
ROOT=original.ROOT;OUT=original.OUT;HISTORY=ROOT/'outputs/tables/phase3a_recovery';LOW_RAM=ROOT/'outputs/tables/phase3a_low_ram'
REC=LOW_RAM if (LOW_RAM/'configuration_provenance.json').exists() else HISTORY
CODE_CHANGES={'scripts/phase3a.py','scripts/phase3a_worker.py','scripts/verify_phase3a.py','scripts/qa/phase3a_partial_cases.py'}
WATCHED=sorted(CODE_CHANGES|{'scripts/phase3a_recovery.py','scripts/preflight_phase3a_recovery.py','tests/test_phase3a_recovery.py','tests/test_phase3a_low_ram.py','src/agniscope/seasonal_clustering.py','configs/phase3a.toml'})
AGGREGATES=set('all_fit_statistics.csv cluster_size_extent_summaries.parquet confidence_sensitivity.csv daily_reference_observation_distribution.csv daynight_reference_observation_distribution.csv dbscan_parameter_comparison.csv district_primary_clustered_matrix.csv district_reference_observation_distribution.csv experiment_status.json experimental_run_ledger.csv experimental_run_ledger.json extended_season_reference_summary.csv figure_manifest.json figure_visual_qa.json final_integrity_verification.json geographic_plot_points.csv hdbscan_parameter_comparison.csv landcover_sensitivity.csv plot_boundaries.json primary_season_observation_inventory.csv reference_cluster_overlap_edges.csv reference_cluster_sizes_for_plot.csv reference_fit_statistics.csv reference_FRP_summary.csv reference_model_comparison.csv reference_monthly_observation_distribution.csv reference_point_comparisons.parquet reference_source_category_composition.csv regression_test_record.json seasonal_population_inventory.csv spatial_artifact_index.csv temporal_cluster_sampling_dependence.csv temporal_sensitivity.csv'.split())

class RecoveryBlocked(StoreConflict):pass

def baseline():return json.loads(((HISTORY if REC==LOW_RAM else REC)/'preflight.json').read_text())

def configuration_preserved(before):
    current=ROOT/'configs/phase3a.toml'
    if sha256(current)==before['config_sha256']:return False
    provenance=json.loads((LOW_RAM/'configuration_provenance.json').read_text())
    old=ROOT/provenance['prior_config_archive']
    assert sha256(old)==before['config_sha256']==provenance['prior_config_sha256'] and sha256(current)==provenance['revised_config_sha256']
    prior=tomllib.loads(old.read_text());revised=tomllib.loads(current.read_text())
    assert revised['resources'].pop('minimum_startup_available_ram_bytes')==896*1024**2 and revised==prior
    return True
def load_plan():
    before=baseline();assert sha256(OUT/'experiment_plan.json')==before['plan_sha256']
    return original.validate_plan(json.loads((OUT/'experiment_plan.json').read_text()))

def preserved():
    before=baseline();original.protected()
    revised=configuration_preserved(before)
    assert sha256(ROOT/'src/agniscope/seasonal_clustering.py')==before['scientific_core_sha256'] and sha256(ROOT/'environment.lock.win-64.txt')==before['environment_lock_sha256']
    archive=(HISTORY if REC==LOW_RAM else REC)/'baseline/files';mapping={}
    reconciliation=(HISTORY if REC==LOW_RAM else REC)/'staging_reconciliation.json'
    if reconciliation.exists():mapping=json.loads(reconciliation.read_text())['relocations']
    for entry in before['phase3a_original_files']:
        name=entry['path'];p=ROOT/name;relative=Path(name)
        if name=='outputs/tables/phase3a/executor.lock':
            # This advisory control file is intentionally locked by this process.
            # The previous seal included it; preserve its historical bytes without
            # trying to read a live exclusive lock. Scientific files are unchanged.
            assert sha256(archive/name)==entry['sha256'];continue
        mutable=name in CODE_CHANGES or (revised and name=='configs/phase3a.toml') or name=='docs/phase3a_seasonal_clustering_report.md' or name.startswith('outputs/figures/phase3a/') or (relative.parent.as_posix()=='outputs/tables/phase3a' and relative.name in AGGREGATES)
        for old,new in mapping.items():
            if name.startswith(old+'/'):p=ROOT/(new+name[len(old):]);break
        if mutable:assert sha256(archive/name)==entry['sha256']
        else:assert sha256(p)==entry['sha256']
    for spec in before['committed_fit_specs']:assert sha256(OUT/'runs'/spec['run_id'][:20]/'receipt.json')==before['committed_receipt_hashes'][spec['run_id']]
    if REC==LOW_RAM:
        prior=json.loads((REC/'preflight.json').read_text());archives={e['original_path']:e for e in prior['archived_files']}
        for e in prior['prior_inventory']:
            p=ROOT/(archives[e['path']]['archive_path'] if e['path'] in archives else e['path'])
            assert sha256(p)==e['sha256']
    return {'phase0_2b':4286,'original_phase3a_inventory':len(before['phase3a_original_files']),'original_committed_receipts_unchanged':115}

def certify(check_only=False):
    xml=REC/'pytest_regression.xml';path=REC/'executor_test_gate.json'
    if not check_only:
        suite=ET.parse(xml).getroot().find('testsuite');assert int(suite.attrib['tests'])>=507 and all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped'))
        atomic_json(path,{'certified_utc':utc_now(),'tests':int(suite.attrib['tests']),'xml_sha256':sha256(xml),'plan_sha256':sha256(OUT/'experiment_plan.json'),'files':{name:sha256(ROOT/name) for name in WATCHED}})
    gate=json.loads(path.read_text());assert sha256(xml)==gate['xml_sha256'] and sha256(OUT/'experiment_plan.json')==gate['plan_sha256']
    assert all(sha256(ROOT/k)==v for k,v in gate['files'].items());return gate

def stable_headroom(name,*,samples=7,interval=5.,snapshot=resource_snapshot,sleeper=time.sleep):
    rows=[];r=original.config()['resources'];minimum=r.get('minimum_startup_available_ram_bytes',1024**3)
    for i in range(samples):
        rows.append({'utc':utc_now(),**snapshot()})
        if i+1<samples:sleeper(interval)
    falling=len(rows)>1 and all(b['available_physical_bytes']<a['available_physical_bytes'] for a,b in zip(rows,rows[1:]))
    passed=not falling and all(v['available_physical_bytes']>=minimum and v['disk_free_bytes']>=r['minimum_disk_free_bytes'] for v in rows)
    result={'status':'passed' if passed else 'BLOCKED','gate':name,'minimum_stable_available_bytes':minimum,'runtime_reserve_bytes':r['minimum_available_ram_bytes'],'minimum_measured_available_bytes':min(v['available_physical_bytes'] for v in rows),'continuously_dropping':falling,'sampling_seconds':(samples-1)*interval,'samples':rows,'configuration_sha256':sha256(ROOT/'configs/phase3a.toml'),'controller_numeric_modules_loaded':any(k in sys.modules for k in ('numpy','pandas','sklearn','geopandas'))}
    stamp=utc_now().replace(':','').replace('.','').replace('+','_');path=REC/'resource_gates'/f'{name}_{stamp}.json';atomic_json(path,result)
    if not passed:raise RecoveryBlocked('Configured stable startup RAM/disk/trend gate failed; zero unsafe workers launched.')
    return result

def reconcile():
    record_path=REC/'staging_reconciliation.json'
    if REC==LOW_RAM:
        result=json.loads((HISTORY/'staging_reconciliation.json').read_text())
        for f in result['files']:assert sha256(ROOT/f['isolated_path'])==f['sha256']
        return result
    if record_path.exists():return json.loads(record_path.read_text())
    before=baseline();moves={};files=[];runroot=(OUT/'runs').resolve()
    for entry in before['interrupted']:
        source=(ROOT/entry['path']).resolve();target=source.with_name(source.name.removesuffix('.staged')+'.interrupted_recovery01')
        if source.parent!=runroot or target.parent!=runroot or not source.is_relative_to(ROOT.resolve()) or not target.is_relative_to(ROOT.resolve()):raise StoreConflict('Unsafe staging relocation target.')
        if target.exists() or (source/'receipt.json').exists():raise StoreConflict('Reconciliation conflict; no files overwritten.')
        for f in entry['files']:assert sha256(ROOT/f['path'])==f['sha256']
        source.rename(target);moves[entry['path']]=target.relative_to(ROOT).as_posix()
        for f in entry['files']:
            new=target/Path(f['path']).name;assert sha256(new)==f['sha256'];files.append({'original_path':f['path'],'isolated_path':new.relative_to(ROOT).as_posix(),'sha256':f['sha256']})
    result={'status':'passed','reconciled_utc':utc_now(),'relocations':moves,'files':files,'original_diagnostics_preserved':True,'interrupted_not_accepted_as_complete':True};atomic_json(record_path,result);return result

def classify(plan):
    rows=[];old={p['run_id'] for p in baseline()['committed_fit_specs']}
    for p in plan:
        d=OUT/'runs'/p['run_id'][:20];state='pending'
        if (d/'receipt.json').exists():
            receipt=json.loads((d/'receipt.json').read_text());assert receipt['spec']==p
            assert all(sha256(ROOT/e['path'])==e['sha256'] for e in receipt['files']);state='verified_complete'
        elif (d.with_name(d.name+'.staged')/'failure.json').exists():state='interrupted_uncommitted'
        rows.append({**{k:p[k] for k in ('run_id','season_id','subset','role')},'parameters':p['parameters'],'scientific':p['role'] not in ('high_volume_probe','extended_probe'),'classification':state,'reused_prompt14':p['run_id'] in old})
    return rows

def write_progress(plan):
    rows=classify(plan);atomic_json(REC/'classification_current.json',rows)
    with (REC/'classification_current.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

def primary_ready(plan):
    gate=json.loads((REC/'primary_gate.json').read_text());assert gate['status']=='passed' and gate['scientific_fits']==189 and gate['plan_sha256']==sha256(OUT/'experiment_plan.json')
    assert all(sha256(ROOT/k)==v for k,v in gate['receipts'].items())

def select(plan,mode):
    if mode=='primary':return [p for p in plan if p['role'] in ('primary_grid','primary_sensitivity')]
    if mode=='extended':primary_ready(plan);return sorted([p for p in plan if p['role'] in ('extended_probe','extended_reference')],key=lambda p:p['role']!='extended_probe')
    if mode=='repeat':return [p for p in plan if p['season_id'] in ('CORE_2020','CORE_2021') and p['subset']=='all' and p['role']=='primary_grid' and p['reference']]
    raise ValueError('Unknown group')

def run(mode):
    preserved();certify(True);plan=load_plan();selected=select(plan,mode);stable_headroom(mode);reconcile();results=[];large_checked=set();reused=0
    for p in selected:
        directory=OUT/'runs'/p['run_id'][:20]
        if mode!='repeat' and (directory/'receipt.json').exists():
            r=original.execute(p);assert r['reused'];reused+=1;continue
        if mode=='repeat' and (directory.with_name(directory.name+'.repeat')/'repeat_verification.json').exists():
            proof=json.loads((directory.with_name(directory.name+'.repeat')/'repeat_verification.json').read_text());assert proof['status']=='passed';reused+=1;continue
        if p['n_observations']>20000 and p['season_id'] not in large_checked:stable_headroom('large_'+p['season_id']);large_checked.add(p['season_id'])
        assert sha256(ROOT/p['input_path'])==p['input_sha256'];snapshot=resource_snapshot();r=original.config()['resources']
        # Existing worker peaks are ~225 MiB. Reserve at least 256 MiB additional
        # headroom before a launch; this strengthens, never lowers, the 512 MiB guard.
        if snapshot['available_physical_bytes']<r['minimum_available_ram_bytes']+256*1024**2 or snapshot['disk_free_bytes']<r['minimum_disk_free_bytes']:raise RecoveryBlocked('Insufficient conservative pre-worker RAM/disk headroom.')
        result=original.execute(p,repeat=mode=='repeat',verify_before_commit=True);results.append(result)
        atomic_json(REC/f'{mode}_execution.json',{'mode':mode,'updated_utc':utc_now(),'configuration_sha256':sha256(ROOT/'configs/phase3a.toml'),'reused_receipts':reused,'attempts':results});write_progress(plan)
        print(json.dumps({'mode':mode,'new_attempts':len(results),'reused':reused,'season':p['season_id'],'subset':p['subset'],'parameters':p['parameters'],'status':result['status']}),flush=True)
        if result['status']!='completed':raise RecoveryBlocked('Resource/scientific guard stopped the attempt; staging preserved.')
        assert result['resources']['worker_process_reaped'] and not result['resources']['controller_numeric_modules_loaded']
    preserved();return {'status':'passed','mode':mode,'new_attempts':len(results),'reused':reused}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--certify-tests',action='store_true');parser.add_argument('--check-headroom',action='store_true');parser.add_argument('--reconcile',action='store_true');parser.add_argument('--run',choices=('primary','extended','repeat'));args=parser.parse_args()
    try:
        if args.certify_tests:result=certify()
        elif args.check_headroom:result=stable_headroom('manual_recovery_check')
        elif args.reconcile:preserved();result=reconcile()
        elif args.run:
            with store_lock(OUT):result=run(args.run)
        else:
            rows=classify(load_plan());science=[r for r in rows if r['scientific']]
            result={'mode':'dry_run','fits':0,'classification':{s:sum(r['classification']==s for r in rows) for s in ('verified_complete','interrupted_uncommitted','pending','failed_verification')},
              'completed_scientific':sum(r['classification']=='verified_complete' for r in science),'remaining_scientific':sum(r['classification']!='verified_complete' for r in science),
              'primary_remaining':sum(r['classification']!='verified_complete' and r['role'] in ('primary_grid','primary_sensitivity') for r in science),
              'extended_remaining':sum(r['classification']!='verified_complete' and r['role']=='extended_reference' for r in science)}
        print(json.dumps(result,indent=2))
    except RecoveryBlocked as exc:
        plan=load_plan();rows=write_progress(plan);record={'status':'BLOCKED','reason':str(exc),'recorded_utc':utc_now(),'new_scientific_complete':sum(r['scientific'] and r['classification']=='verified_complete' and not r['reused_prompt14'] for r in rows),'remaining_scientific':sum(r['scientific'] and r['classification']!='verified_complete' for r in rows),'resources':resource_snapshot()};atomic_json(REC/'resource_stop.json',record);print(json.dumps(record,indent=2));raise SystemExit(3)
if __name__=='__main__':main()
