"""Mocked recovery gates plus exact scientific serialization equivalence."""
from pathlib import Path
import sys,json,importlib.util,subprocess,shutil
import pytest,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import phase3a_recovery as recovery
import phase3a as driver
from agniscope.provenance import sha256
from agniscope import seasonal_clustering as s

@pytest.mark.parametrize('ram,disk,passed',[(1024**3,20*1024**3,True),(1024**3-1,20*1024**3,False),(2*1024**3,15*1024**3-1,False),(2*1024**3,15*1024**3,True)])
def test_stable_restart_gate_preserves_both_limits(tmp_path,monkeypatch,ram,disk,passed):
    legacy=recovery.original.config();legacy['resources'].pop('minimum_startup_available_ram_bytes',None);monkeypatch.setattr(recovery.original,'config',lambda:legacy)
    monkeypatch.setattr(recovery,'REC',tmp_path);snap=lambda:{'available_physical_bytes':ram,'disk_free_bytes':disk}
    if passed:assert recovery.stable_headroom('fixture',samples=3,interval=0,snapshot=snap,sleeper=lambda _:None)['status']=='passed'
    else:
        with pytest.raises(recovery.RecoveryBlocked):recovery.stable_headroom('fixture',samples=3,interval=0,snapshot=snap,sleeper=lambda _:None)

def test_one_low_sample_rejects_restart_even_if_average_is_safe(tmp_path,monkeypatch):
    legacy=recovery.original.config();legacy['resources'].pop('minimum_startup_available_ram_bytes',None);monkeypatch.setattr(recovery.original,'config',lambda:legacy)
    monkeypatch.setattr(recovery,'REC',tmp_path);values=iter([2*1024**3,900*1024**2,2*1024**3])
    with pytest.raises(recovery.RecoveryBlocked):recovery.stable_headroom('fixture',samples=3,interval=0,snapshot=lambda:{'available_physical_bytes':next(values),'disk_free_bytes':20*1024**3},sleeper=lambda _:None)

def test_controller_remains_lightweight_in_fresh_interpreter():
    root=Path(__file__).resolve().parents[1];code="import sys; sys.path.insert(0,'scripts'); import phase3a_recovery as r; r.original.config(); assert not any(m in sys.modules for m in ('numpy','pandas','sklearn','geopandas','pyarrow')); print('passed')"
    assert subprocess.run([sys.executable,'-B','-c',code],cwd=root,capture_output=True,text=True,check=True,timeout=30).stdout.strip()=='passed'

def test_preservation_uses_archived_control_lock_without_reading_live_lock(tmp_path,monkeypatch):
    from agniscope.acquisition_store import store_lock
    rec=tmp_path/'recovery';out=tmp_path/'outputs/tables/phase3a';out.mkdir(parents=True)
    names=['configs/phase3a.toml','src/agniscope/seasonal_clustering.py','environment.lock.win-64.txt']
    for name in names:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('SYNTHETIC')
    lock_name='outputs/tables/phase3a/executor.lock';lock=tmp_path/lock_name;lock.write_bytes(b'');archived=rec/'baseline/files'/lock_name;archived.parent.mkdir(parents=True);archived.write_bytes(b'')
    before={'config_sha256':sha256(tmp_path/names[0]),'scientific_core_sha256':sha256(tmp_path/names[1]),'environment_lock_sha256':sha256(tmp_path/names[2]),'phase3a_original_files':[{'path':lock_name,'sha256':sha256(archived)}],'committed_fit_specs':[],'committed_receipt_hashes':{}}
    monkeypatch.setattr(recovery,'ROOT',tmp_path);monkeypatch.setattr(recovery,'OUT',out);monkeypatch.setattr(recovery,'REC',rec);monkeypatch.setattr(recovery,'baseline',lambda:before);monkeypatch.setattr(recovery.original,'protected',lambda:0)
    actual_sha=sha256
    def checked(path):
        if Path(path)==lock:raise PermissionError('Live lock must not be hashed')
        return actual_sha(path)
    monkeypatch.setattr(recovery,'sha256',checked)
    with store_lock(out):assert recovery.preserved()['original_phase3a_inventory']==1
    archived.write_bytes(b'CHANGED')
    with pytest.raises(AssertionError):recovery.preserved()

def test_reconciliation_preserves_every_byte_and_is_idempotent(tmp_path,monkeypatch):
    out=tmp_path/'outputs/tables/phase3a';rec=tmp_path/'recovery';stage=out/'runs/fit_fixture.staged';stage.mkdir(parents=True);rec.mkdir()
    (stage/'spec.json').write_text('{"source":"SYNTHETIC"}');(stage/'failure.json').write_text('{"failure":"available_RAM_below_reserve"}')
    files=[{'path':p.relative_to(tmp_path).as_posix(),'sha256':sha256(p)} for p in stage.iterdir()];before={'interrupted':[{'path':stage.relative_to(tmp_path).as_posix(),'files':files}]}
    monkeypatch.setattr(recovery,'ROOT',tmp_path);monkeypatch.setattr(recovery,'OUT',out);monkeypatch.setattr(recovery,'REC',rec);monkeypatch.setattr(recovery,'baseline',lambda:before)
    result=recovery.reconcile();assert not stage.exists() and result['original_diagnostics_preserved']
    assert recovery.reconcile()==result and all(sha256(tmp_path/r['isolated_path'])==r['sha256'] for r in result['files'])

@pytest.mark.parametrize('conflict',['receipt','destination'])
def test_reconciliation_rejects_conflicts_without_moving(tmp_path,monkeypatch,conflict):
    out=tmp_path/'out';stage=out/'runs/fit_fixture.staged';stage.mkdir(parents=True);(tmp_path/'rec').mkdir()
    if conflict=='receipt':(stage/'receipt.json').write_text('{}')
    else:stage.with_name('fit_fixture.interrupted_recovery01').mkdir()
    monkeypatch.setattr(recovery,'ROOT',tmp_path);monkeypatch.setattr(recovery,'OUT',out);monkeypatch.setattr(recovery,'REC',tmp_path/'rec');monkeypatch.setattr(recovery,'baseline',lambda:{'interrupted':[{'path':stage.relative_to(tmp_path).as_posix(),'files':[]}]})
    with pytest.raises(recovery.StoreConflict):recovery.reconcile()
    assert stage.exists()

def test_remaining_primary_order_and_probe_separation():
    plan=recovery.load_plan();before=recovery.baseline();old={p['run_id'] for p in before['committed_fit_specs']}
    remaining=[p for p in recovery.select(plan,'primary') if p['run_id'] not in old]
    assert len(remaining)==80 and [p['season_id'] for p in remaining]==['CORE_2023']*26+['CORE_2024']*27+['CORE_2025']*27
    assert sum(p['role'] not in ('high_volume_probe','extended_probe') for p in plan)==201
    assert len(recovery.select(plan,'repeat'))==4

def test_extended_remains_closed_until_independent_primary_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'REC',tmp_path)
    with pytest.raises(FileNotFoundError):recovery.select([], 'extended')

def test_precommit_verification_does_not_change_legacy_scientific_bytes(tmp_path,monkeypatch):
    import phase3a_worker as worker,verify_phase3a as verifier
    from test_seasonal_clustering import points
    monkeypatch.setattr(worker,'ROOT',tmp_path);monkeypatch.setattr(verifier,'ROOT',tmp_path);monkeypatch.setattr(verifier,'OUT',tmp_path/'outputs/tables/phase3a')
    frame=points([[0,0],[100,0],[200,0],[10000,0]]);source=tmp_path/'synthetic.parquet';frame.to_parquet(source,index=False)
    spec={'input_path':'synthetic.parquet','input_sha256':sha256(source),'n_observations':4,'input_feature_identity':s.feature_identity(frame),'season_id':'SYNTHETIC','subset':'all','parameters':driver.REFS[0],'reference':True,'run_id':'fit_SYNTHETIC'}
    dirs=[]
    for name,flag in [('legacy',False),('checked',True)]:
        d=tmp_path/'outputs/tables/phase3a/runs'/name;d.mkdir(parents=True);(d/'spec.json').write_text(json.dumps(spec));dirs.append(d)
        monkeypatch.setattr(sys,'argv',['worker',str(d.relative_to(tmp_path))]+(['--verify-before-commit'] if flag else []));worker.main()
    for p in dirs[0].iterdir():
        if p.name!='timing.json':assert sha256(p)==sha256(dirs[1]/p.name)
    proof=json.loads((dirs[1]/'precommit_verification.json').read_text());assert proof['status']=='passed' and proof['n']==4
    assert not (dirs[1]/'receipt.json').exists()
    bad=dirs[1]/'labels_compact.npz';np.savez_compressed(bad,labels=np.full(4,-1),strength=np.full(4,np.nan),core=np.zeros(4,bool))
    with pytest.raises(AssertionError):verifier.verify_spec(spec,directory=dirs[1],require_receipt=False)

@pytest.mark.parametrize('proof_status',['passed','failed'])
def test_controller_requires_precommit_proof_before_atomic_placement(tmp_path,monkeypatch,proof_status):
    out=tmp_path/'outputs/tables/phase3a';monkeypatch.setattr(driver,'ROOT',tmp_path);monkeypatch.setattr(driver,'OUT',out)
    resource={'minimum_available_ram_bytes':512*1024**2,'minimum_disk_free_bytes':15*1024**3,'maximum_worker_working_set_bytes':512*1024**2,'maximum_fit_seconds':180,'monitor_interval_seconds':.1}
    monkeypatch.setattr(driver,'config',lambda:{'resources':resource});monkeypatch.setattr(driver,'resource_snapshot',lambda:{'available_physical_bytes':2*1024**3,'disk_free_bytes':30*1024**3})
    spec={'run_id':'fit_SYNTHETIC'}
    class FakeProcess:
        returncode=0;pid=999999
        def __init__(self,command,**kwargs):
            assert command[-1]=='--verify-before-commit';d=tmp_path/command[3];(d/'precommit_verification.json').write_text(json.dumps({'status':proof_status,'run_id':spec['run_id']}))
        def poll(self):return 0
        def communicate(self):return b'{}',b''
    monkeypatch.setattr(driver.subprocess,'Popen',FakeProcess)
    if proof_status=='passed':
        r=driver.execute(spec,verify_before_commit=True);assert r['resources']['worker_process_reaped'] and (out/'runs/fit_SYNTHETIC/receipt.json').exists()
    else:
        with pytest.raises(driver.StoreConflict):driver.execute(spec,verify_before_commit=True)
        assert not (out/'runs/fit_SYNTHETIC/receipt.json').exists() and (out/'runs/fit_SYNTHETIC.staged').exists()
