"""Authorized startup change cannot relax runtime, cap, disk or serial execution."""
from pathlib import Path
import sys,json,tomllib
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import phase3a_recovery as recovery
import phase3a as driver
MIB=1024**2

@pytest.mark.parametrize('available,passed',[(896*MIB,True),(896*MIB-1,False),(1024*MIB,True)])
def test_configured_startup_boundary(tmp_path,monkeypatch,available,passed):
    monkeypatch.setattr(recovery,'REC',tmp_path)
    snap=lambda:{'available_physical_bytes':available,'disk_free_bytes':20*1024**3}
    if passed:
        result=recovery.stable_headroom('fixture',samples=2,interval=0,snapshot=snap,sleeper=lambda _:None)
        assert result['minimum_stable_available_bytes']==896*MIB and result['runtime_reserve_bytes']==512*MIB
    else:
        with pytest.raises(recovery.RecoveryBlocked):recovery.stable_headroom('fixture',samples=2,interval=0,snapshot=snap,sleeper=lambda _:None)

def test_continuously_falling_memory_rejects_even_above_threshold(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'REC',tmp_path);values=iter([1000,990,980,970,960,950,940])
    with pytest.raises(recovery.RecoveryBlocked):
        recovery.stable_headroom('falling',snapshot=lambda:{'available_physical_bytes':next(values)*MIB,'disk_free_bytes':20*1024**3},sleeper=lambda _:None)
    result=json.loads(next((tmp_path/'resource_gates').glob('*.json')).read_text());assert result['continuously_dropping'] and result['sampling_seconds']==30

def test_full_stability_observation_and_unchanged_disk_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'REC',tmp_path);sleeps=[]
    result=recovery.stable_headroom('duration',snapshot=lambda:{'available_physical_bytes':900*MIB,'disk_free_bytes':15*1024**3},sleeper=sleeps.append)
    assert sleeps==[5.]*6 and len(result['samples'])==7 and result['sampling_seconds']==30 and not result['continuously_dropping']
    with pytest.raises(recovery.RecoveryBlocked):recovery.stable_headroom('disk',samples=2,interval=0,snapshot=lambda:{'available_physical_bytes':900*MIB,'disk_free_bytes':15*1024**3-1},sleeper=lambda _:None)

def test_only_authorized_configuration_value_changed():
    root=driver.ROOT
    before=tomllib.loads((root/'outputs/tables/phase3a_low_ram/baseline/files/configs/phase3a.toml').read_text())
    after=driver.config();assert after['resources'].pop('minimum_startup_available_ram_bytes')==896*MIB and after==before
    r=driver.config()['resources'];assert r['minimum_available_ram_bytes']==512*MIB and r['maximum_worker_working_set_bytes']==512*MIB and r['minimum_disk_free_bytes']==15*1024**3 and r['concurrency']==1 and r['monitor_interval_seconds']==.1

def test_runtime_reserve_still_terminates_worker_without_committing(tmp_path,monkeypatch):
    out=tmp_path/'out';monkeypatch.setattr(driver,'OUT',out);monkeypatch.setattr(driver,'ROOT',tmp_path)
    resources={'minimum_available_ram_bytes':512*MIB,'maximum_worker_working_set_bytes':512*MIB,'minimum_disk_free_bytes':15*1024**3,'maximum_fit_seconds':180,'monitor_interval_seconds':.1,'minimum_startup_available_ram_bytes':896*MIB}
    monkeypatch.setattr(driver,'config',lambda:{'resources':resources})
    available=iter([2*1024**3,511*MIB,2*1024**3])
    monkeypatch.setattr(driver,'resource_snapshot',lambda:{'available_physical_bytes':next(available),'disk_free_bytes':20*1024**3})
    monkeypatch.setattr(driver,'worker_memory',lambda _: {'working':100*MIB,'peak':100*MIB})
    class FakeProcess:
        returncode=None;pid=999999
        def __init__(self,*args,**kwargs):self.terminated=False
        def poll(self):return self.returncode
        def terminate(self):self.terminated=True;self.returncode=1
        def communicate(self):assert self.terminated;return b'{}',b''
    monkeypatch.setattr(driver.subprocess,'Popen',FakeProcess)
    result=driver.execute({'run_id':'fit_RUNTIME_FIXTURE'},verify_before_commit=True)
    assert result['status']=='failed' and result['failure']=='available_RAM_below_reserve'
    assert result['resources']['worker_cap_bytes']==512*MIB and result['resources']['minimum_available_ram_bytes']==511*MIB
    assert result['resources']['worker_process_reaped'] and not (out/'runs/fit_RUNTIME_FIXTURE/receipt.json').exists()
