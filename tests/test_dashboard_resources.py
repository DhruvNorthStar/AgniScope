"""Dashboard-only RAM override, diagnostic stops and unchanged ML protection."""
from pathlib import Path
import json,sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
from dashboard import resources as r
import phase3b_preflight as science
import dashboard_server as server
MIB=1024**2
GIB=1024**3

def sample(ram=1000,disk=20):return {"available_physical_bytes":int(ram*MIB),"disk_free_bytes":int(disk*GIB)}

def test_dashboard_override_keeps_scientific_contract_unchanged():
    before=dict(science.settings()["resources"]);limits=r.load_limits()
    assert limits["minimum_available_ram_bytes"]==200*MIB
    assert limits["minimum_startup_available_ram_bytes"]==896*MIB
    assert limits["minimum_disk_free_bytes"]==15*GIB
    assert limits["maximum_worker_working_set_bytes"]==512*MIB
    assert limits["maximum_server_peak_working_set_bytes"]==1024*MIB
    assert limits["concurrency"]==1 and limits["monitor_interval_seconds"]==.1
    assert science.settings()["resources"]==before and before["minimum_available_ram_bytes"]==512*MIB
    assert not science.runtime_safe(sample(300),before)
    assert r.runtime_violations(sample(300),limits,100*MIB,1000*MIB)==[]

@pytest.mark.parametrize("ram,failed",[(199.99,True),(200,False),(256,False),(468.23,False),(512,False)])
def test_runtime_system_ram_boundary(ram,failed):
    result=r.runtime_violations(sample(ram),r.load_limits(),100*MIB,1100*MIB)
    assert bool(result)==failed
    if failed:assert result[0]["resource"]=="system_available_ram_bytes" and result[0]["threshold_bytes"]==200*MIB

@pytest.mark.parametrize("peak,launch,failed",[(1024,2400,False),(1024.01,2400,True),(600,1200,False),(600.01,1200,True),(450,900,False),(450.01,900,True)])
def test_server_cap_is_separate_and_retains_half_launch_rule(peak,launch,failed):
    result=r.runtime_violations(sample(300),r.load_limits(),int(peak*MIB),int(launch*MIB))
    assert bool(result)==failed
    if failed:assert result[0]["resource"]=="server_peak_working_set_bytes"

@pytest.mark.parametrize("disk,failed",[(14.999,True),(15,False)])
def test_disk_reserve_unchanged(disk,failed):
    result=r.runtime_violations(sample(1000,disk),r.load_limits(),100*MIB,1000*MIB)
    assert bool(result)==failed
    if failed:assert result[0]["resource"]=="disk_free_bytes" and result[0]["threshold_bytes"]==15*GIB

def test_all_failing_resources_and_exact_values_are_reported():
    failures=r.runtime_violations(sample(199,14),r.load_limits(),1025*MIB,2400*MIB)
    assert {f["resource"] for f in failures}=={"system_available_ram_bytes","disk_free_bytes","server_peak_working_set_bytes"}
    text="\n".join(r.describe_violation(f) for f in failures)
    assert "199.00 MiB" in text and ">= 200.00 MiB" in text and "<= 1024.00 MiB" in text
    assert "observed_bytes=208666624" in text and "threshold_bytes=209715200" in text

@pytest.mark.parametrize("ram,band",[(199,"stop"),(200,"critical"),(256,"critical"),(256.01,"low"),(511.99,"low"),(512,"normal")])
def test_warning_bands(ram,band):assert r.memory_band(int(ram*MIB),r.load_limits())==band

@pytest.mark.parametrize("ram,status",[(895.99,"BLOCKED"),(896,"passed")])
def test_startup_still_requires_every_reading_at_896_for_30_seconds(ram,status):
    waits=[];gate=science.stable_headroom(r.load_limits(),snapshot=lambda:sample(ram),sleeper=waits.append)
    assert gate["status"]==status and waits==[5]*6 and len(gate["samples"])==7
    failures=r.startup_violations(gate,r.load_limits())
    if status=="BLOCKED":assert failures[0]["threshold_bytes"]==896*MIB

def test_continuously_falling_startup_ram_still_blocks():
    values=iter([1006,1005,1004,1003,1002,1001,1000])
    gate=science.stable_headroom(r.load_limits(),snapshot=lambda:sample(next(values)),sleeper=lambda _:None)
    assert gate["status"]=="BLOCKED" and r.startup_violations(gate,r.load_limits())[0]["resource"]=="startup_continuously_falling_ram"

def test_real_monitor_accepts_300_but_stops_at_199_and_saves_diagnostics(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(server,"ROOT",tmp_path);monkeypatch.setattr(server,"OUT",tmp_path/"evidence")
    config=tmp_path/"configs/dashboard_resources.toml";config.parent.mkdir();config.write_bytes(r.CONFIG.read_bytes());monkeypatch.setattr(server,"CONFIG",config)
    monkeypatch.setattr(server,"utc_now",lambda:"fixture")
    monkeypatch.setattr(server,"stable_headroom",lambda limits:{"status":"passed","samples":[sample()],"continuously_dropping":False})
    snapshots=iter([sample(),sample(300),sample(199)])
    monkeypatch.setattr(server,"resource_snapshot",lambda:next(snapshots))
    monkeypatch.setattr(server,"worker_memory",lambda pid:{"peak":100*MIB,"working":100*MIB})
    monkeypatch.setattr(server.time,"sleep",lambda _:None)
    class Worker:
        pid=123;stopped=False
        def poll(self):return -15 if self.stopped else None
        def terminate(self):self.stopped=True
        def wait(self,timeout):assert self.stopped;return -15
    worker=Worker();monkeypatch.setattr(server.subprocess,"Popen",lambda *args,**kwargs:worker)
    assert server.main([])==2 and worker.stopped
    folder=tmp_path/"evidence/server_sessions/fixture";receipt=json.loads((folder/"session.json").read_text())
    assert receipt["minimum_available_bytes"]==199*MIB and receipt["cap_bytes"]==500*MIB
    assert receipt["warning_counts"]["low"]==1 and receipt["failures"][0]["resource"]=="system_available_ram_bytes"
    events=[json.loads(line) for line in (folder/"resource_monitor.jsonl").read_text().splitlines()]
    assert any(e["event"]=="sample" and e["available_physical_bytes"]==300*MIB for e in events)
    assert any(e["event"]=="STOP" and e["failures"][0]["observed_bytes"]==199*MIB for e in events)
    console=capsys.readouterr().out
    assert "LOW" in console and "199.00 MiB" in console and "200.00 MiB" in console

@pytest.mark.parametrize("ram,kind",[(400,"warning"),(250,"error")])
def test_browser_warning_uses_dashboard_reserve_without_changing_metrics(monkeypatch,ram,kind):
    from streamlit.testing.v1 import AppTest
    import dashboard.resource_status as status
    monkeypatch.setattr(status,"resource_snapshot",lambda:sample(ram))
    app=AppTest.from_file(str(ROOT/"dashboard/app.py"),default_timeout=60).run()
    assert not app.exception
    alerts=app.warning if kind=="warning" else app.error
    assert any("200 MiB" in a.value and "server process cap remains separate" in a.value for a in alerts)
    assert any(m.label=="Validated observations" and m.value=="95,052" for m in app.metric)


def test_actual_monitor_uses_dashboard_1024_cap_and_stops_above_it(tmp_path,monkeypatch):
    monkeypatch.setattr(server,"ROOT",tmp_path);monkeypatch.setattr(server,"OUT",tmp_path/"evidence")
    config=tmp_path/"configs/dashboard_resources.toml";config.parent.mkdir();config.write_bytes(r.CONFIG.read_bytes());monkeypatch.setattr(server,"CONFIG",config)
    monkeypatch.setattr(server,"utc_now",lambda:"cap_fixture")
    monkeypatch.setattr(server,"stable_headroom",lambda limits:{"status":"passed","samples":[sample(2400)],"continuously_dropping":False})
    monkeypatch.setattr(server,"resource_snapshot",lambda:sample(2400))
    peaks=iter([700*MIB,1025*MIB]);monkeypatch.setattr(server,"worker_memory",lambda pid:{"peak":next(peaks),"working":100*MIB})
    monkeypatch.setattr(server.time,"sleep",lambda _:None)
    class Worker:
        pid=123;stopped=False
        def poll(self):return -15 if self.stopped else None
        def terminate(self):self.stopped=True
        def wait(self,timeout):assert self.stopped;return -15
    worker=Worker();monkeypatch.setattr(server.subprocess,"Popen",lambda *args,**kwargs:worker)
    assert server.main([])==2 and worker.stopped
    folder=tmp_path/"evidence/server_sessions/cap_fixture"
    receipt=json.loads((folder/"session.json").read_text())
    assert receipt["cap_bytes"]==1024*MIB and receipt["peak_server_bytes"]==1025*MIB
    assert receipt["failures"]==[dict(resource="server_peak_working_set_bytes",observed_bytes=1025*MIB,threshold_bytes=1024*MIB,requirement="maximum",unit="MiB")]
    events=[json.loads(line) for line in (folder/"resource_monitor.jsonl").read_text().splitlines()]
    assert any(e["event"]=="sample" and e["server_peak_bytes"]==700*MIB for e in events)
