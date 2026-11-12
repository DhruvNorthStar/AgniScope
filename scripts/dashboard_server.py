"""Local-only dashboard monitor; scientific resource guards remain unchanged."""
from pathlib import Path
import argparse,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from agniscope.acquisition_store import atomic_json,store_lock
from agniscope.provenance import utc_now,sha256
from agniscope.resources import resource_snapshot
from phase3b_preflight import stable_headroom
from phase3a import worker_memory
from dashboard.resources import CONFIG,load_limits,process_cap,runtime_violations,memory_band,describe_violation,startup_violations,MIB
OUT=ROOT/"outputs/tables/phase4b"

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("--port",type=int,default=8501)
    parser.add_argument("--max-seconds",type=int,default=0,help="Optional bounded QA duration; zero runs until Ctrl+C or safety stop.")
    args=parser.parse_args(argv);assert 1024<=args.port<=65535 and args.max_seconds>=0
    stamp=utc_now().replace(":","").replace(".","").replace("+","_")
    folder=OUT/"server_sessions"/stamp;folder.mkdir(parents=True,exist_ok=False)
    limits=load_limits();policy={"configuration":CONFIG.relative_to(ROOT).as_posix(),"sha256":sha256(CONFIG),"limits":limits,
        "scope":"dashboard only; scientific worker RAM minimum remains 512 MiB"}
    gate=stable_headroom(limits);atomic_json(folder/"startup_gate.json",gate)
    if gate["status"]!="passed":
        failures=startup_violations(gate,limits)
        atomic_json(folder/"session.json",{"status":"BLOCKED","reason":"startup_resource_gate","server_started":False,"failures":failures,"policy":policy})
        for failure in failures:
            print("STOP: "+(describe_violation(failure) if "observed_bytes" in failure else "available RAM fell continuously through the 30-second gate"),flush=True)
        return 2
    before=resource_snapshot();cap=process_cap(limits,before["available_physical_bytes"])
    launch_failures=startup_violations({"samples":[before],"continuously_dropping":False},limits)
    if launch_failures:
        atomic_json(folder/"session.json",{"status":"BLOCKED","reason":"immediate_prelaunch_resource_check","server_started":False,"failures":launch_failures,"policy":policy})
        for failure in launch_failures:print("STOP: "+describe_violation(failure),flush=True)
        return 2
    command=[sys.executable,"-B","-m","streamlit","run",str(ROOT/"dashboard/app.py"),"--server.address=127.0.0.1",f"--server.port={args.port}","--server.headless=true","--server.fileWatcherType=none","--browser.gatherUsageStats=false"]
    started=time.monotonic();peak=0;minimum=before["available_physical_bytes"];disk=before["disk_free_bytes"];reason=None;failures=[];warning_counts={"low":0,"critical":0};previous_band="normal";last_sample=-5
    log=folder/"server.log";monitor=folder/"resource_monitor.jsonl"
    with store_lock(OUT/"dashboard_server.lock"):
        with log.open("x",encoding="utf-8") as handle,monitor.open("x",encoding="utf-8") as events:
            def record(event,**values):
                events.write(json.dumps({"utc":utc_now(),"event":event,**values})+"\n");events.flush()
            process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            atomic_json(folder/"running.json",{"pid":process.pid,"port":args.port,"session_started_utc":utc_now(),"launch_available_bytes":before["available_physical_bytes"],"cap_bytes":cap,"server_log":log.relative_to(ROOT).as_posix(),"policy":policy})
            print(f"AgniScope: http://127.0.0.1:{args.port} — Ctrl+C stops the local app.",flush=True)
            print(f"Dashboard system RAM reserve: {limits['minimum_available_ram_bytes']/MIB:.0f} MiB; separate server peak-working-set cap: {cap/MIB:.2f} MiB. Startup 896 MiB; disk 15 GiB.",flush=True)
            print("CAUTION: a 200 MiB system reserve can cause heavy paging, slow responses or a Windows freeze. Low-memory warnings begin below 512 MiB; critical warnings at 256 MiB.",flush=True)
            try:
                while process.poll() is None:
                    sample=resource_snapshot();memory=worker_memory(process.pid)
                    if memory:peak=max(peak,memory["peak"])
                    minimum=min(minimum,sample["available_physical_bytes"]);disk=min(disk,sample["disk_free_bytes"])
                    failures=runtime_violations(sample,limits,peak,before["available_physical_bytes"])
                    band=memory_band(sample["available_physical_bytes"],limits)
                    if failures:
                        reason="runtime_resource_limit";record("STOP",failures=failures,available_physical_bytes=sample["available_physical_bytes"],disk_free_bytes=sample["disk_free_bytes"],server_peak_bytes=peak,cap_bytes=cap)
                        for failure in failures:print("STOP: "+describe_violation(failure),flush=True)
                        break
                    if band!=previous_band:
                        if band in warning_counts:
                            warning_counts[band]+=1
                            print(f"WARNING — {'DANGEROUSLY LOW' if band=='critical' else 'LOW'} system RAM: {sample['available_physical_bytes']/MIB:.2f} MiB available; stop threshold {limits['minimum_available_ram_bytes']/MIB:.0f} MiB. Close unneeded applications; paging or freezing is possible.",flush=True)
                        record("memory_band_changed",band=band,available_physical_bytes=sample["available_physical_bytes"],stop_threshold_bytes=limits["minimum_available_ram_bytes"])
                        previous_band=band
                    elapsed=time.monotonic()-started
                    if elapsed-last_sample>=5:
                        record("sample",available_physical_bytes=sample["available_physical_bytes"],disk_free_bytes=sample["disk_free_bytes"],server_peak_bytes=peak,cap_bytes=cap);last_sample=elapsed
                    if args.max_seconds and elapsed>=args.max_seconds:reason="bounded_QA_duration_reached";break
                    time.sleep(limits["monitor_interval_seconds"])
            except KeyboardInterrupt:reason="user_stopped"
            finally:
                if process.poll() is None:process.terminate()
                code=process.wait(timeout=30)
                record("server_stopped",reason=reason or "server_exited_unexpectedly",peak_server_bytes=peak,minimum_available_bytes=minimum,minimum_disk_free_bytes=disk)
    if reason is None:reason="server_exited_unexpectedly"
    status="safely_stopped" if reason in ("user_stopped","bounded_QA_duration_reached") else "BLOCKED"
    atomic_json(folder/"session.json",{"status":status,"reason":reason,"failures":failures,"policy":policy,"warning_counts":warning_counts,"exit_code":code,"elapsed_seconds":time.monotonic()-started,
        "peak_server_bytes":peak,"minimum_available_bytes":minimum,"minimum_disk_free_bytes":disk,"cap_bytes":cap,
        "startup_gate_sha256":sha256(folder/"startup_gate.json"),"server_log_sha256":sha256(log),"monitor_log_sha256":sha256(monitor),"server_started":True,
        "observation_HTTP_calls":0,"credential_reads":0,"clustering_refits":0})
    print(json.dumps({"status":status,"reason":reason,"failures":failures,"session":folder.relative_to(ROOT).as_posix(),"peak_server_MiB":peak/MIB,"minimum_available_MiB":minimum/MIB,"minimum_free_disk_GiB":disk/1024**3},indent=2),flush=True)
    return 2 if status=="BLOCKED" else 0

if __name__=="__main__":raise SystemExit(main())
