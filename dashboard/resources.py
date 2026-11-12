"""Dashboard resource overrides; never mutates the scientific worker contract."""
from pathlib import Path
import tomllib
from scripts.phase3b_preflight import settings
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/dashboard_resources.toml"
MIB=1024**2
GIB=1024**3

def load_limits(path=CONFIG):
    values=tomllib.loads(Path(path).read_text(encoding="utf-8"))["dashboard"]
    assert set(values)=={"minimum_runtime_available_ram_bytes","maximum_server_peak_working_set_bytes","low_memory_warning_bytes","critical_memory_warning_bytes"}
    assert all(type(value) is int for value in values.values())
    assert values["maximum_server_peak_working_set_bytes"]>0
    limits=dict(settings()["resources"])
    assert 0<values["minimum_runtime_available_ram_bytes"]<=values["critical_memory_warning_bytes"]<=values["low_memory_warning_bytes"]<limits["minimum_startup_available_ram_bytes"]
    limits["minimum_available_ram_bytes"]=values["minimum_runtime_available_ram_bytes"]
    limits.update({key:value for key,value in values.items() if key!="minimum_runtime_available_ram_bytes"})
    return limits

def process_cap(limits,launch_available_bytes):
    return min(limits["maximum_server_peak_working_set_bytes"],launch_available_bytes//2)

def runtime_violations(sample,limits,server_peak_bytes,launch_available_bytes):
    rules=[("system_available_ram_bytes",sample["available_physical_bytes"],limits["minimum_available_ram_bytes"],"minimum","MiB"),
        ("disk_free_bytes",sample["disk_free_bytes"],limits["minimum_disk_free_bytes"],"minimum","GiB"),
        ("server_peak_working_set_bytes",server_peak_bytes,process_cap(limits,launch_available_bytes),"maximum","MiB")]
    return [dict(resource=name,observed_bytes=actual,threshold_bytes=threshold,requirement=direction,unit=unit)
        for name,actual,threshold,direction,unit in rules if (actual<threshold if direction=="minimum" else actual>threshold)]

def memory_band(available_bytes,limits):
    if available_bytes<limits["minimum_available_ram_bytes"]:return "stop"
    if available_bytes<=limits["critical_memory_warning_bytes"]:return "critical"
    if available_bytes<limits["low_memory_warning_bytes"]:return "low"
    return "normal"

def describe_violation(failure):
    divisor=GIB if failure["unit"]=="GiB" else MIB
    relation=">=" if failure["requirement"]=="minimum" else "<="
    return (f'{failure["resource"]}: observed {failure["observed_bytes"]/divisor:.2f} {failure["unit"]}; '
        f'required {relation} {failure["threshold_bytes"]/divisor:.2f} {failure["unit"]} '
        f'(observed_bytes={failure["observed_bytes"]}, threshold_bytes={failure["threshold_bytes"]})')

def startup_violations(gate,limits):
    lowest_ram=min(r["available_physical_bytes"] for r in gate["samples"])
    lowest_disk=min(r["disk_free_bytes"] for r in gate["samples"])
    result=[]
    for name,actual,threshold,unit in (("startup_system_available_ram_bytes",lowest_ram,limits["minimum_startup_available_ram_bytes"],"MiB"),
            ("disk_free_bytes",lowest_disk,limits["minimum_disk_free_bytes"] ,"GiB")):
        if actual<threshold:result.append(dict(resource=name,observed_bytes=actual,threshold_bytes=threshold,requirement="minimum",unit=unit))
    if gate["continuously_dropping"]:result.append(dict(resource="startup_continuously_falling_ram",observed=True,allowed=False))
    return result
