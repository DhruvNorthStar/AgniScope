"""Read-only Windows memory and workspace disk measurements."""

import ctypes
from ctypes import wintypes
import shutil
import sys

from .config import ROOT


def resource_snapshot() -> dict:
    report = {"disk_total_bytes": shutil.disk_usage(ROOT).total, "disk_free_bytes": shutil.disk_usage(ROOT).free}
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [(name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            report.update(total_physical_bytes=status.total_phys, available_physical_bytes=status.avail_phys, physical_memory_load_percent=status.load)
        class ProcessCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "peak_paged_pool", "paged_pool", "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile")]
        counters = ProcessCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if get_info(get_process(), ctypes.byref(counters), counters.cb):
            report.update(process_peak_working_set_bytes=counters.peak_working_set, process_working_set_bytes=counters.working_set)
    return report
