"""Read-only browser warning; the external monitor alone enforces termination."""
import streamlit as st
from agniscope.resources import resource_snapshot
from dashboard.resources import load_limits,memory_band,MIB

@st.fragment(run_every="5s")
def show_warning():
    limits=load_limits();sample=resource_snapshot();available=sample["available_physical_bytes"]
    band=memory_band(available,limits)
    if band!="normal":
        message=(f'System memory is {"dangerously low" if band in ("critical","stop") else "low"}: {available/MIB:.0f} MiB available. '
            f'The dashboard stops below {limits["minimum_available_ram_bytes"]/MIB:.0f} MiB. '
            'Close unneeded applications; paging, slow responses or a Windows freeze are possible. The server process cap remains separate.')
        st.error(message) if band in ("critical","stop") else st.warning(message)
