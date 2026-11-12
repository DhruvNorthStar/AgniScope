"""Exactly six read-only research pages; startup never fits or acquires data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if str(ROOT/"src") not in sys.path:sys.path.insert(0,str(ROOT/"src"))
import streamlit as st
from dashboard import data,ui
from dashboard.content import PAGES,APP_NAME
from dashboard.resource_status import show_warning

st.set_page_config(page_title=APP_NAME+" | Recurrent Thermal Hotspots, 2019–2025",page_icon="🔥",layout="wide",initial_sidebar_state="expanded")
ui.style()
st.logo(str(ROOT/"dashboard/assets/wordmark.svg"),size="large",icon_image=str(ROOT/"dashboard/assets/mark.svg"))
try:
    data.catalog() # Small immutable bindings only; page-specific data stays lazy.
    pages=[st.Page(Path(__file__).parent/"pages"/filename,title=title,url_path=url,default=index==0) for index,(title,filename,url) in enumerate(PAGES)]
    navigation=st.navigation(pages,position="sidebar")
    st.sidebar.divider();st.sidebar.caption("UTTARAKHAND · 2019–2025\n\nNASA FIRMS / Standard Processing\n\nExplore the research. All results are saved and read-only.")
    with st.sidebar:show_warning()
    navigation.run()
except data.DataIntegrityError as error:
    st.error("Scientific source integrity check failed. "+str(error))
    st.info("Restore the verified source or run the offline integrity verifier. No substitute data have been loaded.")
    st.stop()
except (FileNotFoundError,ValueError) as error:
    st.error("This view could not be displayed. "+str(error))
    st.info("Check the selected filters and the local verified output files. Missing results are not zero observations.")
    st.stop()
