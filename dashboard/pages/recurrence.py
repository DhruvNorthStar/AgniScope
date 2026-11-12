import streamlit as st
from dashboard import data,maps,ui
from dashboard.content import METHODS,RECURRENCE_LABEL

ui.heading("Recurring Hotspot Map",RECURRENCE_LABEL+". A place qualifies when its grid cell has support in several primary seasons.","04 / The recurrence atlas")
with st.container(border=True):
    a,b,c,d=st.columns(4)
    resolution=a.selectbox("Grid resolution (m)",[500,1000,2000],index=1,key="recurrence_resolution",format_func=lambda x:"500 m" if x==500 else f"{x//1000} km")
    method=b.selectbox("Support definition",list(METHODS),format_func=METHODS.get,key="recurrence_method")
    k=c.selectbox("Minimum active seasons",[2,3,4,5],index=2,format_func=lambda x:f"{x} of 7",key="recurrence_k")
    selected=d.selectbox("District support",["All districts"]+data.district_names(),key="recurrence_district")
district=None if selected=="All districts" else selected
summary=data.recurrence_summary(resolution,method,k,district)
components=data.component_summary(resolution,method,k)
ui.metrics([("Recurring cells",f'{summary["cells"]:,}',"All qualifying saved cells in this selected scope. District cell counts can overlap."),
    ("Clipped recurring support",f'{summary["area_km2"]:,.2f} km²',"Actual saved cell/footprint intersections; for a district, exact district-intersection areas."),
    ("Share of approved area",f'{summary["share_percent"]:.2f}%',"Selected recurring support divided by the same saved analytical footprint or district area."),
    ("Connected areas · statewide",f'{int(components["components"]):,}',"Saved shared-edge components for this resolution/method/k, always statewide even if a district filter is active.")])
ui.section(f'{"500 m" if resolution==500 else str(resolution//1000)+" km"} · {METHODS[method]} · at least {k} of 7 seasons',"Colour shows the observed active-season count, not a probability. The reference view is 1 km / all detections / four of seven.")
ui.frequency_legend(k)
cells=data.recurrence_cells(resolution,method,k,district)
if cells.empty:st.info("No saved cells meet this selection. This is not evidence of fire absence.")
else:
    webmap,shown=maps.recurrence_map(cells,resolution,district=district)
    ui.render_map(webmap,"recurrence_map",height=650);ui.sample_caption(shown,len(cells),"qualifying cells")
    mask=0
    for value in cells.season_mask:mask|=int(value)
    st.caption("Active years represented in the full selection: "+", ".join(map(str,data.years_from_mask(mask)))+". Hover identifies the years for each displayed cell.")
if district:st.info("District area uses saved exact intersections. Boundary-straddling cells are drawn as their complete approved-footprint-clipped geometry and may extend into an adjacent district. District cell counts are nonadditive; connected-area metrics remain statewide.")
ui.caution("Recurring grid cells and connected areas describe observed spatial support. They are not exact wildfire boundaries, independent fires, burned area or future wildfire risk.")
ui.section("One reference, three definitions","The fixed statewide comparison below stays the same while you explore the map.")
overlap=data.single(data.table("overlap"),resolution_m=1000,subset="all",k=4,left="dbscan",right="hdbscan")
items=[(METHODS[m],f'{data.recurrence_summary(1000,m,4)["cells"]:,}',"Fixed statewide reference; unchanged by the controls above.") for m in METHODS]
items.append(("DBSCAN / HDBSCAN shared",f'{int(overlap["intersection_cells"]):,}',"Intersection of the two saved cluster-supported reference cell sets."))
ui.metrics(items)
with st.expander("Source, geometry and threshold definition"):
    st.markdown("A cell receives one binary vote for a primary season with saved support. The denominator is seven study seasons; independently complete satellite opportunity remains unknown. All confidence levels and land-cover contexts are retained in this baseline.")
    st.code(data.catalog()["sources"][f"cells_{resolution}"]["path"])
    st.caption("EPSG:32644 · origin (0,0) metres · half-open floor assignment · positive-area district-union clipping. Display geometries are transformed to EPSG:4326. No recurrence is recomputed by this interface.")
    st.caption(f'Statewide shared-edge components crossing districts: {int(components["cross_district_components"]):,}. Largest component: {int(components["largest_component_cells"]):,} cells / {components["largest_component_area_km2"]:.2f} km² of clipped support.')
ui.footer()
