import pandas as pd
import streamlit as st
from dashboard import data,ui
from dashboard.content import METHODS

ui.heading("Scientific Validation & Sensitivity","The pattern depends on how it is measured. Read the saved comparisons below to see which choices matter, and where the evidence stops.","05 / Reading the uncertainty")
st.caption("A robustness review of existing results, not validation against independent ground-truth fire incidents.")
a,b=st.columns(2)
method=a.selectbox("Support definition",list(METHODS),format_func=METHODS.get,key="sensitivity_method")
k=b.selectbox("Minimum active seasons",[2,3,4,5],index=2,key="sensitivity_k")
thresholds=data.table("thresholds");baseline=thresholds.loc[(thresholds.subset=="all")&(thresholds.method==method)]
ui.section("01 · Scale and threshold","Coarser cells can accumulate seasonal support from different places within the same cell.")
a,b=st.columns(2)
with a:
    st.subheader("Resolution changes the support")
    ui.chart(baseline.loc[baseline.k==k],"resolution_m:O","recurrent_area_km2:Q",x_title="Grid resolution (m)",y_title="Clipped recurring support (km²)")
with b:
    st.subheader("Threshold sensitivity")
    view=baseline.copy();view["resolution_m"]=view.resolution_m.astype(str)+" m"
    ui.chart(view,"k:O","recurrent_area_km2:Q",color="resolution_m:N",kind="line",x_title="Minimum of seven active seasons",y_title="Clipped recurring support (km²)")
ui.caution("At k = 4, detection-supported area is 81.25 km² (500 m), 1,576.46 km² (1 km) and 10,684.76 km² (2 km). Coarser cells union observations from different subcell locations across years. These figures are thresholded grid support, not burned area.")
with st.expander("Read the independent scale audit"):
    qa=data.record("map_QA");rows=[r for r in qa["scale_union_audit"] if r["method"]=="detection"]
    display=pd.DataFrame(rows)[["fine_m","coarse_m","coarse_qualifying_cells","with_at_least_one_qualifying_fine_cell","qualify_by_union_of_different_subcell_years_only"]]
    display=display.rename(columns={"fine_m":"Fine grid (m)","coarse_m":"Coarse grid (m)","coarse_qualifying_cells":"Coarse cells",
        "with_at_least_one_qualifying_fine_cell":"Recurrent child","qualify_by_union_of_different_subcell_years_only":"Union only"})
    st.dataframe(display,hide_index=True,width="stretch")
    st.caption("Recurrent child: a coarse cell contains at least one qualifying finer cell. Union only: it qualifies by combining years observed in different finer cells.")
    st.markdown("The Phase 4A independent audit reproduced every reported area and showed that child-season unions exactly equal direct coarse assignments. This is the scale component of the **modifiable areal unit problem**. The grid origin was held fixed; alternative zoning was not tested.")
st.divider()
ui.section("02 · Which observations count?","The 1 km reference compared with nominal confidence, tree cover and one observation per cell per UTC day.")
rows=[]
for key,label in (("nominal_sensitivity","Nominal confidence"),("tree_sensitivity","WorldCover class 10"),("temporal_sensitivity","One observation / reference cell / UTC day")):
    frame=data.table(key);selected=frame.loc[(frame.resolution_m==1000)&(frame.method==method)&(frame.k==k)].copy()
    if len(selected)!=1:raise data.DataIntegrityError("A saved sensitivity comparison is unavailable or duplicated.")
    row=selected.iloc[0];rows.append({"definition":label,"baseline_cells":int(row.baseline_recurrent_cells),"subset_cells":int(row.subset_recurrent_cells),
        "baseline_support_km2":row.baseline_area_km2,"subset_support_km2":row.subset_area_km2,"cell_Jaccard":row.cell_jaccard,"saved_semantics":row.cluster_semantics})
details=pd.DataFrame(rows)
st.dataframe(details.drop(columns="saved_semantics").rename(columns={"definition":"Definition","baseline_cells":"Baseline cells","subset_cells":"Subset cells",
    "baseline_support_km2":"Baseline (km²)","subset_support_km2":"Subset (km²)","cell_Jaccard":"Cell Jaccard"}),hide_index=True,width="stretch",
    column_config={"Baseline (km²)":st.column_config.NumberColumn(format="%.2f"),"Subset (km²)":st.column_config.NumberColumn(format="%.2f"),"Cell Jaccard":st.column_config.NumberColumn(format="%.3f")})
with st.expander("Saved subset definitions"):
    st.dataframe(details[["definition","saved_semantics"]].rename(columns={"definition":"Definition","saved_semantics":"Recorded scientific semantics"}),hide_index=True,width="stretch")
st.caption("These are saved 1 km comparisons at the selected threshold. Tree cover is a static 2021 contextual subset. Cluster-supported subset semantics come from the recorded Phase 3A sensitivity fits; no new filtering is passed to a refit.")
st.divider()
ui.section("03 · Detection support and method agreement","All detections and the two saved cluster-supported populations use the same 1 km grid.")
support=thresholds.loc[(thresholds.resolution_m==1000)&(thresholds.subset=="all")&(thresholds.k==k)]
ui.chart(support,"method:N","recurrent_cells:Q",x_title="Saved support definition · 1 km",y_title="Qualifying cells")
overlap=data.table("overlap");overlap=overlap.loc[(overlap.resolution_m==1000)&(overlap.subset=="all")&(overlap.k==k)]
with st.expander("Pairwise overlap values"):
    st.dataframe(overlap[["left","right","intersection_cells","union_cells","cell_jaccard","area_jaccard"]],hide_index=True,width="stretch",
        column_config={"cell_jaccard":st.column_config.NumberColumn(format="%.3f"),"area_jaccard":st.column_config.NumberColumn(format="%.3f")})
st.caption("Method agreement is descriptive compatibility, not accuracy against independent fire incidents.")
st.divider()
ui.section("04 · The district perspective","Share of each district covered by qualifying support, using its own area as the denominator.")
districts=data.table("district_recurrence");districts=districts.loc[(districts.resolution_m==1000)&(districts.subset=="all")&(districts.method==method)&(districts.k==k)]
ui.chart(districts,"district_display:N","district_area_percent:Q",x_title=None,y_title="District area qualifying (%)",horizontal=True,height=380)
st.caption("Areas use exact saved cell–district intersections. Qualifying cell counts can be shared across districts; area percentages use district-specific denominators.")
st.divider()
ui.section("05 · What remains uncertain")
st.markdown("A dense spatial group can chain across large distances, multiple observation dates and district boundaries. The 10 km group-diameter flag is a diagnostic, not a fire-event boundary. Parameter choices, repeat overpasses and contextual filtering change the outputs.")
coverage=data.table("primary_coverage")
with st.expander("Primary-season retrieval ledger"):
    st.dataframe(coverage,hide_index=True,width="stretch")
st.warning("All primary dates have retrieval accounting, but independently verified satellite observation opportunity is zero in the saved ledger. Clouds, overpasses, detection thresholds and upstream processing concerns limit interannual interpretation. A zero return is not a fire-free day.")
with st.expander("Verified QGIS resolution comparison"):
    ui.optional_image("sensitivity_figure","Phase 4A native QGIS export · the original verified map is reused unchanged.")
ui.footer()
