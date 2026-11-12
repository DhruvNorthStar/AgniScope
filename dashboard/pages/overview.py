import streamlit as st
from dashboard import data,maps,ui
from dashboard.content import QUESTION,APP_NAME,FORMAL_TITLE

ui.heading(APP_NAME,FORMAL_TITLE.split(': ',1)[1],"01 / Overview")
st.markdown("Explore seven years of satellite observations through seasonal clustering and recurring spatial activity.")
annual=data.annual();reference=data.recurrence_summary()
ui.metrics([("Raw satellite observations",f'{annual.raw.sum():,}',"Original bounding-box observations; geographic exclusions are preserved."),
    ("Validated observations",f'{annual.retained.sum():,}',"Retained in the approved thirteen-district analytical footprint."),
    ("Primary seasons","7","February 15–June 15 UTC in each year, 2019–2025."),
    ("Recurring 1 km cells",f'{reference["cells"]:,}',"Detection-supported cells active in at least four of seven primary seasons.")])
ui.section("Uttarakhand · the study area","Thirteen labelled districts within the approved analytical footprint. Tints distinguish districts; they do not represent thermal activity or elevation.")
ui.render_map(maps.intro_map(),"overview_map",height=520)
st.caption("The state silhouette follows the union of the approved 2021 district polygons. Display boundaries alone are simplified by at most 60 m; source geometries and analytical areas are unchanged.")
ui.caution()
left,right=st.columns(2,gap="large")
with left:
    ui.section("The annual record","Original bounding-box detections and the retained district population.")
    series=annual[["utc_year","raw","retained"]].rename(columns={"utc_year":"year","raw":"Raw bounding box","retained":"Validated footprint"}).melt(id_vars="year",var_name="population",value_name="observations")
    ui.chart(series,"year:O","observations:Q",color="population:N",x_title="UTC year",height=300)
    st.caption("Different geographic denominators. Changes in detection counts do not establish changes in true wildfire incidence.")
with right:
    ui.section("The rhythm of the year","Retained observations by UTC month, summed across all seven years.")
    monthly=data.table("monthly");columns=[f"retained_{year}" for year in data.config()["application"]["years"]]
    monthly=monthly.assign(observations=monthly[columns].sum(axis=1))[["month","observations"]]
    ui.chart(monthly,"month:O","observations:Q",x_title="UTC month, all seven years",height=300)
st.markdown("**The question behind the maps** · "+QUESTION)
st.markdown("The study follows the same thirteen-district footprint from 2019 to 2025. It starts with NASA observations, compares two ways of grouping them, and asks which places show support in several primary seasons. Every view uses saved results.")
st.caption("The recurrence reference uses 1 km cells with support in at least four of seven February 15–June 15 seasons. This is a declared analytical threshold.")
with st.expander("How to explore this study"):
    st.markdown("**Historical Fire Explorer** follows individual observations. **DBSCAN vs HDBSCAN** compares 201 saved scientific fits, without retraining. **Recurring Hotspot Map** shows the seasonal grid support, while **Scientific Validation & Sensitivity** tests how its interpretation depends on choices.")
    st.caption("All confidence categories and land-cover contexts are retained. WorldCover 2021 is static context, not a universal forest-fire label. NASA VIIRS_SNPP_SP Standard Processing uses UTC acquisition timestamps and separate IST reporting dates.")
ui.footer()
