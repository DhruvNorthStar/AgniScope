import pandas as pd
import streamlit as st
from dashboard import data,maps,ui
from dashboard.content import LANDCOVER

ui.heading("Historical Fire Explorer","Choose a year and place, then explore the NASA observations behind the study. Statistics describe every matching record; the map may show a bounded sample.","02 / Observations")
with st.container(border=True):
    a,b,c=st.columns(3)
    year=a.selectbox("UTC year",data.config()["application"]["years"],index=5,key="explorer_year")
    period=b.selectbox("UTC period",["Full calendar year","Primary season · Feb 15–Jun 15"]+[f"Month {n:02d}" for n in range(1,13)],key="explorer_period")
    district=c.selectbox("District",["All districts"]+data.district_names()+["Unassigned / ambiguous"],key="explorer_district")
    with st.expander("Refine confidence, day / night and land cover"):
        st.caption("All categories are included initially. These controls narrow the displayed population; they do not change the saved dataset.")
        a,b=st.columns(2)
        confidence=a.multiselect("Confidence",["low","nominal","high"],default=["low","nominal","high"],key="explorer_confidence")
        daynight=b.multiselect("Day / night",["D","N"],default=["D","N"],format_func=lambda n:{"D":"Day","N":"Night"}[n],key="explorer_daynight")
        landcover=st.multiselect("WorldCover 2021 class",list(LANDCOVER),default=list(LANDCOVER),format_func=lambda n:f"{n} · {LANDCOVER[n]}",key="explorer_landcover")
selection=data.ObservationFilter(year=year,month=int(period[-2:]) if period.startswith("Month ") else None,
    primary=period.startswith("Primary"),district=None if district=="All districts" else district,
    confidence=tuple(confidence),daynight=tuple(daynight),landcover=None if set(landcover)==set(LANDCOVER) else tuple(landcover))
frame=data.observations(selection);frp=data.frp_summary(frame)
ui.metrics([("Selected observations",f"{len(frame):,}","Full selected population, not the displayed sample."),
    ("UTC detection dates",f'{frame["agniscope_utc_date"].nunique():,}',"Distinct dates with at least one retained detection; not verified observation opportunity."),
    ("Median FRP",f'{frp["median_MW"]:.2f} MW' if frp["median_MW"] is not None else "Unavailable","Fire radiative power is not burned area or fire-event severity."),
    ("FRP available",f'{frp["available"]:,}',f'{frp["missing"]:,} selected observations lack a numeric FRP value.')])
if frame.empty:
    st.info("No observations match these filters. Empty selections represent no matching records, not confirmed fire-free conditions.")
    st.caption("Select at least one confidence, day/night and land-cover category, or broaden the district/period.")
else:
    webmap,shown=maps.point_map(frame,district=selection.district if selection.district in data.district_names() else None)
    ui.section("Observations on the map",f"{year} · {period} · {district}")
    ui.render_map(webmap,"historical_map",height=590);ui.sample_caption(shown,len(frame))
    st.caption("Colours: teal nominal confidence · orange low · purple high. All categories are retained by default. Hover includes UTC acquisition and a separate IST reporting date.")
    a,b=st.columns(2)
    with a:
        ui.section("When detections were recorded")
        mode=st.radio("Temporal chart",["Monthly","Daily"],horizontal=True,key="explorer_temporal")
        if mode=="Monthly":
            counts=pd.to_datetime(frame["agniscope_acquired_utc"],utc=True).dt.month.value_counts().reindex(range(1,13),fill_value=0).rename_axis("month").reset_index(name="observations")
            ui.chart(counts,"month:O","observations:Q",x_title="UTC month")
        else:
            counts=frame.groupby("agniscope_utc_date").size().rename("observations").reset_index().rename(columns={"agniscope_utc_date":"date"})
            ui.chart(counts,"date:T","observations:Q",kind="line",x_title="UTC acquisition date")
    with b:
        st.subheader("District distribution")
        district_counts=data.distribution(frame,"agniscope_district_display")
        if "Unassigned / ambiguous" not in district_counts["agniscope_district_display"].tolist():
            district_counts=pd.concat([district_counts,pd.DataFrame({"agniscope_district_display":["Unassigned / ambiguous"],"observations":[0]})],ignore_index=True)
        ui.chart(district_counts,"agniscope_district_display:N","observations:Q",x_title=None,horizontal=True,height=365)
    with st.expander("Confidence and day / night breakdown"):
        a,b=st.columns(2)
        with a:
            st.subheader("Confidence categories")
            ui.chart(data.distribution(frame,"agniscope_confidence"),"agniscope_confidence:N","observations:Q",x_title="NASA confidence")
        with b:
            st.subheader("Day / night")
            ui.chart(data.distribution(frame,"daynight"),"daynight:N","observations:Q",x_title="D = day · N = night")
    with st.expander("FRP descriptive values and source rows"):
        st.dataframe(pd.DataFrame([frp]),hide_index=True,width="stretch")
        st.caption("Numeric FRP summaries use available values from the full selection. A maximum is a detection-level value, not an independent incident estimate.")
        cols=["agniscope_observation_id","agniscope_acquired_utc","agniscope_ist_date","agniscope_district_display","agniscope_confidence","daynight","frp","agniscope_worldcover_2021_class","version"]
        st.dataframe(frame[cols].head(200),hide_index=True,width="stretch")
        st.caption("Table preview is capped at the first 200 source-ID-ordered rows; its cap does not affect statistics.")
ui.caution();ui.footer()
