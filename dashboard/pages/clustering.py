import pandas as pd
import streamlit as st
from dashboard import data,maps,ui

ui.heading("DBSCAN vs HDBSCAN","Two ways to group the same seasonal observations. DBSCAN uses a fixed neighbourhood; HDBSCAN looks for groups across density levels. These are saved results, with no fitting in the app.","03 / Comparing spatial groups")
seasons=sorted({row["season"] for row in data.catalog()["references"]})
a,b=st.columns([1,1])
season=a.selectbox("Saved season",seasons,index=seasons.index("CORE_2024"),key="cluster_season")
view=b.radio("Algorithm view",["Side by side","DBSCAN","HDBSCAN"],horizontal=True,key="cluster_view")
if season.startswith("CORE_"):st.caption("Primary season: February 15–June 15 UTC. One shared input population, two saved reference methods.")
else:st.caption("Extended season: November 1–June 30 UTC across two years. Extended and primary populations overlap and must not be added.")
db=data.model_stats(season,"dbscan");hb=data.model_stats(season,"hdbscan")
left,right=st.columns(2)
for column,algorithm,stats in ((left,"dbscan",db),(right,"hdbscan",hb)):
    with column:
        st.subheader(algorithm.upper())
        st.caption("Fixed-neighbourhood density: ε = 1 km, min_samples = 3 (includes self)." if algorithm=="dbscan" else "Hierarchical density: min_cluster_size = 5, min_samples = 5, EOM selection.")
        ui.metrics([("Saved spatial groups",f'{stats["n_clusters"]:,}',"Cluster labels identify analytical point groups, not independent fire incidents."),
            ("Clustered points",f'{100*stats["n_clustered"]/stats["n_observations"]:.1f}%',f'{stats["n_clustered"]:,} of {stats["n_observations"]:,} input observations.')])
        st.caption(f'Noise: {stats["n_noise_minus1"]:,} ({100*stats["n_noise_minus1"]/stats["n_observations"]:.1f}%). Other excluded inputs: {stats["n_nonmember"]-stats["n_noise_minus1"]:,}.')
ui.caution("Numerical cluster labels do not correspond between seasons or algorithms. Clusters are spatial groups of thermal detections, not independent fire incidents; membership strength is not fire probability.")
algorithms=["dbscan","hdbscan"] if view=="Side by side" else [view.lower()]
columns=st.columns(len(algorithms))
for column,algorithm in zip(columns,algorithms):
    with column:
        ui.section(algorithm.upper()+" assignments")
        clusters=data.cluster_table(season,algorithm)
        ordered=clusters.sort_values(["n_observations","cluster_label"],ascending=[False,True])
        options=["All assignments","Noise (-1)"]+[str(value) for value in ordered.cluster_label]
        chosen=st.selectbox("Display group",options,key="cluster_group_"+algorithm,format_func=lambda s:s if s in ("All assignments","Noise (-1)") else "Saved label "+s)
        points=data.assignments(season,algorithm)
        if chosen!="All assignments":points=points.loc[points.cluster_label==(-1 if chosen=="Noise (-1)" else int(chosen))]
        if points.empty:st.info("No saved observations in this group.")
        else:
            webmap,shown=maps.point_map(points,algorithm=algorithm)
            ui.render_map(webmap,"cluster_map_"+algorithm,height=520);ui.sample_caption(shown,len(points))
        st.caption("Blue DBSCAN / orange HDBSCAN members; grey noise. The metrics above describe the full saved fit even when the map selects one group.")
        stats=db if algorithm=="dbscan" else hb
        if stats["broad_clusters_over_10km"]:
            st.warning(f'{stats["broad_clusters_over_10km"]:,} saved groups exceed 10 km diameter. Maximum: {stats["diameter_max_m"]/1000:.2f} km. Broad groups can reflect spatial chaining and multiple dates, not a single fire footprint.')
        else:st.caption(f'Maximum saved diameter: {stats["diameter_max_m"]/1000:.2f} km; no group exceeds the declared 10 km diagnostic.')
        with st.expander("Cluster sizes, extent and chaining diagnostics"):
            st.markdown("**How large are the saved groups?**")
            sizes=clusters.groupby("n_observations").size().rename("groups").reset_index()
            ui.chart(sizes,"n_observations:Q","groups:Q",x_title="Observations per spatial group",y_title="Saved groups",height=230)
            fields=["cluster_label","n_observations","diameter_m","radius_p95_m","district_count","first_utc","last_utc","distinct_UTC_dates"]
            st.dataframe(ordered[fields].head(30),hide_index=True,width="stretch")
            st.caption(f'{stats["cross_district_clusters"]:,} groups cross district boundaries. Top 30 by observation count shown; extents come from saved geometry diagnostics.')
st.divider()
ui.section("Where the methods agree","Comparison uses the same source observation identities, not the numeric group labels.")
comparison=data.model_comparison(season)
ui.metrics([("Membership agreement",f'{comparison["membership_agreement"]*100:.1f}%',"Agreement on clustered versus nonmember status, not accuracy."),
    ("Clustered-population Jaccard",f'{comparison["clustered_population_jaccard"]:.3f}',"Overlap of clustered observation identities."),
    ("Co-clustered-pair Jaccard",f'{comparison["coclustered_pair_jaccard"]:.3f}',"Pair agreement from saved comparisons; not a supervised validation score.")])
st.caption("Agreement metrics describe compatibility between methods. No independent ground-truth fire labels exist here.")
ui.section("What changes with the parameters?","Each curve comes from completed Phase 3A fits. Changing this view does not start a model.")
statistics=data.table("fit_statistics");grid=statistics.loc[(statistics.season_id==season)&(statistics.role=="primary_grid")&(statistics.subset=="all")].copy()
if grid.empty:st.info("A parameter grid was not executed for this extended window. Only its saved reference fits are available.")
else:
    a,b=st.columns(2)
    for column,algorithm,x,title in ((a,"dbscan","eps_km:Q","DBSCAN ε (km)"),(b,"hdbscan","min_cluster_size:Q","HDBSCAN minimum group size")):
        with column:
            subset=grid.loc[grid.algorithm==algorithm].copy();subset["min_samples"]=subset.min_samples.astype(int).astype(str)
            ui.chart(subset,x,"n_clusters:Q",kind="line",color="min_samples:N",x_title=title,y_title="Saved spatial groups")
    st.caption("All curves are existing completed fits. Reference settings were declared in Phase 3A, not selected by a claimed ground-truth accuracy score.")
ui.footer()
