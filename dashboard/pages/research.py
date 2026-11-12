import streamlit as st
from dashboard import data,ui,ROOT
from dashboard.content import QUESTION

ui.heading("Research findings & methods","The research in one place: the question, the evidence, the choices behind the maps and the limits on what they can tell us.","06 / The research record")
ui.section("The question")
st.markdown(QUESTION)
st.markdown('<div class="workflow"><span><b>01 / OBSERVE</b>Authentic FIRMS records</span><span><b>02 / PREPARE</b>One validated footprint</span><span><b>03 / GROUP</b>Seasonal DBSCAN &amp; HDBSCAN</span><span><b>04 / COMPARE</b>Grid votes &amp; sensitivity</span><span><b>05 / EXPLORE</b>Saved maps &amp; evidence</span></div>',unsafe_allow_html=True)
ui.section("How the study was built","Open a method note for source definitions and reference settings.")
with st.expander("NASA source and historical construction"):
    summary=data.record("master_summary")
    st.markdown(f'**VIIRS_SNPP_SP, Standard Processing, 2019–2025.** {summary["totals"]["raw"]:,} original bounding-box observations reconcile with {summary["totals"]["retained"]:,} retained observations and {summary["totals"]["geographic_exclusions"]:,} geographic exclusions. Original CSVs, collection strings and source identities are preserved.')
    st.markdown("All 2,557 UTC calendar dates have acquisition accounting. Acquisition success is separate from independently verified satellite opportunity. Original source-row IDs link observations to authenticated acquisition provenance without exposing credentials. The app operates offline and does not read the NASA key.")
    st.markdown("The analytical footprint is the union of thirteen approved geoBoundaries 2021 districts. The different 2011 state outline is not an additional clipping step. Observations retain UTC acquisition and separate IST reporting dates, original district labels and display aliases.")
with st.expander("Clustering methods and saved reference choices"):
    st.markdown("**DBSCAN:** ε = 1 km, min_samples = 3, including the point itself. **HDBSCAN:** minimum cluster size = 5, min_samples = 5, EOM selection. Features are unweighted EPSG:32644 XY coordinates in metres; time, FRP, confidence and land cover are metadata, not these fits’ feature dimensions.")
    st.markdown("All 201 historical scientific fits are complete and immutable. The dashboard reads saved reference assignments, group extents, parameter grids and method comparisons. Labels have local meaning within a fit. Spatial groups can span many dates and are not independent confirmed fires. No accuracy, precision, recall or F1 score is claimed.")
with st.expander("Fixed-grid recurrence methodology"):
    st.markdown("Grids are fixed at origin (0,0) m in EPSG:32644, with 500 m, 1 km and 2 km support. Half-open floor indexing assigns points consistently. Eligible support has strictly positive area inside the approved district union. Actual clipped areas are used.")
    st.markdown("Each February 15–June 15 UTC primary season contributes one binary activity vote per supported cell. The reference is at least four of seven seasons; k = 2–5 provides sensitivity. Support can be all detections, saved DBSCAN membership or saved HDBSCAN membership. Shared-edge components describe connectivity of these cells, not event perimeters.")
st.divider()
ui.section("What the study found","The shared 1 km reference: support in at least four of seven primary seasons.")
reference=[{"support":name,"cells":data.recurrence_summary(1000,method,4)["cells"],"clipped_support_km2":data.recurrence_summary(1000,method,4)["area_km2"]} for method,name in (("detection","All detections"),("dbscan","DBSCAN"),("hdbscan","HDBSCAN"))]
st.dataframe(reference,hide_index=True,width="stretch")
st.markdown("**390 cells** share both cluster-supported definitions at 1 km / four-of-seven. Phase 4A independently reconstructed the reference totals, clipping and resolution effects. Agreement is not independent validation.")
a,b=st.columns(2)
with a:
    st.markdown("**Most defensible interpretations**\n\n- Observed detections and their source lineage are reproducible.\n- Repeated spatial support is present under declared choices.\n- Maps permit transparent comparison of saved support definitions.\n- Grid size and threshold materially influence reported support.")
with b:
    st.markdown("**Interpretations the study does not establish**\n\n- Independent incident or burned-area counts.\n- A validated future-risk or susceptibility model.\n- Complete observation of every study day.\n- Detection-count changes as proof of changes in true wildfire incidence.")
st.divider()
ui.section("What to keep in mind","These limits apply even when a map looks spatially precise.")
st.markdown("1. **Unknown satellite opportunity:** cloud, overpass timing, detection limits and contextual outages remain distinct from successful API retrieval.\n2. **Thermal detection versus incident:** repeated overpasses and nonforest heat sources can contribute; observations are not independent confirmed forest-fire events.\n3. **WorldCover 2021:** point-sampled static land-cover context is not a time-varying vegetation history or universal forest-fire label.\n4. **Collection/type lineage:** original source collection values are preserved; unresolved hotspot-type revision lineage is not used as ground-truth classification.\n5. **Parameters and chaining:** group membership and broad spatial connections depend on density settings and input population.\n6. **Modifiable areal unit effects:** support resolution, grid alignment and thresholds affect spatial recurrence. The saved origin is fixed; arbitrary alternative zoning was not tested.\n7. **Between-year comparison:** different observation opportunity, processing and detection conditions prevent interpreting count changes as established changes in actual fire incidence.")
ui.section("Could Indian data offer a second comparison?")
st.markdown("The reviewed FSI archival form advertises 1 December 2004–30 June 2025, including all seven primary windows. Sensor-specific Uttarakhand completeness, a usable historical delivery and redistribution rights remain unverified. Bhuvan includes multiple alert and older aggregate-regime products; their dates, resolutions and filters must not be conflated.")
st.markdown("FSI/NRSC may process the same satellite measurements as NASA, with different near-real-time and forest/industrial/confidence filters. A future authorized comparison could assess concordance on common support; it would not create independent ground truth. No external validation score or Indian point-data download is claimed.")
st.divider()
ui.section("Follow the evidence","Reports, source notes and the dedicated QGIS project stay with the local research files.")
for key,label in (("phase3a_report","Seasonal clustering report"),("phase3b_report","Fixed-grid recurrence report"),("phase4a_report","QGIS scientific map review"),("indian_feasibility","Indian-source feasibility")):
    path=data.source_file(key)
    with st.expander(label):
        st.code(str(path));st.download_button("Download report",path.read_bytes(),file_name=path.name,mime="text/markdown",key="download_"+key)
qgis=data.source_file("qgis_project")
st.markdown("**Dedicated QGIS project** · open manually in QGIS 3.44.12; the dashboard does not launch a desktop process.")
st.code(str(qgis));st.caption("Keep project-relative data folders together. Thirty verified layers and six native layouts are available.")
with st.expander("Verified static recurrence map"):
    ui.optional_image("recurrence_figure","Observed satellite thermal-detection recurrence, 2019–2025 · Phase 4A export.")
with st.expander("Reproducibility and source integrity"):
    st.markdown("The original Conda scientific lock is unchanged. Dashboard additions have their own hashed wheel lock. Sources are read-only and checked against a catalogue bound to prior scientific inventories. Small summary caches and column-selective, year-filtered reads bound memory; displayed map samples never replace analytical populations.")
    st.code('.\\scripts\\launch_dashboard.ps1')
    st.code('.\\requirements.lock.txt')
    st.code('outputs/tables/phase4b/file_integrity_manifest.json (generated locally)')
    st.caption("The launch wrapper retains the established available-RAM, disk and single-worker safeguards. No NASA credential, acquisition, scientific fit or recurrence build is needed.")
st.subheader("Attribution")
st.markdown("NASA FIRMS / S-NPP VIIRS Standard Processing; geoBoundaries ADM2 2021, Pathways Data / lgdirectory.gov.in (ODbL 1.0); ESA WorldCover 2021 v200 (CC BY 4.0). Retain the original data notices and review applicable redistribution terms before publication.")
ui.caution();ui.footer()
