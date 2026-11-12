"""Bounded, column-selective access to immutable, hash-bound scientific sources.

No network, credentials, raster access, model fitting or recurrence construction.
Cache keys include current file metadata and verified source hashes. Changed sources
are hashed before acceptance; a mismatch is an explicit scientific integrity error.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib,json,tomllib
from pathlib import Path
import streamlit as st
from dashboard import ROOT

class DataIntegrityError(RuntimeError):pass

@lru_cache(maxsize=1)
def config():
    return tomllib.loads((ROOT/"configs/dashboard.toml").read_text(encoding="utf-8"))

@lru_cache(maxsize=1)
def catalog():
    result=json.loads((ROOT/config()["application"]["catalog"]).read_text(encoding="utf-8"))
    if result["status"]!="verified_source_bindings":raise DataIntegrityError("The source catalogue is not verified.")
    digest=hashlib.sha256((ROOT/"configs/dashboard.toml").read_bytes()).hexdigest()
    if digest!=result["configuration_sha256"]:raise DataIntegrityError("The dashboard source configuration changed after verification.")
    return result

@lru_cache(maxsize=256)
def _hash_matches(path,expected,size,mtime_ns):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    if digest.hexdigest()!=expected:raise DataIntegrityError("A verified scientific source has a checksum mismatch: "+Path(path).name)
    return True

def source_file(key):
    try:row=catalog()["sources"][key]
    except KeyError as error:raise DataIntegrityError("Requested source is absent from the verified catalogue.") from error
    path=(ROOT/row["path"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():raise DataIntegrityError("A verified source is missing: "+row["path"])
    stat=path.stat()
    if stat.st_size!=row["size_bytes"]:raise DataIntegrityError("A verified source has an unexpected size: "+row["path"])
    _hash_matches(str(path),row["sha256"],stat.st_size,stat.st_mtime_ns)
    return path

def token(key):
    path=source_file(key);stat=path.stat()
    return str(path),stat.st_size,stat.st_mtime_ns,catalog()["sources"][key]["sha256"]

@st.cache_data(show_spinner=False,max_entries=24)
def _csv(identity):
    import pandas as pd
    try:return pd.read_csv(identity[0])
    except Exception as error:raise DataIntegrityError("A scientific summary could not be decoded: "+Path(identity[0]).name) from error

def table(key):return _csv(token(key))

@st.cache_data(show_spinner=False,max_entries=16)
def _json(identity):
    try:return json.loads(Path(identity[0]).read_text(encoding="utf-8"))
    except Exception as error:raise DataIntegrityError("A scientific record could not be decoded.") from error

def record(key):return _json(token(key))

def require(frame,columns,context):
    if not set(columns)<=set(frame.columns):raise DataIntegrityError(context+" has an incompatible schema.")
    return frame

def single(frame,context="Scientific summary",**filters):
    for column,value in filters.items():frame=frame.loc[frame[column]==value]
    if len(frame)!=1:raise DataIntegrityError(context+" does not contain exactly one matching saved result.")
    return frame.iloc[0]

def annual():
    import pandas as pd
    result=pd.DataFrame(record("master_summary")["annual"])
    if result["retained"].sum()!=95052 or result["raw"].sum()!=158959:raise DataIntegrityError("Verified annual totals do not reconcile.")
    return result

@dataclass(frozen=True)
class ObservationFilter:
    year:int=2024
    month:int|None=None
    primary:bool=False
    district:str|None=None
    confidence:tuple[str,...]=("low","nominal","high")
    daynight:tuple[str,...]=("D","N")
    landcover:tuple[int,...]|None=None

OBS_COLUMNS=("agniscope_observation_id","agniscope_latitude","agniscope_longitude","agniscope_x_m","agniscope_y_m",
    "agniscope_acquired_utc","agniscope_acquired_ist","agniscope_utc_date","agniscope_ist_date","agniscope_calendar_year",
    "agniscope_district_display","agniscope_district_assignment","agniscope_confidence","agniscope_worldcover_2021_class",
    "agniscope_landcover_status","daynight","frp","version")

@st.cache_data(show_spinner=False,max_entries=2)
def _observations(identity,selection):
    import pyarrow.dataset as ds
    import pandas as pd
    if selection.year not in config()["application"]["years"] or selection.month not in (None,*range(1,13)):
        raise ValueError("Choose a study year and a valid UTC month.")
    if not selection.confidence or not selection.daynight or selection.landcover==():
        return ds.dataset(identity[0],format="parquet").head(0,columns=list(OBS_COLUMNS)).to_pandas()
    expression=ds.field("agniscope_calendar_year")==selection.year
    expression=expression & ds.field("agniscope_confidence").isin(selection.confidence) & ds.field("daynight").isin(selection.daynight)
    if selection.district=="Unassigned / ambiguous":
        expression=expression & ds.field("agniscope_district_display").is_null()
    elif selection.district:expression=expression & (ds.field("agniscope_district_display")==selection.district)
    if selection.landcover is not None:
        classes=[value for value in selection.landcover if value!=-1]
        condition=ds.field("agniscope_worldcover_2021_class").isin(classes)
        if -1 in selection.landcover:condition=condition | ds.field("agniscope_worldcover_2021_class").is_null() | (ds.field("agniscope_worldcover_2021_class")==0)
        expression=expression & condition
    try:frame=ds.dataset(identity[0],format="parquet").to_table(columns=list(OBS_COLUMNS),filter=expression).to_pandas()
    except Exception as error:raise DataIntegrityError("The observation Parquet could not be read. Verify the local Arrow installation and source integrity.") from error
    if not frame.empty:
        utc=pd.to_datetime(frame["agniscope_acquired_utc"],utc=True)
        if selection.month is not None:frame=frame.loc[utc.dt.month==selection.month]
        if selection.primary:frame=frame.loc[(utc.dt.strftime("%m-%d")>="02-15") & (utc.dt.strftime("%m-%d")<="06-15")]
    return frame.sort_values("agniscope_observation_id",kind="stable").reset_index(drop=True)

def observations(selection):return _observations(token("master"),selection)

def distribution(frame,column):
    return frame[column].fillna("Unassigned / ambiguous").value_counts(dropna=False).rename_axis(column).reset_index(name="observations")

def frp_summary(frame):
    import pandas as pd
    values=pd.to_numeric(frame["frp"],errors="coerce").dropna()
    return {"available":len(values),"missing":len(frame)-len(values),"median_MW":float(values.median()) if len(values) else None,
        "mean_MW":float(values.mean()) if len(values) else None,"minimum_MW":float(values.min()) if len(values) else None,
        "maximum_MW":float(values.max()) if len(values) else None,"q25_MW":float(values.quantile(.25)) if len(values) else None,
        "q75_MW":float(values.quantile(.75)) if len(values) else None}

def district_names():
    return sorted(table("district_recurrence")["district_display"].dropna().unique().tolist())

def reference(season,algorithm):
    matches=[r for r in catalog()["references"] if r["season"]==season and r["algorithm"]==algorithm]
    if len(matches)!=1:raise DataIntegrityError("No verified saved reference exists for this season/method.")
    row=matches[0];expected={"algorithm":"dbscan","eps_km":1.,"min_samples":3} if algorithm=="dbscan" else {"algorithm":"hdbscan","min_cluster_size":5,"min_samples":5}
    if row["parameters"]!=expected:raise DataIntegrityError("Saved model parameters differ from the scientific reference contract.")
    return row

def model_stats(season,algorithm):
    ref=reference(season,algorithm);saved=record("statistics_"+ref["key"])
    if saved["spec"]["input_sha256"]!=ref["input_sha256"] or saved["spec"]["parameters"]!=ref["parameters"]:
        raise DataIntegrityError("Saved model provenance differs from its verified source.")
    metrics=saved["metrics"]
    if metrics["n_observations"]!=ref["n"] or metrics["n_clustered"]+metrics["n_nonmember"]!=ref["n"]:
        raise DataIntegrityError("Saved model row accounting failed.")
    if algorithm=="hdbscan":
        settings=tomllib.loads(source_file("model_configuration").read_text(encoding="utf-8"))
        if settings["hdbscan"]["cluster_selection_method"]!="eom":raise DataIntegrityError("HDBSCAN selection is not the declared EOM reference.")
    return metrics

@st.cache_data(show_spinner=False,max_entries=2)
def _parquet(identity,columns):
    import pyarrow.parquet as pq
    try:return pq.read_table(identity[0],columns=list(columns)).to_pandas()
    except Exception as error:raise DataIntegrityError("A saved model table could not be read.") from error

def cluster_table(season,algorithm):
    key="clusters_"+reference(season,algorithm)["key"]
    cols=("cluster_label","scoped_cluster_id","n_observations","diameter_m","radius_p95_m","broad_over_10km","district_count","first_utc","last_utc","distinct_UTC_dates")
    return _parquet(token(key),cols)

def assignments(season,algorithm):
    key="assignments_"+reference(season,algorithm)["key"]
    cols=("agniscope_observation_id","agniscope_x_m","agniscope_y_m","agniscope_acquired_utc","agniscope_ist_date",
        "agniscope_district_display","agniscope_confidence","frp","cluster_label","membership_strength")
    return _parquet(token(key),cols)

def model_comparison(season):
    key="reference_comparison" if season.startswith("CORE_") else "extended_comparison"
    return single(table(key),"Saved method comparison",season_id=season)

def recurrence_summary(resolution=1000,method="detection",k=4,district=None):
    validate_recurrence(resolution,method,k)
    if district:
        row=single(table("district_recurrence"),"Saved district recurrence",resolution_m=resolution,subset="all",method=method,k=k,district_display=district)
        return {"cells":int(row["qualifying_cells"]),"area_km2":float(row["qualifying_area_km2"]),"share_percent":float(row["district_area_percent"])}
    row=single(table("thresholds"),"Saved recurrence",resolution_m=resolution,subset="all",method=method,k=k)
    return {"cells":int(row["recurrent_cells"]),"area_km2":float(row["recurrent_area_km2"]),"share_percent":float(row["eligible_area_percent"])}

def validate_recurrence(resolution,method,k):
    if resolution not in (500,1000,2000) or method not in ("detection","dbscan","hdbscan") or k not in (2,3,4,5):
        raise ValueError("Unsupported resolution, support definition or recurrence threshold.")

@st.cache_data(show_spinner=False,max_entries=1)
def _cells(identity,intersection_identity,resolution,method,k,district):
    import pyarrow.dataset as ds
    field="all_"+method+"_active_seasons";mask="all_"+method+"_season_mask"
    cols=["cell_id","geometry","clipped_area_m2",field,mask]
    expression=ds.field(field)>=k
    try:frame=ds.dataset(identity[0],format="parquet").to_table(columns=cols,filter=expression).to_pandas()
    except Exception as error:raise DataIntegrityError("The selected recurrence GeoParquet is unavailable or incompatible.") from error
    if district:
        shares=ds.dataset(intersection_identity[0],format="parquet").to_table(
            columns=["cell_id","intersection_area_m2"],filter=ds.field("district_display")==district).to_pandas()
        frame=frame.merge(shares,on="cell_id",how="inner",validate="one_to_one")
        frame["display_scope_area_m2"]=frame["intersection_area_m2"]
    else:frame["display_scope_area_m2"]=frame["clipped_area_m2"]
    return frame.rename(columns={field:"active_seasons",mask:"season_mask"}).sort_values("cell_id",kind="stable").reset_index(drop=True)

def recurrence_cells(resolution=1000,method="detection",k=4,district=None):
    validate_recurrence(resolution,method,k)
    frame=_cells(token(f"cells_{resolution}"),token(f"district_cells_{resolution}") if district else None,resolution,method,k,district)
    summary=recurrence_summary(resolution,method,k,district)
    if len(frame)!=summary["cells"] or abs(frame["display_scope_area_m2"].sum()/1e6-summary["area_km2"])>1e-6:
        raise DataIntegrityError("Map rows or exact clipped areas disagree with the saved scientific summary.")
    return frame

def component_summary(resolution,method,k):
    return single(table("component_inventory"),"Saved component inventory",resolution_m=resolution,method=method,k=k,adjacency=4)

def years_from_mask(mask):return [year for bit,year in enumerate(range(2019,2026)) if int(mask)&(1<<bit)]

def deterministic_sample(frame,limit,identity):
    """Evenly spaced positions in a stable source-ID order; no random state or analytical filtering."""
    if limit<1:raise ValueError("Display limit must be positive.")
    ordered=frame.sort_values(identity,kind="stable")
    if len(ordered)<=limit:return ordered.copy()
    positions=[i*(len(ordered)-1)//(limit-1) for i in range(limit)] if limit>1 else [0]
    return ordered.iloc[positions].copy()
