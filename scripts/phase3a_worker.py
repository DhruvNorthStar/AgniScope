"""One resource-monitored offline fit; no dense distance or silhouette helpers."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[variable]='1'
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
import pyarrow as pa,pyarrow.parquet as pq
from agniscope.config import ROOT
from agniscope.provenance import sha256
from agniscope.acquisition_store import atomic_json
from agniscope import seasonal_clustering as s

def main():
    directory=(ROOT/sys.argv[1]).resolve();assert directory.is_relative_to((ROOT/'outputs/tables/phase3a/runs').resolve())
    spec=json.loads((directory/'spec.json').read_text());source=ROOT/spec['input_path'];assert sha256(source)==spec['input_sha256']
    frame=s.canonical_points(pd.read_parquet(source));assert len(frame)==spec['n_observations'] and s.feature_identity(frame)==spec['input_feature_identity']
    start=time.perf_counter();labels,strength,core,details=s.fit_points(frame,spec['parameters']);fit_seconds=time.perf_counter()-start
    scope=spec['season_id']+'__'+spec['subset']+'__'+spec['parameters']['algorithm']+'__'+spec['run_id'][4:20]
    clusters=s.cluster_statistics(frame,labels,strength,scope);stats=s.accounting(frame,labels,strength,clusters,core)
    if spec['parameters']['algorithm']=='hdbscan':stats['n_core']=None
    np.savez_compressed(directory/'labels_compact.npz',labels=labels.astype(np.int32),strength=strength,core=core)
    atomic_json(directory/'statistics.json',{'spec':spec,'metrics':stats,'estimator':details,'features':'EPSG:32644 XY metres only, unweighted','input_feature_identity':spec['input_feature_identity'],'scientific_fit_timing_excluded_from_deterministic_artifacts':True})
    # Runtime is kept separate from deterministic scientific artifacts.
    atomic_json(directory/'timing.json',{'fit_seconds':fit_seconds})
    pd.DataFrame(clusters,columns=list(clusters[0]) if clusters else ['cluster_label','scoped_cluster_id','n_observations','diameter_m']).to_parquet(directory/'clusters.parquet',index=False,compression='zstd')
    if spec['reference']:
        assignments=frame.copy();assignments['cluster_label']=labels;assignments['membership_strength']=strength;assignments['point_role']=np.where(labels<0,'nonmember',np.where(core,'core','border')) if spec['parameters']['algorithm']=='dbscan' else np.where(labels<0,'nonmember','member')
        table=pa.Table.from_pandas(assignments,preserve_index=False).replace_schema_metadata({b'agniscope_fit_scope':scope.encode(),b'input_checksum':spec['input_sha256'].encode(),b'parameters':json.dumps(spec['parameters'],sort_keys=True).encode()})
        pq.write_table(table,directory/'assignments.parquet',compression='zstd',row_group_size=4096)
        import geopandas as gpd,shapely
        geometries=[];members=[]
        for row in clusters:
            selected=frame.loc[labels==row['cluster_label']];geometries.append(shapely.MultiPoint(selected[s.XY].to_numpy()).convex_hull);members.append(selected[s.ID].tolist())
        geo=gpd.GeoDataFrame(clusters if clusters else {'cluster_label':pd.Series(dtype='int64')},geometry=geometries,crs=32644)
        geo['source_observation_ids']=members;geo['season_id']=spec['season_id'];geo['subset']=spec['subset'];geo['algorithm']=spec['parameters']['algorithm'];geo['parameter_identity']=json.dumps(spec['parameters'],sort_keys=True);geo['input_sha256']=spec['input_sha256']
        geo.to_parquet(directory/'cluster_geometries.parquet',index=False,compression='zstd')
        membership=assignments.assign(membership=np.where(labels>=0,'clustered','nonmember'))
        for name,column in [('district',s.DISTRICT),('daily','agniscope_utc_date'),('daynight','daynight')]:
            membership.groupby([column,'membership'],dropna=False).size().rename('observations').reset_index().to_parquet(directory/(name+'_diagnostics.parquet'),index=False)
    atomic_json(directory/'worker_finished.json',{'status':'passed','input_observations':len(frame),'scope':scope,'observation_requests':0,'credential_reads':0})
    if '--verify-before-commit' in sys.argv[2:]:
        from verify_phase3a import verify_spec
        proof=verify_spec(spec,deep=spec['reference'],directory=directory,require_receipt=False)
        atomic_json(directory/'precommit_verification.json',proof)
    print(json.dumps({'status':'passed','n':len(frame),'clusters':stats['n_clusters']}))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'status':'failed','error_class':type(exc).__name__,'safe_context':'offline scientific worker; inspect preserved staging'}));raise SystemExit(2)
