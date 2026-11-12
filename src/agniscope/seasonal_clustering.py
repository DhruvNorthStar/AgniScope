"""Bounded seasonal spatial clustering; no source edits, acquisition or recurrence."""
from collections import deque
import hashlib,json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score
from . import dbscan as pilot_db,hdbscan as pilot_hdb

ID='agniscope_observation_id'
XY=['agniscope_x_m','agniscope_y_m']
UTC='agniscope_acquired_utc'
DISTRICT='agniscope_district_original'
PARAM_DB=[{'algorithm':'dbscan','eps_km':eps,'min_samples':m} for eps in (.5,1.,2.) for m in (3,5,10)]
PARAM_HDB=[{'algorithm':'hdbscan','min_cluster_size':size,'min_samples':m} for size in (3,5,10,15) for m in (3,5,10)]
HDB_FIXED={'cluster_selection_method':'eom','cluster_selection_epsilon':0.,'metric':'euclidean','alpha':1.,'algorithm':'kd_tree','leaf_size':40,'allow_single_cluster':False,'copy':True,'n_jobs':1}

def canonical_points(frame):
    if frame[ID].isna().any() or frame[ID].duplicated().any():raise ValueError('Source IDs must be unique and present.')
    result=frame.copy();split=result[ID].str.rsplit(':',n=1,expand=True)
    if split.shape[1]!=2:raise ValueError('Expected original SHA256:record identities.')
    result['_raw_sort']=split[0];result['_record_sort']=pd.to_numeric(split[1],errors='raise')
    result=result.sort_values(['_raw_sort','_record_sort'],kind='stable').drop(columns=['_raw_sort','_record_sort']).reset_index(drop=True)
    if not np.isfinite(result[XY].to_numpy(float)).all():raise ValueError('Finite projected coordinates required.')
    result.attrs['analysis_crs']='EPSG:32644'
    return result

def feature_identity(frame):
    digest=hashlib.sha256()
    for identity in frame[ID]:digest.update((identity+'\n').encode())
    digest.update(np.asarray(frame[XY],dtype='<f8').tobytes(order='C'))
    return digest.hexdigest()

def temporal_sample(frame):
    """Reuse pilot sampler, mapping stable global identities to ordered integers.

    Earliest UTC instant, then raw SHA256/numeric original-row order; origin 0,0.
    Original historic source-record fields are never replaced by adapter ordinals.
    """
    points=canonical_points(frame)
    adapter=points[[*XY,UTC]].copy();adapter[pilot_db.ID]=np.arange(1,len(points)+1,dtype=np.int64)
    adapter['utc_date']=pd.to_datetime(adapter[UTC],utc=True).dt.strftime('%Y-%m-%d')
    reduced,ledger=pilot_db.daily_grid_sample(adapter,1000.,(0.,0.))
    ids=points[ID].to_numpy();selected=reduced[pilot_db.ID].to_numpy(int)-1
    ledger['observation_id']=ids[ledger[pilot_db.ID].to_numpy(int)-1]
    ledger['representative_observation_id']=ids[ledger.representative_source_record.to_numpy(int)-1]
    return points.iloc[selected].reset_index(drop=True),ledger[['observation_id','grid_x','grid_y','utc_date','representative_observation_id','selected']]

def populations(frame):
    points=canonical_points(frame);reduced,ledger=temporal_sample(points)
    return {'all':points,'grid_day':reduced,'nominal':points.loc[points.agniscope_confidence.eq('nominal')].reset_index(drop=True),
            'tree':points.loc[points.agniscope_worldcover_2021_class.eq(10)].reset_index(drop=True)},ledger

def streaming_dbscan(xy,eps_km,min_samples):
    """Exact unweighted DBSCAN, cKDTree p=2/eps=0, one neighbourhood in memory.

    Counts include self. Components are discovered in canonical input order;
    shared borders belong to the first reachable core component, as sklearn.
    """
    pilot_db.parameter_grid([eps_km],[min_samples]);xy=np.asarray(xy,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('Finite two-dimensional metre coordinates required.')
    n=len(xy);labels=np.full(n,-1,dtype=np.int64)
    if not n:return labels,np.zeros(0,dtype=bool)
    radius=pilot_db.eps_metres(eps_km);tree=cKDTree(xy)
    core=tree.query_ball_point(xy,radius,p=2,eps=0,workers=1,return_length=True)>=min_samples
    label=0
    for seed in range(n):
        if not core[seed] or labels[seed]>=0:continue
        labels[seed]=label;queue=deque([seed])
        while queue:
            current=queue.popleft()
            for other in tree.query_ball_point(xy[current],radius,p=2,eps=0,workers=1,return_sorted=True):
                if labels[other]<0:
                    labels[other]=label
                    if core[other]:queue.append(other)
        label+=1
    return labels,core

def fit_points(frame,parameters):
    if frame.attrs.get('analysis_crs')!='EPSG:32644':raise ValueError('EPSG:32644 marker is required.')
    if not frame[ID].is_unique:raise ValueError('Unique identities required.')
    if parameters not in PARAM_DB+PARAM_HDB:raise ValueError('Parameters outside the declared grids.')
    xy=frame[XY].to_numpy(float);original=xy.copy()
    if parameters['algorithm']=='dbscan':
        labels,core=streaming_dbscan(xy,parameters['eps_km'],parameters['min_samples'])
        strength=np.full(len(frame),np.nan);details={'neighbour_search':'scipy.spatial.cKDTree exact radius, p=2, eps=0, workers=1; one queried neighbourhood at a time','epsilon_m':pilot_db.eps_metres(parameters['eps_km']),'min_samples_includes_self':True}
    else:
        # Reuse the protected Phase 1B estimator/validation implementation only;
        # its pilot pairwise summary and silhouette helpers are deliberately avoided.
        adapter=pd.DataFrame({pilot_db.ID:np.arange(1,len(frame)+1),XY[0]:xy[:,0],XY[1]:xy[:,1]});adapter.attrs['analysis_crs']='EPSG:32644'
        fitted,actual=pilot_hdb.fit_configuration(adapter,parameters['min_cluster_size'],parameters['min_samples'],HDB_FIXED)
        labels=fitted.cluster_id.to_numpy(np.int64);strength=fitted.membership_probability.to_numpy(float);core=np.zeros(len(frame),bool)
        details={'estimator':'sklearn.cluster.HDBSCAN','actual_estimator_parameters':actual,'min_samples_includes_self':True,'strength_interpretation':'algorithmic membership diagnostic, not fire probability'}
    if not np.array_equal(xy,original):raise ValueError('Estimator changed input coordinates.')
    return labels,strength,core,details

def convex_diameter(xy):
    """Exact convex-hull diameter with rotating calipers; no quadratic matrix."""
    values=np.unique(np.asarray(xy,dtype=float),axis=0)
    if len(values)<2:return 0.
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for p in values:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in values[::-1]:
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    hull=np.asarray(lower[:-1]+upper[:-1]);n=len(hull)
    if n==2:return float(np.linalg.norm(hull[1]-hull[0]))
    best=0.;j=1
    for i in range(n):
        nxt=(i+1)%n
        while abs(cross(hull[i],hull[nxt],hull[(j+1)%n]))>abs(cross(hull[i],hull[nxt],hull[j])):
            j=(j+1)%n
        best=max(best,float(np.sum((hull[i]-hull[j])**2)),float(np.sum((hull[nxt]-hull[j])**2)))
    return float(np.sqrt(best))

def describe(values,prefix):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    return {prefix+'available':len(values),**{prefix+name:(float(np.quantile(values,q)) if len(values) else None) for name,q in [('min',0),('q25',.25),('median',.5),('q75',.75),('max',1)]},
            prefix+'mean':float(values.mean()) if len(values) else None,prefix+'sample_std':float(values.std(ddof=1)) if len(values)>1 else None}

def cluster_statistics(frame,labels,strength,scope,threshold=10000.):
    rows=[]
    for label in np.unique(labels[labels>=0]):
        mask=labels==label;part=frame.loc[mask];xy=part[XY].to_numpy(float);centre=xy.mean(axis=0);distance=np.linalg.norm(xy-centre,axis=1)
        representative=part.iloc[int(np.argmin(distance))][ID];times=pd.to_datetime(part[UTC],utc=True);diameter=convex_diameter(xy)
        row={'cluster_label':int(label),'scoped_cluster_id':scope+'__cluster_'+str(label),'n_observations':len(part),'centroid_x_m':float(centre[0]),'centroid_y_m':float(centre[1]),
          'representative_observation_id':representative,'x_min_m':float(xy[:,0].min()),'x_max_m':float(xy[:,0].max()),'y_min_m':float(xy[:,1].min()),'y_max_m':float(xy[:,1].max()),
          'diameter_m':diameter,'radius_p95_m':float(np.quantile(distance,.95)),'broad_over_10km':diameter>threshold,
          'districts_original':';'.join(sorted(part[DISTRICT].unique())),'district_count':int(part[DISTRICT].nunique()),
          'first_utc':times.min().isoformat(),'last_utc':times.max().isoformat(),'span_days':float((times.max()-times.min()).total_seconds()/86400),'distinct_UTC_dates':int(times.dt.date.nunique()),
          'n_day':int(part.daynight.eq('D').sum()),'n_night':int(part.daynight.eq('N').sum()),'n_nominal':int(part.agniscope_confidence.eq('nominal').sum()),'n_tree_context':int(part.agniscope_worldcover_2021_class.eq(10).sum()),
          **describe(pd.to_numeric(part.frp), 'frp_MW_'),**describe(strength[mask],'strength_')}
        rows.append(row)
    return rows

def accounting(frame,labels,strength,clusters,core):
    n=len(frame);labels=np.asarray(labels);clustered=labels>=0;sizes=[r['n_observations'] for r in clusters]
    if len(labels)!=n or sum(sizes)!=int(clustered.sum()) or int(clustered.sum())+int((~clustered).sum())!=n:raise ValueError('Cluster accounting failed.')
    if set(labels[labels<0])-{-1,-2,-3}:raise ValueError('Undocumented negative labels.')
    return {'n_observations':n,'n_clusters':len(sizes),'n_clustered':int(clustered.sum()),'n_noise_minus1':int((labels==-1).sum()),'n_infinite_minus2':int((labels==-2).sum()),'n_missing_minus3':int((labels==-3).sum()),
      'n_nonmember':int((~clustered).sum()),'nonmember_pct':100*float((~clustered).mean()) if n else None,'size_min':min(sizes) if sizes else None,'size_median':float(np.median(sizes)) if sizes else None,
      'size_max':max(sizes) if sizes else None,'diameter_median_m':float(np.median([r['diameter_m'] for r in clusters])) if sizes else None,'diameter_max_m':max([r['diameter_m'] for r in clusters],default=None),
      'broad_clusters_over_10km':sum(r['broad_over_10km'] for r in clusters),'cross_district_clusters':sum(r['district_count']>1 for r in clusters),'largest_share_of_input':max(sizes)/n if n and sizes else None,
      'n_core':int(core.sum()),'silhouette_status':'omitted_resource_and_density_scale_interpretation; no dense pairwise distances',
      **describe(strength[clustered],'member_strength_'),**describe(pd.to_numeric(frame.loc[clustered,'frp']),'clustered_frp_MW_')}

def compare_labels(ids_left,labels_left,ids_right,labels_right):
    """Identity alignment and contingency counts, never all observation pairs."""
    left=pd.DataFrame({ID:ids_left,'left':labels_left});right=pd.DataFrame({ID:ids_right,'right':labels_right})
    if not left[ID].is_unique or not right[ID].is_unique:raise ValueError('Comparison identities must be unique.')
    joined=left.merge(right,on=ID,validate='one_to_one');n=len(joined)
    if not n:raise ValueError('No common source identities.')
    a=np.where(joined.left.to_numpy()<0,-1,joined.left.to_numpy());b=np.where(joined.right.to_numpy()<0,-1,joined.right.to_numpy());ac=a>=0;bc=b>=0;both=ac&bc;union=ac|bc
    fraction=lambda x,y:float(x/y) if y else None
    ari=lambda mask:float(adjusted_rand_score(a[mask],b[mask])) if mask.sum()>=2 else None
    choose2=lambda x:int(x)*(int(x)-1)//2
    pair_a=sum(choose2(v) for v in pd.Series(a[ac]).value_counts());pair_b=sum(choose2(v) for v in pd.Series(b[bc]).value_counts())
    intersections=pd.DataFrame({'left':a[both],'right':b[both]}).value_counts().rename('shared_observations').reset_index()
    shared_pairs=sum(choose2(v) for v in intersections.shared_observations)
    stats={'left_input_n':len(left),'right_input_n':len(right),'n_common':n,'left_not_common':len(left)-n,'right_not_common':len(right)-n,
      'clustered_both':int(both.sum()),'left_only_clustered':int((ac&~bc).sum()),'right_only_clustered':int((~ac&bc).sum()),'nonmember_both':int((~union).sum()),'clustered_union_n':int(union.sum()),
      'membership_agreement':float((ac==bc).mean()),'clustered_population_jaccard':fraction(both.sum(),union.sum()),'ARI_all_common_shared_noise':ari(np.ones(n,bool)),
      'ARI_clustered_union':ari(union),'ARI_clustered_intersection':ari(both),'shared_coclustered_pairs':shared_pairs,'left_coclustered_pairs':pair_a,'right_coclustered_pairs':pair_b,
      'coclustered_pair_jaccard':fraction(shared_pairs,pair_a+pair_b-shared_pairs),'ARI_degenerate_shared_noise_warning':bool(not union.any()),
      'left_clusters_split_on_common':int(intersections.groupby('left').size().gt(1).sum()) if len(intersections) else 0,'right_clusters_merge_on_common':int(intersections.groupby('right').size().gt(1).sum()) if len(intersections) else 0}
    joined['membership_transition']=np.select([both,ac&~bc,~ac&bc],['clustered_both','left_only_clustered','right_only_clustered'],default='nonmember_both')
    return stats,joined,intersections

def neighbour_cost(xy):
    values=np.asarray(xy,float);tree=cKDTree(values);rows=[]
    for km in (.5,1.,2.):
        counts=tree.query_ball_point(values,km*1000,p=2,eps=0,workers=1,return_length=True)
        rows.append({'eps_km':km,'observations':len(values),'neighbour_entries_including_self':int(counts.sum()),'max_neighbourhood':int(counts.max()) if len(counts) else 0,
         'median_neighbourhood':float(np.median(counts)) if len(counts) else 0,'p95_neighbourhood':float(np.quantile(counts,.95)) if len(counts) else 0,
         'bulk_neighbour_index_lower_bound_bytes':int(counts.sum())*8,'streamed_single_neighbour_index_upper_bound_bytes':len(values)*8})
    return rows
