"""Synthetic Phase 3A science/resource contracts; no NASA network or source writes."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np,pandas as pd
import pytest
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN,HDBSCAN
from sklearn.metrics import adjusted_rand_score
from agniscope import seasonal_clustering as s
from agniscope.seasonal_inputs import season_bounds

def points(xy):
    n=len(xy);frame=pd.DataFrame({s.ID:[f'SYNTHETIC_SOURCE:{i+1}' for i in range(n)],s.XY[0]:np.array(xy)[:,0],s.XY[1]:np.array(xy)[:,1],
     s.UTC:pd.to_datetime(['2024-04-01T06:00:00Z']*n),s.DISTRICT:['SYNTHETIC_DISTRICT_A' if i%2 else 'SYNTHETIC_DISTRICT_B' for i in range(n)],
     'agniscope_confidence':['nominal']*n,'agniscope_worldcover_2021_class':[10]*n,'daynight':['D']*n,'frp':['1']*n,'agniscope_utc_date':['2024-04-01']*n})
    return s.canonical_points(frame)

@pytest.mark.parametrize('eps',[.5,1.,2.])
@pytest.mark.parametrize('minimum',[3,5,10])
def test_streaming_dbscan_exact_sklearn_noise_core_and_partitions(eps,minimum):
    rng=np.random.default_rng(1729)
    xy=np.concatenate([rng.normal([0,0],300,(100,2)),rng.normal([5000,0],700,(100,2)),rng.uniform(-10000,10000,(40,2)),np.zeros((12,2))])
    labels,core=s.streaming_dbscan(xy,eps,minimum);model=DBSCAN(eps=eps*1000,min_samples=minimum,algorithm='kd_tree',n_jobs=1).fit(xy)
    assert np.array_equal(labels<0,model.labels_<0) and adjusted_rand_score(labels,model.labels_)==1
    expected=np.zeros(len(xy),bool);expected[model.core_sample_indices_]=True
    assert np.array_equal(core,expected)

def test_border_chaining_duplicate_coordinates_and_exact_epsilon():
    xy=np.array([[0,0],[0,0],[1000,0],[2000,0],[3000,0],[4000,0],[10000,0]],float)
    labels,core=s.streaming_dbscan(xy,1.,3)
    truth=DBSCAN(eps=1000,min_samples=3,algorithm='kd_tree').fit(xy)
    assert np.array_equal(labels,truth.labels_) and labels[-1]==-1 and s.convex_diameter(xy[labels==0])>1000

@pytest.mark.parametrize('seed',range(5))
def test_rotating_calipers_diameter_matches_independent_small_pairwise(seed):
    xy=np.random.default_rng(seed).normal(size=(30,2))*1000
    assert np.isclose(s.convex_diameter(xy),pdist(xy).max(),rtol=1e-12,atol=1e-9)

@pytest.mark.parametrize('xy,expected',[([[0,0]],0),([[0,0],[3,4]],5),([[0,0],[0,0]],0),([[0,0],[2,0],[1,0]],2),([[0,0],[3,0],[3,4],[0,4]],5)])
def test_degenerate_and_rectangle_extents(xy,expected):assert s.convex_diameter(xy)==expected

def test_hdbscan_uses_existing_sklearn_semantics_and_protects_coordinates():
    frame=points(np.random.default_rng(3).normal(size=(80,2))*1000);before=frame.copy(deep=True)
    params={'algorithm':'hdbscan','min_cluster_size':3,'min_samples':10}
    labels,strength,_,details=s.fit_points(frame,params)
    model=HDBSCAN(min_cluster_size=3,min_samples=10,**s.HDB_FIXED).fit(frame[s.XY])
    assert np.array_equal(labels,model.labels_) and np.array_equal(strength,model.probabilities_)
    assert details['actual_estimator_parameters']['min_samples']==10 and details['actual_estimator_parameters']['min_cluster_size']==3
    pd.testing.assert_frame_equal(frame,before)

def test_core_projection_marker_and_grid_scope_rejection():
    frame=points([[0,0],[1,0],[2,0]])
    frame.attrs={}
    with pytest.raises(ValueError):s.fit_points(frame,s.PARAM_DB[0])
    frame.attrs['analysis_crs']='EPSG:32644'
    with pytest.raises(ValueError):s.fit_points(frame,{'algorithm':'dbscan','eps_km':3,'min_samples':3})
    with pytest.raises(ValueError):s.streaming_dbscan([[np.inf,1]],1,3)

def test_grid_count_and_references():
    assert len(s.PARAM_DB)==9 and len(s.PARAM_HDB)==12 and 7*(9+12)+7*3*2+6*2==201
    assert [r['eps_km'] for r in s.PARAM_DB[::3]]==[.5,1.,2.]

def test_temporal_reuse_earliest_then_original_numeric_identity_and_day_boundary():
    frame=points([[100,100],[200,200],[300,300],[400,400]])
    frame[s.ID]=['SYNTHETIC:10','SYNTHETIC:2','SYNTHETIC:1','SYNTHETIC:3']
    frame[s.UTC]=pd.to_datetime(['2024-04-01T00:00:00Z','2024-04-01T00:00:00Z','2024-04-01T01:00:00Z','2024-04-02T00:00:00Z'])
    sampled,ledger=s.temporal_sample(frame);assert set(sampled[s.ID])=={'SYNTHETIC:2','SYNTHETIC:3'}
    assert ledger.selected.sum()==2 and ledger.groupby(['grid_x','grid_y','utc_date']).representative_observation_id.nunique().eq(1).all()
    again,_=s.temporal_sample(frame.iloc[::-1]);assert sampled[s.ID].tolist()==again[s.ID].tolist()

def test_confidence_and_tree_populations_preserve_full_baseline():
    frame=points([[0,0],[1,0],[2,0],[3,0]]);frame.agniscope_confidence=['low','nominal','high','nominal'];frame.agniscope_worldcover_2021_class=[10,40,10,10]
    original=frame.copy();pop,ledger=s.populations(frame)
    assert len(pop['all'])==4 and len(pop['nominal'])==2 and len(pop['tree'])==3
    assert set(pop['nominal'][s.ID])==set(frame.loc[frame.agniscope_confidence.eq('nominal'),s.ID]);pd.testing.assert_frame_equal(frame,original)

def test_cluster_accounting_cross_district_identity_extent_and_time():
    frame=points([[0,0],[100,0],[200,0],[11000,0]])
    labels=np.array([0,0,0,-1]);strength=np.array([1,.8,.5,0]);rows=s.cluster_statistics(frame,labels,strength,'SYNTHETIC_SCOPE')
    assert rows[0]['n_observations']==3 and rows[0]['district_count']==2 and rows[0]['diameter_m']==200
    assert rows[0]['scoped_cluster_id']=='SYNTHETIC_SCOPE__cluster_0' and rows[0]['distinct_UTC_dates']==1
    metrics=s.accounting(frame,labels,strength,rows,np.array([1,1,1,0],bool));assert metrics['n_clustered']==3 and metrics['n_nonmember']==1
    with pytest.raises(ValueError):s.accounting(frame,labels,strength,[],np.zeros(4,bool))

def test_no_cluster_and_negative_label_accounting():
    frame=points([[0,0],[1,0],[2,0]])
    labels=np.array([-1,-2,-3]);rows=s.cluster_statistics(frame,labels,np.array([0,0,np.nan]),'SYNTHETIC')
    stats=s.accounting(frame,labels,np.array([0,0,np.nan]),rows,np.zeros(3,bool))
    assert stats['n_clusters']==0 and stats['n_nonmember']==3 and stats['n_noise_minus1']==stats['n_missing_minus3']==stats['n_infinite_minus2']==1 and stats['diameter_max_m'] is None

def test_broad_cluster_flag_is_description_not_split():
    frame=points([[0,0],[5000,0],[11000,0]])
    row=s.cluster_statistics(frame,np.zeros(3,int),np.ones(3),'SYNTHETIC')[0]
    assert row['broad_over_10km'] and row['n_observations']==3 and row['diameter_m']==11000

def test_comparisons_label_invariance_noise_denominators_and_pairs():
    ids=['a','b','c','d','e'];left=np.array([0,0,-1,1,1]);right=np.array([5,5,-1,6,6])
    stats,transitions,edges=s.compare_labels(ids,left,ids,right)
    assert stats['ARI_all_common_shared_noise']==1 and stats['clustered_population_jaccard']==1 and stats['coclustered_pair_jaccard']==1 and stats['shared_coclustered_pairs']==2
    subset,_,_=s.compare_labels(ids,left,ids[:3],np.array([7,-1,-3]))
    assert subset['n_common']==3 and subset['left_not_common']==2 and subset['clustered_population_jaccard']==.5
    empty,_,_=s.compare_labels(ids,[-1]*5,ids,[-1]*5)
    assert empty['clustered_population_jaccard'] is None and empty['ARI_clustered_union'] is None and empty['ARI_degenerate_shared_noise_warning']

@pytest.mark.parametrize('year,expected',[(2019,121),(2020,122),(2021,121),(2022,121),(2023,121),(2024,122),(2025,121)])
def test_primary_season_UTC_and_leaps(year,expected):
    b=season_bounds('core',year);assert (b['end_exclusive_utc']-b['start_utc']).days==expected and b['basis']=='UTC'

def test_extent_and_membership_do_not_allocate_observation_pair_matrix(monkeypatch):
    import scipy.spatial.distance as distance
    monkeypatch.setattr(distance,'pdist',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Dense pairs forbidden')))
    frame=points(np.random.default_rng(3).normal(size=(100,2))*1000)
    labels,strength,core,_=s.fit_points(frame,s.PARAM_DB[3]);s.cluster_statistics(frame,labels,strength,'SYNTHETIC')
    s.compare_labels(frame[s.ID],labels,frame[s.ID],labels)

def test_worker_zero_cluster_serialization_is_complete_and_scoped(tmp_path,monkeypatch):
    spec_import=importlib.util.spec_from_file_location('phase3worker',Path(__file__).resolve().parents[1]/'scripts/phase3a_worker.py');module=importlib.util.module_from_spec(spec_import);spec_import.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    directory=tmp_path/'outputs/tables/phase3a/runs/SYNTHETIC.staged';directory.mkdir(parents=True)
    frame=points([[0,0],[10000,0],[20000,0]]);source=tmp_path/'synthetic.parquet';frame.to_parquet(source,index=False)
    from agniscope.provenance import sha256
    spec={'input_path':'synthetic.parquet','input_sha256':sha256(source),'n_observations':3,'input_feature_identity':s.feature_identity(frame),
      'season_id':'SYNTHETIC_SEASON','subset':'all','parameters':s.PARAM_DB[5],'reference':True,'run_id':'fit_SYNTHETIC'}
    (directory/'spec.json').write_text(json.dumps(spec));monkeypatch.setattr(sys,'argv',['worker',str(directory.relative_to(tmp_path))])
    module.main();assert json.loads((directory/'statistics.json').read_text())['metrics']['n_clusters']==0
    assert (directory/'assignments.parquet').exists() and (directory/'cluster_geometries.parquet').exists()
    compact=np.load(directory/'labels_compact.npz');assert np.array_equal(compact['labels'],[-1,-1,-1])

def driver():
    spec=importlib.util.spec_from_file_location('phase3driver',Path(__file__).resolve().parents[1]/'scripts/phase3a.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_declared_runtime_plan_is_exact_and_unique():
    module=driver();plan=json.loads((module.OUT/'experiment_plan.json').read_text());assert len(module.validate_plan(plan))==209

@pytest.mark.parametrize('mutation',['extra','missing','identity','year','subset'])
def test_plan_expansion_corruption_and_unauthorized_seasons_are_rejected(mutation):
    import hashlib
    module=driver();plan=json.loads((module.OUT/'experiment_plan.json').read_text())
    if mutation=='extra':plan.append(plan[0])
    elif mutation=='missing':plan.pop()
    elif mutation=='identity':plan[0]['run_id']='bad'
    else:
        plan[0]['season_id' if mutation=='year' else 'subset']='UNAUTHORIZED'
        body={k:v for k,v in plan[0].items() if k!='run_id'};plan[0]['run_id']='fit_'+hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with pytest.raises(module.StoreConflict):module.validate_plan(plan)

def test_ledger_reports_unstarted_failed_completed_without_hiding_runs(tmp_path,monkeypatch):
    module=driver();monkeypatch.setattr(module,'OUT',tmp_path)
    specs=[{'run_id':'fit_'+c*64,'season_id':'SYNTHETIC','subset':'all','role':'primary_grid','parameters':{}} for c in 'abc']
    for spec,status in zip(specs,['completed','failed']):
        directory=tmp_path/'runs'/spec['run_id'][:20]
        if status=='failed':directory=directory.with_name(directory.name+'.staged')
        directory.mkdir(parents=True);(directory/('receipt.json' if status=='completed' else 'failure.json')).write_text('{}')
    assert [r['status'] for r in module.ledger(specs)]==['completed','failed','not_started']

def test_unsafe_resources_prevent_worker_launch_and_leave_inputs_immutable(tmp_path,monkeypatch):
    module=driver();monkeypatch.setattr(module,'OUT',tmp_path)
    monkeypatch.setattr(module,'resource_snapshot',lambda:{'available_physical_bytes':1,'disk_free_bytes':10**12})
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:pytest.fail('Unsafe worker launched'))
    with pytest.raises(module.StoreConflict):module.execute({'run_id':'fit_SYNTHETIC'})
    assert not (tmp_path/'runs').exists()

def test_primary_requires_independent_feasibility_and_equivalence_gates(tmp_path,monkeypatch):
    module=driver();plan=json.loads((module.OUT/'experiment_plan.json').read_text());monkeypatch.setattr(module,'OUT',tmp_path)
    (tmp_path/'experiment_plan.json').write_text(json.dumps(plan));(tmp_path/'feasibility_gate.json').write_text('{"status":"PASS"}')
    monkeypatch.setattr(module,'protected',lambda:0);monkeypatch.setattr(module,'test_gate',lambda:None)
    monkeypatch.setattr(module,'execute',lambda *a,**k:pytest.fail('Unverified grid launched'))
    with pytest.raises(FileNotFoundError):module.run_group('primary')

def test_serial_controller_does_not_load_numeric_libraries_into_parent():
    import subprocess
    root=Path(__file__).resolve().parents[1]
    code="import runpy,sys; d=runpy.run_path('scripts/phase3a.py'); d['config'](); assert not any(m in sys.modules for m in ('numpy','pandas','sklearn','geopandas')); print('small_controller_passed')"
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=root,capture_output=True,text=True,timeout=30,check=True)
    assert result.stdout.strip()=='small_controller_passed'
