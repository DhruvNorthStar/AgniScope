"""Explicit C4/2021 then gated C5/2020; offline by default, one serial batch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agniscope import acquisition as a,stage_b as b,stage_b_qc as q,paired_acquisition as p
from agniscope.acquisition_store import StoreConflict,atomic_json,store_lock
from agniscope.config import workspace_path
from agniscope.provenance import sha256,utc_now
from agniscope.resources import resource_snapshot

def metadata_check(settings):
    key=a.local_map_key()
    if not key:raise StoreConflict('A configured local credential is required.')
    http={};body=a._bounded_get(f'{a.API_BASE}/data_availability/csv/{key}/VIIRS_SNPP_SP',65536,response_metadata=http)
    available=a._availability(body,'VIIRS_SNPP_SP')
    if available['min_date']>'2020-01-01' or available['max_date']<'2021-12-31':raise StoreConflict('Availability excludes approved dates.')
    quota=a._transaction_status(key)
    if quota['transaction_limit']-quota['current_transactions']<max(1000,quota['transaction_limit']*.2):raise StoreConflict('Transaction reserve insufficient.')
    evidence={'checked_utc':utc_now(),'source':'VIIRS_SNPP_SP','availability':available,'http':http,
      'availability_response_sha256':hashlib.sha256(body).hexdigest(),'quota':quota,'observation_requests':0,
      'upstream_supplement_sha256':sha256(workspace_path('configs/phase2b_c4_c5_upstream_issues.csv'))}
    public_http={}
    try:
        public=a._bounded_get('https://firms.modaps.eosdis.nasa.gov/api/missing_data/',524288,response_metadata=public_http)
        evidence['global_missing_metadata']={'public_url':'https://firms.modaps.eosdis.nasa.gov/api/missing_data/','http':public_http,
          'size_bytes':len(public),'sha256':hashlib.sha256(public).hexdigest(),'failed_to_load_message_present':b'Failed to load data' in public,'daily_calendar_verified':False}
    except a.AcquisitionError:evidence['global_missing_metadata']={'status':'unavailable','daily_calendar_verified':False}
    atomic_json(workspace_path(settings['paths']['output'])/'upstream_metadata_evidence.json',evidence)
    return evidence

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--year',type=int,choices=(2021,2020),default=2021)
    parser.add_argument('--batch-size',type=int,default=10)
    modes=parser.add_mutually_exclusive_group()
    for name in ('execute-c4-c5','check-metadata','certify-tests','open-gate0','open-2020-gate','integrate','audit-only'):modes.add_argument('--'+name,action='store_true')
    args=parser.parse_args(argv)
    try:
        plans=p.combined_plan();settings=p.configuration(args.year);out=workspace_path(settings['paths']['output'])
        b.protected_integrity(p.configuration(2021))
        if args.certify_tests:
            print(json.dumps({'tests':p.test_gate(args.year,True)['tests'],'observation_requests':0}));return 0
        if args.check_metadata:
            print(json.dumps(metadata_check(settings),indent=2));return 0
        if args.open_gate0:
            result=p.open_gate0();print(json.dumps({'gate0':result['status'],'calendar_days':731,'intervals':147,'observation_requests':0}));return 0
        if args.open_2020_gate:
            if args.year!=2021:raise StoreConflict('The transition gate belongs to completed C4.')
            metadata_check(p.configuration(2020))
            result=p.open_c4_gate();print(json.dumps({'gate1':result['gate1'],'gate2':result['gate2'],'authorized_next_year':2020,'observation_requests':0}));return 0
        if not any((args.execute_c4_c5,args.integrate,args.audit_only)):
            print(json.dumps({'mode':'dry_run','year_order':[2021,2020],'calendar_days':731,'intervals':147,
              'years':{y:b.run_batch(p.configuration(y),batch_size=args.batch_size) for y in plans},'observation_requests':0},indent=2));return 0
        p.test_gate(args.year);p.require_authorization(settings)
        if args.execute_c4_c5:
            metadata=json.loads((out/'upstream_metadata_evidence.json').read_text())
            if metadata['source']!='VIIRS_SNPP_SP' or metadata['availability']['min_date']>settings['study']['start'] or metadata['availability']['max_date']<settings['study']['end']:raise StoreConflict('Per-year availability evidence required.')
            with store_lock(workspace_path(p.JOINT)):
                result=b.run_batch(settings,execute=True,batch_size=args.batch_size)
        else:result={'status':'local_audit','observation_requests':0}
        catalog=b.load_catalog(settings)
        if catalog is None:raise StoreConflict('No annual manifest exists; local audit cannot initiate acquisition.')
        q.write_table(out/f'daily_coverage_{args.year}.csv',q.coverage_rows(settings,catalog,plans[args.year]))
        if args.integrate:
            before=resource_snapshot();summary,reused=q.integrate_year(settings,catalog,plans[args.year])
            atomic_json(out/'integration_resources.json',{'before':before,'after':resource_snapshot()})
            result={'year':args.year,'status':summary[f"stage_{settings['study']['stage'].lower()}_status"],'counts':summary['counts'],'annual_dataset_reused':reused,'observation_requests':0}
        print(json.dumps(result,indent=2));return 2 if catalog['halt_reason'] else 0
    except (a.AcquisitionError,StoreConflict,OSError,ValueError,KeyError):
        print('C4+C5 stopped by an authorization, gate, integrity, resource, service or QC check. Sensitive details withheld.');return 2

if __name__=='__main__':raise SystemExit(main())
