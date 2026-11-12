"""Offline, bounded seven-year schema normalization and deterministic source ordering.

No network or credential functions are imported here. Original annual products are
read-only inputs. A year-specific Boolean is renamed, without changing its values.
"""
from datetime import date, timedelta
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from .acquisition_store import StoreConflict

YEARS=tuple(range(2019,2026))
CORE='agniscope_core_year'

def normalized_schema(schema,year):
    old=f'agniscope_core_{year}'
    if len(schema)!=49 or old not in schema.names or CORE in schema.names:
        raise StoreConflict('Annual input does not match the declared 49-field schema.')
    return pa.schema([pa.field(CORE if f.name==old else f.name,f.type,nullable=f.nullable) for f in schema])

def normalized_table(batch,year,schema):
    table=pa.Table.from_batches([batch]) if isinstance(batch,pa.RecordBatch) else batch
    actual=normalized_schema(table.schema,year)
    if actual!=schema:raise StoreConflict('Annual field names or types differ.')
    return table.rename_columns(schema.names).replace_schema_metadata(None)

def independent_calendar(first=date(2019,1,1),last=date(2025,12,31)):
    if last<first:raise ValueError('Invalid calendar bounds.')
    return [str(first+timedelta(days=n)) for n in range((last-first).days+1)]

def reconcile_counts(rows):
    total={'raw':0,'retained':0,'geographic_exclusions':0,'other_exclusions':0}
    for row in rows:
        values={k:row[k] for k in total}
        if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in values.values()):
            raise StoreConflict('Accounting requires nonnegative integer counts.')
        if values['raw']!=values['retained']+values['geographic_exclusions']+values['other_exclusions']:
            raise StoreConflict('Raw-to-retained accounting does not reconcile.')
        for key,value in values.items():total[key]+=value
    return total

def build_sorted_parquet(sources,destination,index_path,*,batch_size=4096):
    """External SQLite ordering; memory bounded by one Arrow batch and 4096 rows.

    Identity and payload collisions are errors; source records are never silently
    removed. Dates, times and original source strings are preserved. Timestamps
    enter SQLite as integer nanoseconds and are restored with their exact zones.
    """
    if batch_size<1 or batch_size>4096:raise ValueError('Invalid bounded batch size.')
    destination,index_path=Path(destination),Path(index_path)
    if destination.exists() or index_path.exists():raise StoreConflict('Master staging must be new.')
    if sorted(sources)!=list(YEARS):raise StoreConflict('Exactly seven annual partitions are required.')
    schema=normalized_schema(pq.read_schema(sources[2019]),2019)
    destination.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(index_path)
    db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE records(id TEXT PRIMARY KEY,payload TEXT UNIQUE,utc_ns INTEGER,body TEXT)')
    count=0;writer=None
    try:
        timestamp_names=[f.name for f in schema if pa.types.is_timestamp(f.type)]
        for year,path in sorted(sources.items()):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                table=normalized_table(batch,year,schema)
                values=table.to_pydict()
                for name in timestamp_names:values[name]=table[name].cast(pa.int64()).to_pylist()
                data=[]
                for n in range(table.num_rows):
                    row={name:values[name][n] for name in schema.names}
                    if row['agniscope_calendar_year']!=year or row['agniscope_utc_date'][:4]!=str(year):
                        raise StoreConflict('Annual provenance or UTC year mismatch.')
                    if not row['agniscope_in_footprint'] or row['agniscope_disposition']!='retained':
                        raise StoreConflict('Master input contains an observation outside accepted QC.')
                    data.append((row['agniscope_observation_id'],row['agniscope_source_payload_sha256'],row['agniscope_acquired_utc'],json.dumps(row,sort_keys=True,allow_nan=False,separators=(',',':'))))
                try:db.executemany('INSERT INTO records VALUES(?,?,?,?)',data)
                except sqlite3.IntegrityError:raise StoreConflict('Duplicate source identity or exact source payload in annual inputs.') from None
                count+=table.num_rows
                db.commit()
        db.execute('CREATE INDEX ordered_records ON records(utc_ns,id)');db.commit()
        writer=pq.ParquetWriter(destination,schema,compression='zstd',version='2.6',use_dictionary=True,write_statistics=True)
        cursor=db.execute('SELECT body FROM records ORDER BY utc_ns,id')
        while records:=cursor.fetchmany(batch_size):
            rows=[json.loads(r[0]) for r in records]
            table=pa.Table.from_pylist(rows,schema=schema)
            writer.write_table(table,row_group_size=batch_size)
        writer.close();writer=None
        return {'retained_rows':count,'field_count':len(schema),'schema':[{ 'name':f.name,'type':str(f.type)} for f in schema],
                'duplicate_identities':0,'duplicate_payloads':0,'ordering':['agniscope_acquired_utc','agniscope_observation_id'],
                'batch_size':batch_size,'timestamp_roundtrip':'integer nanoseconds with original timezone types',
                'core_mapping':'agniscope_core_YYYY -> agniscope_core_year; original year retained in agniscope_calendar_year'}
    finally:
        if writer is not None:writer.close()
        db.close()
