"""Deterministic ML input batches; no clustering, labels or feature fitting."""
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

STUDY_START=datetime(2019,1,1,tzinfo=timezone.utc)
STUDY_END=datetime(2026,1,1,tzinfo=timezone.utc)

def season_bounds(kind,year,*,basis='UTC'):
    if basis not in ('UTC','IST'):raise ValueError('Declare UTC or IST explicitly.')
    if year not in range(2019,2027):raise ValueError('Season year is outside the supported study interface.')
    zone=timezone.utc if basis=='UTC' else ZoneInfo('Asia/Kolkata')
    if kind=='calendar':first,last=(year,1,1),(year+1,1,1);identifier=f'CY_{year}'
    elif kind=='core':first,last=(year,2,15),(year,6,16);identifier=f'CORE_{year}'
    elif kind=='extended':first,last=(year-1,11,1),(year,7,1);identifier=f'EXT_{year-1}_{year}'
    else:raise ValueError('Use calendar, core or extended.')
    start=datetime(*first,tzinfo=zone).astimezone(timezone.utc)
    end=datetime(*last,tzinfo=zone).astimezone(timezone.utc)
    return {'season_id':identifier,'basis':basis,'start_utc':start,'end_exclusive_utc':end,
            'calendar_supported':start>=STUDY_START and end<=STUDY_END,
            'left_truncated':start<STUDY_START,'right_truncated':end>STUDY_END,
            'satellite_opportunity':'unverified'}

def iter_season(path,kind,year,*,basis='UTC',allow_partial=False,batch_size=4096):
    """Yield all original 49 fields plus explicit interface identifiers.

    Extended partial flags in annual inputs remain unchanged: the new selection
    metadata states whether the combined calendar window is supported. No filters
    on confidence, FRP, land cover or Type are introduced.
    """
    if not 1<=batch_size<=4096:raise ValueError('Batch size must be 1..4096.')
    bounds=season_bounds(kind,year,basis=basis)
    if not bounds['calendar_supported'] and not allow_partial:raise ValueError('Partial season requires explicit allow_partial=True.')
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        table=pa.Table.from_batches([batch]);time=table['agniscope_acquired_utc'];dtype=time.type
        mask=pc.and_(pc.greater_equal(time,pa.scalar(bounds['start_utc'],type=dtype)),pc.less(time,pa.scalar(bounds['end_exclusive_utc'],type=dtype)))
        selected=table.filter(mask)
        if not selected.num_rows:continue
        selected=selected.append_column('ml_season_id',pa.array([bounds['season_id']]*selected.num_rows,type=pa.large_string()))
        selected=selected.append_column('ml_date_basis',pa.array([basis]*selected.num_rows,type=pa.large_string()))
        selected=selected.append_column('ml_calendar_supported',pa.array([bounds['calendar_supported']]*selected.num_rows,type=pa.bool_()))
        yield selected
