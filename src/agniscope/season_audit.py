"""Pure UTC November-June boundary and daily-accounting checks; no acquisition."""
import pandas as pd


def utc_extended_mask(timestamps, ending_year=2023):
    values = pd.Series(pd.to_datetime(timestamps, utc=True, format="mixed"))
    start = pd.Timestamp(f"{ending_year-1}-11-01", tz="UTC")
    end = pd.Timestamp(f"{ending_year}-07-01", tz="UTC")
    return values.ge(start) & values.lt(end)


def audit_extended_daily(frame, ending_year=2023):
    """Missing/failed dates remain unknown; duplicate dates are an integrity error."""
    start, last = f"{ending_year-1}-11-01", f"{ending_year}-06-30"
    dates = pd.to_datetime(frame["utc_date"], utc=True, errors="raise")
    selected = frame.loc[utc_extended_mask(dates, ending_year).to_numpy()].copy()
    selected["utc_date"] = pd.to_datetime(selected.utc_date, utc=True).dt.strftime("%Y-%m-%d")
    if selected.utc_date.duplicated().any():
        raise ValueError("Extended season contains duplicate UTC dates.")
    expected = pd.date_range(start, last).strftime("%Y-%m-%d").tolist()
    missing = sorted(set(expected) - set(selected.utc_date))
    valid = selected.retrieval_state.isin(["positive_detections", "zero_returned_detections"])
    valid &= selected.qc_status.eq("passed") & selected.provenance_status.eq("hash_verified")
    for column in ("raw_observations", "retained_observations"):
        numeric = pd.to_numeric(selected[column], errors="coerce")
        if (valid & (numeric.isna() | numeric.lt(0) | numeric.mod(1).ne(0))).any():
            raise ValueError("Validated daily counts must be nonnegative integers.")
        selected[column] = numeric
    successful = selected.loc[valid].copy()
    if (successful.retained_observations.astype(int) > successful.raw_observations.astype(int)).any():
        raise ValueError("Retained daily counts exceed their raw denominator.")
    monthly = []
    for month in pd.period_range(start, last, freq="M").astype(str):
        part = selected.loc[selected.utc_date.str.startswith(month)]
        accepted = successful.loc[successful.utc_date.str.startswith(month)]
        monthly.append({"month": month, "represented_dates": len(part), "validated_dates": len(accepted),
                        "raw_observations": int(accepted.raw_observations.sum()), "retained_observations": int(accepted.retained_observations.sum())})
    return {"season_id": f"EXT_{ending_year-1}_{ending_year}", "utc_start": start, "utc_end_inclusive": last,
            "utc_end_exclusive": f"{ending_year}-07-01", "expected_dates": len(expected), "represented_dates": len(selected),
            "validated_dates": len(successful), "missing_dates": missing, "failed_or_unvalidated_dates": sorted(selected.loc[~valid, "utc_date"]),
            "calendar_retrieval_eligible": not missing and bool(valid.all()), "duplicate_dates": 0,
            "raw_observations": int(successful.raw_observations.sum()), "retained_observations": int(successful.retained_observations.sum()),
            "satellite_opportunity_unknown_dates": int(selected.satellite_coverage.eq("unknown").sum()),
            "satellite_completeness": "unverified", "monthly_contributions": monthly}
