# -------------------------------------
# AMT schedules - Schedule expansion
# -------------------------------------
"""
Schedule expansion and straddle building utilities.

Handles reading schedules from AMT files, expanding across year/month ranges,
and packing into straddle strings.
"""
import hashlib
import re
from pathlib import Path
from typing import Any, List
import calendar
import datetime 
import itertools
from . import loader 
from . import _numba_kernels
import numpy as np
from .table import table_to_arrow, table_nrows, _import_pyarrow
pa, pc = _import_pyarrow()

# memoization

_MEMOIZE_ENABLED: bool = True

# Forward declare caches (initialized later, referenced by clear_schedule_caches)
_SCHEDULE_CACHE: dict = {}
_EXPAND_YM_CACHE: dict = {}
_DAYS_YM_CACHE: dict = {}
_STRADDLE_DAYS_CACHE: dict = {}

def set_memoize_enabled(enabled: bool) -> None:
    """Enable or disable memoization for schedule functions."""
    global _MEMOIZE_ENABLED
    _MEMOIZE_ENABLED = enabled

def clear_schedule_caches() -> None:
    """Clear all schedule-related caches."""
    _SCHEDULE_CACHE.clear()
    _EXPAND_YM_CACHE.clear()
    _DAYS_YM_CACHE.clear()
    _STRADDLE_DAYS_CACHE.clear()



# -------------------------------------
#  AMT schedule -> by asset, packed schedule
# -------------------------------------


# Pattern to match lowercase a, b, c, d (the values that need fixing)
_ABCD_PATTERN = re.compile(r"^([abcd])$")


def _underlying_hash(underlying: str) -> int:
    """
    Compute a deterministic integer hash from an underlying string.

    Uses MD5 to get a consistent hash across runs, then takes modulo
    to get a small integer suitable for the fix calculation.

    Args:
        underlying: The underlying string (e.g., "LA Comdty")

    Returns:
        Integer hash value (0-999999)
    """
    h = hashlib.md5(underlying.encode()).hexdigest()
    return int(h[:8], 16) % 1000000


def _fix_value(value: str, assid: int, schcnt: int, schid: int) -> str:
    """Fix a/b/c/d values to computed day numbers."""
    if not _ABCD_PATTERN.match(value): return value
    if schcnt > 0:
        day_offset = int(assid % 5 + 1)
        day_stride = int(20 / (schcnt + 1))
        fixed = int(schid - 1) * day_stride + day_offset
    else:
        fixed = int(assid % 5 + 1)
    return str(fixed)


def _split_code_value(s: str) -> tuple[str, str]:
    """Split a string into uppercase alphabetic code prefix and trailing value."""
    m = re.match(r"^([A-Z]+)(.*)$", s)
    if m:return m.group(1), m.group(2)
    return s, ""


def _fix_schedule(underlying: str, schedule: list[str]) -> list[str]:
    """Fix a/b/c/d values in schedule entries for a given underlying."""
    assid = _underlying_hash(underlying)
    schcnt = len(schedule)
    fixed_schedule = []

    for schid, component in enumerate(schedule, start=1):
        parts = component.split("_")
        if len(parts) >= 2:
            entry = parts[0]
            expiry = parts[1]
            rest = parts[2:] if len(parts) > 2 else []
            # Split and fix entry value
            ntrc, ntrv = _split_code_value(entry)
            ntrv = _fix_value(ntrv, assid, schcnt, schid)
            # Split and fix expiry value
            xprc, xprv = _split_code_value(expiry)
            xprv = _fix_value(xprv, assid, schcnt, schid)
            # Reconstruct the component
            fixed_entry = f"{ntrc}{ntrv}"
            fixed_expiry = f"{xprc}{xprv}"
            fixed_parts = [fixed_entry, fixed_expiry] + rest
            fixed_schedule.append("_".join(fixed_parts))
        else:
            fixed_schedule.append(component)

    return fixed_schedule



def _schedule_to_rows(underlying: str, schedule: list[str] | None) -> list[list[Any]]:
    """Convert a schedule list to table rows with fixed values."""
    rows = []
    if isinstance(schedule, list) and schedule:
        fixed = _fix_schedule(underlying, schedule)
        schcnt = len(fixed)
        for schid, component in enumerate(fixed, start=1):
            parts = component.split("_")
            entry = parts[0] if len(parts) > 0 else ""
            expiry = parts[1] if len(parts) > 1 else ""
            wgt = parts[2] if len(parts) > 2 else ""
            ntrc, ntrv = _split_code_value(entry)
            xprc, xprv = _split_code_value(expiry)
            rows.append([schcnt, schid, underlying, ntrc, ntrv, xprc, xprv, wgt])
    else:
        rows.append([0, 0, underlying, "", "", "", "", ""])
    return rows

def get_schedule_nocache(path: str | Path, underlying: str) -> dict[str, Any]:
    data = loader.load_amt(path)
    asset_data = loader.get_asset(path, underlying)
    if not asset_data: return { "orientation": "row", "columns": _SCHEDULE_COLUMNS, "rows": [], }
    schedule_name = asset_data.get("Options")
    scheds = data.get("expiry_schedules", {})
    schedule = scheds.get(schedule_name) if schedule_name else None
    rows = _schedule_to_rows(underlying, schedule)
    return {"orientation": "row", "columns": _SCHEDULE_COLUMNS, "rows": rows,}

_SCHEDULE_COLUMNS = ["schcnt", "schid", "asset", "ntrc", "ntrv", "xprc", "xprv", "wgt"]
def get_schedule(path: str | Path, underlying: str) -> dict[str, Any]:
    """Get the expiry schedule for an asset by its Underlying value."""
    path_str = str(Path(path).resolve())
    cache_key = (path_str, underlying)
    if _MEMOIZE_ENABLED and cache_key in _SCHEDULE_CACHE: 
        return _SCHEDULE_CACHE[cache_key]
    result = get_schedule_nocache(path,underlying)
    if _MEMOIZE_ENABLED:
        _SCHEDULE_CACHE[cache_key] = result
    return result

def get_schedule_count(path: str | Path, underlying: str) -> int:
    return len(get_schedule(path,underlying)["rows"])


def find_schedules(path: str | Path, pattern: str,live_only: bool = True) -> dict[str, Any]:
    """Find assets matching a regex pattern and return their schedule components."""
    rows = []
    assets = [asset for _, asset in loader._iter_assets(path, live_only=live_only, pattern=pattern)]
    for asset in assets:
        shedule=get_schedule(path,asset)["rows"]
        rows.extend(shedule)
    return { "orientation": "row", "columns": _SCHEDULE_COLUMNS, "rows": rows, }

# -------------------------------------
# Straddle string parsing
# -------------------------------------
# Straddle format: |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|
# Example: |2023-12|2024-01|N|0|OVERRIDE||33.3|

def _parse_straddle(s: str) -> tuple[str, str, str, str, str, str, str]:
    """Parse a straddle string into its 7 components.
    Returns: (ntr, xpr, ntrc, ntrv, xprc, xprv, wgt)
    """
    if s.startswith("|") and s.endswith("|"):
        parts = s[1:-1].split("|")
    else:
        raise ValueError(f"Invalid straddle format: expect |straddle|")
    if len(parts) != 7:
        raise ValueError(f"Invalid straddle format: expected 7 parts, got {len(parts)}")
    return tuple(parts)

def ntr(s: str) -> str:
    """Extract entry date string (YYYY-MM) from straddle."""
    return s[1:8]

def ntry(s: str) -> int:
    """Extract entry year from straddle."""
    return int(s[1:5])

def ntrm(s: str) -> int:
    """Extract entry month from straddle."""
    return int(s[6:8])

def xpr(s: str) -> str:
    """Extract expiry date string (YYYY-MM) from straddle."""
    return s[9:16]

def xpry(s: str) -> int:
    """Extract expiry year from straddle."""
    return int(s[9:13])

def xprm(s: str) -> int:
    """Extract expiry month from straddle."""
    return int(s[14:16])


def ntrc(s: str) -> str:
    """Extract entry code from straddle."""
    return _parse_straddle(s)[2]


def ntrv(s: str) -> str:
    """Extract entry value from straddle."""
    return _parse_straddle(s)[3]


def xprc(s: str) -> str:
    """Extract expiry code from straddle."""
    return _parse_straddle(s)[4]


def xprv(s: str) -> str:
    """Extract expiry value from straddle."""
    return _parse_straddle(s)[5]


def wgt(s: str) -> str:
    """Extract weight from straddle."""
    return _parse_straddle(s)[6]

def _year_month_plus(year:int,month:int,offset:int):
    total_months = year * 12 + (month - 1) - offset
    new_year = total_months // 12
    new_month = (total_months % 12) + 1
    return new_year,new_month

def _schedule2straddle(
    xpry:int,
    xprm:int,
    ntrc:str,
    ntrv:str,
    xprc:str,
    xprv:str,
    wgt:float
):
    offset = {"N": 1, "F": 2}.get(ntrc, 0)
    ntry , ntrm = _year_month_plus(xpry,xprm,offset)
    straddle = (
        f"|{ntry}-{ntrm:02d}"
        f"|{xpry}-{xprm:02d}"
        f"|{ntrc}|{ntrv}"
        f"|{xprc}|{xprv}"
        f"|{wgt}|"
    )
    return straddle
 
def _schedules2straddles(table: dict[str, Any], xpry: int, xprm: int) -> list[list]:
    """pack schedules table for a specific year/month into straddle strings."""
    input_rows = table["rows"]
    cols = table["columns"]
    asset_idx = cols.index("asset")
    ntrc_idx = cols.index("ntrc")
    ntrv_idx = cols.index("ntrv")
    xprc_idx = cols.index("xprc")
    xprv_idx = cols.index("xprv")
    wgt_idx = cols.index("wgt")

    rows = []
    for asset_row in input_rows:
        asset = asset_row[asset_idx]
        straddle = _schedule2straddle(
            xpry,xprm,
            asset_row[ntrc_idx],
            asset_row[ntrv_idx],
            asset_row[xprc_idx],
            asset_row[xprv_idx],
            asset_row[wgt_idx]
        )
        rows.append([asset, straddle])

    return {"orientation": "row", "columns": ["asset", "straddle"], "rows": rows}

def _schedules2straddle_yrs(
    table: dict[str, Any],
    start_year: int,
    end_year: int
) -> dict[str, Any]:
    """Expand a schedules table across a year/month range and pack into straddle strings."""
    rows = []
    for xpry in range(start_year, end_year + 1):
        for xprm in range(1, 13):
            packed_table = _schedules2straddles(table, xpry, xprm)
            rows.extend(packed_table["rows"])
    return {"orientation": "row", "columns": ["asset", "straddle"], "rows": rows}


def get_straddle_yrs(path: str | Path, underlying: str, start_year: int, end_year: int) -> dict[str, Any]:
    """Expand a single asset's schedule across a year range into straddle strings."""
    schedule = get_schedule(path, underlying)
    straddles = _schedules2straddle_yrs(schedule, start_year, end_year)
    return straddles

def find_straddle_ym(path: str | Path, year: int, month: int,pattern: str = ".", live_only: bool = True) -> dict[str, Any]:
    """Expand live schedules for a specific year/month into straddle strings."""
    found = find_schedules( path, pattern=pattern, live_only=live_only )
    straddles = _schedules2straddles(found, year, month)
    return straddles

def find_straddle_yrs(path: str | Path, start_year: int, end_year: int, pattern: str = ".", live_only: bool = True) -> dict[str, Any]:
    """Expand all live schedules across a year/month range into straddle strings."""
    found = find_schedules(path,pattern=pattern,live_only=live_only)
    straddles =  _schedules2straddle_yrs(found, start_year, end_year)
    return  straddles




# -------------------------------------
# Expand YM cache
# -------------------------------------

def get_expand_ym(path: str | Path, underlying: str, year: int, month: int) -> dict[str, Any]:
    """Expand a single asset's schedule for a specific year/month into straddle strings."""
    path_str = str(Path(path).resolve())
    cache_key = (path_str, underlying, year, month)
    if _MEMOIZE_ENABLED and cache_key in _EXPAND_YM_CACHE:
        return _EXPAND_YM_CACHE[cache_key]

    schedule = get_schedule(path, underlying)
    straddles = _schedules2straddles(schedule, year, month)
    if _MEMOIZE_ENABLED:
        _EXPAND_YM_CACHE[cache_key] = straddles
    return straddles


# -------------------------------------
# Days-in-month cache
# -------------------------------------

def get_days_ym(year: int, month: int) -> list:
    """Return cached list of all days in a given year/month.

    Args:
        year: The year (e.g., 2024)
        month: The month (1-12)

    Returns:
        List of date objects for every day in that month
    """
    key = f"{year}-{month:02d}"
    if _MEMOIZE_ENABLED and key in _DAYS_YM_CACHE:
        return _DAYS_YM_CACHE[key]

    _, num_days = calendar.monthrange(year, month)
    days = [datetime.date(year, month, day) for day in range(1, num_days + 1)]

    if _MEMOIZE_ENABLED:
        _DAYS_YM_CACHE[key] = days
    return days


def clear_days_cache() -> None:
    """Clear the days-in-month cache."""
    _DAYS_YM_CACHE.clear()


# -------------------------------------
# year month -> year month days
# -------------------------------------

def count_straddle_days(straddle: str) -> int:
    ny = int(straddle[1:5])
    nm = int(straddle[6:8])
    xy = int(straddle[9:13])
    xm = int(straddle[14:16])
    start = datetime.date(ny, nm, 1)
    end = datetime.date(xy, xm, calendar.monthrange(xy, xm)[1])
    n = (end - start).days + 1
    return(n)


def straddle_days(straddle: str) -> list[datetime.date]:
    key=straddle[1:16]
    if _MEMOIZE_ENABLED:
        hit = _STRADDLE_DAYS_CACHE.get(key)
        if hit is not None:
            return hit

    sy = int(straddle[1:5])
    sm = int(straddle[6:8])
    ey = int(straddle[9:13])
    em = int(straddle[14:16])

    start = datetime.date(sy, sm, 1)
    end = datetime.date(ey, em, calendar.monthrange(ey, em)[1])

    n = (end - start).days + 1
    base = start.toordinal()
    days = [datetime.date.fromordinal(base + i) for i in range(n)]

    if _MEMOIZE_ENABLED:
        _STRADDLE_DAYS_CACHE[key] = days
    return days

def count_straddles_days(
    straddles: dict
):
    cols = straddles["columns"]
    straddle_idx = cols.index("straddle")
    days=0
    for row in straddles["rows"]:
        days=days+count_straddle_days(row[straddle_idx])
    return days
    
# cols = list(map(list, zip(*rows)))
# rows = list(map(list,zip(*cols)))


# -------------------------------------
# Arrow-based straddle date expansion
# -------------------------------------
# These functions provide a vectorized implementation using PyArrow.
# The calendar is built once and cached, then used for fast lookups.

# Cache for month calendar: (start_year, end_year, calendar_table)
_MONTH_CALENDAR_CACHE: tuple[int, int, dict[str, Any]] | None = None

# Cache for NTRC calendar: (start_year, end_year, calendar_table)
_NTRC_CALENDAR_CACHE: tuple[int, int, dict[str, Any]] | None = None


def clear_calendar_cache() -> None:
    """Clear the cached calendar tables."""
    global _MONTH_CALENDAR_CACHE, _NTRC_CALENDAR_CACHE
    _MONTH_CALENDAR_CACHE = None
    _NTRC_CALENDAR_CACHE = None


def _build_month_calendar_arrow(start_year: int, end_year: int) -> dict[str, Any]:
    """Build calendar table mapping yearmonth to date list for that month.

    Returns:
        Arrow table with columns ["ym", "dates"]
        - ym: "2024-01", "2024-02", etc.
        - dates: list of dates for that month (pa.list_(pa.date32()))
    """
    from .table import _import_pyarrow

    pa, _ = _import_pyarrow()

    yms = []
    date_lists = []

    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            ym = f"{y}-{m:02d}"
            _, last_day = calendar.monthrange(y, m)
            dates = [datetime.date(y, m, d) for d in range(1, last_day + 1)]
            yms.append(ym)
            date_lists.append(dates)

    return {
        "orientation": "arrow",
        "columns": ["ym", "dates"],
        "rows": [
            pa.array(yms),
            pa.array(date_lists, type=pa.list_(pa.date32())),
        ],
    }


def _get_month_calendar(start_year: int, end_year: int) -> dict[str, Any]:
    """Get or build the month calendar, expanding range if needed.

    The cache tracks the year range. If a wider range is requested,
    the calendar is rebuilt to cover it.
    """
    global _MONTH_CALENDAR_CACHE

    if _MONTH_CALENDAR_CACHE is not None:
        cached_start, cached_end, cached_cal = _MONTH_CALENDAR_CACHE
        if cached_start <= start_year and cached_end >= end_year:
            return cached_cal

    # Build new calendar (either first time or range expanded)
    # Pad range slightly to reduce rebuilds for small range changes
    padded_start = start_year - 1
    padded_end = end_year + 2
    cal = _build_month_calendar_arrow(padded_start, padded_end)
    _MONTH_CALENDAR_CACHE = (padded_start, padded_end, cal)
    return cal


def _build_ntrc_calendar_from_months(month_cal: dict[str, Any]) -> dict[str, Any]:
    """Build NTRC-keyed calendar by concatenating month date lists.

    Takes base month calendar and creates span-keyed entries:
    - "2024-01-2" → concat(dates["2024-01"], dates["2024-02"])
    - "2024-01-3" → concat(dates["2024-01"], dates["2024-02"], dates["2024-03"])

    Returns:
        Arrow table with columns ["key", "dates"]
    """
    from .table import _import_pyarrow

    pa, _ = _import_pyarrow()

    # Build lookup: ym -> dates array
    yms = month_cal["rows"][0].to_pylist()
    dates_col = month_cal["rows"][1]
    ym_to_idx = {ym: i for i, ym in enumerate(yms)}

    def next_ym(ym: str) -> str:
        y, m = int(ym[:4]), int(ym[5:7])
        m += 1
        if m > 12:
            m, y = 1, y + 1
        return f"{y}-{m:02d}"

    keys = []
    date_lists = []

    for ym in yms:
        for span in (2, 3):
            # Collect months for this span
            months_to_concat = [ym]
            curr = ym
            for _ in range(span - 1):
                curr = next_ym(curr)
                if curr in ym_to_idx:
                    months_to_concat.append(curr)

            # Skip if we don't have all months (edge of range)
            if len(months_to_concat) < span:
                continue

            # Concatenate date arrays for these months
            # .values gets the underlying flat array from the list scalar
            arrays_to_concat = [
                dates_col[ym_to_idx[m]].values
                for m in months_to_concat
            ]
            combined = pa.concat_arrays(arrays_to_concat)

            keys.append(f"{ym}-{span}")
            date_lists.append(combined)

    return {
        "orientation": "arrow",
        "columns": ["key", "dates"],
        "rows": [
            pa.array(keys),
            pa.array(date_lists, type=pa.list_(pa.date32())),
        ],
    }


def _get_ntrc_calendar(start_year: int, end_year: int) -> dict[str, Any]:
    """Get or build the NTRC-keyed calendar, with caching."""
    global _NTRC_CALENDAR_CACHE

    if _NTRC_CALENDAR_CACHE is not None:
        cached_start, cached_end, cached_cal = _NTRC_CALENDAR_CACHE
        if cached_start <= start_year and cached_end >= end_year:
            return cached_cal

    # Build from month calendar
    month_cal = _get_month_calendar(start_year, end_year)
    ntrc_cal = _build_ntrc_calendar_from_months(month_cal)

    # Cache with same range as month calendar
    if _MONTH_CALENDAR_CACHE is not None:
        cached_start, cached_end, _ = _MONTH_CALENDAR_CACHE
        _NTRC_CALENDAR_CACHE = (cached_start, cached_end, ntrc_cal)

    return ntrc_cal


def _ntrc_to_span(ntrc) -> Any:
    """Map NTRC values to span lengths, with validation.

    Args:
        ntrc: Arrow array of NTRC values ("N" or "F")

    Returns:
        Arrow array of span strings ("2" or "3")

    Raises:
        ValueError: If any NTRC value is not "N" or "F"
    """
    from .table import _import_pyarrow

    _, pc = _import_pyarrow()

    is_n = pc.equal(ntrc, "N")
    is_f = pc.equal(ntrc, "F")
    is_valid = pc.or_(is_n, is_f)

    if not pc.all(is_valid).as_py():
        # Find first invalid value for error message
        invalid_mask = pc.invert(is_valid)
        invalid_indices = pc.indices_nonzero(invalid_mask)
        if len(invalid_indices) > 0:
            first_invalid = ntrc[invalid_indices[0].as_py()].as_py()
            raise ValueError(f"Invalid NTRC value: {first_invalid!r} (expected 'N' or 'F')")

    return pc.if_else(is_n, "2", "3")


def find_straddle_days(
    path: str | Path,
    start_year: int,
    end_year: int,
    pattern: str = ".",
    live_only: bool = True,
) -> dict[str, Any]:
    """Expand straddles to daily rows.

    Uses optimized Python loop with memoized date generation. For 99.8%+ cache
    hit ratios typical of production data, this outperforms Arrow vectorization
    due to lower conversion overhead.

    Args:
        path: Path to AMT YAML file
        start_year: Start year for straddles
        end_year: End year for straddles
        pattern: Regex pattern to filter assets
        live_only: Only include live straddles

    Returns:
        Column-oriented table with columns ["asset", "straddle", "date"]
    """
    straddles = find_straddle_yrs(path, start_year, end_year, pattern, live_only)
    cols = straddles["columns"]
    asset_idx = cols.index("asset")
    straddle_idx = cols.index("straddle")
    asset_col = []
    straddle_col = []
    date_col = []

    for row in straddles["rows"]:
        asset = row[asset_idx]
        straddle = row[straddle_idx]
        days = straddle_days(straddle)
        n = len(days)
        asset_col.extend(itertools.repeat(asset, n))
        straddle_col.extend(itertools.repeat(straddle, n))
        date_col.extend(days)

    return {
        "orientation": "column",
        "columns": ["asset", "straddle", "date"],
        "rows": [asset_col, straddle_col, date_col]
    }


def find_straddle_days_arrow(
    path: str | Path,
    start_year: int,
    end_year: int,
    pattern: str = ".",
    live_only: bool = True,
    validate_ntrc: bool = True,
) -> dict[str, Any]:
    """Expand straddles to daily rows using Arrow operations.

    This is a vectorized implementation that uses:
    - Arrow string slicing to extract entry yearmonth and NTRC
    - A precomputed calendar table for date lookups
    - Arrow-native list explosion for expanding dates

    Note: For datasets where most straddles share date ranges (high cache hit ratio),
    find_straddle_days() is faster due to lower conversion overhead. Use this function
    when you need Arrow-format output or when processing very large datasets where
    vectorization benefits outweigh conversion costs.

    Args:
        path: Path to AMT YAML file
        start_year: Start year for straddles
        end_year: End year for straddles
        pattern: Regex pattern to filter assets
        live_only: Only include live straddles
        validate_ntrc: If True, validate NTRC values are 'N' or 'F'

    Returns:
        Arrow-oriented table with columns ["asset", "straddle", "date"]
    """
    from .table import (
        table_to_arrow, table_explode_arrow, table_select_columns,
        table_nrows, _import_pyarrow
    )

    pa, pc = _import_pyarrow()

    # Step 1: Get straddles table and convert to arrow
    straddles = find_straddle_yrs(path, start_year, end_year, pattern, live_only)
    straddles_arrow = table_to_arrow(straddles)

    if table_nrows(straddles_arrow) == 0:
        # Return empty table with correct schema
        return {
            "orientation": "arrow",
            "columns": ["asset", "straddle", "date"],
            "rows": [pa.array([]), pa.array([]), pa.array([], type=pa.date32())],
        }

    # Step 2: Extract entry_ym and ntrc using vectorized string slicing
    straddle_idx = straddles_arrow["columns"].index("straddle")
    straddle_col = straddles_arrow["rows"][straddle_idx]

    # Straddle format: |2024-01|2024-03|N|5|F||33.3|
    #                   ^      ^       ^
    #                   1-8    9-16    17 (NTRC: N=2months, F=3months)
    entry_ym = pc.utf8_slice_codeunits(straddle_col, 1, 8)    # "YYYY-MM"
    ntrc = pc.utf8_slice_codeunits(straddle_col, 17, 18)      # "N" or "F"

    # Step 3: Map NTRC to span with validation
    if validate_ntrc:
        span = _ntrc_to_span(ntrc)
    else:
        # Fast path without validation (use with caution)
        span = pc.if_else(pc.equal(ntrc, "N"), "2", "3")

    # Step 4: Build lookup key: "YYYY-MM-2" for N, "YYYY-MM-3" for F
    cal_key = pc.binary_join_element_wise(entry_ym, span, "-")

    # Step 5: Lookup dates using index_in + take (faster than join for small calendar)
    ntrc_cal = _get_ntrc_calendar(start_year - 1, end_year + 2)

    cal_keys = ntrc_cal["rows"][ntrc_cal["columns"].index("key")]
    cal_dates = ntrc_cal["rows"][ntrc_cal["columns"].index("dates")]

    # index_in returns index into cal_keys for each cal_key, or null if not found
    indices = pc.index_in(cal_key, value_set=cal_keys)
    dates = pc.take(cal_dates, indices)

    # Verify no missing calendar entries
    if pc.any(pc.is_null(dates)).as_py():
        # Find first missing key for error message
        null_mask = pc.is_null(dates)
        null_indices = pc.indices_nonzero(null_mask)
        if len(null_indices) > 0:
            missing_key = cal_key[null_indices[0].as_py()].as_py()
            raise ValueError(f"Calendar missing entry for key: {missing_key!r}")

    # Step 6: Add dates column to straddles table
    straddles_with_dates = {
        "orientation": "arrow",
        "columns": straddles_arrow["columns"] + ["dates"],
        "rows": straddles_arrow["rows"] + [dates],
    }

    # Step 7: Explode dates list → one row per date per straddle
    exploded = table_explode_arrow(straddles_with_dates, "dates")
    # Rename "dates" to "date" (now single values)
    exploded["columns"] = ["date" if c == "dates" else c for c in exploded["columns"]]

    # Step 8: Select final columns
    result = table_select_columns(exploded, ["asset", "straddle", "date"])

    return result


def find_straddle_days_numba(
    path: str | Path,
    start_year: int,
    end_year: int,
    pattern: str = ".",
    live_only: bool = True,
    parallel: bool = False,
) -> dict[str, Any]:
    """Expand straddles to daily rows using Numba-accelerated kernel.

    This is the fastest implementation for large datasets. It uses:
    - Numba JIT-compiled date expansion kernel
    - Howard Hinnant's O(1) date conversion algorithm
    - Optional parallel execution for millions+ output rows

    Performance characteristics:
    - 10-100x faster than Python loops
    - 2-10x faster than Arrow calendar lookup approach
    - First call has JIT compilation overhead (~100-200ms, cached thereafter)

    Args:
        path: Path to AMT YAML file
        start_year: Start year for straddles
        end_year: End year for straddles
        pattern: Regex pattern to filter assets
        live_only: Only include live straddles
        parallel: If True, use parallel version (best for millions+ output rows)

    Returns:
        Arrow-oriented table with columns ["asset", "straddle", "date"]

    Raises:
        ImportError: If Numba is not installed
    """
    

    # Step 1: Get straddles table and convert to arrow
    straddles = find_straddle_yrs(path, start_year, end_year, pattern, live_only)
    straddles_arrow = table_to_arrow(straddles)

    n_straddles = table_nrows(straddles_arrow)
    if n_straddles == 0:
        # Return empty table with correct schema
        return {
            "orientation": "arrow",
            "columns": ["asset", "straddle", "date"],
            "rows": [pa.array([]), pa.array([]), pa.array([], type=pa.date32())],
        }

    # Step 2: Extract entry_year, entry_month, ntrc from straddle strings
    straddle_idx = straddles_arrow["columns"].index("straddle")
    straddle_col = straddles_arrow["rows"][straddle_idx]

    # Vectorized string slicing to get components
    # Straddle format: |2024-01|2024-03|N|5|F||33.3|
    #                   ^      ^       ^
    #                   1-8    9-16    17
    entry_ym = pc.utf8_slice_codeunits(straddle_col, 1, 8)  # "YYYY-MM"
    ntrc = pc.utf8_slice_codeunits(straddle_col, 17, 18)  # "N" or "F"

    # Parse year/month from "YYYY-MM" strings
    entry_year_str = pc.utf8_slice_codeunits(entry_ym, 0, 4)
    entry_month_str = pc.utf8_slice_codeunits(entry_ym, 5, 7)

    # Convert to numpy int32 arrays
    year_np = np.asarray(
        pc.cast(entry_year_str, pa.int32()).to_numpy(), dtype=np.int32
    )
    month_np = np.asarray(
        pc.cast(entry_month_str, pa.int32()).to_numpy(), dtype=np.int32
    )

    # Map NTRC to span: N→2, F→3
    ntrc_np = ntrc.to_numpy(zero_copy_only=False)
    # Handle both bytes and str depending on Arrow version
    if len(ntrc_np) > 0:
        if isinstance(ntrc_np[0], bytes):
            month_count_np = np.where(ntrc_np == b"N", 2, 3).astype(np.int32)
        else:
            month_count_np = np.where(ntrc_np == "N", 2, 3).astype(np.int32)
    else:
        month_count_np = np.array([], dtype=np.int32)

    # Step 3: Run Numba kernel
    if parallel:
        date32, parent_idx = _numba_kernels.expand_months_to_date32_parallel(
            year_np, month_np, month_count_np
        )
    else:
        date32, parent_idx = _numba_kernels.expand_months_to_date32(
            year_np, month_np, month_count_np
        )

    # Step 4: Build output table by gathering from source arrays using parent_idx
    asset_idx = straddles_arrow["columns"].index("asset")
    asset_col = straddles_arrow["rows"][asset_idx]

    # Use parent_idx to repeat asset/straddle values
    parent_arr = pa.array(parent_idx)
    out_asset = pc.take(asset_col, parent_arr)
    out_straddle = pc.take(straddle_col, parent_arr)

    # date32 output is already Arrow date32 compatible (days since epoch)
    out_date = pa.array(date32, type=pa.date32())

    return {
        "orientation": "arrow",
        "columns": ["asset", "straddle", "date"],
        "rows": [out_asset, out_straddle, out_date],
    }


# Keep for backwards compatibility (alias to Arrow version)
_find_straddle_days_legacy = find_straddle_days

# -------------------------------------
# CLI
# -------------------------------------


def _main() -> int:
    import argparse
    from .loader import print_table

    p = argparse.ArgumentParser(
        description="Schedule expansion and straddle building utilities.",
    )
    p.add_argument("path", help="Path to AMT YAML file")

    # Commands
    p.add_argument("--get", "-g", metavar="UNDERLYING",
                   help="Get expiry schedule for asset by Underlying value")
    p.add_argument("--find", "-f", metavar="PATTERN",
                   help="Find schedules by regex pattern on Underlying")
    p.add_argument("--live", "-l", action="store_true",
                   help="List all live assets with their schedules")
    p.add_argument("--expand", "-e", nargs=2, type=int, metavar=("START_YEAR", "END_YEAR"),
                   help="Expand live schedules into straddle strings")
    p.add_argument("--expand-ym", nargs=2, type=int, metavar=("YEAR", "MONTH"),
                   help="Expand live schedules for a specific year/month")
    p.add_argument("--get-expand", nargs=3, metavar=("UNDERLYING", "START_YEAR", "END_YEAR"),
                   help="Expand a single asset's schedule into straddle strings")
    p.add_argument("--get-expand-ym", nargs=3, metavar=("UNDERLYING", "YEAR", "MONTH"),
                   help="Expand a single asset's schedule for a specific year/month")

    args = p.parse_args()

    if args.get:
        table = get_schedule(args.path, args.get)
        if not table["rows"]:
            print(f"No schedule found for: {args.get}")
            return 1
        print_table(table)

    elif args.find:
        table = find_schedules(args.path, args.find)
        if not table["rows"]:
            print(f"No assets found matching: {args.find}")
            return 1
        print_table(table)

    elif args.live:
        table = find_schedules(args.path, ".", live_only=True)
        print_table(table)

    elif args.expand:
        start_year, end_year = args.expand
        table = find_straddle_yrs(args.path, start_year, end_year)
        print_table(table)

    elif args.expand_ym:
        year, month = args.expand_ym
        table = find_straddle_ym(args.path, year, month)
        print_table(table)

    elif args.get_expand:
        underlying, start_year, end_year = args.get_expand
        table = get_straddle_yrs(args.path, underlying, int(start_year), int(end_year))
        if not table["rows"]:
            print(f"No schedule found for: {underlying}")
            return 1
        print_table(table)

    elif args.get_expand_ym:
        underlying, year, month = args.get_expand_ym
        table = get_expand_ym(args.path, underlying, int(year), int(month))
        if not table["rows"]:
            print(f"No schedule found for: {underlying}")
            return 1
        print_table(table)

    else:
        p.print_help()

    return 0


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(_main())
