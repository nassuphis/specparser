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
from . import loader 


# memoization 

_MEMOIZE_ENABLED: bool = False

def set_memoize_enabled(enabled: bool) -> None:
    """Enable or disable memoization for schedule functions."""
    global _MEMOIZE_ENABLED
    _MEMOIZE_ENABLED = enabled

def clear_schedule_caches() -> None:
    """Clear all schedule-related caches."""
    _SCHEDULE_CACHE.clear()
    _EXPAND_YM_CACHE.clear()
    _DAYS_YM_CACHE.clear()



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




_SCHEDULE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_SCHEDULE_COLUMNS = ["schcnt", "schid", "asset", "ntrc", "ntrv", "xprc", "xprv", "wgt"]
def get_schedule(path: str | Path, underlying: str) -> dict[str, Any]:
    """Get the expiry schedule for an asset by its Underlying value."""
    path_str = str(Path(path).resolve())
    cache_key = (path_str, underlying)
    if _MEMOIZE_ENABLED and cache_key in _SCHEDULE_CACHE: 
        return _SCHEDULE_CACHE[cache_key]

    data = loader.load_amt(path)
    asset_data = loader.get_asset(path, underlying)
    columns = _SCHEDULE_COLUMNS
    if not asset_data:
        result = { "columns": columns, "rows": [], }
        if _MEMOIZE_ENABLED:
            _SCHEDULE_CACHE[cache_key] = result
        return result
    schedule_name = asset_data.get("Options")
    scheds = data.get("expiry_schedules", {})
    schedule = scheds.get(schedule_name) if schedule_name else None
    rows = _schedule_to_rows(underlying, schedule)
    result = { "columns": columns, "rows": rows, }
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
    return { "columns": _SCHEDULE_COLUMNS, "rows": rows, }

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


def _pack_ym(table: dict[str, Any], xpry: int, xprm: int) -> list[list]:
    """pack schedules table for a specific year/month into straddle strings."""
    asset_rows = table["rows"]
    cols = table["columns"]
    asset_idx = cols.index("asset")
    ntrc_idx = cols.index("ntrc")
    ntrv_idx = cols.index("ntrv")
    xprc_idx = cols.index("xprc")
    xprv_idx = cols.index("xprv")
    wgt_idx = cols.index("wgt")

    rows = []
    for asset_row in asset_rows:
        asset = asset_row[asset_idx]
        ntrc = asset_row[ntrc_idx]
        ntrv = asset_row[ntrv_idx]
        xprc = asset_row[xprc_idx]
        xprv = asset_row[xprv_idx]
        wgt = asset_row[wgt_idx]

        # Compute entry year/month based on Near (N) or Far (F)
        if ntrc == "N":
            offset = 1
        elif ntrc == "F":
            offset = 2
        else:
            offset = 0
        total_months = xpry * 12 + (xprm - 1) - offset
        ntry = total_months // 12
        ntrm = (total_months % 12) + 1

        # Format: |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|
        straddle = f"|{ntry}-{ntrm:02d}|{xpry}-{xprm:02d}|{ntrc}|{ntrv}|{xprc}|{xprv}|{wgt}|"
        rows.append([asset, straddle])

    return {"columns": ["asset", "straddle"], "rows": rows}


def _expand_and_pack(
    table: dict[str, Any], 
    start_year: int, 
    end_year: int
) -> dict[str, Any]:
    """Expand a schedules table across a year/month range and pack into straddle strings."""
    rows = []
    for xpry in range(start_year, end_year + 1):
        for xprm in range(1, 13):
            packed_table = _pack_ym(table, xpry, xprm)
            rows.extend(packed_table["rows"])
    return {"columns": ["asset", "straddle"], "rows": rows}


def expand(path: str | Path, start_year: int, end_year: int, pattern: str = ".", live_only: bool = True) -> dict[str, Any]:
    """Expand all live schedules across a year/month range into straddle strings."""
    found = find_schedules(path,pattern=pattern,live_only=live_only)
    straddles =  _expand_and_pack(found, start_year, end_year)
    return  straddles

def expand_ym(path: str | Path, year: int, month: int,pattern: str = ".", live_only: bool = True) -> dict[str, Any]:
    """Expand live schedules for a specific year/month into straddle strings."""
    found = find_schedules( path, pattern=pattern, live_only=live_only )
    straddles = _pack_ym(found, year, month)
    return straddles

def get_expand(path: str | Path, underlying: str, start_year: int, end_year: int) -> dict[str, Any]:
    """Expand a single asset's schedule across a year range into straddle strings."""
    schedule = get_schedule(path, underlying)
    straddles = _expand_and_pack(schedule, start_year, end_year)
    return straddles


# -------------------------------------
# Expand YM cache
# -------------------------------------

_EXPAND_YM_CACHE: dict[tuple[str, str, int, int], dict[str, Any]] = {}
def get_expand_ym(path: str | Path, underlying: str, year: int, month: int) -> dict[str, Any]:
    """Expand a single asset's schedule for a specific year/month into straddle strings."""
    path_str = str(Path(path).resolve())
    cache_key = (path_str, underlying, year, month)
    if _MEMOIZE_ENABLED and cache_key in _EXPAND_YM_CACHE:
        return _EXPAND_YM_CACHE[cache_key]

    schedule = get_schedule(path, underlying)
    straddles = _pack_ym(schedule, year, month)
    if _MEMOIZE_ENABLED:
        _EXPAND_YM_CACHE[cache_key] = straddles
    return straddles


# -------------------------------------
# Days-in-month cache
# -------------------------------------

_DAYS_YM_CACHE: dict[str, list] = {}


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


def year_month_days(straddle) -> List[str]:
    """
    Return all calendar days from the 1st of (start_year,start_month)
    through the last day of (end_year,end_month), inclusive.
    """
    start_year=int(straddle[1:5])
    start_month=int(straddle[6:8])
    end_year=int(straddle[9:13])
    end_month=int(straddle[14:16])
    if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
        raise ValueError("month must be in 1..12")
    key = f"{start_year}-{start_month:02d}:{end_year}-{end_month:02d}"
    if _MEMOIZE_ENABLED and key in _DAYS_YM_CACHE:
        return _DAYS_YM_CACHE[key]

    start = datetime.date(start_year, start_month, 1)
    last_day = calendar.monthrange(end_year, end_month)[1]
    end = datetime.date(end_year, end_month, last_day)
    if start > end: raise ValueError("start must be <= end")

    n_days = (end - start).days + 1
    days = [start.fromordinal(start.toordinal() + i) for i in range(n_days)]
    if _MEMOIZE_ENABLED:
        _DAYS_YM_CACHE[key] = days

    return days


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
        table = expand(args.path, start_year, end_year)
        print_table(table)

    elif args.expand_ym:
        year, month = args.expand_ym
        table = expand_ym(args.path, year, month)
        print_table(table)

    elif args.get_expand:
        underlying, start_year, end_year = args.get_expand
        table = get_expand(args.path, underlying, int(start_year), int(end_year))
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
