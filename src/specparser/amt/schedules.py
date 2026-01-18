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
from typing import Any

from .loader import load_amt, get_asset, find_underlyings, _iter_assets


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
    """
    Fix a/b/c/d values to computed day numbers.

    If value is 'a', 'b', 'c', or 'd', compute:
        (schedule_id - 1) * day_stride + day_offset
    where:
        day_offset = asset_id % 5 + 1
        day_stride = 20 / (schedule_count + 1)

    Args:
        value: The value to potentially fix (e.g., "5", "a", "b")
        assid: Asset ID (hash of underlying)
        schcnt: Schedule count (number of schedule components)
        schid: Schedule ID (1-based index of this component)

    Returns:
        Fixed value as string, or original value if not a/b/c/d
    """
    if not _ABCD_PATTERN.match(value):
        return value

    if schcnt > 0:
        day_offset = int(assid % 5 + 1)
        day_stride = int(20 / (schcnt + 1))
        fixed = int(schid - 1) * day_stride + day_offset
    else:
        fixed = int(assid % 5 + 1)

    return str(fixed)


def _split_code_value(s: str) -> tuple[str, str]:
    """
    Split a string into uppercase alphabetic code prefix and trailing value.

    The code is the leading uppercase letters. The value is everything after
    (digits or lowercase letters like a,b,c,d for fix_expiry).

    Examples:
        'N5' -> ('N', '5')
        'BD15' -> ('BD', '15')
        'LBD' -> ('LBD', '')
        'F3' -> ('F', '3')
        'BDa' -> ('BD', 'a')
        'BDb' -> ('BD', 'b')
    """
    m = re.match(r"^([A-Z]+)(.*)$", s)
    if m:
        return m.group(1), m.group(2)
    return s, ""


def _fix_schedule(underlying: str, schedule: list[str]) -> list[str]:
    """
    Fix a/b/c/d values in schedule entries for a given underlying.

    Takes a list of schedule components (e.g., ['Na_BDa_33.3', 'Nb_BDb_33.3'])
    and replaces a/b/c/d values with computed day numbers based on the underlying.

    Args:
        underlying: The underlying string (e.g., "LA Comdty")
        schedule: List of schedule components to fix

    Returns:
        List of fixed schedule components

    Example:
        >>> _fix_schedule("LA Comdty", ['Na_BDa_33.3', 'Nb_BDb_33.3'])
        ['N1_BD1_33.3', 'N5_BD5_33.3']
    """
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


def get_schedule(path: str | Path, underlying: str) -> list[str] | None:
    """
    Get the expiry schedule for an asset by its Underlying value.

    Looks up the asset's 'Options' field to get the schedule name,
    then returns the schedule from 'expiry_schedules'. Values like
    'a', 'b', 'c', 'd' are automatically fixed to computed day numbers.

    Args:
        path: Path to the AMT YAML file
        underlying: The Underlying value to search for

    Returns:
        List of schedule entries (with fixed values), or None if asset or schedule not found

    Example:
        >>> get_schedule("data/amt.yml", "LA Comdty")
        ['N0_OVERRIDE_33.3', 'N5_OVERRIDE_33.3', 'F10_OVERRIDE_12.5', 'F15_OVERRIDE_12.5']
    """
    asset = get_asset(path, underlying)
    if not asset:
        return None

    schedule_name = asset.get("Options")
    if not schedule_name:
        return None

    data = load_amt(path)
    schedules = data.get("expiry_schedules", {})
    schedule = schedules.get(schedule_name)

    if not isinstance(schedule, list):
        return None

    return _fix_schedule(underlying, schedule)


def _schedule_to_rows(underlying: str, schedule: list[str] | None) -> list[list[Any]]:
    """
    Convert a schedule list to table rows with fixed values.

    Args:
        underlying: The underlying string
        schedule: List of schedule components, or None

    Returns:
        List of rows for the schedule table
    """
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


def find_schedules(path: str | Path, pattern: str) -> dict[str, Any]:
    """
    Find assets matching a regex pattern and return their schedule components.

    Each schedule component is expanded to its own row, with the component split
    on '_' into entry code/value and expiry code/value columns. Values like
    'a', 'b', 'c', 'd' are automatically fixed to computed day numbers.

    Args:
        path: Path to the AMT YAML file
        pattern: Regex pattern to match against Underlying values

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['schcnt', 'schid', 'asset', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']

    Example:
        >>> table = find_schedules("data/amt.yml", "^LA.*")
        >>> table['columns']
        ['schcnt', 'schid', 'asset', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']
    """
    data = load_amt(path)
    schedules = data.get("expiry_schedules", {})
    underlyings = find_underlyings(path, pattern)

    rows = []
    for underlying in underlyings:
        asset_data = get_asset(path, underlying)
        if not asset_data:
            continue
        schedule_name = asset_data.get("Options")
        schedule = schedules.get(schedule_name) if schedule_name else None
        rows.extend(_schedule_to_rows(underlying, schedule))

    return {
        "columns": ["schcnt", "schid", "asset", "ntrc", "ntrv", "xprc", "xprv", "wgt"],
        "rows": rows,
    }


def live_schedules(path: str | Path) -> dict[str, Any]:
    """
    Get all live assets (WeightCap > 0) with their schedule components.

    Each schedule component is expanded to its own row, with the component split
    on '_' into entry code/value and expiry code/value columns. Values like
    'a', 'b', 'c', 'd' are automatically fixed to computed day numbers.

    Args:
        path: Path to the AMT YAML file

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['schcnt', 'schid', 'asset', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']

    Example:
        >>> table = live_schedules("data/amt.yml")
        >>> table['columns']
        ['schcnt', 'schid', 'asset', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']
    """
    data = load_amt(path)
    schedules = data.get("expiry_schedules", {})

    rows = []
    for _, _, asset_data, underlying, _ in _iter_assets(path, live_only=True):
        schedule_name = asset_data.get("Options")
        schedule = schedules.get(schedule_name) if schedule_name else None
        rows.extend(_schedule_to_rows(underlying, schedule))

    return {
        "columns": ["schcnt", "schid", "asset", "ntrc", "ntrv", "xprc", "xprv", "wgt"],
        "rows": rows,
    }


def _expand_and_pack(table: dict[str, Any], start_year: int, end_year: int) -> dict[str, Any]:
    """
    Expand a schedules table across a year/month range and pack into straddle strings.

    Computes the cartesian product of:
    - years: start_year to end_year (inclusive)
    - months: 1 to 12
    - rows from the input table

    Then packs each row into a pipe-delimited straddle string.

    Args:
        table: Dict with 'columns' and 'rows' from live_schedules() or find_schedules()
        start_year: Start year (inclusive)
        end_year: End year (inclusive)

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['asset', 'straddle']
    """
    asset_rows = table["rows"]
    cols = table["columns"]
    asset_idx = cols.index("asset")
    ntrc_idx = cols.index("ntrc")
    ntrv_idx = cols.index("ntrv")
    xprc_idx = cols.index("xprc")
    xprv_idx = cols.index("xprv")
    wgt_idx = cols.index("wgt")

    rows = []
    for xpry in range(start_year, end_year + 1):
        for xprm in range(1, 13):
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

    return {
        "columns": ["asset", "straddle"],
        "rows": rows,
    }


def expand(path: str | Path, start_year: int, end_year: int) -> dict[str, Any]:
    """
    Expand all live schedules across a year/month range into straddle strings.

    Computes the cartesian product of:
    - years: start_year to end_year (inclusive)
    - months: 1 to 12
    - rows from live_schedules() (already expanded by schedule component)

    Each row is packed into a pipe-delimited straddle string:
        |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|

    Args:
        path: Path to the AMT YAML file
        start_year: Start year (inclusive)
        end_year: End year (inclusive)

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['asset', 'straddle']

    Example:
        >>> table = expand("data/amt.yml", 2024, 2025)
        >>> table['columns']
        ['asset', 'straddle']
    """
    live = live_schedules(path)
    return _expand_and_pack(live, start_year, end_year)


def find_expand(path: str | Path, pattern: str, start_year: int, end_year: int) -> dict[str, Any]:
    """
    Expand schedules matching a regex pattern across a year/month range into straddle strings.

    Computes the cartesian product of:
    - years: start_year to end_year (inclusive)
    - months: 1 to 12
    - rows from find_schedules() for assets matching the pattern

    Each row is packed into a pipe-delimited straddle string:
        |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|

    Args:
        path: Path to the AMT YAML file
        pattern: Regex pattern to match against Underlying values
        start_year: Start year (inclusive)
        end_year: End year (inclusive)

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['asset', 'straddle']

    Example:
        >>> table = find_expand("data/amt.yml", "^LA.*", 2024, 2025)
        >>> table['columns']
        ['asset', 'straddle']
    """
    found = find_schedules(path, pattern)
    return _expand_and_pack(found, start_year, end_year)
