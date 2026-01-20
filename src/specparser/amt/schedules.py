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
from . import loader 

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

def get_schedule(path: str | Path, underlying: str) -> dict[str, Any]:
    """Get the expiry schedule for an asset by its Underlying value."""
    data = loader.load_amt(path)
    asset_data = loader.get_asset(path, underlying)
    columns = ["schcnt", "schid", "asset", "ntrc", "ntrv", "xprc", "xprv", "wgt"]
    if not asset_data: return { "columns": columns, "rows": [], }
    schedule_name = asset_data.get("Options")
    schedules = data.get("expiry_schedules", {})
    schedule = schedules.get(schedule_name) if schedule_name else None
    rows = _schedule_to_rows(underlying, schedule)
    return { "columns": columns, "rows": rows, }


def find_schedules(path: str | Path, pattern: str,live_only: bool = True) -> dict[str, Any]:
    """Find assets matching a regex pattern and return their schedule components."""
    amt = loader.load_amt(path)
    schedules = amt.get("expiry_schedules", {})
    columns = ["schcnt", "schid", "asset", "ntrc", "ntrv", "xprc", "xprv", "wgt"]
    rows = []
    for asset_data, underlying in loader._iter_assets(path,live_only=live_only,pattern=pattern):
        if not asset_data: continue
        schedule_name = asset_data.get("Options")
        schedule = schedules.get(schedule_name) if schedule_name else None
        rows.extend(_schedule_to_rows(underlying, schedule))
    return { "columns": columns, "rows": rows, }


# -------------------------------------
# Straddle string parsing
# -------------------------------------
# Straddle format: |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|
# Example: |2023-12|2024-01|N|0|OVERRIDE||33.3|

def _parse_straddle(s: str) -> tuple[str, str, str, str, str, str, str]:
    """Parse a straddle string into its 7 components.

    Returns: (ntr, xpr, ntrc, ntrv, xprc, xprv, wgt)
    """
    # Remove only the leading and trailing pipe, preserving empty internal parts
    # Using s[1:-1] instead of strip("|") to keep empty fields like ||
    if s.startswith("|") and s.endswith("|"):
        parts = s[1:-1].split("|")
    else:
        parts = s.split("|")
    if len(parts) != 7:
        raise ValueError(f"Invalid straddle format: expected 7 parts, got {len(parts)}")
    return tuple(parts)


def ntr(s: str) -> str:
    """Extract entry date string (YYYY-MM) from straddle."""
    return _parse_straddle(s)[0]


def ntry(s: str) -> int:
    """Extract entry year from straddle."""
    return int(_parse_straddle(s)[0].split("-")[0])


def ntrm(s: str) -> int:
    """Extract entry month from straddle."""
    return int(_parse_straddle(s)[0].split("-")[1])


def xpr(s: str) -> str:
    """Extract expiry date string (YYYY-MM) from straddle."""
    return _parse_straddle(s)[1]


def xpry(s: str) -> int:
    """Extract expiry year from straddle."""
    return int(_parse_straddle(s)[1].split("-")[0])


def xprm(s: str) -> int:
    """Extract expiry month from straddle."""
    return int(_parse_straddle(s)[1].split("-")[1])


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

def get_expand_ym(path: str | Path, underlying: str, year: int, month: int) -> dict[str, Any]:
    """Expand a single asset's schedule for a specific year/month into straddle strings."""
    schedule = get_schedule(path, underlying)
    straddles = _pack_ym(schedule, year, month)
    return straddles

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
