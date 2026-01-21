# -------------------------------------
# AMT strings - Fast straddle string building
# -------------------------------------
"""
High-performance straddle string building.

The key insight: pre-compute everything that doesn't depend on expiry year/month,
then the inner loop is just dict lookups and string concatenation.

Straddle format: |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|
Example: |2023-12|2024-01|N|0|BD|5|25|

Performance (178k straddles):
- schedules.expand(): ~400ms (parse + fix + pack per straddle)
- expand_fast(): ~35ms (pre-computed lookups only)
- Speedup: ~12x
"""
import re
from typing import Any


# -------------------------------------
# Template pre-computation
# -------------------------------------

def precompute_templates(
    schedules_dict: dict[str, list[str]],
    assets: list[tuple[dict, str]],
    underlying_hash_fn,
    fix_value_fn,
) -> list[tuple[str, int, str]]:
    """
    Pre-compute schedule templates from AMT data.

    Args:
        schedules_dict: expiry_schedules from AMT (schedule_name -> entries)
        assets: List of (asset_data, underlying) tuples for live assets
        underlying_hash_fn: Function to compute hash from underlying
        fix_value_fn: Function to fix a/b/c/d values

    Returns:
        List of (underlying, entry_offset, suffix) tuples where:
        - underlying: asset name (e.g., "LA Comdty")
        - entry_offset: months before expiry (-1 for N, -2 for F, 0 otherwise)
        - suffix: pre-computed "ntrc|ntrv|xprc|xprv|wgt|" string
    """
    templates = []
    code_pattern = re.compile(r"^([A-Z]+)(.*)$")

    for asset_data, underlying in assets:
        schedule_name = asset_data.get("Options")
        if not schedule_name:
            continue

        schedule = schedules_dict.get(schedule_name)
        if not schedule:
            continue

        assid = underlying_hash_fn(underlying)
        schcnt = len(schedule)

        for schid, entry in enumerate(schedule, 1):
            # Parse entry: "N0_BDa_25" -> ntrc=N, ntrv=0, xprc=BD, xprv=a, wgt=25
            parts = entry.split("_")
            if len(parts) < 2:
                continue

            # Extract components
            m1 = code_pattern.match(parts[0])
            if m1:
                ntrc, ntrv = m1.group(1), m1.group(2)
            else:
                ntrc, ntrv = parts[0], ""

            m2 = code_pattern.match(parts[1])
            if m2:
                xprc, xprv = m2.group(1), m2.group(2)
            else:
                xprc, xprv = parts[1], ""

            wgt = parts[2] if len(parts) > 2 else ""

            # Fix a/b/c/d values
            ntrv = fix_value_fn(ntrv, assid, schcnt, schid)
            xprv = fix_value_fn(xprv, assid, schcnt, schid)

            # Compute entry offset from ntrc
            entry_offset = {"N": -1, "F": -2}.get(ntrc, 0)

            # Build suffix: "ntrc|ntrv|xprc|xprv|wgt|"
            suffix = f"{ntrc}|{ntrv}|{xprc}|{xprv}|{wgt}|"

            templates.append((underlying, entry_offset, suffix))

    return templates


# -------------------------------------
# Fast expansion
# -------------------------------------

def expand_fast(
    templates: list[tuple[str, int, str]],
    start_year: int,
    end_year: int
) -> dict[str, Any]:
    """
    Fast straddle expansion using pre-computed templates.

    The inner loop does only dict lookups and string concatenation - no
    arithmetic, no parsing, no hash computation.

    Args:
        templates: List of (underlying, entry_offset, suffix) from precompute_templates()
        start_year: First expiry year
        end_year: Last expiry year (inclusive)

    Returns:
        Table dict with columns ["asset", "straddle"] and rows
    """
    n_templates = len(templates)
    if n_templates == 0:
        return {"columns": ["asset", "straddle"], "rows": []}

    n_years = end_year - start_year + 1
    total = n_years * 12 * n_templates

    # Pre-compute all entry date strings for each (offset, xpry, xprm)
    # Offsets are typically -1 (N) or -2 (F), so we only have 2-3 unique values
    unique_offsets = set(t[1] for t in templates)
    entry_strs = {}  # (offset, xpry, xprm) -> "|ntry-ntrm"

    for offset in unique_offsets:
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                total_months = y * 12 + (m - 1) + offset
                ntry = total_months // 12
                ntrm = (total_months % 12) + 1
                entry_strs[(offset, y, m)] = f"|{ntry}-{ntrm:02d}"

    # Pre-compute expiry strings
    xpr_strs = {}
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            xpr_strs[(y, m)] = f"|{y}-{m:02d}|"

    # Build rows with only lookups, no arithmetic
    rows = [None] * total
    idx = 0
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            xpr_str = xpr_strs[(y, m)]
            for underlying, offset, suffix in templates:
                entry_str = entry_strs[(offset, y, m)]
                rows[idx] = [underlying, f"{entry_str}{xpr_str}{suffix}"]
                idx += 1

    return {"columns": ["asset", "straddle"], "rows": rows}


# -------------------------------------
# Benchmark (run with: python -m specparser.amt.strings)
# -------------------------------------

def _benchmark():
    """Benchmark against real AMT data."""
    import time
    from . import loader, schedules

    print("Fast straddle expansion benchmark")
    print("=" * 50)

    amt_path = "data/amt.yml"
    start_year = 2005
    end_year = 2024

    # Current implementation
    t0 = time.perf_counter()
    current_result = schedules.find_straddle_yrs(amt_path, start_year, end_year, ".", True)
    t1 = time.perf_counter()
    current_time = t1 - t0
    print(f"schedules.find_straddle_yrs(): {current_time:.3f}s ({len(current_result['rows']):,} rows)")

    # New implementation
    t0 = time.perf_counter()
    amt = loader.load_amt(amt_path)
    schedules_dict = amt.get("expiry_schedules", {})
    assets = list(loader._iter_assets(amt_path, live_only=True, pattern="."))
    templates = precompute_templates(
        schedules_dict, assets,
        schedules._underlying_hash, schedules._fix_value,
    )
    t_precompute = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_result = expand_fast(templates, start_year, end_year)
    t_expand = time.perf_counter() - t0

    total_new = t_precompute + t_expand
    print(f"expand_fast():      {total_new:.3f}s ({len(new_result['rows']):,} rows)")
    print(f"  - precompute:     {t_precompute:.3f}s ({len(templates):,} templates)")
    print(f"  - expand:         {t_expand:.3f}s")
    print(f"Speedup:            {current_time/total_new:.1f}x")

    # Verify
    current_set = set((r[0], r[1]) for r in current_result["rows"])
    new_set = set((r[0], r[1]) for r in new_result["rows"])
    print(f"Match: {'✓' if current_set == new_set else '✗'}")


if __name__ == "__main__":
    _benchmark()
