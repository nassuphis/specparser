# Plan: Add u8m (uint8 Matrix) Functions to schedules.py

## Overview

Integrate fast uint8 matrix operations from `strings.py` into `schedules.py` by creating `_u8m` variants of existing functions. This allows downstream code to work with uint8 matrices throughout the pipeline, avoiding expensive Python string conversions until output.

## Current Flow (Python strings)

```
get_schedule_nocache() → _schedule_to_rows() → dict with string rows
    ↓
find_schedules() → dict with string rows
    ↓
_schedules2straddles() → _schedule2straddle() → straddle string "|YYYY-MM|YYYY-MM|...|"
    ↓
find_straddle_yrs() → dict with ["asset", "straddle"] string columns
```

## Proposed Flow (u8m variants)

```
get_schedule_nocache_u8m() → _schedule_to_u8m() → dict with u8m columns
    ↓
find_schedules_u8m() → dict with u8m columns
    ↓
_schedules_u8m2straddles_u8m() → u8m straddle matrix
    ↓
find_straddle_yrs_u8m() → dict with u8m ["asset_u8m", "straddle_u8m"] columns
```

---

## Function Mapping

| Current Function | u8m Variant | Description |
|------------------|-------------|-------------|
| `_schedule_to_rows()` | `_schedule_to_u8m()` | Convert schedule to u8m columns |
| `get_schedule_nocache()` | `get_schedule_nocache_u8m()` | Get schedule as u8m (no cache) |
| `get_schedule()` | `get_schedule_u8m()` | Get schedule as u8m (cached) |
| `find_schedules()` | `find_schedules_u8m()` | Find schedules as u8m |
| `_schedule2straddle()` | (vectorized in batch) | N/A - handled in batch |
| `_schedules2straddles()` | `_schedules_u8m2straddles_u8m()` | Pack schedules u8m to straddle u8m |
| `_schedules2straddle_yrs()` | `_schedules_u8m2straddle_yrs_u8m()` | Expand year range |
| `find_straddle_yrs()` | `find_straddle_yrs_u8m()` | Find all straddles as u8m |

---

## Data Structures

### Schedule u8m Table

```python
{
    "orientation": "u8m",
    "columns": ["asset", "ntrc", "ntrv", "xprc", "xprv", "wgt"],
    "rows": [asset_u8m, ntrc_u8m, ntrv_u8m, xprc_u8m, xprv_u8m, wgt_u8m],
}
```

Each u8m column is a `np.ndarray` of dtype `uint8` with shape `(nrows, field_width)`.

### Design Decision: schcnt and schid Columns

**Question**: Should we include `schcnt` (schedule count) and `schid` (schedule index) as columns?

**Analysis**:

1. **Value ranges are tiny**:
   - `schcnt`: 1-4 (number of schedule components per asset)
   - `schid`: 1 to schcnt (1-based index within schedule)

2. **Options considered**:

   | Option | Storage | Pros | Cons |
   |--------|---------|------|------|
   | Separate int arrays (`_schcnt`, `_schid`) | 8 bytes/row each | Native numpy ops | Breaks u8m uniformity, extra fields |
   | u8m columns (1 byte width) | 1 byte/row each | Uniform structure | Wasteful for 2-bit values |
   | Single u8m column `|schcnt|schid|` | 4 bytes/row | Uniform, compact | Extra parsing |
   | **Infer from asset column** | 0 bytes | No storage overhead | Requires scan |
   | **Don't store at all** | 0 bytes | Simplest | Lose the information |

3. **Key insight**: After expansion to straddles, `schcnt` and `schid` are **not needed**:
   - `schcnt` is only used in `_fix_value()` during schedule parsing (already done)
   - `schid` is only used in `_fix_value()` during schedule parsing (already done)
   - Downstream code (straddle expansion, valuation) never uses these fields

4. **Verification** - checking usage in the codebase:
   - `_schedule_to_rows()` outputs `[schcnt, schid, asset, ntrc, ntrv, xprc, xprv, wgt]`
   - `find_schedules()` returns these columns but downstream only uses `asset`, `ntrc`, `ntrv`, `xprc`, `xprv`, `wgt`
   - `_schedules2straddles()` at line ~240 extracts: `asset_idx`, `ntrc_idx`, `ntrv_idx`, `xprc_idx`, `xprv_idx`, `wgt_idx` - **no schcnt/schid**

**Decision**: **Don't include schcnt/schid in u8m tables**

**Reasoning**:
- These values are only needed during the `_fix_schedule()` step which happens before u8m conversion
- By the time we have u8m data, the a/b/c/d values have already been fixed to day numbers
- Storing them would add complexity with no benefit
- If ever needed, `schcnt` can be inferred by counting rows with the same asset (run-length encoding)
- If ever needed, `schid` can be inferred as position within each asset's run (1, 2, 3, ...)

**Alternative if needed later**: Store as a single packed u8m column `|C|I|` where C=schcnt, I=schid (2 bytes/row). But this is unnecessary given the analysis above.

### Straddle u8m Table

```python
{
    "orientation": "u8m",
    "columns": ["asset", "straddle"],
    "rows": [asset_u8m, straddle_u8m],
    # straddle_u8m shape: (nrows, straddle_width)
    # straddle format: |YYYY-MM|YYYY-MM|NTRC|NTRV|XPRC|XPRV|WGT|
}
```

---

## Implementation Steps

### Step 1: Add u8m Schedule Functions

**1a. `_schedule_to_u8m(underlying, schedule)`**

Convert a single asset's schedule to u8m columns:

```python
def _schedule_to_u8m(underlying: str, schedule: list[str] | None) -> dict[str, Any]:
    """Convert schedule to u8m format.

    Returns dict with:
        - asset_u8m: (N, asset_width) uint8 matrix
        - ntrc_u8m: (N, 1) uint8 matrix (N or F)
        - ntrv_u8m: (N, ntrv_width) uint8 matrix
        - xprc_u8m: (N, xprc_width) uint8 matrix
        - xprv_u8m: (N, xprv_width) uint8 matrix
        - wgt_u8m: (N, wgt_width) uint8 matrix
        - schcnt: int
        - schid: (N,) int64 array
    """
```

**1b. `get_schedule_nocache_u8m(path, underlying)`**

```python
def get_schedule_nocache_u8m(path: str | Path, underlying: str) -> dict[str, Any]:
    """Get schedule as u8m format (no caching)."""
    data = loader.load_amt(path)
    asset_data = loader.get_asset(path, underlying)
    # ... extract schedule ...
    return _schedule_to_u8m(underlying, schedule)
```

**1c. `get_schedule_u8m(path, underlying)` with cache**

```python
_SCHEDULE_U8M_CACHE: dict = {}

def get_schedule_u8m(path: str | Path, underlying: str) -> dict[str, Any]:
    """Get schedule as u8m format (cached)."""
    path_str = str(Path(path).resolve())
    cache_key = (path_str, underlying)
    if _MEMOIZE_ENABLED and cache_key in _SCHEDULE_U8M_CACHE:
        return _SCHEDULE_U8M_CACHE[cache_key]
    result = get_schedule_nocache_u8m(path, underlying)
    if _MEMOIZE_ENABLED:
        _SCHEDULE_U8M_CACHE[cache_key] = result
    return result
```

### Step 2: Add u8m Find Schedules

**`find_schedules_u8m(path, pattern, live_only)`**

```python
def find_schedules_u8m(path: str | Path, pattern: str, live_only: bool = True) -> dict[str, Any]:
    """Find assets matching pattern and return schedules as u8m.

    Returns dict with u8m columns concatenated from all matching assets.
    """
    assets = [asset for _, asset in loader._iter_assets(path, live_only=live_only, pattern=pattern)]

    # Collect u8m tables from each asset
    tables = [get_schedule_u8m(path, asset) for asset in assets]

    # Concatenate u8m columns vertically
    return _concat_schedule_u8m_tables(tables)
```

### Step 3: Add u8m Straddle Expansion

**3a. `_schedules_u8m2straddles_u8m(table_u8m, xpry, xprm)`**

Expand schedules for a single year-month:

```python
def _schedules_u8m2straddles_u8m(table_u8m: dict, xpry: int, xprm: int) -> dict[str, Any]:
    """Pack u8m schedules for year/month into u8m straddles.

    Uses strings.py functions:
    - make_ym_matrix() for expiry year-month
    - add_months2specs_inplace() for entry date computation
    - np.hstack() for straddle assembly
    """
```

**3b. `_schedules_u8m2straddle_yrs_u8m(table_u8m, start_year, end_year)`**

Expand across year range using cartesian product:

```python
def _schedules_u8m2straddle_yrs_u8m(table_u8m: dict, start_year: int, end_year: int) -> dict[str, Any]:
    """Expand u8m schedules across year range.

    Uses:
    - make_ym_matrix((start_year, 1, end_year, 12)) for all year-months
    - np.repeat/tile for cartesian product
    - add_months2specs_inplace() for vectorized entry date computation
    """
```

**3c. `find_straddle_yrs_u8m(path, start_year, end_year, pattern, live_only)`**

Top-level function:

```python
def find_straddle_yrs_u8m(
    path: str | Path,
    start_year: int,
    end_year: int,
    pattern: str = ".",
    live_only: bool = True,
) -> dict[str, Any]:
    """Find all straddles as u8m matrices.

    Returns:
        {
            "orientation": "u8m",
            "columns": ["asset", "straddle"],
            "rows": [asset_u8m, straddle_u8m],
        }
    """
    schedules = find_schedules_u8m(path, pattern=pattern, live_only=live_only)
    return _schedules_u8m2straddle_yrs_u8m(schedules, start_year, end_year)
```

### Step 4: Add Conversion Functions

**`straddles_u8m_to_strings(table_u8m)`**

Convert u8m table back to string table (for output/compatibility):

```python
def straddles_u8m_to_strings(table_u8m: dict) -> dict[str, Any]:
    """Convert u8m straddle table to string table.

    Uses u8m2s() from strings.py for conversion.
    """
    from .strings import u8m2s

    asset_u8m = table_u8m["rows"][0]
    straddle_u8m = table_u8m["rows"][1]

    assets = u8m2s(asset_u8m).tolist()
    straddles = u8m2s(straddle_u8m).tolist()

    rows = [[a.strip(), s.strip()] for a, s in zip(assets, straddles)]
    return {"orientation": "row", "columns": ["asset", "straddle"], "rows": rows}
```

---

## Cache Management

Add new cache and update `clear_schedule_caches()`:

```python
_SCHEDULE_U8M_CACHE: dict = {}

def clear_schedule_caches() -> None:
    """Clear all schedule-related caches."""
    _SCHEDULE_CACHE.clear()
    _SCHEDULE_U8M_CACHE.clear()  # NEW
    _EXPAND_YM_CACHE.clear()
    _DAYS_YM_CACHE.clear()
    _STRADDLE_DAYS_CACHE.clear()
```

---

## Imports Required

Add to `schedules.py`:

```python
from .strings import (
    strs2u8mat,
    u8m2s,
    make_ym_matrix,
    add_months2specs_inplace,
    ASCII_PIPE,
)
```

---

## Testing

```bash
# Test u8m schedule retrieval
uv run python -c "
from specparser.amt.schedules import get_schedule_u8m, find_schedules_u8m
s = get_schedule_u8m('data/amt.yml', 'LA Comdty')
print('Schedule u8m columns:', s['columns'])
print('Asset shape:', s['rows'][0].shape)
"

# Test u8m straddle expansion
uv run python -c "
from specparser.amt.schedules import find_straddle_yrs_u8m, straddles_u8m_to_strings
u8m = find_straddle_yrs_u8m('data/amt.yml', 2024, 2024, '^LA Comdty')
print('Straddle u8m shape:', u8m['rows'][1].shape)
strings = straddles_u8m_to_strings(u8m)
print('First row:', strings['rows'][0])
"

# Benchmark comparison
uv run python -c "
import time
from specparser.amt.schedules import find_straddle_yrs, find_straddle_yrs_u8m, straddles_u8m_to_strings

# Warmup
_ = find_straddle_yrs('data/amt.yml', 2024, 2024, '^LA Comdty')
_ = find_straddle_yrs_u8m('data/amt.yml', 2024, 2024, '^LA Comdty')

# Standard
t0 = time.perf_counter()
for _ in range(10):
    r1 = find_straddle_yrs('data/amt.yml', 2001, 2024, '^LA Comdty')
t1 = time.perf_counter()
print(f'Standard: {(t1-t0)*100:.2f}ms, rows={len(r1[\"rows\"])}')

# u8m (without string conversion)
t0 = time.perf_counter()
for _ in range(10):
    r2 = find_straddle_yrs_u8m('data/amt.yml', 2001, 2024, '^LA Comdty')
t1 = time.perf_counter()
print(f'u8m only: {(t1-t0)*100:.2f}ms')

# u8m + string conversion
t0 = time.perf_counter()
for _ in range(10):
    r2 = find_straddle_yrs_u8m('data/amt.yml', 2001, 2024, '^LA Comdty')
    r2s = straddles_u8m_to_strings(r2)
t1 = time.perf_counter()
print(f'u8m+conv: {(t1-t0)*100:.2f}ms')
"
```

---

## Implementation Order

1. [x] Add imports from `strings.py`
2. [x] Add `_SCHEDULE_U8M_CACHE` and update `clear_schedule_caches()`
3. [x] Implement `_schedule_to_u8m()`
4. [x] Implement `get_schedule_nocache_u8m()` and `get_schedule_u8m()`
5. [x] Implement `find_schedules_u8m()` with `_concat_schedule_u8m_tables()`
6. [x] Implement `_schedules_u8m2straddles_u8m()`
7. [x] Implement `_schedules_u8m2straddle_yrs_u8m()`
8. [x] Implement `find_straddle_yrs_u8m()`
9. [x] Implement `straddles_u8m_to_strings()`
10. [x] Add tests (verified against standard implementation)
11. [ ] Update `backtest_strings.py` to use the new functions (optional - remove duplication)

---

## Files Modified

| File | Changes |
|------|---------|
| `src/specparser/amt/schedules.py` | Added all `_u8m` functions |
| `src/specparser/amt/strings.py` | Added `sub_months2specs_inplace_NF()` |

---

## Implementation Status: ✅ COMPLETE

All functions implemented and tested. Results verified against standard implementation.

---

## Actual Performance

Benchmark results (LA Comdty 2001-2024, 2,304 straddles, 10 iterations):

| Approach | Time | Speedup |
|----------|------|---------|
| Standard (`find_straddle_yrs`) | 4.25ms | 1.0x (baseline) |
| **u8m only** (`find_straddle_yrs_u8m`) | **0.29ms** | **14.8x** |
| u8m + string conversion | 2.14ms | 2.0x |

**Key findings:**
- u8m operations are **14.8x faster** than Python string approach
- String conversion (`straddles_u8m_to_strings`) is the bottleneck
- For downstream code that can work directly with u8m matrices, the speedup is significant
- Even with string conversion, still 2x faster than standard approach
