# Plan: Add find_straddle_days_u8m Function to schedules.py

## Overview

Create a `find_straddle_days_u8m()` function that takes u8m straddle tables and expands them to daily rows, staying entirely in u8m format until output is needed.

**Key Decisions:**
- **Date format:** int32 (days since epoch) - reuses existing optimized kernel
- **Parsing kernel location:** In `schedules.py` - straddle-specific, not general string ops
- **Existing functions:** Keep `find_straddle_days`, `find_straddle_days_arrow`, `find_straddle_days_numba` for comparison

## Current Flow Analysis

### Existing `find_straddle_days_numba()` Flow

```
find_straddle_yrs() → straddles (Python strings)
    ↓
table_to_arrow() → Arrow strings
    ↓
pc.utf8_slice_codeunits() → Extract year, month, ntrc as Arrow strings
    ↓
pc.cast() + .to_numpy() → Convert to numpy int32 arrays
    ↓
schedules_numba.expand_months_to_date32() → date32, parent_idx
    ↓
pc.take() → Expand asset/straddle using parent_idx
    ↓
Arrow table with ["asset", "straddle", "date"]
```

### Problem
- Starts with `find_straddle_yrs()` which returns Python strings
- Converts to Arrow for string slicing
- Then converts back to numpy for numba kernel
- Multiple format conversions add overhead

### Proposed u8m Flow

```
find_straddle_yrs_u8m() → u8m table (asset_u8m, straddle_u8m)
    ↓
_parse_straddle_u8m_dates() → (ntry, ntrm, ntrc) as int32/uint8 arrays
    ↓
_ntrc_to_month_span() → month_span array (N→2, F→3)
    ↓
schedules_numba.expand_months_to_date32() → date32, parent_idx
    ↓
numpy fancy indexing → expand asset_u8m, straddle_u8m
    ↓
u8m table with ["asset", "straddle", "date"]
```

## Key Insight: Expiry + NTRC is the Source of Truth

The straddle u8m format: `|YYYY-MM|YYYY-MM|ntrc|ntrv|xprc|xprv|wgt|`
- Entry date at positions 1-7: `YYYY-MM` ← **Derived from expiry - ntrc offset**
- Expiry date at positions 9-15: `YYYY-MM` ← **Primary (schedules defined by expiry)**
- NTRC code at position 17: `N` or `F`

**Why expiry + ntrc should be the central codes:**

1. **Schedules are defined by expiry month** - `find_straddle_yrs` iterates over expiry year-months
2. **Entry is derived**: `entry = expiry - ntrc_offset` (N=1mo, F=2mo)
3. **Consistency**: The straddle already stores expiry as the second date field

**NTRC relationships:**
- `N` code: entry = expiry - 1 month → span = 2 months
- `F` code: entry = expiry - 2 months → span = 3 months

**For date expansion, we need:**
1. **Expiry year/month** (parse from positions 9-12, 14-15) - the anchor
2. **NTRC code** (parse from position 17) → determines entry offset and span

**Derivation for expand_months_to_date32 kernel:**
```
offset = 1 if ntrc == 'N' else 2   # months before expiry
span = offset + 1                   # N→2, F→3 months total
entry = expiry - offset             # compute entry from expiry
expand_months_to_date32(entry_year, entry_month, span)
```

**This means we parse 3 values (xpry, xprm, ntrc), compute entry, then expand.**

## Design Decisions

### Decision 1: Date format → **int32 (days since epoch)**

```python
{
    "orientation": "u8m",
    "columns": ["asset", "straddle", "date"],
    "rows": [asset_u8m, straddle_u8m, date_int32],
    # date_int32 shape: (n_days,) - int32 days since 1970-01-01
}
```

**Reasoning:**
- Reuses existing optimized `schedules_numba.expand_months_to_date32()`
- Compact storage (4 bytes vs 10 for `YYYY-MM-DD`)
- Direct Arrow date32 conversion when needed
- Downstream code (valuation) needs date arithmetic - int32 is ideal

### Decision 2: Parsing kernel location → **schedules.py**

The `_parse_straddle_u8m_dates()` kernel goes in `schedules.py` because:
- It's specific to straddle format (`|YYYY-MM|YYYY-MM|N|...|`)
- Not a general-purpose string operation
- Keeps related straddle code together

### Decision 3: Keep existing functions → **Yes, for comparison**

Do NOT delete:
- `find_straddle_days()` - Python loop with memoization
- `find_straddle_days_arrow()` - Arrow calendar lookup
- `find_straddle_days_numba()` - Numba kernel with Arrow input

These remain for benchmarking and backward compatibility.

---

## New Helpers for strings.py

The following generic u8m helpers should be added to `strings.py` (reusable across the codebase):

### 1. `sub_months_vec()` - Vectorized month subtraction (NEW)

```python
@njit(cache=True)
def sub_months_vec(years, months, offsets):
    """Vectorized: subtract offsets from year-month pairs.

    Uses existing add_months() logic but vectorized over arrays.

    Args:
        years: int32 array of years
        months: int32 array of months (1-12)
        offsets: int32 array of month offsets to subtract

    Returns:
        out_years: int32 array of result years
        out_months: int32 array of result months (1-12)
    """
    n = len(years)
    out_years = np.empty(n, dtype=np.int32)
    out_months = np.empty(n, dtype=np.int32)

    for i in range(n):
        # Reuse add_months with negative offset
        out_years[i], out_months[i] = add_months(years[i], months[i], -offsets[i])

    return out_years, out_months
```

### 2. `ntrc_to_offset_span()` - N/F code to offset and span (NEW)

```python
@njit(cache=True)
def ntrc_to_offset_span(ntrc):
    """Convert NTRC codes (uint8) to offset and span arrays.

    N (ord 78) → offset=1, span=2
    F (ord 70) → offset=2, span=3

    Args:
        ntrc: uint8 array of N/F codes

    Returns:
        offset: int32 array - months before expiry for entry
        span: int32 array - total months to expand
    """
    n = len(ntrc)
    offset = np.empty(n, dtype=np.int32)
    span = np.empty(n, dtype=np.int32)
    for i in range(n):
        if ntrc[i] == 78:  # ord('N')
            offset[i] = 1
            span[i] = 2
        else:  # F
            offset[i] = 2
            span[i] = 3
    return offset, span
```

**Why in strings.py**: These are generic month/date arithmetic operations, not straddle-specific. They can be reused by any code working with N/F codes and year-month pairs.

---

## Implementation Plan

### Step 1: Add helpers to strings.py

Add these to `strings.py`:
- `sub_months_vec(years, months, offsets)` - vectorized month subtraction
- `ntrc_to_offset_span(ntrc)` - convert N/F codes to offset/span

### Step 2: Add `_parse_straddle_u8m_xpr()` (numba kernel in schedules.py)

Parse expiry year/month + ntrc from straddle u8m column. This is **straddle-specific** (knows field positions), so stays in schedules.py:

```python
from numba import njit
from .strings import read_4digits, read_2digits

@njit(cache=True)
def _parse_straddle_u8m_xpr(straddle_u8m):
    """Extract expiry year, month, and ntrc from straddle u8m matrix.

    Straddle format: |YYYY-MM|YYYY-MM|N|...|
                      ^      ^       ^
                      1      9       17
    Positions:
        - Entry year: 1-4 (derived, not parsed here)
        - Entry month: 6-7 (derived, not parsed here)
        - Expiry year: 9-12 ← **parsed**
        - Expiry month: 14-15 ← **parsed**
        - NTRC: 17 ← **parsed**

    Returns:
        xpry: int32 array of expiry years
        xprm: int32 array of expiry months (1-12)
        ntrc: uint8 array of ntrc codes (ord('N')=78 or ord('F')=70)

    Note: Entry is derived from expiry - ntrc_offset, not parsed directly.
    """
    n = straddle_u8m.shape[0]
    xpry = np.empty(n, dtype=np.int32)
    xprm = np.empty(n, dtype=np.int32)
    ntrc = np.empty(n, dtype=np.uint8)

    for i in range(n):
        row = straddle_u8m[i]
        # Expiry year at position 9-12
        xpry[i] = read_4digits(row, 9)
        # Expiry month at position 14-15
        xprm[i] = read_2digits(row, 14)
        # NTRC code at position 17
        ntrc[i] = row[17]

    return xpry, xprm, ntrc
```

**Note:** We parse expiry (the anchor) + ntrc. Entry is derived using helpers from strings.py.

### Step 3: Add `find_straddle_days_u8m()`

Main function - uses helpers from strings.py:

```python
from .strings import ntrc_to_offset_span, sub_months_vec

def find_straddle_days_u8m(
    path: str | Path,
    start_year: int,
    end_year: int,
    pattern: str = ".",
    live_only: bool = True,
    parallel: bool = False,
) -> dict[str, Any]:
    """Expand straddles to daily rows using u8m format.

    This is the fastest implementation for large datasets when working
    with u8m data throughout the pipeline.

    Uses expiry + ntrc as source of truth, derives entry for expansion.

    Args:
        path: Path to AMT YAML file
        start_year: Start year for straddles
        end_year: End year for straddles
        pattern: Regex pattern to filter assets
        live_only: Only include live straddles
        parallel: If True, use parallel kernel (best for millions+ output rows)

    Returns:
        Table with columns ["asset", "straddle", "date"]
        - asset: u8m matrix (n_days, asset_width) uint8
        - straddle: u8m matrix (n_days, straddle_width) uint8
        - date: int32 array (n_days,) - days since 1970-01-01
    """
    # Step 1: Get u8m straddles
    straddles = find_straddle_yrs_u8m(path, start_year, end_year, pattern, live_only)

    asset_u8m = straddles["rows"][0]
    straddle_u8m = straddles["rows"][1]
    n = asset_u8m.shape[0]

    if n == 0:
        return {
            "orientation": "u8m",
            "columns": ["asset", "straddle", "date"],
            "rows": [
                np.empty((0, 1), dtype=np.uint8),
                np.empty((0, 1), dtype=np.uint8),
                np.empty(0, dtype=np.int32),
            ],
        }

    # Step 2: Parse EXPIRY year/month + ntrc from straddle u8m (expiry is the anchor)
    xpry, xprm, ntrc = _parse_straddle_u8m_xpr(straddle_u8m)

    # Step 3: Compute offset and span from ntrc using strings.py helper
    offset, month_span = ntrc_to_offset_span(ntrc)

    # Step 4: Derive ENTRY from expiry - offset using strings.py helper
    ntry, ntrm = sub_months_vec(xpry, xprm, offset)

    # Step 5: Run numba date expansion kernel (starts from entry, expands span months)
    if parallel:
        date32, parent_idx = schedules_numba.expand_months_to_date32_parallel(
            ntry, ntrm, month_span
        )
    else:
        date32, parent_idx = schedules_numba.expand_months_to_date32(
            ntry, ntrm, month_span
        )

    # Step 6: Expand asset and straddle columns using numpy fancy indexing
    expanded_asset = asset_u8m[parent_idx]
    expanded_straddle = straddle_u8m[parent_idx]

    return {
        "orientation": "u8m",
        "columns": ["asset", "straddle", "date"],
        "rows": [expanded_asset, expanded_straddle, date32],
    }
```

### Step 4: Add `straddle_days_u8m_to_arrow()` (optional conversion)

Convert u8m result to Arrow for compatibility with existing code:

```python
def straddle_days_u8m_to_arrow(table_u8m: dict[str, Any]) -> dict[str, Any]:
    """Convert u8m straddle days table to Arrow format.

    Use this when you need Arrow-format output for downstream Arrow operations
    or for compatibility with existing code expecting Arrow tables.
    """
    from .table import _import_pyarrow
    pa, _ = _import_pyarrow()

    asset_u8m = table_u8m["rows"][0]
    straddle_u8m = table_u8m["rows"][1]
    date32 = table_u8m["rows"][2]

    n = asset_u8m.shape[0]
    if n == 0:
        return {
            "orientation": "arrow",
            "columns": ["asset", "straddle", "date"],
            "rows": [pa.array([]), pa.array([]), pa.array([], type=pa.date32())],
        }

    # Convert u8m to string arrays
    assets = u8m2s(asset_u8m)
    straddles = u8m2s(straddle_u8m)

    # Clean straddle strings (strip spaces around pipes)
    def clean_straddle(s):
        parts = s.split('|')
        return '|'.join(p.strip() for p in parts)

    return {
        "orientation": "arrow",
        "columns": ["asset", "straddle", "date"],
        "rows": [
            pa.array([a.strip() for a in assets.tolist()]),
            pa.array([clean_straddle(s) for s in straddles.tolist()]),
            pa.array(date32, type=pa.date32()),
        ],
    }
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/specparser/amt/strings.py` | Add `sub_months_vec()`, `ntrc_to_offset_span()` (generic numba helpers) |
| `src/specparser/amt/schedules.py` | Add `_parse_straddle_u8m_xpr()`, `find_straddle_days_u8m()`, `straddle_days_u8m_to_arrow()` |
| `src/specparser/amt/__init__.py` | Export `find_straddle_days_u8m`, `straddle_days_u8m_to_arrow`, `sub_months_vec`, `ntrc_to_offset_span` |
| `tests/test_amt.py` | Add tests for `find_straddle_days_u8m()` |

---

## Required Imports

Add to schedules.py imports:

```python
from numba import njit
from .strings import read_4digits, read_2digits, ntrc_to_offset_span, sub_months_vec
```

---

## Performance Expectations

Based on existing benchmarks:
- `find_straddle_yrs_u8m()`: 0.29ms for 2,304 straddles (14.8x faster than strings)
- `schedules_numba.expand_months_to_date32()`: Already optimized with Howard Hinnant algorithm
- NumPy fancy indexing `u8m[parent_idx]`: Very fast, memory-bound

Expected speedup vs `find_straddle_days_numba()`:
- Eliminates Arrow string conversion overhead
- Direct u8m parsing is cheaper than Arrow utf8_slice
- **Estimated 2-3x faster** for the parsing/setup phase

---

## Testing

```bash
# Test u8m straddle days expansion
uv run python -c "
from specparser.amt.schedules import find_straddle_days_u8m
result = find_straddle_days_u8m('data/amt.yml', 2024, 2024, '^LA Comdty')
print('Columns:', result['columns'])
print('Asset shape:', result['rows'][0].shape)
print('Straddle shape:', result['rows'][1].shape)
print('Date array shape:', result['rows'][2].shape)
"

# Verify matches find_straddle_days_numba output
uv run python -c "
from specparser.amt.schedules import find_straddle_days_numba, find_straddle_days_u8m, straddle_days_u8m_to_arrow
from specparser.amt import u8m2s

result_numba = find_straddle_days_numba('data/amt.yml', 2024, 2024, '^LA Comdty')
result_u8m = find_straddle_days_u8m('data/amt.yml', 2024, 2024, '^LA Comdty')

# Compare counts
print(f'Numba rows: {len(result_numba[\"rows\"][0])}')
print(f'u8m rows: {result_u8m[\"rows\"][0].shape[0]}')

# Verify dates match
import numpy as np
numba_dates = result_numba['rows'][2].to_numpy()
u8m_dates = result_u8m['rows'][2]
print(f'Dates match: {np.array_equal(numba_dates, u8m_dates)}')
"

# Benchmark all 4 approaches
uv run python -c "
import time
from specparser.amt.schedules import (
    find_straddle_days, find_straddle_days_arrow,
    find_straddle_days_numba, find_straddle_days_u8m,
    clear_schedule_caches, clear_calendar_cache
)

def bench(name, fn, path, start, end, pattern, iterations=10):
    clear_schedule_caches()
    clear_calendar_cache()
    # Warmup
    _ = fn(path, start, end, pattern)
    # Benchmark
    t0 = time.perf_counter()
    for _ in range(iterations):
        result = fn(path, start, end, pattern)
    elapsed = (time.perf_counter() - t0) / iterations
    # Get row count
    if isinstance(result['rows'][0], list):
        n = len(result['rows'][0])
    else:
        n = len(result['rows'][0]) if hasattr(result['rows'][0], '__len__') else result['rows'][0].shape[0]
    print(f'{name:30s} {elapsed*1000:8.2f}ms  ({n:,} rows)')
    return elapsed

path = 'data/amt.yml'
pattern = '^LA Comdty'

print('Benchmark: LA Comdty 2001-2024')
print('=' * 60)
t_loop = bench('find_straddle_days (loop)', find_straddle_days, path, 2001, 2024, pattern)
t_arrow = bench('find_straddle_days_arrow', find_straddle_days_arrow, path, 2001, 2024, pattern)
t_numba = bench('find_straddle_days_numba', find_straddle_days_numba, path, 2001, 2024, pattern)
t_u8m = bench('find_straddle_days_u8m', find_straddle_days_u8m, path, 2001, 2024, pattern)
print('=' * 60)
print(f'u8m speedup vs loop:  {t_loop/t_u8m:.1f}x')
print(f'u8m speedup vs arrow: {t_arrow/t_u8m:.1f}x')
print(f'u8m speedup vs numba: {t_numba/t_u8m:.1f}x')
"
```

---

## Implementation Order

### strings.py (generic helpers)
1. [x] Add `sub_months_vec()` - vectorized month subtraction (uses existing `add_months`)
2. [x] Add `ntrc_to_offset_span()` - N/F code to offset/span conversion

### schedules.py (straddle-specific)
3. [x] Add import: `from .strings import ntrc_to_offset_span, sub_months_vec, read_4digits, read_2digits`
4. [x] Add `_parse_straddle_u8m_xpr()` (numba jit - parse expiry + ntrc from straddle positions)
5. [x] Add `find_straddle_days_u8m()` - uses helpers from strings.py
6. [x] Add `straddle_days_u8m_to_arrow()` for Arrow compatibility

### __init__.py
7. [x] Add exports: `find_straddle_days_u8m`, `straddle_days_u8m_to_arrow`, `sub_months_vec`, `ntrc_to_offset_span`

### Testing
8. [x] Add tests for new functions
9. [x] Benchmark all 4 approaches and verify correctness

---

## Implementation Status: ✅ COMPLETE

All functions implemented and tests passing (453 tests).

---

## Benchmark Results

LA Comdty 2001-2024 (175,320 rows, 10 iterations):

| Approach | Time (ms) | Speedup vs Loop |
|----------|-----------|-----------------|
| Loop     | 7.54      | 1.0x (baseline) |
| Arrow    | 5.07      | 1.5x            |
| Numba    | 4.14      | 1.8x            |
| **u8m**  | **3.75**  | **2.0x**        |

The u8m approach is the fastest, achieving ~2x speedup over the loop approach by staying in uint8 matrix format throughout the pipeline.

---

## Existing Functions (KEEP - do not delete)

| Function | Input | Output | Use Case |
|----------|-------|--------|----------|
| `find_straddle_days()` | string table | column table | Python loop, good cache hit ratio |
| `find_straddle_days_arrow()` | string table | Arrow table | Arrow-native pipelines |
| `find_straddle_days_numba()` | string table | Arrow table | Fast with Arrow input |
| **`find_straddle_days_u8m()`** | u8m table | u8m table | **Fastest, pure u8m pipeline** |
