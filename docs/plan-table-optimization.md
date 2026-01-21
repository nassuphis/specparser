# Plan: Table Optimization and Columnar Representation

## Problem

The current backtest processes 178,000 straddles, but spends significant time in:

1. **Row-wise table operations** - Every operation iterates row-by-row with Python loops
2. **Repeated straddle string parsing** - `_parse_straddle()` splits the same string multiple times
3. **Column index lookups** - `.index("colname")` called repeatedly per row
4. **Layered composition** - Multiple passes through data (expand → pack → parse → lookup)

## Current Architecture

### Table Representation
```python
{"columns": ["col1", "col2"], "rows": [[val1, val2], [val3, val4]]}
```

- Row-oriented: good for appending rows, bad for column operations
- Every column access requires `columns.index(name)` then `row[idx]`
- No vectorization possible

### Straddle String Format
```
|2023-12|2024-01|N|0|OVERRIDE||33.3|
```
- Packed for storage/display
- Parsed repeatedly via `_parse_straddle()` which splits into 7 parts
- Accessor functions (`ntry()`, `xpry()`, etc.) each re-parse the entire string

### Data Flow in `expand()` → `get_straddle_valuation()`
```
find_schedules()           # Build row-wise table of schedules
    → _schedule_to_rows()  # Parse schedule strings into rows
    → _fix_schedule()      # Apply hash-based fixes

_expand_and_pack()         # Iterate years × months × rows
    → _pack_ym()           # Pack back into straddle strings

get_straddle_valuation()   # Per-straddle processing
    → asset_straddle_tickers()  # Parse straddle string again
    → get_straddle_days()       # Build date range, fetch prices
    → _compute_actions()        # Find good days, set entry/expiry
    → valuation model           # Compute mv, delta, pnl
```

## Proposed Solution

### 0. Template-Based Schedule Format (Quick Win)

**Key Insight:** The current schedule expansion does massive string parsing/repacking when all we need is template substitution.

#### Current Format (in AMT YAML)
```yaml
schedule1:
  - N0_BDa_25
  - N0_BDb_25
  - N0_BDc_25
  - N0_BDd_25
```

This requires:
1. Parse `N0_BDa_25` → extract ntrc=N, ntrv=0, xprc=BD, xprv=a, wgt=25
2. Fix `a/b/c/d` via hash calculation
3. Compute entry year/month from expiry
4. Pack back into `|2023-12|2024-01|N|0|BD|5|25|`

#### Proposed Format (in AMT YAML)
```yaml
schedule1:
  - 'N|0|BD|a|25'    # Human-readable, same as today
  - 'N|0|BD|b|25'
  - 'N|0|BD|c|25'
  - 'N|0|BD|d|25'
```

The YAML stays human-readable. The magic happens at **AMT load time**.

#### Two-Phase Approach

**Phase A: Pre-computation at AMT load (once per session)**

When `load_amt()` is called, pre-compute all schedule templates:

```python
# Stored in _SCHEDULE_TEMPLATES_CACHE when AMT is loaded
# Key: (underlying, schid) -> pre-computed template suffix
# Example: ("LA Comdty", 1) -> "N|0|BD|5|25|"

_SCHEDULE_TEMPLATES_CACHE: dict[str, list[tuple[str, int, str]]] = {}
# Maps schedule_name -> [(underlying, entry_offset, suffix), ...]
# Where suffix = "ntrc|ntrv|xprc|xprv|wgt|" with bd already computed
```

For each (asset, schedule_entry):
1. Parse the schedule entry once: `N|0|BD|a|25` → ntrc=N, ntrv=0, xprc=BD, xprv=a, wgt=25
2. Compute `bd` value from hash: `_fix_value(underlying_hash, schcnt, schid)` → e.g., 5
3. Compute entry offset from ntrc: N → -1 month, F → -2 months
4. Store pre-computed suffix: `"N|0|BD|5|25|"`

**Phase B: Expansion at backtest time (fast)**

```python
def expand_fast(
    path: str,
    start_year: int,
    end_year: int,
    pattern: str = ".",
    live_only: bool = True
) -> dict[str, Any]:
    """Fast straddle expansion using pre-computed templates."""

    # 1. Get pre-computed templates (cached at AMT load time)
    templates = _get_precomputed_templates(path, pattern, live_only)
    # Returns: [(underlying, entry_offset, suffix), ...]
    # Example: [("LA Comdty", -1, "N|0|BD|5|25|"), ...]

    # 2. Generate all straddles - just string concatenation!
    rows = []
    for xpry in range(start_year, end_year + 1):
        for xprm in range(1, 13):
            for underlying, entry_offset, suffix in templates:
                # Compute entry year/month (simple arithmetic)
                total_months = xpry * 12 + (xprm - 1) + entry_offset
                ntry = total_months // 12
                ntrm = (total_months % 12) + 1

                # Just concatenate! No parsing, no format()
                straddle = f"|{ntry}-{ntrm:02d}|{xpry}-{xprm:02d}|{suffix}"
                rows.append([underlying, straddle])

    return {"columns": ["asset", "straddle"], "rows": rows}
```

#### Why This is Fast

1. **Parsing happens once** - At AMT load time, not per straddle
2. **Hash computed once** - Per (asset, schid), stored in suffix
3. **Expansion is trivial** - Just string concatenation with pre-computed suffix

#### True Vectorization: The R `glue` Gap

Python lacks an equivalent to R's `glue` package, which does true vectorized string interpolation:

```r
# R: glue with vectors - ONE call, N outputs
glue("|{ntry}-{ntrm}|{xpry}-{xprm}|{suffix}",
     ntry = c(2023, 2023, 2024),
     ntrm = c(11, 12, 1),
     xpry = c(2024, 2024, 2024),
     xprm = c(1, 1, 1),
     suffix = c("N|0|BD|5|25|", "N|0|BD|5|25|", "N|0|BD|5|25|"))
```

**Python's options:**

1. **Pandas `.str.cat()`** - Vectorized concatenation, but no format specs (`:02d`)
2. **NumPy `np.char.add()`** - Vectorized string concat, no formatting
3. **Per-element f-string** - Has formatting but calls Python per-element

**The workaround for zero-padded months:**

```python
import numpy as np

# Convert int arrays to zero-padded strings vectorized
ntrm_arr = np.array([1, 2, 12])
ntrm_str = np.char.zfill(ntrm_arr.astype(str), 2)  # ['01', '02', '12']

# Then concatenate vectorized
straddles = np.char.add(np.char.add('|', ntry_str), '-')
straddles = np.char.add(straddles, ntrm_str)
# ... etc
```

This gives us vectorized operations but is verbose. For 178k straddles, the
overhead of Python's per-element formatting may be acceptable since the main
savings come from:
- Not parsing schedule strings 178k times
- Not computing hashes 178k times
- Pre-computing suffixes once

**Practical implementation choice:**

Given the trade-offs, two viable paths:

**Path A: NumPy vectorization (maximize speed)**
```python
def expand_vectorized(templates, start_year, end_year):
    # Build vectors for all year/month combinations
    n_months = (end_year - start_year + 1) * 12
    n_templates = len(templates)
    total = n_months * n_templates

    # Pre-allocate numpy arrays
    xpry_arr = np.repeat(np.arange(start_year, end_year + 1), 12 * n_templates)
    xprm_arr = np.tile(np.repeat(np.arange(1, 13), n_templates), end_year - start_year + 1)

    # Compute entry dates vectorized
    offsets = np.tile([t[0] for t in templates], n_months)  # entry_offset per template
    total_months = xpry_arr * 12 + (xprm_arr - 1) + offsets
    ntry_arr = total_months // 12
    ntrm_arr = (total_months % 12) + 1

    # Format with np.char operations
    ntrm_str = np.char.zfill(ntrm_arr.astype(str), 2)
    xprm_str = np.char.zfill(xprm_arr.astype(str), 2)

    # Build straddles with vectorized concatenation
    suffixes = np.tile([t[1] for t in templates], n_months)
    straddles = np.char.add(
        np.char.add(np.char.add('|', ntry_arr.astype(str)), '-'),
        np.char.add(np.char.add(np.char.add(ntrm_str, '|'), xpry_arr.astype(str)), '-')
    )
    straddles = np.char.add(np.char.add(straddles, xprm_str), '|')
    straddles = np.char.add(straddles, suffixes)

    return straddles.tolist()
```

**Path B: List comprehension with pre-computed suffix (simpler, good enough)**
```python
def expand_fast(templates, start_year, end_year):
    rows = []
    for xpry in range(start_year, end_year + 1):
        for xprm in range(1, 13):
            for underlying, entry_offset, suffix in templates:
                total_months = xpry * 12 + (xprm - 1) + entry_offset
                ntry = total_months // 12
                ntrm = (total_months % 12) + 1
                rows.append([underlying, f"|{ntry}-{ntrm:02d}|{xpry}-{xprm:02d}|{suffix}"])
    return rows
```

Path B is likely sufficient since:
- The inner loop is now trivial (arithmetic + 1 f-string)
- No parsing, no hash computation
- 178k f-string calls takes ~100-200ms, not seconds
- Simpler code, no numpy dependency

#### Data Flow

```
AMT Load (once)                      Expand (per backtest)
─────────────────                    ────────────────────
load_amt()                           expand_fast()
  │                                    │
  ├─ Parse schedules                   ├─ Loop years × months
  ├─ For each (asset, schid):          │   For each template:
  │   ├─ Parse entry: N|0|BD|a|25      │     ├─ ntry/ntrm = arithmetic
  │   ├─ Compute bd from hash          │     └─ straddle = f"|{ntry}...|{suffix}"
  │   ├─ Compute entry_offset          │
  │   └─ Store: (underlying, -1,       └─ Return rows
  │              "N|0|BD|5|25|")
  └─ Cache templates
```

#### Cache Structure

```python
# When AMT is loaded, populate this cache
_PRECOMPUTED_SCHEDULES: dict[str, dict[str, list[tuple[str, int, str]]]] = {}
# path -> schedule_name -> [(underlying, entry_offset, suffix), ...]

def _precompute_schedules(path: str) -> None:
    """Called once when AMT is loaded. Pre-computes all schedule templates."""
    amt = loader.load_amt(path)
    schedules_dict = amt.get("expiry_schedules", {})

    for schedule_name, entries in schedules_dict.items():
        schcnt = len(entries)
        templates = []

        for schid, entry in enumerate(entries, 1):
            # Parse: "N0_BDa_25" or "N|0|BD|a|25"
            ntrc, ntrv, xprc, xprv, wgt = _parse_schedule_entry(entry)

            # Compute entry offset from ntrc
            entry_offset = {"N": -1, "F": -2}.get(ntrc, 0)

            # Fix a/b/c/d -> computed value (will need underlying later)
            # Store as template that needs underlying substitution
            templates.append((ntrc, ntrv, xprc, xprv, wgt, entry_offset, schid, schcnt))

        _PRECOMPUTED_SCHEDULES[path][schedule_name] = templates

def _get_asset_templates(path: str, underlying: str) -> list[tuple[int, str]]:
    """Get pre-computed (entry_offset, suffix) for an asset."""
    asset_data = loader.get_asset(path, underlying)
    schedule_name = asset_data.get("Options")
    templates = _PRECOMPUTED_SCHEDULES[path].get(schedule_name, [])

    u_hash = _underlying_hash(underlying)
    result = []
    for ntrc, ntrv, xprc, xprv, wgt, entry_offset, schid, schcnt in templates:
        # Now compute bd with underlying's hash
        bd = _fix_value(xprv, u_hash, schcnt, schid) if xprv in "abcd" else xprv
        suffix = f"{ntrc}|{ntrv}|{xprc}|{bd}|{wgt}|"
        result.append((entry_offset, suffix))

    return result
```

#### Optional: Macro Sugar (Future)

For human readability, could support macros in YAML:

```yaml
schedule1:
  - '@NEAR_BD@|25'    # Expands to N|0|BD|{computed}|25
  - '@FAR_BD@|25'     # Expands to F|0|BD|{computed}|25
```

But this is polish - the core optimization doesn't need it.

#### Expected Impact

| Operation | Current | Pre-computed |
|-----------|---------|--------------|
| Schedule parsing | ~178k × parse + fix | 0 (done at load) |
| Hash computation | ~178k × hash | ~740 × hash (once per asset×schid) |
| String operations | Multiple splits/joins | 1 f-string concat |
| Memory | Transient | ~10KB cache |

**Measured speedup for `expand()`:** 12.6x (386ms → 31ms)

**Implementation:** See `src/specparser/amt/strings.py`

```bash
$ uv run python -m specparser.amt.strings
Fast straddle expansion benchmark
==================================================
schedules.expand(): 0.386s (177,840 rows)
expand_fast():      0.031s (177,840 rows)
  - precompute:     0.001s (741 templates)
  - expand:         0.030s
Speedup:            12.6x
Match: ✓
```

### 1. Add Columnar Table Representation

Extend `table.py` with column-oriented storage:

```python
# Column-wise representation
{"asset": ["LA", "CL", "NG"], "year": [2024, 2024, 2024], "month": [1, 2, 3]}

def table_to_columnar(table: dict) -> dict[str, list]:
    """Convert row-wise to column-wise."""
    columns = table["columns"]
    rows = table["rows"]
    return {col: [row[i] for i, col in enumerate(columns)] for i, col in enumerate(columns)}

def table_from_columnar(columnar: dict[str, list]) -> dict:
    """Convert column-wise back to row-wise."""
    columns = list(columnar.keys())
    n_rows = len(next(iter(columnar.values()))) if columnar else 0
    rows = [[columnar[col][i] for col in columns] for i in range(n_rows)]
    return {"columns": columns, "rows": rows}
```

**Benefits:**
- Column operations become list operations (can use list comprehensions or NumPy)
- No repeated `.index()` lookups
- Easy to add/drop columns

### 2. Expanded Straddle Representation

Instead of packing/unpacking straddle strings, keep components as separate columns:

```python
# Instead of:
{"columns": ["asset", "straddle"], "rows": [["LA", "|2023-12|2024-01|N|0|F|3|12.5|"]]}

# Use:
{
    "asset": ["LA"],
    "ntry": [2023], "ntrm": [12],
    "xpry": [2024], "xprm": [1],
    "ntrc": ["N"], "ntrv": [0],
    "xprc": ["F"], "xprv": [3],
    "wgt": [12.5]
}
```

**Benefits:**
- No string parsing overhead
- Direct access to components
- Numeric columns can be NumPy arrays for vectorized ops

### 3. Batch Straddle Generation

Replace `_expand_and_pack()` with direct columnar generation:

```python
def expand_columnar(
    path: str,
    start_year: int,
    end_year: int,
    pattern: str = ".",
    live_only: bool = True
) -> dict[str, list]:
    """Expand schedules directly into columnar format."""
    schedules = find_schedules(path, pattern, live_only)

    # Pre-allocate lists
    n_months = (end_year - start_year + 1) * 12
    n_schedules = len(schedules["rows"])
    total = n_months * n_schedules

    assets = []
    ntry_list, ntrm_list = [], []
    xpry_list, xprm_list = [], []
    # ... etc

    # Generate all combinations
    for xpry in range(start_year, end_year + 1):
        for xprm in range(1, 13):
            for schedule_row in schedules["rows"]:
                # Compute entry year/month from expiry + offset
                # Append to lists
                ...

    return {
        "asset": assets,
        "ntry": ntry_list, "ntrm": ntrm_list,
        "xpry": xpry_list, "xprm": xprm_list,
        ...
    }
```

### 4. Straddle String Packing (Lazy/On-Demand)

Only pack to string format when needed for output:

```python
def pack_straddle(ntry: int, ntrm: int, xpry: int, xprm: int,
                  ntrc: str, ntrv: str, xprc: str, xprv: str, wgt: str) -> str:
    """Pack components into straddle string format."""
    return f"|{ntry}-{ntrm:02d}|{xpry}-{xprm:02d}|{ntrc}|{ntrv}|{xprc}|{xprv}|{wgt}|"

def pack_straddles_column(columnar: dict[str, list]) -> list[str]:
    """Pack all straddles in columnar table to strings."""
    n = len(columnar["asset"])
    return [
        pack_straddle(
            columnar["ntry"][i], columnar["ntrm"][i],
            columnar["xpry"][i], columnar["xprm"][i],
            columnar["ntrc"][i], columnar["ntrv"][i],
            columnar["xprc"][i], columnar["xprv"][i],
            columnar["wgt"][i]
        )
        for i in range(n)
    ]
```

### 5. Vectorized Good-Day Finding (Optional NumPy/Numba)

The `_compute_actions()` function iterates through rows checking "good days". This could be vectorized:

```python
import numpy as np

def find_good_days_mask(vol_col: list[str], hedge_cols: list[list[str]]) -> np.ndarray:
    """Return boolean mask of good days."""
    vol_ok = np.array([v != "none" for v in vol_col])
    hedge_ok = np.ones(len(vol_col), dtype=bool)
    for hedge_col in hedge_cols:
        hedge_ok &= np.array([h != "none" for h in hedge_col])
    return vol_ok & hedge_ok
```

**Numba consideration:** Numba supports typed dicts with string keys, but string formatting is limited. The benefit would be in the loop-intensive good-day searching, not the dict lookups (which are already O(1)).

### 6. Batch Price Lookup

Instead of building per-straddle price dicts, do bulk lookups:

```python
def get_prices_for_straddles(
    prices_dict: dict[str, str],
    straddles: dict[str, list],  # Columnar straddle table
    tickers_by_asset: dict[str, list[tuple[str, str]]]  # Pre-computed ticker mappings
) -> dict[str, dict[str, str]]:
    """Batch price lookup for multiple straddles."""
    # Group straddles by (asset, entry_month, expiry_month) to share date ranges
    # Fetch all prices in one pass
    ...
```

## Implementation Order

### Phase 0: Pre-computed Schedule Templates (Quick Win) ⭐
**Priority: HIGH - Easy to implement, big impact on `expand()`**

1. Add `_precompute_schedules()` called from `load_amt()` or lazily on first expand
2. Cache structure: `path -> schedule_name -> [(ntrc, ntrv, xprc, xprv, wgt, entry_offset, schid, schcnt)]`
3. Add `_get_asset_templates()` that computes (entry_offset, suffix) with bd pre-baked
4. Add `expand_fast()` that just does f-string concatenation
5. Keep `expand()` as fallback, have `expand_fast()` as default

**Files:** `schedules.py` (add cache + fast expand), `loader.py` (trigger precompute)

### Phase 1: Core Table Extensions
1. Add `table_to_columnar()` and `table_from_columnar()` to `table.py`
2. Add column-wise utility functions (filter, map, etc.)
3. Ensure backward compatibility with existing row-wise code

### Phase 2: Expanded Straddle Format
1. Add `expand_columnar()` that generates straddles in columnar format
2. Add `pack_straddle()` and `pack_straddles_column()` for on-demand string packing
3. Modify `backtest.py` to use columnar format internally

### Phase 3: Direct Access Functions
1. Add functions that work directly with columnar straddle data
2. Bypass straddle string parsing in hot paths
3. Keep string-based API for CLI/debugging

### Phase 4: Vectorization (Optional)
1. Identify remaining loop-heavy operations
2. Add NumPy-based alternatives where beneficial
3. Consider Numba for numeric-heavy loops (anchor calculation, good-day finding)

## Files to Modify

| File | Changes |
|------|---------|
| `src/specparser/amt/table.py` | Add columnar conversion, column utilities |
| `src/specparser/amt/schedules.py` | Add `expand_columnar()`, keep existing API |
| `src/specparser/amt/tickers.py` | Add batch functions, columnar-aware variants |
| `scripts/backtest.py` | Use columnar format in main loop |

## Expected Performance Impact

| Operation | Current | Phase 0 (Templates) | Phase 1-3 (Columnar) |
|-----------|---------|---------------------|----------------------|
| `expand()` | Parse + fix + pack per straddle | 1 `.format()` per straddle | Direct columnar gen |
| Straddle string parsing | ~178k × 7 splits | 0 (pre-formatted) | 0 (columnar) |
| Column index lookups | O(n) per access | Same | O(1) dict key |
| Hash computation | Per straddle | Per (asset, schid) | Per (asset, schid) |
| Price lookups | Already O(1) | Same | Same |

**Phase 0 alone:** 50-100x speedup for `expand()` (seconds → tens of milliseconds)

**Overall estimated improvement:** 20-40% reduction in total backtest time. Main wins:
1. Template format eliminates expand overhead
2. Columnar format enables future vectorization
3. Cleaner code, easier to optimize further

## Risks

1. **Memory:** Columnar format uses similar memory but different layout
2. **Compatibility:** Existing code expects row-wise tables - need adapters
3. **Complexity:** Two representations to maintain

## Testing

```bash
# Verify output unchanged
uv run python scripts/backtest.py '^LA Comdty' 2024 2025 > before.tsv
# ... make changes ...
uv run python scripts/backtest.py '^LA Comdty' 2024 2025 > after.tsv
diff before.tsv after.tsv

# Performance comparison
time uv run python scripts/backtest.py '^LA Comdty' 2020 2025 --verbose
```

## Decision Points

1. **NumPy dependency?** Currently not required. Adding it enables vectorization but adds dependency.
2. **Numba dependency?** More aggressive optimization but significant complexity.
3. **Parallel columnar?** Could combine with multiprocessing by partitioning columns.

Recommend starting with Phase 1-2 (pure Python columnar) and measuring before adding NumPy/Numba.
