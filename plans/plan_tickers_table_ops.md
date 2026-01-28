# Plan: Migrate tickers.py to Table Operations

## Status: Phase B COMPLETED ✅

Last updated: 2025-01 (after `overrides_csv` bug fix)

---

## Function Naming Clarification

**IMPORTANT**: There are TWO modules with similarly-named functions. This section clarifies which is which:

### schedules.py - Date Expansion (NO prices)

| Function | Purpose | Output |
|----------|---------|--------|
| `straddle_days(straddle)` | Get dates for ONE straddle | `list[datetime.date]` |
| `find_straddle_days(path, start_year, end_year)` | Expand MANY straddles to daily rows | Table: `[asset, straddle, date]` |
| `find_straddle_days_arrow(...)` | Arrow-optimized version | Arrow table |
| `find_straddle_days_numba(...)` | Numba-optimized version | Arrow table |

### tickers.py - Price Lookup + Actions

| Function | Purpose | Output |
|----------|---------|--------|
| `get_prices(underlying, year, month, i, path, ...)` | Get prices for ONE straddle | Table: `[asset, straddle, date, vol, hedge, ...]` |
| `actions(prices_table, path, overrides_csv)` | Add action/strike columns | Table with action, model, strike columns added |
| `get_straddle_actions(underlying, year, month, i, path, ...)` | Convenience: `get_prices()` + `actions()` | Full table with prices + actions |
| `get_straddle_valuation(...)` | Add valuation (mv, delta, pnl) | Full table with valuation columns |

### Key Difference

- **`schedules.find_straddle_days()`** → Many straddles → dates only (for batch processing)
- **`tickers.get_straddle_actions()`** → One straddle → prices + actions (for single straddle analysis)

The `tickers.get_prices()` function internally calls `schedules.straddle_days()` for date generation.

---

## Overview

The `tickers.py` module contains functions that work with daily price data for straddles. The key functions are:

1. **`tickers.get_straddle_actions()`** - Gets prices + actions for a single straddle
2. **`tickers.get_straddle_valuation()`** - Adds valuation columns (mv, delta, pnl)
3. **`tickers._compute_actions()`** - Determines entry/expiry trigger dates
4. Helper functions for "good day" detection and date arithmetic

This plan documented the migration of monolithic functions to composable table operations.

---

## Review Feedback Summary

Key insights from code review:

1. **Two-track approach**: Keep existing per-straddle path (fast for ~60-90 rows), add batch-oriented path for many straddles
2. **Main bottleneck**: Per-cell Python dict lookups in `get_straddle_actions()` - small per straddle but expensive at scale
3. **Multi-key join**: Current `table_left_join` is single-key; batch path needs (ticker, field, date) join
4. **Pivot scaling**: `table_pivot_wider()` should be used on small partitions; for batch, prefer DuckDB pivot
5. **Cross join caution**: Avoid Cartesian explosions; prefer join-by-(asset,straddle) to get only needed combinations
6. **Lag semantics**: Must define partition_by and order_by (typically `(asset, straddle)` partitioned, `date` ordered)
7. **Null handling**: Arrow paths should use nulls (not "none" string) for validity bitmap benefits
8. **Actions computation**: Keep scalar, but consume precomputed `is_good_day` mask column
9. **Prices dict**: Current path is fast for single-straddle; don't replace unless batching

---

## Key Insight: Separation of Concerns

The old monolithic `tickers.get_straddle_actions()` (formerly `get_straddle_days`) mixed multiple responsibilities. It has been decomposed following single responsibility:

| Responsibility | Module | Function | Status |
|---------------|--------|----------|--------|
| Generate dates for ONE straddle | `schedules` | `straddle_days(straddle)` | ✅ Reused by tickers |
| Expand MANY straddles to dates | `schedules` | `find_straddle_days(path, ...)` | ✅ Separate function |
| Find tickers for straddle | `tickers` | `filter_tickers()` | ✅ Unchanged |
| Lookup prices for dates × tickers | `tickers` | `get_prices()` | ✅ NEW |
| Compute entry/expiry actions | `tickers` | `actions()` | ✅ NEW |
| Convenience: prices + actions | `tickers` | `get_straddle_actions()` | ✅ Composes above |
| Compute valuation (mv, pnl) | `tickers` | `get_straddle_valuation()` | ✅ Uses get_straddle_actions |

### Implemented Function Decomposition ✅

```
tickers.get_prices(underlying, year, month, i, path, chain_csv, prices_parquet)
    │
    ├─ Calls filter_tickers() for ticker info
    ├─ Calls schedules.straddle_days() for date generation  ← uses schedules module
    ├─ Builds ticker_map: param → (ticker, field)
    ├─ Lookups prices from _PRICES_DICT or DuckDB
    ├─ Pivots to wide format (one column per param)
    │
    └─ Output: [asset, straddle, date, vol, hedge, ...]

tickers.actions(prices_table, path, overrides_csv=None)  ← overrides_csv added!
    │
    ├─ Input: prices table from get_prices()
    ├─ Extracts underlying from 'asset' column, straddle from 'straddle' column
    │
    ├─ Calls _compute_actions() for ntry/xpry detection
    │   └─ Uses overrides_csv for assets with xprc='OVERRIDE'
    ├─ Adds action column ("-", "ntry", "xpry")
    ├─ Adds model column from AMT config
    ├─ Adds strike columns (values from ntry row)
    │
    └─ Output: [asset, straddle, date, ..., action, model, strike_vol, strike, expiry]

tickers.get_straddle_actions(underlying, year, month, i, path, ..., overrides_csv=None)
    │
    ├─ Calls get_prices() for price table
    ├─ Calls actions() to add action/strike columns
    │
    └─ Output: Full table with prices + actions

tickers.get_straddle_valuation(underlying, year, month, i, path, ..., overrides_csv=None)
    │
    ├─ Calls get_straddle_actions() for base table
    │
    ├─ Computes mv, delta using model function
    ├─ Tracks prev_mv, prev_delta, prev_hedge for PnL
    ├─ Computes opnl, hpnl, pnl
    │
    └─ Output: [..., mv, delta, opnl, hpnl, pnl]
```

---

## Current Architecture Analysis

### Data Flow in `tickers.get_straddle_actions()` (BEFORE decomposition)

This section describes the OLD monolithic implementation. It's kept for historical reference.

```
filter_tickers(asset, year, month, i)
    │
    ▼ returns ticker table: [asset, straddle, param, source, ticker, field]
    │                       e.g., [("CL", "|2024-01|...|", "vol", "BBG", "CLX4", "PX_LAST"),
    │                              ("CL", "|2024-01|...|", "hedge", "BBG", "CLX4", "PX_LAST")]
    │
    ▼ Parse straddle → entry_ym, expiry_ym
    │
    ▼ Generate dates from entry_month to expiry_month
    │
    ▼ For each (ticker, field) pair: lookup prices from _PRICES_DICT or DuckDB
    │
    ▼ Build output table:
    │   [asset, straddle, date, vol, hedge, hedge1, ...]
    │   One row per day, columns pivoted from params
    │
    ▼ _compute_actions() → find ntry/xpry row indices
    │
    ▼ Add columns: action, model, strike_vol, strike, expiry
    │
    ▼ Return row-oriented table
```

### Key Operations That Need Table Primitives

1. **Pivot/Reshape**: Convert long-form `(param, value)` to wide-form `(vol, hedge, ...)`
2. **Date range generation**: Entry month → expiry month
3. **Price join**: Join dates × tickers with prices table
4. **Good day detection**: `vol != "none" AND hedge != "none"`
5. **Range-based fill**: Fill values only between ntry and xpry rows
6. **Lag/lead operations**: Previous day's mv, delta, hedge for PnL

---

## Proposed Table Operations

### New Primitives Needed

#### 1. `table_pivot_wider()` - Reshape long to wide

Convert rows with a "name" column to multiple columns:

```python
def table_pivot_wider(
    table: dict[str, Any],
    names_from: str,      # Column containing new column names
    values_from: str,     # Column containing values
    id_cols: list[str],   # Columns that identify each row group
    fill_value: Any = None  # Value for missing combinations
) -> dict[str, Any]:
    """Pivot a table from long to wide format.

    Example:
        Input (long):
            date    param   value
            2024-01 vol     100
            2024-01 hedge   50
            2024-02 vol     101
            2024-02 hedge   51

        Output (wide):
            date    vol   hedge
            2024-01 100   50
            2024-02 101   51
    """
```

**Implementation approach:**
- Group by id_cols
- For each group, collect {name: value} mapping
- Build output columns from unique names

#### 2. `table_lag()` / `table_lead()` - Window functions

```python
def table_lag(
    table: dict[str, Any],
    column: str,
    n: int = 1,
    default: Any = None,
    result_column: str | None = None,
    partition_by: list[str] | None = None,  # e.g., ["asset", "straddle"]
    order_by: str | None = None,            # e.g., "date" - table must be pre-sorted
) -> dict[str, Any]:
    """Add column with lagged values.

    IMPORTANT: Table must be pre-sorted by order_by column within each partition.
    Initial implementation assumes single-partition (whole table) and pre-sorted input.

    Example:
        Input:          Output:
        date    vol     date    vol     vol_lag1
        Jan-01  100     Jan-01  100     None
        Jan-02  101     Jan-02  101     100
        Jan-03  102     Jan-03  102     101
    """
```

**Arrow implementation**: Use `pc.list_slice()` with offset on chunked array, or manual shift.

**Partition-aware implementation (future)**:
- Group by partition_by columns
- Apply lag within each group
- Reassemble table preserving original order

#### 3. `table_fill_null_range()` - Conditional fill

```python
def table_fill_null_range(
    table: dict[str, Any],
    column: str,
    fill_value: Any,
    start_mask: str,      # Column with True at range start
    end_mask: str,        # Column with True at range end
    outside_value: Any = None  # Value outside range
) -> dict[str, Any]:
    """Fill values only within a marked range.

    Used for strike columns that only have values between ntry and xpry.
    """
```

#### 4. `table_cummax()` with reset - Rolling good day

Actually, the "good day" detection is more like a filter + first/last operation:

```python
def table_first_where(
    table: dict[str, Any],
    condition: str,       # Column name with boolean mask
    partition_by: list[str] | None = None
) -> dict[str, Any]:
    """Return first row where condition is True (per partition)."""

def table_last_where(
    table: dict[str, Any],
    condition: str,
    partition_by: list[str] | None = None
) -> dict[str, Any]:
    """Return last row where condition is True (per partition)."""
```

---

## Refactoring Plan

### Phase 1: Create Date-Ticker Base Table

Instead of generating dates in Python loop, use table operations:

```python
def _build_straddle_date_table(
    straddle: str,
    ticker_table: dict[str, Any]
) -> dict[str, Any]:
    """Build base table with date × ticker combinations.

    Input ticker_table:
        [asset, straddle, param, source, ticker, field]

    Output:
        [asset, straddle, date, param, ticker, field]
        One row per (date, param) combination
    """
    # Extract entry/expiry from straddle
    entry_ym = straddle[1:8]  # "2024-01"
    expiry_ym = straddle[9:16]  # "2024-03"

    # Use calendar table from schedules module
    dates_table = _get_date_range_table(entry_ym, expiry_ym)
    # Cross join: dates × ticker params
    return table_cross_join(dates_table, ticker_table)
```

**New primitive needed**: `table_cross_join()` (Cartesian product)

### Phase 2: Join with Prices

```python
def _join_prices(
    date_ticker_table: dict[str, Any],
    prices_table: dict[str, Any]  # [ticker, field, date, value]
) -> dict[str, Any]:
    """Left join to get price values.

    Join keys: (ticker, field, date)
    """
    return table_left_join(
        date_ticker_table, prices_table,
        left_on=["ticker", "field", "date"],
        right_on=["ticker", "field", "date"]
    )
```

**Multi-key join options:**

1. **Extend `table_left_join`** to accept list of key columns:
   - Row path: use tuple key in dict
   - Arrow path: use `pa.Table.join` with multiple keys (supported in PyArrow)

2. **Use DuckDB for price join** (recommended for batch):
   - Skip implementing multi-key join in Python
   - DuckDB handles multi-key joins efficiently
   - Better "bulk I/O boundary" than Python dict lookups

**Practical suggestion**: For batch pricing, use DuckDB for join + pivot, then hand off to table layer for downstream ops.

### Phase 3: Pivot to Wide Format

```python
def _pivot_prices_wide(
    joined_table: dict[str, Any]
) -> dict[str, Any]:
    """Pivot param values to columns.

    Input:
        [asset, straddle, date, param, value]
        ("CL", "|...|", "2024-01-02", "vol", "100")
        ("CL", "|...|", "2024-01-02", "hedge", "50")

    Output:
        [asset, straddle, date, vol, hedge]
        ("CL", "|...|", "2024-01-02", "100", "50")
    """
    return table_pivot_wider(
        joined_table,
        names_from="param",
        values_from="value",
        id_cols=["asset", "straddle", "date"],
        fill_value="none"
    )
```

### Phase 4: Compute Good Day Mask

**Current approach (string sentinel):**
```python
def _add_good_day_column(table: dict[str, Any]) -> dict[str, Any]:
    """Add boolean column: vol != "none" AND all hedges != "none"."""
    vol_valid = table_not_equal_arrow(table, "vol", "none", result_column="_vol_valid")
    hedge_valid = table_not_equal_arrow(vol_valid, "hedge", "none", result_column="_hedge_valid")
    return table_and_arrow(hedge_valid, "_vol_valid", "_hedge_valid", result_column="is_good_day")
```

**Better approach for Arrow paths (nulls instead of "none"):**
```python
def _add_good_day_column_arrow(table: dict[str, Any]) -> dict[str, Any]:
    """Add boolean column using Arrow null validity.

    Uses Arrow's native null handling - faster than string comparison.
    Requires prices stored as nulls (not "none" strings).
    """
    import pyarrow.compute as pc

    vol_col = table["rows"][table["columns"].index("vol")]
    hedge_col = table["rows"][table["columns"].index("hedge")]

    # is_valid returns True for non-null values
    vol_valid = pc.is_valid(vol_col)
    hedge_valid = pc.is_valid(hedge_col)

    is_good_day = pc.and_(vol_valid, hedge_valid)
    # ... add column to table
```

**Recommendation**: For Arrow batch paths, store prices as typed arrays with nulls (not "none" strings). Convert to "none" only at JSON output boundary.

### Phase 5: Compute Actions (Entry/Expiry Triggers)

This is the most complex part. The current logic:

1. Compute anchor date from xprc/xprv (e.g., "3rd Friday")
2. Add ntrv calendar days for entry
3. Find first good day at/after anchor
4. If none found, use last good day in month

**Table-based approach:**

```python
def _compute_entry_expiry_indices(
    table: dict[str, Any],
    straddle: str,
    underlying: str
) -> tuple[int | None, int | None]:
    """Find row indices for entry and expiry triggers.

    Returns (ntry_idx, xpry_idx)
    """
    # Parse straddle fields
    xprc = schedules.xprc(straddle)
    xprv = schedules.xprv(straddle)
    ntrv = schedules.ntrv(straddle)
    ntry, ntrm = schedules.ntry(straddle), schedules.ntrm(straddle)
    xpry, xprm = schedules.xpry(straddle), schedules.xprm(straddle)

    # 1. Compute anchor dates (scalar Python - unavoidable)
    entry_anchor = _anchor_day(xprc, xprv, ntry, ntrm, underlying)
    expiry_anchor = _anchor_day(xprc, xprv, xpry, xprm, underlying)

    # 2. Filter table to good days within date ranges
    entry_month_end = f"{ntry}-{ntrm:02d}-{calendar.monthrange(ntry, ntrm)[1]:02d}"

    # 3. Find first good day >= entry_anchor + ntrv days
    target_entry_date = _add_calendar_days(entry_anchor, int(ntrv or 0))

    # Filter: is_good_day AND date >= target_entry_date AND date <= entry_month_end
    entry_candidates = table_filter_arrow(table, ...)  # complex filter

    # Get first row index, or fall back to last good day in month
    ...
```

**Assessment**: The action computation involves date arithmetic (anchor calculation, adding calendar days) that is inherently scalar and date-dependent. This is hard to vectorize without a complete rewrite of the calendar logic.

**Recommendation**: Keep action computation as Python loop, but operate on the table's date and good_day columns directly rather than raw row lists.

### Phase 6: Add Derived Columns

Once we have the base table with prices and action column:

```python
def _add_strike_columns(
    table: dict[str, Any],
    ntry_idx: int,
    xpry_idx: int
) -> dict[str, Any]:
    """Add strike_vol, strike, expiry columns.

    Values come from ntry row, shown only between ntry and xpry.
    """
    # Get values from ntry row
    ntry_row = table_head(table_filter_by_index(table, [ntry_idx]), 1)
    strike_vol = table_column(ntry_row, "vol")[0]
    strike_hedge = table_column(ntry_row, "hedge")[0]

    # Add constant columns
    table = table_add_column(table, "strike_vol", value=strike_vol)
    table = table_add_column(table, "strike", value=strike_hedge)

    # Mask outside ntry-xpry range to "-"
    # Need: table_replace_range() or conditional assignment
    ...
```

### Phase 7: Valuation Computation

The `get_straddle_valuation()` function adds:
- `mv` and `delta` from model function
- `opnl = mv[today] - mv[yesterday]`
- `hpnl = -delta[yesterday] * (hedge[today] - hedge[yesterday]) / strike`
- `pnl = opnl + hpnl`

**Table-based approach with lag:**

```python
def _add_valuation_columns(
    table: dict[str, Any],
    model_fn: Callable,
    ntry_idx: int,
    xpry_idx: int
) -> dict[str, Any]:
    """Add mv, delta, opnl, hpnl, pnl columns."""

    # 1. Add row index column
    table = table_add_column(table, "_row_idx", value=lambda i: i)

    # 2. Compute mv and delta (need UDF support or per-row model call)
    # This is hard to vectorize since model_fn expects dict input
    # Option A: Keep as loop
    # Option B: Implement table_apply() for row-wise UDF

    # 3. Add lag columns
    table = table_lag(table, "mv", n=1, result_column="mv_lag")
    table = table_lag(table, "delta", n=1, result_column="delta_lag")
    table = table_lag(table, "hedge", n=1, result_column="hedge_lag")

    # 4. Compute derived columns
    # opnl = mv - mv_lag
    table = table_subtract_arrow(table, "mv", "mv_lag", result_column="opnl")

    # hpnl = -delta_lag * (hedge - hedge_lag) / strike
    # This needs multiple operations...

    # 5. Mask values outside ntry-xpry range
    ...
```

**Valuation optimization notes:**

1. **Keep model_fn loop** - Python UDF, unlikely to vectorize profitably
2. **Use `table_lag()` for PnL plumbing** - cleaner, less error-prone
3. **Reduce per-row overhead**:
   - Current: `dict(zip(columns, row))` per row is expensive
   - Better: keep numeric columns typed (float arrays), pass pre-indexed lookups
4. **Type handling**:
   - Current: returns row-oriented strings; valuation casts strings to float repeatedly
   - Better: use numeric arrays + nulls internally, convert at output edge

---

## New Table Primitives Summary

| Primitive | Priority | Complexity | Notes |
|-----------|----------|------------|-------|
| `table_pivot_wider()` | High | Medium | Essential for param→columns transform; document duplicate handling |
| `table_lag()` | High | Low | Arrow: shift array, pad with null; document partition/order assumptions |
| `table_cross_join()` | Low | Low | Cartesian product - **use sparingly**, prefer join-by-key to avoid explosion |
| `table_filter_by_index()` | Medium | Low | Select rows by integer indices |
| `table_apply()` | Low | High | Row-wise UDF application |
| `table_replace_range()` | Low | Medium | Conditional column update |

### Primitive Design Notes

**`table_pivot_wider()` semantics to define:**
- Duplicate handling for same (id_cols, name) pair: last wins? first wins? error?
- Column ordering: preserve order of first occurrence (important for params_ordered)
- Use on small per-straddle partitions; for large batch, prefer DuckDB pivot

**`table_lag()` requirements:**
- Document: assumes table is pre-sorted by order column within partition
- Initial version: single-partition only (whole table), pre-sorted input
- Future: add partition_by/order_by parameters with sort enforcement

**Cross join alternative:**
Instead of `table_cross_join(dates, tickers)` which explodes row count, prefer:
```python
# Join straddle-days to per-straddle ticker table on (asset, straddle)
# This yields (asset, straddle, date, param, ticker, field) without full Cartesian
table_left_join(straddle_days, ticker_table, ["asset", "straddle"])
```

---

## Implementation Phases

### Phase A: Core Primitives (Do First)

1. **`table_pivot_wider()`** - Required for reshape
2. **`table_lag()`** - Required for PnL calculations
3. **`table_cross_join()`** - Required for date × ticker expansion

### Phase B: Refactor `get_straddle_actions()`

1. Replace date generation loop with calendar table
2. Replace price lookup loop with table join
3. Replace pivot logic with `table_pivot_wider()`
4. Keep action computation as-is (scalar date logic)
5. Use table operations for derived columns

### Phase C: Refactor `get_straddle_valuation()`

1. Add lag columns with `table_lag()`
2. Compute PnL columns with arithmetic operations
3. Keep model function call as loop (UDF)

---

## Risks and Trade-offs

### 1. Performance

For single-straddle queries (`get_straddle_actions` with one asset/month), the table operation overhead may not be worth it. The current Python loop is simple and fast for ~60-90 rows (2-3 months of daily data).

**Mitigation**: Keep both implementations, use table version for batch operations.

### 2. Complexity

The pivot operation adds complexity. Current code is straightforward: loop over dates, lookup each param.

**Mitigation**: Good documentation, comprehensive tests.

### 3. Action Computation

The anchor date calculation (`_anchor_day`) involves calendar logic that doesn't vectorize well. The current approach (Python loop to find good day) is appropriate.

**Mitigation**: Keep as-is, just clean up the interface to work with table data.

---

## Alternative: Minimal Changes

Instead of full refactoring, we could:

1. **Keep `get_straddle_actions()` mostly as-is** - It's already reasonably clean
2. **Add `table_lag()` for valuation** - Main benefit is cleaner PnL code
3. **Add `table_pivot_wider()` for other use cases** - General utility

This minimizes risk while still providing useful primitives.

---

## Recommended Approach

**Phase A: Core Primitives (DONE)**

1. ✅ Implement `table_pivot_wider()` as a general utility (Python, for small tables)
2. ✅ Implement `table_lag()` / `table_lead()` for window operations (single-partition, pre-sorted)
3. ✅ Skip `table_cross_join()` - prefer join-by-key to avoid Cartesian explosion
4. ✅ Implement `find_straddle_days_numba()` for fast date expansion

**Phase B: Refactor `get_straddle_actions()` (DO NOW)**

The current `get_straddle_actions()` mixes multiple responsibilities and needs to be split into focused, composable functions. This refactoring is necessary to:
- Follow single-responsibility principle
- Enable reuse of components (e.g., price lookup for other use cases)
- Make testing easier (each function can be tested independently)
- Prepare for batch processing by having clean building blocks

See **Concrete Refactoring Steps** section below for detailed implementation plan.

**Future: Batch API (Phase C)**

Once Phase B is complete, add batch processing for many straddles:

```python
def get_straddle_actions_batch(
    path: str,
    straddles: list[tuple[str, int, int, int]],  # [(asset, year, month, i), ...]
    prices_source: str | Path,
) -> dict[str, Any]:
    """Batch process many straddles at once.

    Pipeline:
    1. Get straddle-days from schedules.find_straddle_days() (Arrow)
    2. Get per-straddle tickers (batch variant of filter_tickers)
    3. Fetch prices in bulk via DuckDB (ticker, field, date, value)
    4. Join prices to straddle-days in DuckDB (multi-key join)
    5. Pivot to wide format in DuckDB (SQL pivot / conditional aggregates)
    6. Return Arrow table for downstream processing
    """
```

**Batch path design principles:**
- Use DuckDB for multi-key joins (ticker, field, date) - faster than Python
- Use DuckDB pivot for large datasets - scales better than Python dict approach
- Keep action computation scalar (unavoidable calendar logic)
- Use nulls (not "none" string) internally for Arrow validity bitmap benefits
- Convert to "none" strings only at final output edge (JSON export)

**Batch I/O boundary:**
- Prefer fetching prices as long table `(ticker, field, date, value)` from DuckDB once
- Then join/reshape - better than building Python dicts per straddle

---

## Files to Modify

1. **`src/specparser/amt/table.py`** - Add new primitives:
   - `table_pivot_wider()`
   - `table_lag()` / `table_lead()`
   - `table_cross_join()`

2. **`src/specparser/amt/__init__.py`** - Export new functions

3. **`tests/test_amt.py`** - Add tests for new primitives

4. **`src/specparser/amt/tickers.py`** - Optional refactoring (Phase B/C)

---

## Verification

```bash
# Run existing tests (ensure no regression)
uv run pytest tests/test_amt.py -v

# Test new primitives
uv run pytest tests/test_amt.py -v -k "pivot or lag"

# Verify tickers.get_straddle_actions still works
uv run python -m specparser.amt.tickers data/amt.yml --asset-days "CL Comdty" 2024 6 0
```

---

## Risk Summary

| Risk | Mitigation |
|------|------------|
| Multi-key joins needed for batch | Use DuckDB for price joins, not Python table layer |
| Pivot scaling on large datasets | Use DuckDB pivot for batch; Python pivot only for small partitions |
| Cross join explosion | Prefer join-by-key (asset, straddle) over Cartesian product |
| Lag without partition awareness | Document pre-sort requirement; add partition_by later |
| "none" strings vs nulls | Use nulls internally for Arrow; convert at output edge |
| Over-engineering single-straddle path | Keep current code; add batch API separately |

---

## Action Items (Phase A) - COMPLETED ✅

1. ✅ **Implemented `table_pivot_wider()`**
   - Python implementation for row/column tables
   - Duplicate handling: last wins
   - Column ordering: stable (first occurrence order)

2. ✅ **Implemented `table_lag()` / `table_lead()`**
   - Simple version: single-partition, assumes pre-sorted
   - Arrow-optimized path using `slice()` + `concat_arrays()`

3. ✅ **Skipped `table_cross_join()`**
   - Use join-by-key pattern instead
   - Avoids Cartesian explosion risk

4. ✅ **Implemented `schedules.find_straddle_days_numba()`**
   - Numba kernels in `schedules_numba.py` and `valuation_numba.py`
   - Sequential and parallel versions
   - Note: This is in `schedules.py`, NOT `tickers.py`

---

## Phase B - COMPLETED ✅

The `tickers.get_straddle_actions()` function (price lookup + actions) has been decomposed into focused, composable functions.

**Note**: This is different from `schedules.find_straddle_days()` which expands straddles to dates without prices.

### New Public API (tickers.py)

| Function | Purpose | Lines |
|----------|---------|-------|
| `get_prices()` | Get daily prices for a straddle (dates × tickers → wide table) | 1596-1642 |
| `actions()` | Add action, model, and strike columns to a prices table | 1645-1679 |
| `get_straddle_actions()` | Convenience function composing `get_prices()` + `actions()` | 1682-1728 |
| `get_straddle_valuation()` | Add valuation columns (mv, delta, opnl, hpnl, pnl) | 1708+ |

### Helper Functions (private)

| Function | Purpose |
|----------|---------|
| `_build_ticker_map()` | Extract param → (ticker, field) mapping from ticker table |
| `_lookup_straddle_prices()` | Lookup prices for dates × tickers using dict or DuckDB |
| `_build_prices_table()` | Assemble wide-format table from dates and prices |
| `_add_action_column()` | Compute ntry/xpry actions using `_compute_actions()` |
| `_add_model_column()` | Add model name column from AMT config |
| `_add_strike_columns()` | Add strike_vol, strike, expiry columns (values from ntry row) |
| `_find_action_indices()` | Find ntry/xpry row indices in table |

### Bug Fix: `overrides_csv` Parameter Threading (2024-01)

**Problem discovered**: After the initial decomposition, the `overrides_csv` parameter was not being threaded through the function call chain. This caused `_compute_actions()` to use a hardcoded default path `data/overrides.csv` which may not exist from all working directories (e.g., Jupyter notebooks running from different locations).

**Root cause**: When `actions()` was created, the `overrides_csv` parameter was omitted from:
1. `_add_action_column()` - didn't accept or pass the parameter
2. `actions()` - didn't accept the parameter
3. `get_straddle_actions()` - didn't accept or pass the parameter
4. `get_straddle_valuation()` - didn't accept or pass the parameter

**Fix applied**: Added `overrides_csv: str | Path | None = None` parameter to all functions in the chain:
- `_add_action_column(table, straddle, underlying, overrides_csv=None)` → passes to `_compute_actions()`
- `actions(prices_table, path, overrides_csv=None)` → passes to `_add_action_column()`
- `get_straddle_actions(..., overrides_csv=None)` → passes to `actions()`
- `get_straddle_valuation(..., overrides_csv=None)` → passes to `get_straddle_actions()`

**Lesson learned**: When decomposing functions, audit ALL parameters from the original monolithic function to ensure they are threaded through all layers of the new decomposed functions. This is especially critical for optional configuration parameters that have working defaults in some contexts but fail in others.

**Testing approach**: The fix was verified by:
1. Running all 52 straddle-related tests (passed)
2. Manual verification from command line with explicit `overrides_csv` path
3. Confirming `ntry` and `xpry` actions appear at correct indices

### Composition Pattern

```python
# Low-level: Get just prices
prices = get_prices(underlying, year, month, i, path, chain_csv, prices_parquet)
# → Table: [asset, straddle, date, vol, hedge, ...]

# Mid-level: Add actions and strikes
# NOTE: overrides_csv is REQUIRED for assets with xprc='OVERRIDE' (custom expiry dates)
full = actions(prices, path, overrides_csv)
# → Table: [asset, straddle, date, vol, hedge, ..., action, model, strike_vol, strike, expiry]

# High-level: Convenience function (does both)
full = get_straddle_actions(underlying, year, month, i, path, chain_csv, prices_parquet, overrides_csv)

# Valuation: Add mv, delta, pnl columns
valued = get_straddle_valuation(underlying, year, month, i, path, chain_csv, prices_parquet, overrides_csv)
```

**Important**: For assets with `xprc='OVERRIDE'` in their straddle string, the `overrides_csv` path MUST be provided to correctly compute entry/expiry trigger dates. Without it, the function uses a hardcoded default path which may not exist.

### Exports (specparser.amt)

```python
from specparser.amt import (
    get_prices,           # Price lookup only
    actions,              # Add actions/strikes to prices table
    get_straddle_actions,    # Convenience: prices + actions
    get_straddle_valuation,  # Full valuation with PnL
)
```

### Complete Function Signatures (Updated)

```python
def get_prices(
    underlying: str,
    year: int,
    month: int,
    i: int,
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
) -> dict[str, Any]:
    """Get daily prices for a straddle (no actions/strikes)."""

def actions(
    prices_table: dict[str, Any],
    path: str | Path,
    overrides_csv: str | Path | None = None,  # ← Required for OVERRIDE assets
) -> dict[str, Any]:
    """Add action, model, and strike columns to a prices table.

    The underlying is extracted from the 'asset' column in prices_table.
    The straddle is extracted from the 'straddle' column in prices_table.
    """

def get_straddle_actions(
    underlying: str,
    year: int,
    month: int,
    i: int,
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
    overrides_csv: str | Path | None = None,  # ← Required for OVERRIDE assets
) -> dict[str, Any]:
    """Convenience: get_prices() + actions()."""

def get_straddle_valuation(
    underlying: str,
    year: int,
    month: int,
    i: int,
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
    overrides_csv: str | Path | None = None,  # ← Required for OVERRIDE assets
) -> dict[str, Any]:
    """Full valuation with mv, delta, opnl, hpnl, pnl columns."""
```

### Design Decision: `actions()` Extracts from Table

The `actions()` function was designed to extract `underlying` and `straddle` from the input `prices_table` rather than requiring them as separate parameters. This:

1. **Reduces parameter duplication** - caller doesn't need to pass same info twice
2. **Ensures consistency** - values match what's in the table
3. **Simplifies composition** - `get_straddle_actions()` just pipes output of `get_prices()` to `actions()`

The trade-off is that `prices_table` MUST have 'asset' and 'straddle' columns populated.

---

## Lessons Learned from Phase B Implementation

### 1. Parameter Threading in Decomposed Functions

When decomposing a monolithic function into smaller composable pieces, **audit every parameter** from the original function signature. Optional configuration parameters (like `overrides_csv`) are easy to miss because:
- They have default values that work in some contexts
- Tests may not exercise all code paths that use them
- The bug manifests as silent incorrect behavior (no action column populated), not a crash

**Checklist for function decomposition:**
- [ ] List all parameters from original function
- [ ] For each parameter, trace which internal code path uses it
- [ ] Add parameter to ALL functions in the call chain that need it
- [ ] Add integration tests that exercise the parameter end-to-end

### 2. Relative Path Sensitivity

Functions that accept file paths must work regardless of caller's working directory. The bug was masked because:
- Command-line tests ran from project root where `data/overrides.csv` existed
- Jupyter notebook ran from `notebooks/` where `../data/overrides.csv` was correct but code used hardcoded `data/overrides.csv`

**Best practice:** Never hardcode relative paths in default parameter values. Either:
- Require the path explicitly (no default)
- Default to `None` and raise clear error if needed but not provided
- Use absolute paths derived from `__file__` if truly fixed resources

### 3. Testing Decomposed Functions

After decomposition, test at multiple levels:
1. **Unit tests** for each helper function (e.g., `_add_action_column`)
2. **Integration tests** for public API (e.g., `actions()`)
3. **Regression tests** comparing output before/after refactor
4. **End-to-end tests** from typical caller contexts (CLI, notebook, etc.)

The `overrides_csv` bug would have been caught by an integration test that:
```python
def test_actions_with_override_asset():
    # Load prices for an asset with xprc='OVERRIDE'
    prices = get_prices('CL Comdty', 2024, 6, 0, 'data/amt.yml', 'data/futs.csv')
    # Must provide overrides_csv for OVERRIDE assets
    result = actions(prices, 'data/amt.yml', 'data/overrides.csv')
    # Verify actions are computed
    action_col = table_column(result, 'action')
    assert 'ntry' in action_col
    assert 'xpry' in action_col
```

### 4. Debugging Strategy for Silent Failures

When a function produces wrong output (not an error), systematic debugging approach:
1. **Trace the data flow** - What value is passed at each layer?
2. **Test lowest layer first** - Verify `_compute_actions()` works in isolation
3. **Add intermediate logging** or test each layer with explicit inputs
4. **Compare with known-good execution** - Command line vs notebook

In this case, testing `_compute_actions()` directly revealed it worked fine when given the right path, pointing to the parameter threading issue.

---

## Concrete Refactoring Steps (Phase B) - COMPLETED ✅

### Goal

Split `get_straddle_actions()` (currently ~225 lines, tickers.py:1257-1480) into focused functions following the decomposition in the "Key Insight: Separation of Concerns" section.

### Step 1: Extract `_generate_straddle_dates()`

Extract the date generation logic (lines 1336-1346) into a standalone function that uses `schedules.straddle_days()`.

```python
def _generate_straddle_dates(straddle: str) -> list[datetime.date]:
    """Generate all dates from entry month to expiry month.

    Uses schedules.straddle_days() which is already implemented and cached.

    Args:
        straddle: Straddle string like "|2024-01|2024-03|N|5|F||33.3|"

    Returns:
        List of date objects from entry month through expiry month
    """
    return schedules.straddle_days(straddle)
```

**Note**: `schedules.straddle_days()` already exists and does this! The current code duplicates this logic.

### Step 2: Extract `_lookup_straddle_prices()`

Extract the price lookup logic (lines 1348-1403) into a focused function.

```python
def _lookup_straddle_prices(
    dates: list[datetime.date],
    ticker_map: dict[str, tuple[str, str]],  # param -> (ticker, field)
    prices_dict: dict[str, str] | None = None,
    prices_parquet: str | Path | None = None,
) -> dict[tuple[str, str], dict[str, str]]:
    """Lookup prices for all dates and ticker/field pairs.

    Args:
        dates: List of dates to lookup
        ticker_map: Mapping from param name to (ticker, field)
        prices_dict: Preloaded prices dict (ticker|field|date -> value)
        prices_parquet: Path to parquet for DuckDB fallback

    Returns:
        Dict mapping (ticker, field) -> {date_str -> value}
    """
```

### Step 3: Extract `_build_prices_table()`

Extract the table building logic (lines 1393-1403) into a function that produces a table.

```python
def _build_prices_table(
    asset: str,
    straddle: str,
    dates: list[datetime.date],
    params_ordered: list[str],
    ticker_map: dict[str, tuple[str, str]],
    prices: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    """Build the prices table with one row per day.

    Returns:
        Table with columns: [asset, straddle, date, <params...>]
    """
```

### Step 4: Extract `_add_action_column()`

The action computation (`_compute_actions`) already exists as a separate function.
Create a wrapper that adds the column to a table.

```python
def _add_action_column(
    table: dict[str, Any],
    straddle: str,
    underlying: str,
) -> dict[str, Any]:
    """Add action column to prices table.

    Calls existing _compute_actions() and appends result as column.
    """
```

### Step 5: Extract `_add_strike_columns()`

Extract the strike column logic (lines 1423-1478) into a function.

```python
def _add_strike_columns(
    table: dict[str, Any],
    ntry_idx: int | None,
    xpry_idx: int | None,
) -> dict[str, Any]:
    """Add strike_vol, strike, strike1..., and expiry columns.

    Values come from ntry row, shown only between ntry and xpry.
    """
```

### Step 6: Create `get_prices()` (New Public API) ✅

Implemented as `get_prices()` (cleaner name than originally planned `find_straddle_prices()`).

```python
def get_prices(
    underlying: str,
    year: int,
    month: int,
    i: int,
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
) -> dict[str, Any]:
    """Get daily prices for a straddle from entry to expiry.

    This is a focused function that ONLY handles price lookup.
    Does NOT compute actions, strikes, or model columns.

    Returns:
        Table with columns: [asset, straddle, date, vol, hedge, ...]
    """
    # 1. Get ticker table
    ticker_table = filter_tickers(underlying, year, month, i, path, chain_csv)

    # 2. Extract straddle string
    straddle = ticker_table["rows"][0][1]

    # 3. Generate dates using existing schedules.straddle_days()
    dates = schedules.straddle_days(straddle)

    # 4. Build ticker map
    ticker_map = _build_ticker_map(ticker_table)

    # 5. Lookup prices
    prices = _lookup_straddle_prices(dates, ticker_map, _PRICES_DICT, prices_parquet)

    # 6. Build and return table
    return _build_prices_table(asset, straddle, dates, params_ordered, ticker_map, prices)
```

### Step 7: Create `actions()` (New Public API) ✅

Implemented as `actions()` (cleaner name than originally planned `find_straddle_actions()`).

```python
def actions(
    prices_table: dict[str, Any],
    straddle: str,
    underlying: str,
    path: str | Path,
) -> dict[str, Any]:
    """Add action and strike columns to a prices table.

    Args:
        prices_table: Output from find_straddle_prices()
        straddle: Straddle string
        underlying: Asset underlying
        path: Path to AMT YAML

    Returns:
        Table with added columns: action, model, strike_vol, strike, expiry
    """
    # 1. Add action column
    table = _add_action_column(prices_table, straddle, underlying)

    # 2. Add model column
    table = _add_model_column(table, underlying, path)

    # 3. Add strike columns
    ntry_idx, xpry_idx = _find_action_indices(table)
    table = _add_strike_columns(table, ntry_idx, xpry_idx)

    return table
```

### Step 8: Refactor `get_straddle_actions()` to Compose ✅

Refactored to compose the new building blocks:

```python
def get_straddle_actions(
    underlying: str,
    year: int,
    month: int,
    i: int,
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
) -> dict[str, Any]:
    """Get daily prices for a straddle from entry to expiry month.

    This is a convenience function that composes:
    1. get_prices() - price lookup
    2. actions() - action, model, and strike columns

    For more control, use the individual functions directly.
    """
    # Get prices table
    prices_table = get_prices(
        underlying, year, month, i, path, chain_csv, prices_parquet
    )

    if not prices_table["rows"]:
        return prices_table

    # Add actions, model, and strikes
    return actions(prices_table, underlying, path)
```

### Step 9: Update `get_straddle_valuation()` to Use `table_lag()`

Refactor PnL calculation to use the new `table_lag()` primitive:

```python
def get_straddle_valuation(...) -> dict[str, Any]:
    # ... existing code to get base table and compute mv/delta ...

    # Use table_lag for PnL plumbing (cleaner than manual tracking)
    table = table_lag(table, "mv", n=1, result_column="mv_lag")
    table = table_lag(table, "delta", n=1, result_column="delta_lag")
    table = table_lag(table, "hedge", n=1, result_column="hedge_lag")

    # Compute opnl = mv - mv_lag
    # Compute hpnl = -delta_lag * (hedge - hedge_lag) / strike
    # Compute pnl = opnl + hpnl

    # Drop intermediate lag columns if desired
    ...
```

### Testing Strategy

1. **Regression tests**: Ensure refactored `get_straddle_actions()` produces identical output
2. **Unit tests for each new function**:
   - `_generate_straddle_dates()` - verify date ranges
   - `_lookup_straddle_prices()` - mock prices dict
   - `_build_prices_table()` - verify table structure
   - `find_straddle_prices()` - integration test
   - `find_straddle_actions()` - verify action/strike columns

3. **Test using table_lag in valuation**:
   - Compare output before/after refactor
   - Verify PnL calculations are correct

### Files to Modify

1. **`src/specparser/amt/tickers.py`**:
   - Extract helper functions
   - Add `find_straddle_prices()`
   - Add `find_straddle_actions()`
   - Refactor `get_straddle_actions()` to compose
   - Update `get_straddle_valuation()` to use `table_lag()`

2. **`src/specparser/amt/__init__.py`**:
   - Export `find_straddle_prices`
   - Export `find_straddle_actions`

3. **`tests/test_amt.py`**:
   - Add regression tests
   - Add unit tests for new functions

### Verification Commands

```bash
# Run all tests
uv run pytest tests/test_amt.py -v

# Test specifically tickers functions
uv run pytest tests/test_amt.py -v -k "straddle"

# Manual verification with all parameters
uv run python -c "
from specparser.amt import tickers, load_all_prices

# Load prices first
load_all_prices('data/prices.parquet')

# Test get_prices (no overrides needed - just price lookup)
prices = tickers.get_prices('CL Comdty', 2024, 6, 0, 'data/amt.yml', 'data/futs.csv')
print(f'Prices columns: {prices[\"columns\"]}')
print(f'Prices rows: {len(prices[\"rows\"])}')

# Test actions with overrides (REQUIRED for OVERRIDE assets)
full = tickers.actions(prices, 'data/amt.yml', 'data/overrides.csv')
print(f'Full columns: {full[\"columns\"]}')
action_col_idx = full['columns'].index('action')
actions = set(row[action_col_idx] for row in full['rows'])
print(f'Actions in table: {actions}')
assert 'ntry' in actions, 'Missing ntry action!'
assert 'xpry' in actions, 'Missing xpry action!'
print('✓ Actions computed correctly')
"

# Test from a different working directory (simulating notebook)
cd /tmp && uv run python -c "
import sys
sys.path.insert(0, '/Users/nicknassuphis/specparser/src')
from specparser.amt import tickers, load_all_prices

# Use absolute paths (as notebook would need to)
load_all_prices('/Users/nicknassuphis/specparser/data/prices.parquet')
prices = tickers.get_prices('CL Comdty', 2024, 6, 0,
    '/Users/nicknassuphis/specparser/data/amt.yml',
    '/Users/nicknassuphis/specparser/data/futs.csv')
full = tickers.actions(prices,
    '/Users/nicknassuphis/specparser/data/amt.yml',
    '/Users/nicknassuphis/specparser/data/overrides.csv')
action_col_idx = full['columns'].index('action')
actions = set(row[action_col_idx] for row in full['rows'])
print(f'Actions from /tmp: {actions}')
"
```

---

## Next Steps (Phase B.1 - Testing & Documentation)

With the `overrides_csv` bug fixed, the next immediate tasks are:

### 1. Add Missing Integration Tests

```python
# tests/test_amt.py - add these tests

def test_actions_with_override_asset():
    """Verify actions() correctly handles OVERRIDE expiry code."""
    from specparser.amt import tickers, load_all_prices

    load_all_prices('data/prices.parquet')
    prices = tickers.get_prices('CL Comdty', 2024, 6, 0, 'data/amt.yml', 'data/futs.csv')
    result = tickers.actions(prices, 'data/amt.yml', 'data/overrides.csv')

    action_idx = result['columns'].index('action')
    actions = [row[action_idx] for row in result['rows']]

    assert 'ntry' in actions, "Missing ntry action"
    assert 'xpry' in actions, "Missing xpry action"

def test_get_straddle_actions_with_overrides():
    """Verify get_straddle_actions passes overrides through."""
    from specparser.amt import get_straddle_actions, load_all_prices

    load_all_prices('data/prices.parquet')
    result = get_straddle_actions('CL Comdty', 2024, 6, 0, 'data/amt.yml',
                               'data/futs.csv', None, 'data/overrides.csv')

    action_idx = result['columns'].index('action')
    actions = [row[action_idx] for row in result['rows']]
    assert 'ntry' in actions and 'xpry' in actions

def test_get_straddle_valuation_with_overrides():
    """Verify get_straddle_valuation passes overrides through."""
    from specparser.amt import get_straddle_valuation, load_all_prices

    load_all_prices('data/prices.parquet')
    result = get_straddle_valuation('CL Comdty', 2024, 6, 0, 'data/amt.yml',
                                    'data/futs.csv', None, 'data/overrides.csv')

    # Should have valuation columns
    assert 'mv' in result['columns']
    assert 'pnl' in result['columns']
```

### 2. Update Documentation

Update `docs/amt.md` to document the `overrides_csv` parameter for all affected functions:
- `actions()`
- `get_straddle_actions()`
- `get_straddle_valuation()`

### 3. Consider Deprecation Warning

For assets with `xprc='OVERRIDE'`, consider adding a warning if `overrides_csv` is not provided:

```python
def _add_action_column(table, straddle, underlying, overrides_csv=None):
    xprc = schedules.xprc(straddle)
    if xprc == 'OVERRIDE' and overrides_csv is None:
        import warnings
        warnings.warn(
            f"Asset {underlying} has xprc='OVERRIDE' but overrides_csv not provided. "
            "Entry/expiry dates may be incorrect.",
            UserWarning
        )
    # ... rest of function
```

---

## Future Work (Phase C - Batch API)

Now that Phase B is complete, batch processing can be built on the decomposed functions.

### Proposed Batch API

```python
def get_prices_batch(
    straddles: list[tuple[str, int, int, int]],  # [(underlying, year, month, i), ...]
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
) -> dict[str, Any]:
    """Batch price lookup for many straddles.

    Pipeline:
    1. Get straddle-days from schedules.find_straddle_days_numba() (fastest)
    2. Collect all tickers across straddles
    3. Fetch prices via DuckDB as (ticker, field, date, value) table
    4. Join + pivot in DuckDB (multi-key join, SQL pivot)
    5. Return Arrow table
    """

def get_straddle_actions_batch(
    straddles: list[tuple[str, int, int, int]],
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
    overrides_csv: str | Path | None = None,  # ← Don't forget!
) -> dict[str, Any]:
    """Batch get_straddle_actions for many straddles.

    Composes:
    1. get_prices_batch() for bulk price lookup
    2. Vectorized action computation where possible
    3. Keep scalar calendar logic for entry/expiry triggers

    NOTE: overrides_csv must be provided for assets with xprc='OVERRIDE'.
    The batch version should preload overrides once, not per-straddle.
    """
```

### Batch Override Handling

For batch processing, optimize override lookups:

```python
# Current per-straddle approach (inefficient for batch)
def _override_expiry(underlying, year, month, overrides_path):
    # Loads and caches per-call, but still repeated lookups

# Better batch approach
def _preload_overrides(overrides_csv: str | Path) -> dict[tuple[str, int, int], datetime.date]:
    """Preload all overrides into a (underlying, year, month) → date dict."""
    # Load once, use for all straddles in batch

def get_straddle_actions_batch(..., overrides_csv=None):
    overrides_map = _preload_overrides(overrides_csv) if overrides_csv else {}
    # Pass overrides_map to action computation, not the file path
```

### Implementation Strategy

1. **Use DuckDB for bulk operations**:
   - Multi-key joins (ticker, field, date) - faster than Python
   - SQL pivot for wide format - scales better than dict approach

2. **Use Numba for date expansion**:
   - `find_straddle_days_numba(parallel=True)` for large outputs

3. **Keep action computation scalar**:
   - Calendar logic (anchor dates, good day detection) is inherently row-wise
   - Can precompute `is_good_day` mask column for efficiency

4. **Use nulls internally**:
   - Arrow validity bitmaps for missing prices
   - Convert to "none" strings only at JSON output boundary

### Building Blocks Available (from Phase B)

| Function | Batch Adaptation |
|----------|-----------------|
| `get_prices()` | → `get_prices_batch()` with DuckDB join/pivot |
| `actions()` | → Vectorize where possible, keep calendar scalar |
| `_lookup_straddle_prices()` | → DuckDB query for all tickers at once |
| `_build_prices_table()` | → DuckDB pivot instead of Python dict |

---

## Alternative: Numba-Based Date Expansion

Instead of the Arrow calendar lookup + explode dance, use a custom Numba kernel that generates dates directly from (year, month, span) vectors. This avoids:
- Arrow object creation overhead
- Calendar table building and caching
- `pc.take()` and `pc.list_flatten()` overhead
- Python/Arrow boundary crossings

### Design

**Input**: Three int32 vectors of length N (one per straddle)
- `entry_year`: e.g., [2024, 2024, 2025]
- `entry_month`: e.g., [1, 6, 11]
- `span`: e.g., [2, 3, 2] (derived from NTRC: N→2, F→3)

**Output**: Two int32 vectors (total length = sum of days across all straddles)
- `source_idx`: which input row this day belongs to (for joining back)
- `day_offset`: days since epoch (or year/month/day tuple)

**Example**:
```
Input:  entry_year=[2025], entry_month=[1], span=[2]
        → generates Jan 2025 (31 days) + Feb 2025 (28 days) = 59 days

Output: source_idx = [0, 0, 0, ..., 0]  # 59 zeros
        day_offset = [days since epoch for 2025-01-01 through 2025-02-28]
```

### Numba Implementation

Optimized implementation using:
- Howard Hinnant's `days_from_civil` algorithm for O(1) date→epoch conversion
- `base + j` trick: compute base date once per month, then just add day offset
- Parallel version with `prange` for very large datasets
- Bit test `(year & 3) != 0` for fast leap year check

```python
import numpy as np
from numba import njit, prange

# -----------------------------
# Calendar helpers (Numba-safe)
# -----------------------------

_DAYS_PER_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

@njit(cache=True)
def is_leap_year(year: int) -> bool:
    # Fast leap-year test using bit operations
    if (year & 3) != 0:   # year % 4 != 0
        return False
    if (year % 100) != 0:
        return True
    return (year % 400) == 0

@njit(cache=True)
def last_day_of_month(year: int, month: int) -> int:
    d = _DAYS_PER_MONTH[month - 1]
    if month == 2 and is_leap_year(year):
        return 29
    return d

@njit(cache=True)
def add_months(year: int, month: int, k: int):
    """Add k months to (year, month), handling year rollover."""
    t = year * 12 + (month - 1) + k
    yy = t // 12
    mm = (t - yy * 12) + 1
    return yy, mm

@njit(cache=True)
def ymd_to_date32(year: int, month: int, day: int) -> int:
    """Convert civil date to days since 1970-01-01 (proleptic Gregorian).

    Uses Howard Hinnant's days_from_civil algorithm - O(1), no loops.
    Returns an integer suitable for Arrow date32.
    """
    y = year - (1 if month <= 2 else 0)
    if y >= 0:
        era = y // 400
    else:
        era = (y - 399) // 400
    yoe = y - era * 400
    mp = month + (12 if month <= 2 else 0) - 3  # Mar=0..Feb=11
    doy = (153 * mp + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468  # 719468 aligns to 1970-01-01


# -----------------------------
# Sequential version
# -----------------------------

@njit(cache=True)
def expand_months_to_date32(year, month, month_count):
    """Expand (year, month, month_count) rows into date32 and parent_idx.

    Args:
        year: int32 array of entry years
        month: int32 array of entry months (1-12)
        month_count: int32 array of month spans (2 for N, 3 for F)

    Returns:
        date32: int32 array of days since 1970-01-01 (Arrow date32 compatible)
        parent_idx: int32 array giving input row index for each output date

    Output is grouped into contiguous blocks per input row.
    """
    n = len(year)

    # 1) Compute total output size
    total = 0
    for i in range(n):
        span = month_count[i]
        if span <= 0:
            continue
        y0 = year[i]
        m0 = month[i]
        for k in range(span):
            yy, mm = add_months(y0, m0, k)
            total += last_day_of_month(yy, mm)

    # 2) Allocate outputs
    date32 = np.empty(total, dtype=np.int32)
    parent_idx = np.empty(total, dtype=np.int32)

    # 3) Fill outputs - compute base date once per month, then base + j
    pos = 0
    for i in range(n):
        span = month_count[i]
        if span <= 0:
            continue
        y0 = year[i]
        m0 = month[i]

        for k in range(span):
            yy, mm = add_months(y0, m0, k)
            last = last_day_of_month(yy, mm)

            # Base date for first day of month
            base = ymd_to_date32(yy, mm, 1)

            # Inner loop: sequential writes, just base + offset
            for j in range(last):
                date32[pos] = base + j
                parent_idx[pos] = i
                pos += 1

    return date32, parent_idx


# -----------------------------
# Parallel version (for millions of rows)
# -----------------------------

@njit(cache=True)
def _compute_starts(year, month, month_count):
    """Compute start offset for each input row (prefix sum of output counts)."""
    n = len(year)
    starts = np.empty(n + 1, dtype=np.int64)
    starts[0] = 0
    running = 0
    for i in range(n):
        span = month_count[i]
        cnt = 0
        if span > 0:
            y0 = year[i]
            m0 = month[i]
            for k in range(span):
                yy, mm = add_months(y0, m0, k)
                cnt += last_day_of_month(yy, mm)
        running += cnt
        starts[i + 1] = running
    return starts


@njit(cache=True, parallel=True)
def expand_months_to_date32_parallel(year, month, month_count):
    """Same as expand_months_to_date32, but parallel fill.

    Best when output is very large (millions+ rows).
    Pre-computes start offsets, then fills in parallel with prange.
    """
    n = len(year)
    starts = _compute_starts(year, month, month_count)
    total = starts[n]

    date32 = np.empty(total, dtype=np.int32)
    parent_idx = np.empty(total, dtype=np.int32)

    for i in prange(n):
        pos = starts[i]
        span = month_count[i]
        if span <= 0:
            continue

        y0 = year[i]
        m0 = month[i]

        for k in range(span):
            yy, mm = add_months(y0, m0, k)
            last = last_day_of_month(yy, mm)
            base = ymd_to_date32(yy, mm, 1)

            # Tight inner loop: sequential writes
            for j in range(last):
                date32[pos + j] = base + j
                parent_idx[pos + j] = i

            pos += last

    return date32, parent_idx
```

### Integration with Arrow Tables

```python
import pyarrow as pa

def find_straddle_days_numba(
    path: str | Path,
    start_year: int,
    end_year: int,
    pattern: str = ".",
    live_only: bool = True,
    parallel: bool = False,
) -> dict[str, Any]:
    """Expand straddles to daily rows using Numba kernel.

    Faster than Arrow-based approach for large datasets.

    Args:
        parallel: If True, use parallel version (best for millions+ output rows)
    """
    import pyarrow.compute as pc

    # Get straddles table
    straddles = find_straddle_yrs(path, start_year, end_year, pattern, live_only)
    straddles_arrow = table_to_arrow(straddles)

    n_straddles = table_nrows(straddles_arrow)
    if n_straddles == 0:
        return _empty_straddle_days_table()

    # Extract entry_year, entry_month, ntrc from straddle strings
    straddle_col = straddles_arrow["rows"][straddles_arrow["columns"].index("straddle")]

    # Vectorized string slicing to get components
    entry_ym = pc.utf8_slice_codeunits(straddle_col, 1, 8)  # "YYYY-MM"
    ntrc = pc.utf8_slice_codeunits(straddle_col, 17, 18)    # "N" or "F"

    # Parse year/month from "YYYY-MM" strings
    entry_year = pc.utf8_slice_codeunits(entry_ym, 0, 4)
    entry_month = pc.utf8_slice_codeunits(entry_ym, 5, 7)

    # Convert to numpy int32 arrays
    year_np = np.asarray(pc.cast(entry_year, pa.int32()).to_numpy(), dtype=np.int32)
    month_np = np.asarray(pc.cast(entry_month, pa.int32()).to_numpy(), dtype=np.int32)

    # Map NTRC to span: N→2, F→3
    ntrc_bytes = ntrc.to_numpy()
    month_count_np = np.where(ntrc_bytes == b'N', 2, 3).astype(np.int32)

    # Run Numba kernel
    if parallel:
        date32, parent_idx = expand_months_to_date32_parallel(year_np, month_np, month_count_np)
    else:
        date32, parent_idx = expand_months_to_date32(year_np, month_np, month_count_np)

    # Build output table by gathering from source arrays using parent_idx
    asset_col = straddles_arrow["rows"][straddles_arrow["columns"].index("asset")]

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
```

**Note on Arrow integration:**
- `date32` output is already the exact physical representation Arrow uses (int32 days since epoch)
- `pa.array(date32, type=pa.date32())` is zero-copy - just wraps the numpy buffer
- `pc.take()` for asset/straddle is the same as the Arrow approach - unavoidable

### Performance Analysis

**Arrow approach**:
1. Build calendar table (Python loop, cached)
2. Extract NTRC, build lookup key (Arrow string ops)
3. `pc.index_in()` + `pc.take()` for calendar lookup
4. `pc.list_flatten()` + `pc.list_parent_indices()` for explode
5. `pc.take()` to expand asset/straddle columns

**Numba approach**:
1. Extract year/month/ntrc (Arrow string ops - same)
2. Convert to numpy (one-time copy)
3. Numba kernel: tight loop, bandwidth-bound
4. `pc.take()` to expand asset/straddle columns (same)

**Why Numba is fast**:
- **`base + j` trick**: computes `ymd_to_date32` once per month, inner loop is just `base + j`
- **Inner loop is bandwidth-bound**: just sequential memory writes
- **Single allocation**: pre-computed total size, no intermediate arrays
- **O(1) date conversion**: Howard Hinnant's algorithm, no loops
- **Parallel version**: for very large outputs, uses `prange` with pre-computed offsets

**Expected speedup**:
- Numba kernel: 10-100x faster than Python loops
- vs Arrow approach: 2-10x faster (avoids calendar table, list explode, intermediate allocations)
- Parallel version: additional 2-4x on multi-core for large outputs

**Trade-offs**:
- Requires Numba dependency
- First call has JIT compilation overhead (~100-200ms, cached thereafter)
- `cache=True` persists compiled code across sessions
- Less declarative than Arrow operations (but well-commented)

### Hybrid Strategy

Use Numba for the date expansion (the hot path), Arrow for everything else:

```
find_straddle_days_numba():
    straddles = find_straddle_yrs()           # existing
    year, month, span = extract_from_straddles()  # Arrow string ops
    source_idx, dates = numba_expand(year, month, span)  # Numba kernel
    result = build_output_table(source_idx, dates)  # Arrow take
```

### Testing

```python
import numpy as np
import datetime

def test_numba_expand_basic():
    """Single straddle: Jan 2025, 2 months."""
    year = np.array([2025], dtype=np.int32)
    month = np.array([1], dtype=np.int32)
    month_count = np.array([2], dtype=np.int32)

    date32, parent_idx = expand_months_to_date32(year, month, month_count)

    # Jan has 31 days, Feb 2025 has 28 days (not leap year)
    assert len(date32) == 59
    assert np.all(parent_idx == 0)

    # Verify date range using Arrow
    import pyarrow as pa
    dates = pa.array(date32, type=pa.date32()).to_pylist()
    assert dates[0] == datetime.date(2025, 1, 1)
    assert dates[-1] == datetime.date(2025, 2, 28)


def test_numba_expand_leap_year():
    """Feb 2024 (leap year) should have 29 days."""
    year = np.array([2024], dtype=np.int32)
    month = np.array([2], dtype=np.int32)
    month_count = np.array([1], dtype=np.int32)

    date32, parent_idx = expand_months_to_date32(year, month, month_count)
    assert len(date32) == 29  # Feb 2024 has 29 days


def test_numba_expand_multiple():
    """Two straddles with different spans."""
    year = np.array([2024, 2025], dtype=np.int32)
    month = np.array([1, 6], dtype=np.int32)
    month_count = np.array([2, 3], dtype=np.int32)

    date32, parent_idx = expand_months_to_date32(year, month, month_count)

    # First: Jan(31) + Feb(29) = 60 days (2024 is leap year)
    # Second: Jun(30) + Jul(31) + Aug(31) = 92 days
    assert len(date32) == 60 + 92
    assert np.sum(parent_idx == 0) == 60
    assert np.sum(parent_idx == 1) == 92


def test_numba_parallel_matches_sequential():
    """Parallel version should produce identical output."""
    year = np.array([2024, 2024, 2025], dtype=np.int32)
    month = np.array([1, 6, 11], dtype=np.int32)
    month_count = np.array([2, 3, 2], dtype=np.int32)

    date32_seq, parent_seq = expand_months_to_date32(year, month, month_count)
    date32_par, parent_par = expand_months_to_date32_parallel(year, month, month_count)

    np.testing.assert_array_equal(date32_seq, date32_par)
    np.testing.assert_array_equal(parent_seq, parent_par)


def test_ymd_to_date32_known_dates():
    """Verify ymd_to_date32 against known values."""
    # Unix epoch
    assert ymd_to_date32(1970, 1, 1) == 0
    # Known date: 2024-01-01 = 19723 days since epoch
    assert ymd_to_date32(2024, 1, 1) == 19723
    # Leap day 2024
    assert ymd_to_date32(2024, 2, 29) == 19723 + 31 + 28  # Jan + Feb days


def test_numba_vs_arrow_equivalence():
    """Regression test: Numba output matches Arrow output."""
    path = "data/amt.yml"

    arrow_result = find_straddle_days_arrow(path, 2024, 2024)
    numba_result = find_straddle_days_numba(path, 2024, 2024)

    # Compare sorted (asset, straddle, date) tuples
    def to_tuples(tbl):
        return sorted(zip(
            tbl["rows"][0].to_pylist(),
            tbl["rows"][1].to_pylist(),
            tbl["rows"][2].to_pylist(),
        ))

    assert to_tuples(arrow_result) == to_tuples(numba_result)
```

### Recommendation

Add `find_straddle_days_numba()` as a third implementation option:

| Implementation | Best for | Notes |
|---------------|----------|-------|
| `find_straddle_days()` | Single straddle, small N | Python loop with memoization, simple |
| `find_straddle_days_arrow()` | Medium N, Arrow pipelines | Vectorized, no Numba dependency |
| `find_straddle_days_numba()` | Large N, batch processing | Fastest, requires Numba |
| `find_straddle_days_numba(parallel=True)` | Millions+ output rows | Multi-core parallel fill |

The Numba version should be the default for batch operations once implemented and tested.

### File Organization

The Numba kernels should live in a separate module to:
- Keep the core `schedules.py` clean
- Allow optional Numba dependency
- Enable caching of compiled functions

```
src/specparser/amt/
├── schedules.py          # find_straddle_days(), find_straddle_days_arrow()
├── schedules_numba.py    # Numba date expansion kernels
├── valuation_numba.py    # Numba backtest/valuation kernels
└── ...
```

```python
# In schedules.py
def find_straddle_days_numba(...):
    try:
        from .schedules_numba import expand_months_to_date32, expand_months_to_date32_parallel
    except ImportError:
        raise ImportError("Numba is required for find_straddle_days_numba. Install with: pip install numba")
    ...
```

---

## Code Quality Issues & Fixes

This section documents bugs, security issues, and improvements identified during code review of `tickers.py`.

### Priority 1: Hard Bugs (Must Fix)

#### 1.1 Return Type Mismatch in `_tschema_dict_bbgfc_ym`

**Location**: `tickers.py` line ~285

**Bug**: The `return None` statement returns `None` but the return type annotation claims `dict[str, str]`.

**Current code**:
```python
def _tschema_dict_bbgfc_ym(tschema: str, year: int, month: int) -> dict[str, str]:
    ...
    if not tschema or tschema.startswith("#"):
        return None  # Bug: violates return type
```

**Fix**: Either:
- Return empty dict `{}` for consistency, OR
- Change return type to `dict[str, str] | None` and update all callers

**Impact**: Type checkers will miss bugs; callers may crash with `TypeError` on `.get()`.

#### 1.2 UnboundLocalError in `get_tickers_ym`

**Location**: `tickers.py` lines ~485-510

**Bug**: When the loop body never executes (empty schedules), `result` is never assigned but referenced at `return result`.

**Current code**:
```python
def get_tickers_ym(path, year, month, pattern="."):
    schedules = find_schedules(path, pattern)
    for sched in schedules:
        ...
        result = {...}  # Only assigned inside loop
    return result  # UnboundLocalError if schedules is empty
```

**Fix**:
```python
def get_tickers_ym(path, year, month, pattern="."):
    schedules = find_schedules(path, pattern)
    result = {"orientation": "row", "columns": [...], "rows": []}  # Initialize before loop
    for sched in schedules:
        ...
    return result
```

#### 1.3 CLI Typo: `--staddle-valuation`

**Location**: `tickers.py` line ~2010

**Bug**: CLI flag is misspelled as `--staddle-valuation` instead of `--straddle-valuation`.

**Fix**: Rename to `--straddle-valuation`.

---

### Priority 2: SQL Injection Hazards

#### 2.1 `prices_last()` - Unsanitized Table Names

**Location**: `tickers.py` lines ~1350-1380

**Bug**: `parquet` file path and `ticker` column values are interpolated directly into SQL.

**Current code**:
```python
def prices_last(parquet: str, tickers: list[str], ...):
    ticker_list = ", ".join(f"'{t}'" for t in tickers)  # No escaping!
    query = f"""
        SELECT ticker, ...
        FROM read_parquet('{parquet}')
        WHERE ticker IN ({ticker_list})
    """
```

**Attack vector**: A malicious ticker like `'); DROP TABLE prices; --` could be injected.

**Fix**: Use parameterized queries with DuckDB's `?` placeholders:
```python
def prices_last(parquet: str, tickers: list[str], ...):
    placeholders = ", ".join("?" * len(tickers))
    query = f"""
        SELECT ticker, ...
        FROM read_parquet(?)
        WHERE ticker IN ({placeholders})
    """
    return conn.execute(query, [parquet] + tickers).fetchall()
```

#### 2.2 `_lookup_straddle_prices()` - Same Issue

**Location**: `tickers.py` lines ~1450-1500

**Bug**: Same pattern of string interpolation for SQL queries.

**Fix**: Same parameterized query approach.

---

### Priority 3: Caching & Resource Lifetime

#### 3.1 DuckDB Connection Lifetime

**Location**: `tickers.py` lines ~1300-1320 (`_DUCKDB_CACHE`)

**Issue**: Module-level `duckdb.connect()` cached forever. In long-running processes:
- Connection may become stale
- No cleanup on module unload
- Memory grows unbounded if many different paths used

**Current code**:
```python
_DUCKDB_CACHE: dict[str, duckdb.DuckDBPyConnection] = {}

def _get_duckdb_conn(path: str) -> duckdb.DuckDBPyConnection:
    if path not in _DUCKDB_CACHE:
        _DUCKDB_CACHE[path] = duckdb.connect(path)
    return _DUCKDB_CACHE[path]
```

**Fix options**:
1. Add `clear_duckdb_cache()` function and export it
2. Use context manager pattern for connections
3. Add TTL-based eviction (e.g., `functools.lru_cache` with maxsize)

#### 3.2 Override Cache Never Cleared

**Location**: `tickers.py` `_OVERRIDE_CACHE`

**Issue**: Like DuckDB cache, `_OVERRIDE_CACHE` grows indefinitely. Should add `clear_override_cache()`.

---

### Priority 4: Logic & Semantics Issues

#### 4.1 `_parse_date_constraint` Accepts Invalid Formats

**Location**: `tickers.py` lines ~1550-1580

**Issue**: The function silently accepts malformed dates that happen to parse.

**Current code**:
```python
def _parse_date_constraint(s: str) -> tuple[str, date | None]:
    # Accepts ">=2024" (missing month/day)
    # Accepts "<=abc" (returns None, no error)
```

**Fix**: Add strict validation:
```python
def _parse_date_constraint(s: str) -> tuple[str, date]:
    if not re.match(r'^[<>=]+\d{4}-\d{2}-\d{2}$', s):
        raise ValueError(f"Invalid date constraint: {s}")
    ...
```

#### 4.2 `get_prices()` Returns Different Orientations

**Location**: `tickers.py` lines ~1200-1250

**Issue**: Some code paths return tables without `"orientation"` key, others return column-oriented. Inconsistent.

**Fix**: Always include `"orientation"` key in returned tables. Standardize on one orientation.

---

### Priority 5: Performance & Scalability

#### 5.1 OR Clause Building in `_build_ticker_filter`

**Location**: `tickers.py` lines ~1400-1430

**Issue**: Builds SQL `WHERE ticker = 'A' OR ticker = 'B' OR ...` for N tickers. For large N:
- Query parsing is O(N)
- Query optimizer struggles
- Some DBs have OR clause limits

**Current code**:
```python
def _build_ticker_filter(tickers: list[str]) -> str:
    return " OR ".join(f"ticker = '{t}'" for t in tickers)
```

**Fix**: Use `IN` clause instead:
```python
def _build_ticker_filter(tickers: list[str]) -> str:
    # Plus parameterization per Priority 2
    placeholders = ", ".join("?" * len(tickers))
    return f"ticker IN ({placeholders})"
```

#### 5.2 Repeated `find_schedules()` Calls

**Location**: Multiple functions call `find_schedules()` with same arguments

**Issue**: Each call re-parses YAML, re-filters. Could cache at call site.

**Fix**: Add memoization or pass schedules as parameter to avoid redundant work.

---

### Priority 6: API Consistency

#### 6.1 Missing "orientation" Key

Several functions return tables without the `"orientation"` key that `table.py` functions expect:

- `get_tickers_ym()` - missing orientation
- `find_tickers()` - inconsistent
- `get_prices()` - some paths missing

**Fix**: Audit all table-returning functions, ensure consistent structure:
```python
return {
    "orientation": "row",  # Always include
    "columns": [...],
    "rows": [...]
}
```

#### 6.2 Inconsistent Error Handling

Some functions raise exceptions, others return empty tables, others return `None`.

**Fix**: Document and standardize:
- Invalid input → raise `ValueError`
- No data found → return empty table (not `None`)
- External errors (file not found) → raise with context

---

### Implementation Order

1. **Immediate fixes** (before next release): ✅ DONE
   - ✅ 1.2 UnboundLocalError in `find_tickers()` - Fixed by returning empty table when `tables is None`
   - ✅ 1.3 CLI typo fix - Changed `--staddle-valuation` to `--straddle-valuation`

2. **Next sprint**: ✅ DONE
   - ✅ 1.1 Return type mismatch - Changed `_tschema_dict_bbgfc_ym` return type from `list[dict]` to `dict`
   - ✅ 2.1 SQL injection in `prices_last()` - Now uses parameterized query with `?` placeholder
   - ✅ 2.2 SQL injection in `_lookup_straddle_prices()` - Now uses parameterized query with `?` placeholders
   - ✅ 6.1 Add missing "orientation" keys - Fixed two early returns in `get_straddle_valuation()`

3. **Technical debt**: ✅ PARTIALLY DONE
   - ✅ 3.1 DuckDB cache management - Added `clear_prices_connection_cache()` function and exported
   - ✅ 3.2 Override cache management - Added `clear_override_cache()` function and exported
   - ⏸️ 4.1 Date constraint validation - Deferred (would change API behavior)
   - ⏸️ 5.1 OR clause optimization - Not needed (composite key requires OR pattern; parameterized)
   - ⏸️ 5.2 Repeated `find_schedules()` calls - Deferred (broader architectural change)
   - ⏸️ 6.2 Error handling standardization - Deferred (codebase-wide review needed)

### Tests Added

New tests added to `tests/test_amt.py`:

1. **TestOverrideExpiry::test_clear_override_cache** - Verifies `clear_override_cache()` resets the cache to `None`
2. **TestCacheManagement::test_clear_prices_connection_cache** - Verifies `clear_prices_connection_cache()` clears DuckDB connections
3. **TestCacheManagement::test_find_tickers_empty_year_range** - Verifies `find_tickers()` returns empty table with proper orientation when `start_year > end_year`

All 70 ticker/cache/override related tests pass.
