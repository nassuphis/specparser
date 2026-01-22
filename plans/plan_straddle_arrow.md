# Plan: Arrow-Based Straddle Date Expansion

## Overview

Reimplement `find_straddle_days()` in `schedules.py` to use Arrow table operations instead of Python loops. The current implementation iterates over each straddle, generates dates, and uses `itertools.repeat()` to expand columns - this is inefficient for large datasets.

## Current Implementation Analysis

### Straddle String Format
```
|2024-01|2024-03|N|5|F||33.3|
 ^      ^
 |      +-- expiry year-month (positions 9-16)
 +-- entry year-month (positions 1-8)
```

### Current `find_straddle_days()` (lines 425-455)
```python
for row in straddles["rows"]:
    asset = row[asset_idx]
    straddle = row[straddle_idx]
    days = straddle_days(straddle)  # generates list of dates
    n = len(days)
    asset_col.extend(itertools.repeat(asset, n))
    straddle_col.extend(itertools.repeat(straddle, n))
    date_col.extend(days)
```

**Problems:**
1. Python loop over all straddles - O(n) in Python
2. `straddle_days()` recomputes date ranges even with caching overhead
3. `itertools.repeat` + extend for each row - memory allocation per row

## Proposed Solution

### Strategy: NTRC-Keyed Calendar Table + Single Join + Explode

Instead of computing month spans per-straddle, use the **NTRC flag** to determine span length and join with a precomputed calendar table:

1. **Extract entry yearmonth + NTRC** from straddle strings using Arrow vectorized string slicing
2. **Build lookup key** as `YYYY-MM-N` where N is span length (2 for NTRC=N, 3 for NTRC=F)
3. **Single left-join** with calendar table keyed by `(entry_ym, span_length)` containing date lists
4. **Explode** the dates list using Arrow-native `pc.list_flatten()` + `pc.list_parent_indices()`
5. **Select** final columns

### Key Insights

1. **NTRC determines span**: N = 2 months (entry + next), F = 3 months (entry + next 2)
2. **Single join** replaces: month-list generation → explode → join
3. **Calendar table is small**: ~72 rows per year (36 months × 2 span types), or ~720 rows for 10 years
4. **Arrow-native explode** using `pc.list_flatten()` + `pc.list_parent_indices()` is much faster than Python conversion
5. **No Python loops** for month generation - everything is vectorized
6. **Calendar year range** should be `start_year - 1` to `end_year + 1` to handle offsets

---

## Implementation Plan

### Phase 1: Add Arrow-Native Explode to table.py

Add `table_explode_arrow()` using PyArrow's list functions:

```python
def table_explode_arrow(table: dict[str, Any], column: str | int) -> dict[str, Any]:
    """Explode a list column into multiple rows (Arrow-native).

    Uses pc.list_flatten() and pc.list_parent_indices() for zero-copy expansion.

    Args:
        table: Table with a list-valued column
        column: Column name or index containing lists

    Returns:
        Arrow-oriented table with list column expanded to individual rows.
        Empty lists and null values produce no output rows.

    Raises:
        TypeError: If the column is not a list type.
    """
    _pa, _pc = _import_pyarrow()
    tbl = table_to_arrow(table)
    col_idx = _resolve_column_index(tbl, column)

    list_col = tbl["rows"][col_idx]

    # Validate list type
    if not _pa.types.is_list(list_col.type) and not _pa.types.is_large_list(list_col.type):
        raise TypeError(
            f"Column '{tbl['columns'][col_idx]}' has type {list_col.type}, expected list type"
        )

    parent_indices = _pc.list_parent_indices(list_col)  # parent row for each element
    flat_values = _pc.list_flatten(list_col)            # flattened list values

    out_cols = []
    for i, col in enumerate(tbl["rows"]):
        if i == col_idx:
            out_cols.append(flat_values)
        else:
            out_cols.append(_pc.take(col, parent_indices))  # repeat by parent index

    return {"orientation": "arrow", "columns": tbl["columns"][:], "rows": out_cols}
```

**Note: `table_explode_arrow` vs `table_unchop` semantics**

These functions have different behaviors and should remain separate:

| Behavior | `table_explode_arrow` | `table_unchop` |
|----------|----------------------|----------------|
| Empty list `[]` | No output rows | No output rows |
| Null value | No output rows | One row with null (wraps as `[None]`) |
| Scalar value | TypeError | One row (wraps as `[value]`) |
| Input validation | Strict (must be list type) | Permissive |

**Recommendation:** Keep `table_unchop()` with its existing Python-based semantics for consistency across orientations. Add `table_explode_arrow()` as a separate strict operation for Arrow-native list explosion.

If `table_unchop` is called with an arrow table that has a list column, it can optionally use `table_explode_arrow` internally, but should handle nulls/scalars first:

```python
def table_unchop(table: dict[str, Any], column: str | int) -> dict[str, Any]:
    if _is_arrow(table):
        _pa, _pc = _import_pyarrow()
        col_idx = _resolve_column_index(table, column)
        col = table["rows"][col_idx]

        # Check if column is list type
        if _pa.types.is_list(col.type) or _pa.types.is_large_list(col.type):
            # Use native explode for list columns
            return table_explode_arrow(table, column)
        else:
            # Non-list column: convert to row-oriented and use existing logic
            # (wraps scalars as singletons)
            pass

    # ... existing column/row logic
```

### Phase 2: Create Calendar Tables

#### 2a: Build Base Month Calendar (One-Time Initialization)

Build a simple `(ym, dates)` table with one row per month. This is built once and cached.

```python
def _build_month_calendar_arrow(start_year: int, end_year: int) -> dict[str, Any]:
    """Build calendar table mapping yearmonth to date list for that month.

    Returns:
        Arrow table with columns ["ym", "dates"]
        - ym: "2024-01", "2024-02", etc.
        - dates: list of dates for that month (pa.list_(pa.date32()))
    """
    import pyarrow as pa
    import datetime
    import calendar

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

# Cache at module level (lazy initialization)
# Stores (start_year, end_year, calendar_table)
_MONTH_CALENDAR_CACHE: tuple[int, int, dict[str, Any]] | None = None

def _get_month_calendar(start_year: int, end_year: int) -> dict[str, Any]:
    """Get or build the month calendar, expanding range if needed.

    The cache tracks the year range. If a wider range is requested,
    the calendar is rebuilt to cover it.
    """
    global _MONTH_CALENDAR_CACHE

    if _MONTH_CALENDAR_CACHE is not None:
        cached_start, cached_end, cached_cal = _MONTH_CALENDAR_CACHE
        if cached_start <= start_year and cached_end >= end_year:
            # Cached calendar covers requested range
            return cached_cal

    # Build new calendar (either first time or range expanded)
    # Pad range slightly to reduce rebuilds for small range changes
    padded_start = start_year - 1
    padded_end = end_year + 2
    calendar = _build_month_calendar_arrow(padded_start, padded_end)
    _MONTH_CALENDAR_CACHE = (padded_start, padded_end, calendar)
    return calendar


def clear_calendar_cache() -> None:
    """Clear the cached calendar tables (useful for testing)."""
    global _MONTH_CALENDAR_CACHE, _NTRC_CALENDAR_CACHE
    _MONTH_CALENDAR_CACHE = None
    _NTRC_CALENDAR_CACHE = None


# NTRC calendar cache (keyed by year range)
_NTRC_CALENDAR_CACHE: tuple[int, int, dict[str, Any]] | None = None

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
```

#### 2b: Assemble NTRC-Keyed Calendar from Month Calendar

At query time, assemble the span-keyed calendar by concatenating month date lists:

```python
def _build_ntrc_calendar_from_months(month_cal: dict[str, Any]) -> dict[str, Any]:
    """Build NTRC-keyed calendar by concatenating month date lists.

    Takes base month calendar and creates span-keyed entries:
    - "2024-01-2" → concat(dates["2024-01"], dates["2024-02"])
    - "2024-01-3" → concat(dates["2024-01"], dates["2024-02"], dates["2024-03"])

    Returns:
        Arrow table with columns ["key", "dates"]
    """
    import pyarrow as pa
    import pyarrow.compute as pc

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
            arrays_to_concat = [
                dates_col[ym_to_idx[m]].values  # .values gets the underlying array
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
```

**Benefits of this approach:**
1. Month calendar is built once (~120 rows for 10 years)
2. NTRC calendar assembly uses Arrow `concat_arrays` - no Python date iteration
3. The assembly is fast because it's just concatenating pre-built Arrow arrays
4. Easy to cache either or both calendars

### Phase 3: Implement Arrow-Based `find_straddle_days()`

```python
def _ntrc_to_span(ntrc: pa.Array) -> pa.Array:
    """Map NTRC values to span lengths, with validation.

    Args:
        ntrc: Array of NTRC values ("N" or "F")

    Returns:
        Array of span strings ("2" or "3")

    Raises:
        ValueError: If any NTRC value is not "N" or "F"
    """
    import pyarrow.compute as pc

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


def _validate_span_matches_expiry(
    entry_ym: pa.Array,
    expiry_ym: pa.Array,
    span: pa.Array,
    validate: bool = True
) -> None:
    """Optionally validate that computed span matches entry/expiry difference.

    This catches edge cases where NTRC doesn't match the actual month range.
    """
    if not validate:
        return

    import pyarrow.compute as pc

    # Compute expected expiry from entry + (span - 1) months
    # For now, just check a sample or log warnings rather than full validation
    # Full implementation would compute month arithmetic vectorized
    #
    # This is a sanity check - if it fails, investigate the straddle data
    pass  # TODO: implement if needed after initial deployment


def find_straddle_days(
    path: str | Path,
    start_year: int,
    end_year: int,
    pattern: str = ".",
    live_only: bool = True,
    validate_ntrc: bool = True,
) -> dict[str, Any]:
    """Expand straddles to daily rows using Arrow operations.

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
    import pyarrow as pa
    import pyarrow.compute as pc
    from .table import table_to_arrow, table_explode_arrow, table_select_columns

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

    # Step 2: Extract entry_ym, expiry_ym, and ntrc using vectorized string slicing
    straddle_idx = straddles_arrow["columns"].index("straddle")
    straddle_col = straddles_arrow["rows"][straddle_idx]

    # Straddle format: |2024-01|2024-03|N|5|F||33.3|
    #                   ^      ^       ^
    #                   1-8    9-16    17 (NTRC: N=2months, F=3months)
    entry_ym = pc.utf8_slice_codeunits(straddle_col, 1, 8)    # "YYYY-MM"
    expiry_ym = pc.utf8_slice_codeunits(straddle_col, 9, 16)  # "YYYY-MM" (for validation)
    ntrc = pc.utf8_slice_codeunits(straddle_col, 17, 18)      # "N" or "F"

    # Step 3: Map NTRC to span with validation
    if validate_ntrc:
        span = _ntrc_to_span(ntrc)
    else:
        # Fast path without validation (use with caution)
        span = pc.if_else(pc.equal(ntrc, "N"), "2", "3")

    # Optional: validate span matches entry/expiry difference
    _validate_span_matches_expiry(entry_ym, expiry_ym, span, validate=False)

    # Step 4: Build lookup key: "YYYY-MM-2" for N, "YYYY-MM-3" for F
    cal_key = pc.binary_join_element_wise(entry_ym, span, "-")

    # Step 5: Lookup dates using index_in + take (faster than join for small calendar)
    ntrc_cal = _get_ntrc_calendar(start_year - 1, end_year + 2)  # padded range for span overflow

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
```

### Phase 4: Export New Functions

Add to `__init__.py`:
- `table_explode_arrow` (or update `table_unchop` to use it internally)

---

## Performance Analysis

### Current Implementation
- **Time complexity**: O(S * D) where S = straddles, D = avg days per straddle
- **Memory**: Per-row Python list allocations
- **Bottleneck**: Python loop with `itertools.repeat` + `extend`

### Proposed Implementation
- **Time complexity**: O(S) for key generation + O(S) for join + O(total_days) for explode
- **Memory**: Batch Arrow allocations, zero-copy where possible
- **Benefits**:
  - No Python loops - fully vectorized
  - Single join (vs build list → explode → join)
  - Arrow-native explode

### Expected Speedup
- For large datasets: **10-20x faster** due to:
  - No per-straddle Python iteration
  - Single vectorized join
  - Arrow-native `list_flatten` + `list_parent_indices`

---

## Files to Modify

1. **src/specparser/amt/table.py**
   - Add `table_explode_arrow()`
   - Update `table_unchop()` arrow path to use native explode

2. **src/specparser/amt/schedules.py**
   - Add `_build_month_calendar_arrow()`
   - Add `_get_month_calendar()` (with caching)
   - Add `_build_ntrc_calendar_from_months()`
   - Refactor `find_straddle_days()` to use Arrow operations

3. **src/specparser/amt/__init__.py**
   - Export `table_explode_arrow` if made public

4. **tests/test_amt.py**
   - Add tests for `table_explode_arrow()`
   - Add tests for `_build_month_calendar_arrow()`
   - Add tests for `_build_ntrc_calendar_from_months()`
   - Add regression test for `find_straddle_days()`

---

## Testing Strategy

1. **Unit test for Arrow explode**:
   ```python
   def test_table_explode_arrow():
       import pyarrow as pa
       table = {
           "orientation": "arrow",
           "columns": ["id", "items"],
           "rows": [
               pa.array([1, 2, 3]),
               pa.array([["a", "b"], ["c"], ["d", "e", "f"]], type=pa.list_(pa.string()))
           ]
       }
       result = table_explode_arrow(table, "items")
       assert len(result["rows"][0]) == 6  # 2 + 1 + 3
       assert result["rows"][0].to_pylist() == [1, 1, 2, 3, 3, 3]
       assert result["rows"][1].to_pylist() == ["a", "b", "c", "d", "e", "f"]
   ```

2. **Unit test for month calendar**:
   ```python
   def test_build_month_calendar():
       cal = _build_month_calendar_arrow(2024, 2024)
       # Should have 12 rows (one per month)
       assert len(cal["rows"][0]) == 12
       # Check January 2024
       yms = cal["rows"][0].to_pylist()
       assert yms[0] == "2024-01"
       dates = cal["rows"][1][0].as_py()
       assert len(dates) == 31  # January has 31 days
       # Check February 2024 (leap year)
       assert len(cal["rows"][1][1].as_py()) == 29
   ```

3. **Unit test for NTRC calendar assembly**:
   ```python
   def test_build_ntrc_calendar_from_months():
       month_cal = _build_month_calendar_arrow(2024, 2024)
       ntrc_cal = _build_ntrc_calendar_from_months(month_cal)
       # Check a specific key
       keys = ntrc_cal["rows"][0].to_pylist()
       assert "2024-01-2" in keys
       assert "2024-01-3" in keys
       # Check date counts for 2-month span (Jan + Feb 2024)
       idx = keys.index("2024-01-2")
       dates = ntrc_cal["rows"][1][idx].as_py()
       assert len(dates) == 31 + 29  # Jan + Feb 2024 (leap year)
   ```

4. **Regression test**: Compare output with current implementation
   ```python
   def test_find_straddle_days_regression():
       # Run both implementations and compare
       old_result = _find_straddle_days_legacy(path, 2024, 2024)
       new_result = find_straddle_days(path, 2024, 2024)
       # Convert both to same format and compare
       assert sorted(dates_from(old_result)) == sorted(dates_from(new_result))
   ```

5. **Unit test for NTRC validation**:
   ```python
   def test_ntrc_to_span_valid():
       import pyarrow as pa
       ntrc = pa.array(["N", "F", "N", "F"])
       span = _ntrc_to_span(ntrc)
       assert span.to_pylist() == ["2", "3", "2", "3"]

   def test_ntrc_to_span_invalid():
       import pyarrow as pa
       ntrc = pa.array(["N", "X", "F"])  # "X" is invalid
       with pytest.raises(ValueError, match="Invalid NTRC value: 'X'"):
           _ntrc_to_span(ntrc)
   ```

6. **Edge cases**:
   - Empty straddles table
   - Single-month straddles (entry == expiry)
   - Cross-year straddles (2024-11 to 2025-02)
   - Empty list after explode (shouldn't happen but handle gracefully)
   - Invalid NTRC values (should raise clear error)

---

## Verification Commands

```bash
# Run tests
uv run pytest tests/test_amt.py -v -k "explode or straddle"

# Interactive verification
uv run python -c "
from specparser.amt.schedules import find_straddle_days
import time

t0 = time.perf_counter()
result = find_straddle_days('data/amt.yml', 2024, 2024)
print(f'Time: {time.perf_counter() - t0:.3f}s')
print(f'Columns: {result[\"columns\"]}')
print(f'Rows: {len(result[\"rows\"][0]) if result[\"rows\"] else 0}')
"
```

---

## Dependencies

This plan requires:
1. `table_to_arrow()` - **Already exists**
2. `table_select_columns()` - **Already exists**
3. `table_explode_arrow()` - **NEW - must implement**
4. PyArrow functions:
   - `pc.list_flatten()`, `pc.list_parent_indices()` - **Available in PyArrow 15+**
   - `pc.utf8_slice_codeunits()` - string slicing
   - `pc.if_else()`, `pc.equal()`, `pc.or_()`, `pc.invert()` - conditional/boolean logic
   - `pc.binary_join_element_wise()` - string concatenation
   - `pc.index_in()` - vectorized lookup (returns index into value_set)
   - `pc.take()` - gather by indices
   - `pc.indices_nonzero()` - find indices where mask is true

---

## Open Questions (Resolved)

1. ~~**Output format**~~: Returns arrow-oriented for performance. Use `table_to_columns()` at call site if column-oriented needed.

2. ~~**Arrow optimization**~~: Yes, full Arrow path is worth it - the native explode is the key performance win.

3. ~~**Backward compatibility**~~: Keep old implementation as `_find_straddle_days_legacy()` initially for regression testing, remove later.

---

## Implementation Order

1. **Phase 1**: Add `table_explode_arrow()` to table.py
2. **Phase 2**: Add tests for explode
3. **Phase 3**: Add calendar functions to schedules.py
   - `_build_month_calendar_arrow()`
   - `_get_month_calendar()` with caching
   - `_build_ntrc_calendar_from_months()`
4. **Phase 4**: Add tests for calendar functions
5. **Phase 5**: Refactor `find_straddle_days()` (keep old as `_find_straddle_days_legacy`)
6. **Phase 6**: Run regression tests, benchmark
7. **Phase 7**: Remove legacy implementation once verified

---

## Design Notes

### Why NTRC-Keyed Calendar Works

The NTRC flag (N=Near, F=Far) determines the span:
- **N (Near)**: 2 months - entry month + 1 more
- **F (Far)**: 3 months - entry month + 2 more

By encoding this in the calendar key (`YYYY-MM-2` or `YYYY-MM-3`), we:
1. Eliminate per-straddle month-list generation
2. Replace "build list → explode → join" with "vectorized lookup → explode"
3. Keep the calendar table small (~24 rows per year × 10 years = ~240 rows)

### Why `index_in` + `take` Instead of Join

For this use case, we're doing a simple key→value lookup from a tiny calendar table. Using `pc.index_in()` + `pc.take()` instead of `pa.Table.join()`:

1. **Faster** - No join engine overhead, just hash lookup + gather
2. **Preserves order** - Left table order is naturally preserved
3. **Simpler** - No suffix handling, no extra columns to remove
4. **More robust** - Avoids potential PyArrow join edge cases

```python
# Instead of:
joined = table_left_join(straddles, calendar, "cal_key", "key")

# Use:
indices = pc.index_in(cal_key, value_set=calendar_keys)  # int indices or null
dates = pc.take(calendar_dates, indices)                  # list<date32> per row
```

### Two-Level Calendar Architecture

1. **Base month calendar** (`ym` → `[dates]`):
   - Built once, ~12 rows per year
   - Each row has dates for a single month as an Arrow date array
   - Can be cached at module level

2. **NTRC calendar** (`ym-span` → `[dates]`):
   - Assembled from month calendar using Arrow `concat_arrays`
   - No Python date iteration - just concatenates pre-built arrays
   - ~24 rows per year (12 months × 2 spans)
   - Fast to rebuild if needed, or can be cached

**Why this is efficient:**
- Month calendar built with Python loop (unavoidable for generating dates)
- But each month's dates stored as Arrow array from the start
- NTRC assembly is pure Arrow operations: lookup + concat
- The expensive per-straddle work is fully vectorized

### Assumption

This approach assumes NTRC is always `N` or `F`. If other values exist (e.g., longer spans), the calendar table would need additional keys, or fall back to the original month-list approach for edge cases.

---

## Performance & Robustness Considerations

### 1. Avoid `combine_chunks()` After Joins

**Issue:** PyArrow joins can return chunked arrays. Calling `combine_chunks()` materializes the entire result into contiguous memory, which can cause OOM for large datasets.

**Solution:** Leave chunked arrays as-is. Most PyArrow compute functions handle chunked arrays transparently. Only combine chunks at final output if the consumer requires contiguous arrays.

```python
# DON'T do this after joins:
# for i, col in enumerate(result["rows"]):
#     result["rows"][i] = col.combine_chunks()

# DO: leave as chunked, let downstream operations handle it
```

### 2. Integer Month ID Alternative

**Consideration:** String keys like `"2024-01-2"` incur string comparison overhead. An integer encoding could be faster:

```python
# month_id = year * 12 + (month - 1)  → 2024-01 = 24289
# key = month_id * 10 + span          → 24289 * 10 + 2 = 242892
```

**Trade-off:** Integer keys are faster for comparison but less readable for debugging. The calendar table is small (~240 rows), so the join itself is fast regardless. **Keep string keys for now** - optimize only if profiling shows join as bottleneck.

### 3. Calendar Caching Strategy

The month calendar is already cached at module level. The NTRC calendar can also be cached:

```python
_NTRC_CALENDAR: dict[str, Any] | None = None

def _get_ntrc_calendar(start_year: int, end_year: int) -> dict[str, Any]:
    global _NTRC_CALENDAR
    if _NTRC_CALENDAR is None:
        month_cal = _get_month_calendar(start_year, end_year)
        _NTRC_CALENDAR = _build_ntrc_calendar_from_months(month_cal)
    return _NTRC_CALENDAR
```

**Note:** Current implementation rebuilds NTRC calendar each call. For repeated calls with same year range, caching provides additional speedup.

### 4. Minimize `to_pylist()` Usage

**Issue:** `to_pylist()` in `_build_ntrc_calendar_from_months()` converts Arrow array to Python list, which is expensive for large arrays.

**Current usage:**
```python
yms = month_cal["rows"][0].to_pylist()  # Small: ~120 items for 10 years
```

**Assessment:** The yearmonth array is small (12 × years), so this is acceptable. For larger arrays, use Arrow-native iteration or `pc.dictionary_encode()`.

### 5. Validate List Type in `table_explode_arrow()`

Add defensive check to ensure the column is actually a list type:

```python
def table_explode_arrow(table: dict[str, Any], column: str | int) -> dict[str, Any]:
    _pa, _pc = _import_pyarrow()
    tbl = table_to_arrow(table)
    col_idx = _resolve_column_index(tbl, column)

    list_col = tbl["rows"][col_idx]

    # Validate list type
    if not _pa.types.is_list(list_col.type) and not _pa.types.is_large_list(list_col.type):
        raise TypeError(
            f"Column '{tbl['columns'][col_idx]}' has type {list_col.type}, expected list type"
        )

    # ... rest of implementation
```

### 6. Null List Semantics

Define behavior for null list values:
- **Empty list `[]`**: Produces no output rows (handled by `list_flatten`)
- **Null value `None`**: Should produce no output rows or one row with null?

**Recommendation:** Filter nulls before explode, or document that nulls produce no rows:

```python
# Option A: Filter nulls before explode (safer)
mask = _pc.is_valid(list_col)
if not _pc.all(mask).as_py():
    # Handle nulls - either filter or raise
    pass

# Option B: Document behavior - nulls produce no rows (same as empty list)
```

### 7. Consider Inner Join If Coverage Guaranteed

**Current:** `table_left_join()` preserves all straddles even if no matching calendar entry.

**Alternative:** If calendar is guaranteed to cover all entry months, `inner_join` avoids null checks:

```python
# If calendar coverage is guaranteed:
joined = table_inner_join(straddles_with_key, ntrc_cal, "cal_key", "key")

# Benefits: No null dates to handle after explode
# Risk: Silent data loss if calendar missing entries
```

**Recommendation:** Keep `left_join` for safety. Add assertion to catch missing entries:

```python
# After join, verify no nulls in dates column
dates_col = joined["rows"][joined["columns"].index("dates")]
if _pc.any(_pc.is_null(dates_col)).as_py():
    raise ValueError("Calendar missing entries for some straddles - check year range")
```

### 8. Upstream Row-to-Arrow Conversion Cost

**Issue:** If `find_straddle_yrs()` returns row-oriented table, `table_to_arrow()` conversion adds overhead.

**Mitigation options:**
1. Modify `find_straddle_yrs()` to return arrow-oriented directly
2. Accept the one-time conversion cost (amortized over large result)
3. Profile to determine if this is actually a bottleneck

**Assessment:** The conversion happens once per call, and the straddles table is typically small compared to the expanded output. Likely not a bottleneck.

### 9. Optional: Dictionary Encoding for Repeated Strings

For the `asset` and `straddle` columns which have many repeated values after explode, dictionary encoding can reduce memory:

```python
# After explode, optionally dictionary-encode high-cardinality columns
exploded["rows"][asset_idx] = _pc.dictionary_encode(exploded["rows"][asset_idx])
exploded["rows"][straddle_idx] = _pc.dictionary_encode(exploded["rows"][straddle_idx])
```

**Trade-off:** Memory savings vs. encoding overhead. Only beneficial for very large outputs. **Skip for initial implementation**, add later if memory is a concern.
