# Plan: Arrow Implementation Fixes

## Overview

Address 9 issues with the Arrow integration in `table.py` to improve correctness, performance, and optional PyArrow dependency.

---

## Issues and Solutions

### Issue 1: format_table() / print_table() don't handle arrow correctly

**Problem:** Current logic treats arrow tables like row tables. `len(tbl["rows"])` returns number of columns (not rows), and iteration walks columns.

**Current code (line ~760):**
```python
if _is_column_oriented(tbl):
    tbl = table_to_rows(tbl)
# Then iterates tbl["rows"]
```

**Solution:** Check for arrow orientation and convert to rows before formatting.

```python
def format_table(table: dict[str, Any], limit: int = 100_000) -> str:
    # Convert non-row orientations to row for formatting
    if table.get("orientation") in ("column", "arrow"):
        tbl = table_to_rows(table)
    else:
        tbl = table

    n_rows = len(tbl["rows"])
    if n_rows > limit:
        raise ValueError(f"Table has {n_rows} rows; limit is {limit}")
    # ... rest unchanged
```

**Files:** `table.py` lines 752-791

---

### Issue 2: zip(*rows) transpose scaling problem

**Problem:** `table_to_columns()` uses `list(map(list, zip(*rows)))` which splats every row as an argument - O(n) function arguments, huge overhead for millions of rows.

**Current code (line 173):**
```python
cols = list(map(list, zip(*rows)))
```

**Solution:** Use a manual transpose loop that pre-allocates column lists.

```python
def _transpose_rows_to_cols(rows: list[list], n_cols: int) -> list[list]:
    """Transpose row-oriented data to column-oriented without zip splat."""
    if not rows:
        return [[] for _ in range(n_cols)]

    # Pre-allocate column lists
    cols = [[] for _ in range(n_cols)]
    for row in rows:
        for i, val in enumerate(row):
            cols[i].append(val)
    return cols
```

Replace in `table_to_columns()`:
```python
# Old: cols = list(map(list, zip(*rows)))
# New:
n_cols = len(table["columns"])
cols = _transpose_rows_to_cols(rows, n_cols)
```

**Files:** `table.py` line 173

---

### Issue 3: Arrow constant columns use inefficient pa.array([value] * n_rows)

**Problem:** `table_add_column()` arrow path builds a giant Python list `[value] * n_rows` first.

**Current code (line 388):**
```python
new_col = pa.array([value] * n_rows)
```

**Solution:** Use PyArrow's native construction for constant arrays.

```python
def _make_constant_arrow_array(value: Any, n_rows: int) -> pa.Array:
    """Create Arrow array with constant value, avoiding Python list allocation."""
    if value is None:
        return pa.nulls(n_rows)

    # Use PyArrow's scalar broadcast - create scalar and repeat
    scalar = pa.scalar(value)
    # PyArrow 15+ has efficient repeat; for older versions use chunked approach
    return pa.array([value], type=scalar.type).take(pa.array([0] * n_rows))
    # Or more directly:
    # return pa.repeat(scalar, n_rows)  # if available in pyarrow version
```

Actually, simplest efficient approach:
```python
# Create array with single element and use dictionary encoding concept
# Or use pa.repeat if available (PyArrow 14+)
import pyarrow as pa
if hasattr(pa, 'repeat'):
    new_col = pa.repeat(pa.scalar(value), n_rows)
else:
    # Fallback for older PyArrow - still more efficient than [value] * n_rows
    # Build in chunks to avoid massive list
    chunk_size = 100_000
    chunks = []
    remaining = n_rows
    while remaining > 0:
        size = min(chunk_size, remaining)
        chunks.append(pa.array([value] * size))
        remaining -= size
    new_col = pa.concat_arrays(chunks) if len(chunks) > 1 else chunks[0]
```

**Files:** `table.py` line 388

---

### Issue 4: Arrow bind/concat copies with pa.concat_arrays

**Problem:** `table_bind_rows()` arrow path uses `pa.concat_arrays` per column, which copies data. Could use chunked arrays instead.

**Current code (line 531):**
```python
pa.concat_arrays([t["rows"][i] for t in arrow_tables])
```

**Solution:** Return chunked arrays and only materialize when necessary.

```python
# Option A: Use pa.chunked_array for zero-copy until needed
"rows": [
    pa.chunked_array([t["rows"][i] for t in arrow_tables])
    for i in range(len(cols))
]

# Option B: Use pa.concat_tables which is optimized
pa_tables = [
    pa.table({col: arr for col, arr in zip(t["columns"], t["rows"])})
    for t in arrow_tables
]
combined = pa.concat_tables(pa_tables)
```

**Trade-offs:**
- Chunked arrays: Zero-copy but some operations need `.combine_chunks()`
- concat_tables: PyArrow optimizes internally, cleaner code

**Recommendation:** Use `pa.concat_tables` - it's the idiomatic PyArrow way and handles chunking optimally.

**Files:** `table.py` lines 526-534

---

### Issue 5: Arrow replace_value: old_value=None won't work

**Problem:** `pc.equal(null, null)` yields null (not True), so replacing nulls via equality won't work.

**Current code (line 477):**
```python
mask = pc.equal(col, pa.scalar(old_value, type=col.type))
```

**Solution:** Special-case when `old_value is None`:

```python
def table_replace_value(table, colname, old_value, new_value):
    # ... arrow fast-path
    if old_value is None:
        # Use is_null for null comparison
        mask = pc.is_null(col)
    else:
        mask = pc.equal(col, pa.scalar(old_value, type=col.type))

    new_col = pc.if_else(mask, pa.scalar(new_value), col)
```

**Files:** `table.py` lines 474-481

---

### Issue 6: table_unique_rows() semantics differ for arrow vs non-arrow

**Problem:**
- Non-arrow: preserves last occurrence, deterministic order
- Arrow: uses `group_by().aggregate([])` which is unordered distinct

**Solution:** Document behavior difference, or implement consistent semantics.

**Option A (Document):** Add docstring note that order is not guaranteed for arrow tables.

**Option B (Consistent):** For arrow, use a stable approach:
```python
# Arrow stable unique - keep first occurrence
def _arrow_unique_rows_stable(table):
    pa_table = pa.table({col: arr for col, arr in zip(table["columns"], table["rows"])})

    # Add row index, group by all columns keeping min index, then sort
    n_rows = len(table["rows"][0])
    with_idx = pa_table.append_column("__idx__", pa.array(range(n_rows)))

    # Group and get first index for each unique row
    grouped = with_idx.group_by(table["columns"]).aggregate([("__idx__", "min")])

    # Sort by original index to preserve first-occurrence order
    sorted_table = grouped.sort_by("__idx___min")

    # Remove index column
    return sorted_table.drop("__idx___min")
```

**Recommendation:** Option A initially (document), with Option B as future enhancement.

**Files:** `table.py` lines 556-584

---

### Issue 7: Arrow joins: key column inclusion + ordering differences

**Problem:**
- `pa.Table.join` behavior around key column retention is subtle
- Non-arrow join preserves left table order; arrow join may not

**Solution:**

1. **Key column:** Explicitly select output columns after join:
```python
# After pa.Table.join, explicitly construct output with desired columns
result_pa = left_pa.join(right_pa, keys=left_key, ...)
# Ensure key column appears once, from left table
```

2. **Ordering:** Document that arrow joins may not preserve order, or add explicit sort:
```python
# If order preservation needed:
# Add __left_idx__ before join, sort result by it, then drop
```

**Files:** `table.py` lines 1048-1055

---

### Issue 8: table_to_arrow() can avoid row->column materialization

**Problem:** Current path goes `row -> table_to_columns() -> pa.array()` which is expensive.

**Current code (lines 223-229):**
```python
col_table = table_to_columns(table)
arrays = [pa.array(col) for col in col_table["rows"]]
```

**Solution:** Build Arrow arrays directly from row iteration:

```python
def _rows_to_arrow_arrays(rows: list[list], n_cols: int) -> list[pa.Array]:
    """Convert row-oriented data directly to Arrow arrays."""
    if not rows:
        return [pa.array([]) for _ in range(n_cols)]

    # Transpose in chunks to avoid memory spike
    # Or use PyArrow's RecordBatch for efficient row->columnar

    # Simple direct approach:
    col_data = [[] for _ in range(n_cols)]
    for row in rows:
        for i, val in enumerate(row):
            col_data[i].append(val)

    return [pa.array(col) for col in col_data]
```

This is similar to Issue 2 but avoids the intermediate dict creation in `table_to_columns()`.

**Better approach:** Use `pa.RecordBatch.from_pylist()`:
```python
def table_to_arrow(table):
    if _is_arrow(table):
        return table

    if _is_column_oriented(table):
        # Direct column -> arrow
        ...
    else:
        # Row-oriented: use RecordBatch for efficient conversion
        rows = table["rows"]
        columns = table["columns"]

        # Convert rows to list of dicts for RecordBatch
        # Actually, still need iteration - stick with transpose approach
        # but use the optimized _transpose_rows_to_cols from Issue 2
```

**Files:** `table.py` lines 209-230

---

### Issue 9: PyArrow import is mandatory

**Problem:** Module-level `import pyarrow as pa` makes the whole module fail without pyarrow.

**Solution:** Guard imports and raise only when arrow operations are used.

```python
# At module level
pa = None
pc = None

def _import_pyarrow():
    global pa, pc
    if pa is None:
        try:
            import pyarrow as _pa
            import pyarrow.compute as _pc
            pa = _pa
            pc = _pc
        except ImportError:
            raise ImportError(
                "PyArrow is required for arrow-oriented tables. "
                "Install with: pip install pyarrow"
            )
    return pa, pc

# In functions that need arrow:
def table_to_arrow(table):
    pa, pc = _import_pyarrow()
    # ... use pa and pc
```

**Alternative:** Use `TYPE_CHECKING` for type hints:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pyarrow as pa
```

**Files:** `table.py` lines 25, 467, 1171

---

## Implementation Order

### Phase 1: Critical correctness fixes
1. **Issue 1:** format_table/print_table arrow handling
2. **Issue 5:** replace_value None handling

### Phase 2: Performance improvements
3. **Issue 2:** zip(*rows) transpose
4. **Issue 3:** Constant column allocation
5. **Issue 8:** table_to_arrow direct conversion

### Phase 3: API consistency
6. **Issue 6:** unique_rows semantics (document first)
7. **Issue 7:** Join ordering (document first)

### Phase 4: Memory optimization
8. **Issue 4:** bind_rows chunked arrays

### Phase 5: Optional dependency
9. **Issue 9:** Guarded pyarrow import

---

## Testing

Each fix should include tests:
1. format_table with arrow input
2. replace_value with None as old_value
3. table_to_columns with large row counts (benchmark)
4. table_add_column memory usage (benchmark)
5. table_unique_rows order verification
6. table_left_join order verification
7. table_bind_rows memory usage (benchmark)
8. Import without pyarrow installed

---

## Files to Modify

1. **src/specparser/amt/table.py** - All fixes
2. **tests/test_amt.py** - Add tests for fixes

---

## Verification

```bash
# Run existing tests
uv run pytest tests/test_amt.py -v

# Run notebook to verify benchmarks still work
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/table.ipynb

# Memory profiling (optional)
uv run python -c "
from specparser.amt import table_to_columns, table_add_column, table_bind_rows
import tracemalloc

tracemalloc.start()
# ... test operations
current, peak = tracemalloc.get_traced_memory()
print(f'Peak memory: {peak / 1024 / 1024:.1f} MB')
"
```
