# Plan: Fix Table Notebook and Add Join Benchmarks

## Issues to Fix

### 1. Table Format Section (cell-0)
**Problem:** States orientations are "row" or "column", missing "arrow".

**Fix:** Update the markdown to include all three orientations:
```markdown
- `orientation`: Either `"row"`, `"column"`, or `"arrow"`
```

Add arrow-oriented example to the section.

### 2. Fast Joins with Arrow Section (cell-a745cyeg9jd and cell-aauuflmtqxp)
**Problem:** Code is too long and gets clipped on the right side in PDF.

**Fix:**
- Break long lines into multiple shorter lines
- Use intermediate variables to avoid long chains
- Keep lines under ~70 characters for good PDF rendering

---

## New Section: Join Benchmarks

### Purpose
Demonstrate performance differences between:
- Row-oriented left join (Python dict-based)
- Arrow-oriented left join (PyArrow hash join)

### Data Setup

**Table 1: Items (small)**
- Columns: `item`, `category`
- ~5-10 items with unique names and categories
- Example: products with their product category

**Table 2: Values (large, with duplicates)**
- Columns: `item`, `value`
- Same items but each item appears multiple times with different values
- Simulates time-series data, repeated measurements, or transaction logs

### Benchmark Structure

#### Small Scale Demo (visible output)
```python
# Items table: 3 items
items_small = {
    "orientation": "row",
    "columns": ["item", "category"],
    "rows": [
        ["apple", "fruit"],
        ["carrot", "vegetable"],
        ["bread", "grain"],
    ]
}

# Values table: 9 rows (3 values per item)
values_small = {
    "orientation": "row",
    "columns": ["item", "value"],
    "rows": [
        ["apple", 1.20],
        ["apple", 1.15],
        ["apple", 1.25],
        ["carrot", 0.80],
        ["carrot", 0.75],
        ["carrot", 0.85],
        ["bread", 2.50],
        ["bread", 2.40],
        ["bread", 2.60],
    ]
}

# Show tables
# Show left join result (3 items × 3 values = 9 rows)
```

#### Large Scale Benchmark
```python
import time

# Generate large tables
n_items = 1000
n_values_per_item = 100

# Items table: 1000 unique items
items_large = {
    "orientation": "row",
    "columns": ["item", "category"],
    "rows": [[f"item_{i}", f"cat_{i % 10}"] for i in range(n_items)]
}

# Values table: 100,000 rows (100 values per item)
values_large = {
    "orientation": "row",
    "columns": ["item", "value"],
    "rows": [
        [f"item_{i}", float(j)]
        for i in range(n_items)
        for j in range(n_values_per_item)
    ]
}

print(f"Items: {table_nrows(items_large)} rows")
print(f"Values: {table_nrows(values_large)} rows")

# Benchmark 1: Row-oriented join
t0 = time.perf_counter()
result_row = table_left_join(items_large, values_large, "item")
t_row = time.perf_counter() - t0
print(f"Row join: {t_row:.3f}s, {table_nrows(result_row)} result rows")

# Benchmark 2: Arrow-oriented join
items_arrow = table_to_arrow(items_large)
values_arrow = table_to_arrow(values_large)

t0 = time.perf_counter()
result_arrow = table_left_join(items_arrow, values_arrow, "item")
t_arrow = time.perf_counter() - t0
print(f"Arrow join: {t_arrow:.3f}s, {table_nrows(result_arrow)} result rows")

# Speedup
print(f"Speedup: {t_row / t_arrow:.1f}x faster with Arrow")
```

### Expected Results
- Row join: likely 0.5-2 seconds for 100k rows
- Arrow join: likely 0.01-0.1 seconds
- Speedup: 10-50x typical

---

## Files to Modify

1. **notebooks/table.ipynb**
   - Update cell-0: Add "arrow" orientation to Table Format section
   - Update cell-aauuflmtqxp: Wrap long lines in Fast Joins section
   - Add new section "9. Join Benchmarks" after section 8:
     - Markdown intro explaining what we're benchmarking
     - Small scale demo with visible tables
     - Large scale benchmark with timing
     - Summary markdown with results interpretation

---

## Implementation Order

1. **Fix cell-0**: Update Table Format to mention "arrow" orientation
2. **Fix Fast Joins section**: Break long lines, simplify code
3. **Add more join examples**: Show different join scenarios clearly
4. **Add Section 9: Join Benchmarks**:
   - Cell: Markdown intro
   - Cell: Small demo (items + values + result)
   - Cell: Large scale data generation
   - Cell: Row join timing
   - Cell: Arrow join timing
   - Cell: Summary/comparison table

5. **Regenerate PDF**

---

## Verification

1. Run the notebook to ensure all cells execute
2. Check PDF renders correctly (no clipped code)
3. Verify timing results are reasonable
4. Confirm data preservation (row join and arrow join produce same logical result)
