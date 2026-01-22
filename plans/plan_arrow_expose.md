# Plan: Expose PyArrow Compute Functions in Table Module

## Overview

Add a set of `_arrow` suffixed functions to `table.py` that expose PyArrow's vectorized compute operations. These functions:
- Accept any table orientation (row, column, arrow)
- Convert to arrow internally if needed
- Return arrow-oriented tables
- Provide fast, vectorized operations on columns

## Naming Convention

All functions use `_arrow` suffix to indicate:
1. They use PyArrow compute internally
2. They always return arrow-oriented tables
3. Similar to existing `_byrow` and `_bycol` patterns

---

## Functions to Implement

### 1. Arithmetic Operations (Column Transforms)

Element-wise math on numeric columns. Returns new table with transformed column.

```python
def table_add_arrow(table, col, value_or_col, result_col=None) -> dict:
    """Add value or another column to a column. Result in result_col or replaces col."""

def table_subtract_arrow(table, col, value_or_col, result_col=None) -> dict:
    """Subtract value or another column from a column."""

def table_multiply_arrow(table, col, value_or_col, result_col=None) -> dict:
    """Multiply column by value or another column."""

def table_divide_arrow(table, col, value_or_col, result_col=None) -> dict:
    """Divide column by value or another column."""

def table_negate_arrow(table, col, result_col=None) -> dict:
    """Negate column values."""

def table_abs_arrow(table, col, result_col=None) -> dict:
    """Absolute value of column."""

def table_sign_arrow(table, col, result_col=None) -> dict:
    """Sign of column (-1, 0, 1)."""

def table_power_arrow(table, col, exponent, result_col=None) -> dict:
    """Raise column to power."""

def table_sqrt_arrow(table, col, result_col=None) -> dict:
    """Square root of column."""

def table_exp_arrow(table, col, result_col=None) -> dict:
    """Exponential (e^x) of column."""

def table_ln_arrow(table, col, result_col=None) -> dict:
    """Natural log of column."""

def table_log10_arrow(table, col, result_col=None) -> dict:
    """Base-10 log of column."""

def table_log2_arrow(table, col, result_col=None) -> dict:
    """Base-2 log of column."""

def table_round_arrow(table, col, decimals=0, result_col=None) -> dict:
    """Round column to decimals."""

def table_ceil_arrow(table, col, result_col=None) -> dict:
    """Ceiling of column."""

def table_floor_arrow(table, col, result_col=None) -> dict:
    """Floor of column."""

def table_trunc_arrow(table, col, result_col=None) -> dict:
    """Truncate column toward zero."""
```

### 2. Trigonometric Operations

```python
def table_sin_arrow(table, col, result_col=None) -> dict:
def table_cos_arrow(table, col, result_col=None) -> dict:
def table_tan_arrow(table, col, result_col=None) -> dict:
def table_asin_arrow(table, col, result_col=None) -> dict:
def table_acos_arrow(table, col, result_col=None) -> dict:
def table_atan_arrow(table, col, result_col=None) -> dict:
def table_atan2_arrow(table, y_col, x_col, result_col=None) -> dict:
```

### 3. Comparison Operations (Return Boolean Column)

```python
def table_equal_arrow(table, col, value_or_col, result_col=None) -> dict:
    """Column == value. Returns table with boolean column."""

def table_not_equal_arrow(table, col, value_or_col, result_col=None) -> dict:
    """Column != value."""

def table_less_arrow(table, col, value_or_col, result_col=None) -> dict:
    """Column < value."""

def table_less_equal_arrow(table, col, value_or_col, result_col=None) -> dict:
    """Column <= value."""

def table_greater_arrow(table, col, value_or_col, result_col=None) -> dict:
    """Column > value."""

def table_greater_equal_arrow(table, col, value_or_col, result_col=None) -> dict:
    """Column >= value."""
```

### 4. Null/Value Checks (Return Boolean Column)

```python
def table_is_null_arrow(table, col, result_col=None) -> dict:
    """True where column is null."""

def table_is_valid_arrow(table, col, result_col=None) -> dict:
    """True where column is not null."""

def table_is_nan_arrow(table, col, result_col=None) -> dict:
    """True where column is NaN."""

def table_is_finite_arrow(table, col, result_col=None) -> dict:
    """True where column is finite."""

def table_is_in_arrow(table, col, value_set, result_col=None) -> dict:
    """True where column value is in value_set."""
```

### 5. Logical Operations (Boolean Columns)

```python
def table_and_arrow(table, col1, col2, result_col=None) -> dict:
    """Logical AND of two boolean columns."""

def table_or_arrow(table, col1, col2, result_col=None) -> dict:
    """Logical OR of two boolean columns."""

def table_xor_arrow(table, col1, col2, result_col=None) -> dict:
    """Logical XOR of two boolean columns."""

def table_invert_arrow(table, col, result_col=None) -> dict:
    """Logical NOT of boolean column."""
```

### 6. String Operations

```python
def table_upper_arrow(table, col, result_col=None) -> dict:
    """Uppercase string column."""

def table_lower_arrow(table, col, result_col=None) -> dict:
    """Lowercase string column."""

def table_capitalize_arrow(table, col, result_col=None) -> dict:
    """Capitalize string column."""

def table_title_arrow(table, col, result_col=None) -> dict:
    """Title case string column."""

def table_strip_arrow(table, col, chars=None, result_col=None) -> dict:
    """Strip characters from both ends."""

def table_lstrip_arrow(table, col, chars=None, result_col=None) -> dict:
    """Strip from left."""

def table_rstrip_arrow(table, col, chars=None, result_col=None) -> dict:
    """Strip from right."""

def table_length_arrow(table, col, result_col=None) -> dict:
    """String length."""

def table_starts_with_arrow(table, col, pattern, result_col=None) -> dict:
    """True if starts with pattern."""

def table_ends_with_arrow(table, col, pattern, result_col=None) -> dict:
    """True if ends with pattern."""

def table_contains_arrow(table, col, pattern, result_col=None) -> dict:
    """True if contains pattern."""

def table_replace_arrow(table, col, pattern, replacement, result_col=None) -> dict:
    """Replace pattern with replacement."""

def table_split_arrow(table, col, pattern, result_col=None) -> dict:
    """Split string column into list column."""
```

### 7. Aggregate Functions (Reduce to Scalar per Column)

These summarize columns. Returns a dict or single-row table.

```python
def table_summarize_arrow(
    table,
    aggregations: dict[str, str | list[str]],
) -> dict:
    """
    Summarize table columns with aggregate functions.

    Args:
        table: Input table
        aggregations: Dict mapping column names to aggregation(s)
            e.g., {"price": "mean", "qty": ["sum", "count"]}

    Returns:
        Single-row table with aggregated values.
        Column names: "col_agg" (e.g., "price_mean", "qty_sum")

    Supported aggregations:
        sum, mean, min, max, count, count_distinct,
        stddev, variance, first, last, any, all

    Example:
        >>> table_summarize_arrow(sales, {"price": ["mean", "max"], "qty": "sum"})
        {"columns": ["price_mean", "price_max", "qty_sum"],
         "rows": [[45.5, 100.0, 1500]], ...}
    """
```

Individual aggregate accessors (return scalar):

```python
def table_sum_arrow(table, col) -> float | int:
    """Sum of column."""

def table_mean_arrow(table, col) -> float:
    """Mean of column."""

def table_min_arrow(table, col) -> Any:
    """Minimum of column."""

def table_max_arrow(table, col) -> Any:
    """Maximum of column."""

def table_count_arrow(table, col) -> int:
    """Count non-null values in column."""

def table_count_distinct_arrow(table, col) -> int:
    """Count distinct values in column."""

def table_stddev_arrow(table, col) -> float:
    """Standard deviation of column."""

def table_variance_arrow(table, col) -> float:
    """Variance of column."""

def table_first_arrow(table, col) -> Any:
    """First non-null value in column."""

def table_last_arrow(table, col) -> Any:
    """Last non-null value in column."""

def table_any_arrow(table, col) -> bool:
    """True if any value is true (boolean column)."""

def table_all_arrow(table, col) -> bool:
    """True if all values are true (boolean column)."""
```

### 8. Cumulative Functions (Running Operations)

```python
def table_cumsum_arrow(table, col, result_col=None) -> dict:
    """Cumulative sum of column."""

def table_cumprod_arrow(table, col, result_col=None) -> dict:
    """Cumulative product of column."""

def table_cummin_arrow(table, col, result_col=None) -> dict:
    """Cumulative minimum of column."""

def table_cummax_arrow(table, col, result_col=None) -> dict:
    """Cumulative maximum of column."""

def table_cummean_arrow(table, col, result_col=None) -> dict:
    """Cumulative mean of column."""

def table_diff_arrow(table, col, result_col=None) -> dict:
    """Pairwise differences (like pandas diff)."""
```

### 9. Element-wise Selection

```python
def table_if_else_arrow(table, cond_col, true_col, false_col, result_col=None) -> dict:
    """
    Choose values based on condition.

    Args:
        cond_col: Boolean column name
        true_col: Column name or scalar for true values
        false_col: Column name or scalar for false values
        result_col: Result column name (default: replaces cond_col)
    """

def table_coalesce_arrow(table, cols: list[str], result_col=None) -> dict:
    """First non-null value from list of columns."""

def table_fill_null_arrow(table, col, value, result_col=None) -> dict:
    """Fill null values with value."""

def table_fill_null_forward_arrow(table, col, result_col=None) -> dict:
    """Forward-fill null values."""

def table_fill_null_backward_arrow(table, col, result_col=None) -> dict:
    """Backward-fill null values."""
```

### 10. Filter (Row Selection)

```python
def table_filter_arrow(table, col_or_mask) -> dict:
    """
    Filter table rows by boolean column or mask.

    Args:
        table: Input table
        col_or_mask: Boolean column name or pa.Array mask

    Returns:
        Arrow table with filtered rows.
    """
```

---

## Implementation Strategy

### Internal Helper

```python
def _ensure_arrow(table: dict) -> dict:
    """Convert table to arrow orientation if not already."""
    if _is_arrow(table):
        return table
    return table_to_arrow(table)

def _apply_unary_arrow(table, col, pc_func, result_col=None):
    """Apply unary PyArrow compute function to column."""
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    result = pc_func(arr)

    out_col = result_col if result_col else t["columns"][col_idx]
    # ... build output table with new/replaced column

def _apply_binary_arrow(table, col, value_or_col, pc_func, result_col=None):
    """Apply binary PyArrow compute function."""
    # Handle scalar vs column for second arg
    # ...
```

### File Organization

All functions go in `table.py` in a new section:

```python
# -------------------------------------
# Arrow compute functions
# -------------------------------------
```

### Export

Add to `__init__.py`:
- All `table_*_arrow` functions

---

## Implementation Phases

### Phase 1: Core Infrastructure
- `_ensure_arrow()` helper
- `_apply_unary_arrow()` helper
- `_apply_binary_arrow()` helper

### Phase 2: Arithmetic (15 functions)
- add, subtract, multiply, divide, negate
- abs, sign, power, sqrt
- exp, ln, log10, log2
- round, ceil, floor, trunc

### Phase 3: Trigonometric (7 functions)
- sin, cos, tan, asin, acos, atan, atan2

### Phase 4: Comparison (6 functions)
- equal, not_equal, less, less_equal, greater, greater_equal

### Phase 5: Null/Value Checks (5 functions)
- is_null, is_valid, is_nan, is_finite, is_in

### Phase 6: Logical (4 functions)
- and, or, xor, invert

### Phase 7: String (13 functions)
- upper, lower, capitalize, title
- strip, lstrip, rstrip
- length, starts_with, ends_with, contains
- replace, split

### Phase 8: Aggregates (13 functions)
- summarize (main function)
- sum, mean, min, max, count, count_distinct
- stddev, variance, first, last, any, all

### Phase 9: Cumulative (6 functions)
- cumsum, cumprod, cummin, cummax, cummean, diff

### Phase 10: Selection (5 functions)
- if_else, coalesce, fill_null, fill_null_forward, fill_null_backward

### Phase 11: Filter (1 function)
- filter

### Phase 12: Tests and Documentation
- Unit tests for each function
- Update notebook with examples
- Update docs

---

## Function Count Summary

| Category | Count |
|----------|-------|
| Arithmetic | 15 |
| Trigonometric | 7 |
| Comparison | 6 |
| Null checks | 5 |
| Logical | 4 |
| String | 13 |
| Aggregates | 13 |
| Cumulative | 6 |
| Selection | 5 |
| Filter | 1 |
| **Total** | **75** |

---

## API Design Decisions

1. **Column specification**: Use column name (string) or index (int), consistent with existing functions.

2. **Result column**: Optional `result_col` parameter. If None, replaces input column. If specified, adds new column.

3. **Scalar vs column**: Binary operations accept either scalar or column name for second argument.

4. **Return type**: All functions return arrow-oriented table (except scalar aggregates like `table_sum_arrow`).

5. **Null handling**: Follow PyArrow defaults (nulls propagate). Users can filter nulls first if needed.

6. **Error handling**: Use unchecked versions by default (faster). Add `checked=True` parameter if overflow detection needed.

---

## Example Usage

```python
from specparser.amt import (
    table_to_arrow,
    table_add_arrow,
    table_multiply_arrow,
    table_upper_arrow,
    table_summarize_arrow,
    table_cumsum_arrow,
    table_filter_arrow,
    table_is_null_arrow,
)

# Start with any table
sales = {"orientation": "row", "columns": ["product", "price", "qty"], ...}

# Chain operations (all return arrow tables)
result = table_multiply_arrow(sales, "price", "qty", result_col="total")
result = table_cumsum_arrow(result, "total", result_col="running_total")

# String operations
result = table_upper_arrow(result, "product")

# Summarize
summary = table_summarize_arrow(result, {
    "total": ["sum", "mean", "max"],
    "qty": "count"
})

# Filter
mask_col = table_greater_arrow(result, "total", 100, result_col="_mask")
filtered = table_filter_arrow(mask_col, "_mask")
filtered = table_drop_columns(filtered, ["_mask"])
```

---

## Files to Modify

1. **`src/specparser/amt/table.py`** - Add all 75 functions
2. **`src/specparser/amt/__init__.py`** - Export new functions
3. **`tests/test_amt.py`** - Add tests
4. **`notebooks/table.ipynb`** - Add examples section
5. **`docs/table.md`** - Document new functions

---

## Acceptance Criteria

- [ ] All 75 functions implemented
- [ ] All functions accept any orientation, return arrow
- [ ] All functions have docstrings
- [ ] Unit tests pass
- [ ] Notebook examples work
- [ ] No performance regression for existing functions
- [ ] Benchmarks added to notebook

---

## Notebook Benchmarks

Add a new section "10. Arrow Compute Benchmarks" to `notebooks/table.ipynb` comparing:

### Benchmark 1: Arithmetic Operations
Compare `table_multiply_arrow` vs Python loop for computing `price * qty`.

```python
# Generate large table
n_rows = 100_000
sales_large = {
    "orientation": "row",
    "columns": ["product", "price", "qty"],
    "rows": [[f"prod_{i}", float(i % 100), i % 50] for i in range(n_rows)]
}

# Python loop approach
def multiply_python(table, col1, col2, result_col):
    t = table_to_rows(table)
    idx1 = t["columns"].index(col1)
    idx2 = t["columns"].index(col2)
    new_rows = [row + [row[idx1] * row[idx2]] for row in t["rows"]]
    return {"orientation": "row",
            "columns": t["columns"] + [result_col],
            "rows": new_rows}

# Benchmark Python loop
t0 = time.perf_counter()
result_py = multiply_python(sales_large, "price", "qty", "total")
t_python = time.perf_counter() - t0

# Benchmark Arrow
t0 = time.perf_counter()
result_arrow = table_multiply_arrow(sales_large, "price", "qty", result_col="total")
t_arrow = time.perf_counter() - t0

print(f"Python loop: {t_python:.3f}s")
print(f"Arrow compute: {t_arrow:.3f}s")
print(f"Speedup: {t_python/t_arrow:.1f}x")
```

### Benchmark 2: String Operations
Compare `table_upper_arrow` vs Python loop for uppercase conversion.

```python
# Generate string table
strings_large = {
    "orientation": "row",
    "columns": ["name"],
    "rows": [[f"product_name_{i}_description"] for i in range(n_rows)]
}

# Python loop
def upper_python(table, col):
    t = table_to_rows(table)
    idx = t["columns"].index(col)
    new_rows = [[row[idx].upper()] for row in t["rows"]]
    return {"orientation": "row", "columns": t["columns"], "rows": new_rows}

# Benchmark
t0 = time.perf_counter()
result_py = upper_python(strings_large, "name")
t_python = time.perf_counter() - t0

t0 = time.perf_counter()
result_arrow = table_upper_arrow(strings_large, "name")
t_arrow = time.perf_counter() - t0

print(f"Python loop: {t_python:.3f}s")
print(f"Arrow compute: {t_arrow:.3f}s")
print(f"Speedup: {t_python/t_arrow:.1f}x")
```

### Benchmark 3: Aggregation
Compare `table_summarize_arrow` vs Python sum/mean.

```python
# Python aggregation
def summarize_python(table, col):
    t = table_to_rows(table)
    idx = t["columns"].index(col)
    values = [row[idx] for row in t["rows"]]
    return {"sum": sum(values), "mean": sum(values)/len(values)}

# Benchmark
t0 = time.perf_counter()
result_py = summarize_python(sales_large, "price")
t_python = time.perf_counter() - t0

t0 = time.perf_counter()
result_arrow = table_summarize_arrow(sales_large, {"price": ["sum", "mean"]})
t_arrow = time.perf_counter() - t0

print(f"Python: {t_python:.3f}s")
print(f"Arrow: {t_arrow:.3f}s")
print(f"Speedup: {t_python/t_arrow:.1f}x")
```

### Benchmark 4: Cumulative Sum
Compare `table_cumsum_arrow` vs Python cumulative loop.

```python
# Python cumulative sum
def cumsum_python(table, col, result_col):
    t = table_to_rows(table)
    idx = t["columns"].index(col)
    running = 0
    new_rows = []
    for row in t["rows"]:
        running += row[idx]
        new_rows.append(row + [running])
    return {"orientation": "row",
            "columns": t["columns"] + [result_col],
            "rows": new_rows}

# Benchmark
t0 = time.perf_counter()
result_py = cumsum_python(sales_large, "qty", "cumqty")
t_python = time.perf_counter() - t0

t0 = time.perf_counter()
result_arrow = table_cumsum_arrow(sales_large, "qty", result_col="cumqty")
t_arrow = time.perf_counter() - t0

print(f"Python loop: {t_python:.3f}s")
print(f"Arrow compute: {t_arrow:.3f}s")
print(f"Speedup: {t_python/t_arrow:.1f}x")
```

### Benchmark 5: Filter by Condition
Compare `table_filter_arrow` with `table_greater_arrow` vs Python list comprehension.

```python
# Python filter
def filter_python(table, col, threshold):
    t = table_to_rows(table)
    idx = t["columns"].index(col)
    filtered = [row for row in t["rows"] if row[idx] > threshold]
    return {"orientation": "row", "columns": t["columns"], "rows": filtered}

# Benchmark
t0 = time.perf_counter()
result_py = filter_python(sales_large, "price", 50)
t_python = time.perf_counter() - t0

t0 = time.perf_counter()
mask_table = table_greater_arrow(sales_large, "price", 50, result_col="_mask")
result_arrow = table_filter_arrow(mask_table, "_mask")
result_arrow = table_drop_columns(result_arrow, ["_mask"])
t_arrow = time.perf_counter() - t0

print(f"Python filter: {t_python:.3f}s")
print(f"Arrow filter: {t_arrow:.3f}s")
print(f"Speedup: {t_python/t_arrow:.1f}x")
```

### Summary Table

Display a summary table of all benchmarks:

```python
benchmarks = {
    "orientation": "row",
    "columns": ["Operation", "Python (s)", "Arrow (s)", "Speedup"],
    "rows": [
        ["Multiply columns", f"{t_mult_py:.3f}", f"{t_mult_arrow:.3f}", f"{t_mult_py/t_mult_arrow:.1f}x"],
        ["Uppercase strings", f"{t_upper_py:.3f}", f"{t_upper_arrow:.3f}", f"{t_upper_py/t_upper_arrow:.1f}x"],
        ["Sum + Mean", f"{t_agg_py:.3f}", f"{t_agg_arrow:.3f}", f"{t_agg_py/t_agg_arrow:.1f}x"],
        ["Cumulative sum", f"{t_cum_py:.3f}", f"{t_cum_arrow:.3f}", f"{t_cum_py/t_cum_arrow:.1f}x"],
        ["Filter by value", f"{t_filt_py:.3f}", f"{t_filt_arrow:.3f}", f"{t_filt_py/t_filt_arrow:.1f}x"],
    ]
}
print_table(benchmarks)
```

Expected results: 5-50x speedup depending on operation.
