# Table Utilities

The `specparser.amt.table` module provides functions for manipulating dict-based tables.

## Table Format

Tables are represented as Python dicts with three keys:

```python
{
    "orientation": "row" | "column" | "arrow",
    "columns": ["col1", "col2", ...],
    "rows": [...]
}
```

**Row-oriented** (default):
```python
{"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
# Row 0: [1, 2]
# Row 1: [3, 4]
```

**Column-oriented**:
```python
{"orientation": "column", "columns": ["a", "b"], "rows": [[1, 3], [2, 4]]}
# Column "a": [1, 3]
# Column "b": [2, 4]
```

**Arrow-oriented** (PyArrow arrays):
```python
import pyarrow as pa
{"orientation": "arrow", "columns": ["a", "b"], "rows": [pa.array([1, 3]), pa.array([2, 4])]}
# Column "a": PyArrow array [1, 3]
# Column "b": PyArrow array [2, 4]
```

---

## Table Inspection

### `table_orientation(table)`

Get the orientation of a table.

```python
from specparser.amt import table_orientation

t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2]]}
table_orientation(t)  # "row"
```

### `table_nrows(table)`

Get the number of rows in a table.

```python
from specparser.amt import table_nrows

t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
table_nrows(t)  # 2
```

### `table_validate(table, strict=False)`

Validate table structure. Raises `ValueError` if invalid.

```python
from specparser.amt import table_validate

t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
table_validate(t)  # No error

# With strict=True, checks that all rows have correct length
table_validate(t, strict=True)
```

---

## Orientation Conversion

### `table_to_columns(table)`

Convert a row-oriented table to column-oriented.

```python
from specparser.amt import table_to_columns

t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
result = table_to_columns(t)
# {"orientation": "column", "columns": ["a", "b"], "rows": [[1, 3], [2, 4]]}
```

**Parameters:**
- `table`: Input table dict

**Returns:** Column-oriented table. If already column-oriented, returns the same object.

---

### `table_to_rows(table)`

Convert a column-oriented table to row-oriented.

```python
from specparser.amt import table_to_rows

t = {"orientation": "column", "columns": ["a", "b"], "rows": [[1, 3], [2, 4]]}
result = table_to_rows(t)
# {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
```

**Parameters:**
- `table`: Input table dict

**Returns:** Row-oriented table. If already row-oriented, returns the same object.

---

### `table_to_arrow(table)`

Convert a table to Arrow orientation (PyArrow arrays).

```python
from specparser.amt import table_to_arrow

t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
result = table_to_arrow(t)
# {"orientation": "arrow", "columns": ["a", "b"], "rows": [pa.array([1, 3]), pa.array([2, 4])]}
```

**Parameters:**
- `table`: Input table dict

**Returns:** Arrow-oriented table with PyArrow arrays. If already arrow-oriented, returns the same object.

**Note:** Requires PyArrow to be installed.

---

### `table_to_jsonable(table)`

Convert a table to a JSON-serializable format (converts Arrow arrays to Python lists).

```python
from specparser.amt import table_to_jsonable
import json

t = table_to_arrow({"orientation": "row", "columns": ["a"], "rows": [[1], [2]]})
result = table_to_jsonable(t)
json.dumps(result)  # Works - no PyArrow arrays
```

---

## Column Operations

### `table_column(table, colname)`

Extract a single column as a list.

```python
from specparser.amt import table_column

t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
result = table_column(t, "a")
# [1, 3]
```

**Parameters:**
- `table`: Input table dict
- `colname`: Name of the column to extract (str)

**Returns:** List of values from that column

**Raises:** `ValueError` if column name not found

**Performance:**
- Row-oriented: O(n) - iterates all rows
- Column-oriented: O(1) - direct index access

---

### `table_select_columns(table, columns)`

Select and reorder columns from a table.

```python
from specparser.amt import table_select_columns

t = {"orientation": "row", "columns": ["a", "b", "c"], "rows": [[1, 2, 3], [4, 5, 6]]}
result = table_select_columns(t, ["c", "a"])
# {"orientation": "row", "columns": ["c", "a"], "rows": [[3, 1], [6, 4]]}
```

**Parameters:**
- `table`: Input table dict
- `columns`: List of column names in desired order

**Returns:** New table with only the specified columns (preserves orientation)

**Raises:** `ValueError` if a column name is not found

---

### `table_add_column(table, colname, value=None, position=None)`

Add a new column with a constant value.

```python
from specparser.amt import table_add_column

t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}

# Add at end
result = table_add_column(t, "c", value="X")
# {"orientation": "row", "columns": ["a", "b", "c"], "rows": [[1, 2, "X"], [3, 4, "X"]]}

# Add at position 0
result = table_add_column(t, "c", value="X", position=0)
# {"orientation": "row", "columns": ["c", "a", "b"], "rows": [["X", 1, 2], ["X", 3, 4]]}
```

**Parameters:**
- `table`: Input table dict
- `colname`: Name of the new column
- `value`: Value to fill in for all rows (default: `None`)
- `position`: Index to insert the column (default: append at end)

**Returns:** New table with the added column (preserves orientation)

---

### `table_drop_columns(table, columns)`

Remove specified columns from a table.

```python
from specparser.amt import table_drop_columns

t = {"orientation": "row", "columns": ["a", "b", "c"], "rows": [[1, 2, 3], [4, 5, 6]]}
result = table_drop_columns(t, ["b"])
# {"orientation": "row", "columns": ["a", "c"], "rows": [[1, 3], [4, 6]]}
```

**Parameters:**
- `table`: Input table dict
- `columns`: List of column names to remove

**Returns:** New table without the specified columns (preserves orientation)

**Note:** Non-existent column names are silently ignored.

---

## Row Operations

### `table_bind_rows(*tables)`

Concatenate rows from multiple tables with the same columns.

```python
from specparser.amt import table_bind_rows

t1 = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2]]}
t2 = {"orientation": "row", "columns": ["a", "b"], "rows": [[3, 4], [5, 6]]}
result = table_bind_rows(t1, t2)
# {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4], [5, 6]]}
```

**Parameters:**
- `*tables`: Variable number of tables to bind

**Returns:** New table with all rows concatenated (matches first table's orientation)

**Raises:** `ValueError` if tables have different columns

**Note:** If called with no arguments, returns an empty row-oriented table.

---

### `table_unique_rows(table)`

Remove duplicate rows from a table.

```python
from specparser.amt import table_unique_rows

t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4], [1, 2]]}
result = table_unique_rows(t)
# {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
```

**Parameters:**
- `table`: Input table dict

**Returns:** New table with only unique rows (always row-oriented)

**Note:** Preserves the last occurrence of each duplicate row.

---

### `table_head(table, n=10)`

Return the first n rows of a table.

```python
from specparser.amt import table_head

t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3], [4], [5]]}
result = table_head(t, 3)
# {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3]]}
```

**Parameters:**
- `table`: Input table dict
- `n`: Number of rows to return (default: 10)

**Returns:** New table with only the first n rows (preserves orientation)

---

### `table_sample(table, n=10)`

Return a random sample of n rows from a table (without replacement).

```python
from specparser.amt import table_sample

t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3], [4], [5]]}
result = table_sample(t, 2)
# {"orientation": "row", "columns": ["a"], "rows": [[3], [1]]}  # random
```

**Parameters:**
- `table`: Input table dict
- `n`: Number of rows to sample (default: 10)

**Returns:** New table with sampled rows (always row-oriented)

**Note:** If the table has fewer than n rows, returns all rows.

---

## Value Operations

### `table_replace_value(table, colname, old_value, new_value)`

Replace occurrences of a value in a specific column.

```python
from specparser.amt import table_replace_value

t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, "old"], [2, "old"], [3, "keep"]]}
result = table_replace_value(t, "b", "old", "new")
# {"orientation": "row", "columns": ["a", "b"], "rows": [[1, "new"], [2, "new"], [3, "keep"]]}
```

**Parameters:**
- `table`: Input table dict
- `colname`: Name of the column to modify
- `old_value`: Value to replace
- `new_value`: Replacement value

**Returns:** New table with values replaced (preserves orientation)

**Raises:** `ValueError` if column name not found

---

## Multi-Table Operations

### `table_stack_cols(*tables, key_col=0, copy_data=True)`

Stack columns from multiple tables side-by-side.

Takes the key column from the first table, then appends all non-key columns from each table. All tables must have the same number of rows.

```python
from specparser.amt import table_stack_cols

t1 = {"orientation": "row", "columns": ["key", "a"], "rows": [[1, 10], [2, 20]]}
t2 = {"orientation": "row", "columns": ["key", "b"], "rows": [[1, 100], [2, 200]]}
result = table_stack_cols(t1, t2)
# {"orientation": "column", "columns": ["key", "a", "b"], "rows": [[1, 2], [10, 20], [100, 200]]}
```

**Parameters:**
- `*tables`: Tables to stack
- `key_col`: Index of the key column (default: 0)
- `copy_data`: If `True`, copy column data; if `False`, reference existing lists (default: `True`)

**Returns:** Table with combined columns (always column-oriented)

**Raises:** `ValueError` if tables have different row counts

**Note:** This is NOT a SQL-style join. It assumes tables are already row-aligned.

**Performance:** O(k) where k is the total number of columns (no row iteration needed when inputs are column-oriented).

---

## Reshape Operations

### `table_unchop(table, column)`

Expand rows by unrolling a list-valued column into multiple rows.

```python
from specparser.amt import table_unchop

t = {
    "orientation": "row",
    "columns": ["id", "tags"],
    "rows": [
        ["A", ["x", "y", "z"]],
        ["B", ["p", "q"]],
    ]
}
result = table_unchop(t, "tags")
# {"orientation": "column", "columns": ["id", "tags"], "rows": [["A", "A", "A", "B", "B"], ["x", "y", "z", "p", "q"]]}
```

**Parameters:**
- `table`: Input table dict
- `column`: Column name (str) or index (int) containing list values

**Returns:** New table with expanded rows (always column-oriented for efficiency)

**Notes:**
- If a cell contains an empty list, that row is omitted from output
- If a cell is not a list, it's treated as a single-element list
- Uses `extend()` + `itertools.repeat()` for O(n*m) performance

---

### `table_chop(table, column)`

Collapse rows by grouping on non-target columns and collecting target into lists.

```python
from specparser.amt import table_chop

t = {
    "orientation": "row",
    "columns": ["id", "value"],
    "rows": [
        ["A", 1],
        ["A", 2],
        ["B", 3],
        ["A", 4],
    ]
}
result = table_chop(t, "value")
# {"orientation": "row", "columns": ["id", "value"], "rows": [["A", [1, 2, 4]], ["B", [3]]]}
```

**Parameters:**
- `table`: Input table dict
- `column`: Column name (str) or index (int) to collect into lists

**Returns:** New table with grouped rows (always row-oriented)

**Notes:**
- Groups by all columns EXCEPT the target column
- Preserves order of first occurrence of each group
- Values are collected in the order they appear

---

## Join Operations

### `table_left_join(left, right, left_on, right_on=None)`

Left join two tables on key columns.

```python
from specparser.amt import table_left_join

left = {"orientation": "row", "columns": ["id", "name"], "rows": [[1, "a"], [2, "b"], [3, "c"]]}
right = {"orientation": "row", "columns": ["id", "value"], "rows": [[1, 100], [2, 200]]}

result = table_left_join(left, right, "id")
# All rows from left, matched values from right (None for id=3)
```

**Parameters:**
- `left`: Left table
- `right`: Right table
- `left_on`: Column name in left table to join on
- `right_on`: Column name in right table (defaults to `left_on`)

**Returns:** Joined table with all left rows and matching right columns

### `table_inner_join(left, right, left_on, right_on=None)`

Inner join two tables on key columns.

```python
from specparser.amt import table_inner_join

left = {"orientation": "row", "columns": ["id", "name"], "rows": [[1, "a"], [2, "b"], [3, "c"]]}
right = {"orientation": "row", "columns": ["id", "value"], "rows": [[1, 100], [2, 200]]}

result = table_inner_join(left, right, "id")
# Only rows where id matches in both tables (id=1, id=2)
```

**Parameters:** Same as `table_left_join`

**Returns:** Joined table with only matching rows

---

## Pivot Operations

### `table_pivot_wider(table, names_from, values_from, id_cols=None)`

Pivot a table from long to wide format.

```python
from specparser.amt import table_pivot_wider

t = {
    "orientation": "row",
    "columns": ["date", "metric", "value"],
    "rows": [
        ["2024-01-01", "price", 100],
        ["2024-01-01", "volume", 1000],
        ["2024-01-02", "price", 105],
        ["2024-01-02", "volume", 1200],
    ]
}

result = table_pivot_wider(t, names_from="metric", values_from="value")
# Columns: ["date", "price", "volume"]
# Rows: [["2024-01-01", 100, 1000], ["2024-01-02", 105, 1200]]
```

**Parameters:**
- `table`: Input table in long format
- `names_from`: Column whose values become new column names
- `values_from`: Column whose values fill the new columns
- `id_cols`: Columns to keep as identifiers (default: all other columns)

**Returns:** Wide-format table

---

## Window Operations

### `table_lag(table, column, n=1, result_column=None, default=None)`

Add a lagged column (previous row's value).

```python
from specparser.amt import table_lag

t = {"orientation": "row", "columns": ["date", "value"], "rows": [
    ["2024-01-01", 100],
    ["2024-01-02", 105],
    ["2024-01-03", 110],
]}

result = table_lag(t, "value", n=1, result_column="prev_value")
# Adds column "prev_value" with [None, 100, 105]
```

**Parameters:**
- `table`: Input table (must be pre-sorted)
- `column`: Column to lag
- `n`: Number of rows to lag (default: 1)
- `result_column`: Name for lagged column (default: `{column}_lag`)
- `default`: Value for first n rows (default: None)

**Returns:** Table with lagged column added

### `table_lead(table, column, n=1, result_column=None, default=None)`

Add a lead column (next row's value).

```python
from specparser.amt import table_lead

t = {"orientation": "row", "columns": ["date", "value"], "rows": [
    ["2024-01-01", 100],
    ["2024-01-02", 105],
    ["2024-01-03", 110],
]}

result = table_lead(t, "value", n=1, result_column="next_value")
# Adds column "next_value" with [105, 110, None]
```

**Parameters:** Same as `table_lag`

**Returns:** Table with lead column added

---

## Explode Operations

### `table_explode_arrow(table, column)`

Explode a list-valued column into multiple rows (Arrow-native, strict).

```python
from specparser.amt import table_explode_arrow, table_to_arrow
import pyarrow as pa

t = {
    "orientation": "arrow",
    "columns": ["id", "tags"],
    "rows": [
        pa.array(["A", "B"]),
        pa.array([["x", "y"], ["p", "q", "r"]], type=pa.list_(pa.string()))
    ]
}

result = table_explode_arrow(t, "tags")
# Rows expanded: A->x, A->y, B->p, B->q, B->r
```

**Parameters:**
- `table`: Arrow-oriented table
- `column`: Column name or index containing list values

**Returns:** Table with list column expanded to individual rows

**Raises:** `TypeError` if column is not a list type

**Note:** Empty lists produce no output rows. Nulls also produce no output rows.

---

## Display/Output

### `format_table(table)`

Format a table as a tab-separated string with header.

```python
from specparser.amt import format_table

t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
result = format_table(t)
# "a\tb\n1\t2\n3\t4"
```

**Parameters:**
- `table`: Input table dict

**Returns:** Tab-separated string with header row

**Raises:** `ValueError` if table has more than 100,000 rows

**Notes:**
- Float values are formatted to 3 significant figures
- Numeric strings are also formatted as numbers

---

### `print_table(table)`

Print a table with header and rows to stdout.

```python
from specparser.amt import print_table

t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
print_table(t)
# a	b
# 1	2
# 3	4
```

**Parameters:**
- `table`: Input table dict

**Returns:** None (prints to stdout)

**Raises:** `ValueError` if table has more than 100,000 rows

---

## Constants

### `MAX_FORMAT_ROWS`

Maximum number of rows allowed for `format_table()` and `print_table()`.

```python
from specparser.amt.table import MAX_FORMAT_ROWS
# 100_000
```

---

## When to Use Each Orientation

| Operation | Row-oriented | Column-oriented |
|-----------|--------------|-----------------|
| `table_column()` | O(n) | O(1) |
| `table_add_column()` with constant | O(n) | O(1) |
| `table_select_columns()` | O(n*k) | O(k) |
| `table_bind_rows()` | O(n) | O(n*k) |
| `table_stack_cols()` | needs conversion | O(k) |
| `format_table()` / `print_table()` | natural | needs conversion |
| `table_unchop()` | needs conversion | O(n*m) |

**Column-oriented is better for:**
- Large tables with many column operations
- Building tables by appending columns
- `table_unchop()` operations
- `table_stack_cols()` operations

**Row-oriented is better for:**
- Row-by-row iteration
- Printing/displaying
- `table_bind_rows()` operations
- `table_chop()` operations

**Arrow-oriented is better for:**
- Very large datasets (millions of rows)
- Numeric computations (vectorized operations)
- Interop with PyArrow, Pandas, DuckDB
- High-performance aggregations

---

## Arrow Compute Functions

The table module provides PyArrow-backed compute functions that operate on Arrow-oriented tables. These functions are vectorized and highly efficient for large datasets.

### Arithmetic

| Function | Description |
|----------|-------------|
| `table_add_arrow(t, col1, col2, result)` | Add two columns |
| `table_subtract_arrow(t, col1, col2, result)` | Subtract columns |
| `table_multiply_arrow(t, col1, col2, result)` | Multiply columns |
| `table_divide_arrow(t, col1, col2, result)` | Divide columns |
| `table_negate_arrow(t, col, result)` | Negate column |
| `table_abs_arrow(t, col, result)` | Absolute value |
| `table_sign_arrow(t, col, result)` | Sign (-1, 0, 1) |
| `table_power_arrow(t, col, exp, result)` | Raise to power |
| `table_sqrt_arrow(t, col, result)` | Square root |
| `table_exp_arrow(t, col, result)` | Exponential |
| `table_ln_arrow(t, col, result)` | Natural log |
| `table_log10_arrow(t, col, result)` | Log base 10 |
| `table_log2_arrow(t, col, result)` | Log base 2 |
| `table_round_arrow(t, col, result, decimals=0)` | Round |
| `table_ceil_arrow(t, col, result)` | Ceiling |
| `table_floor_arrow(t, col, result)` | Floor |
| `table_trunc_arrow(t, col, result)` | Truncate |

### Trigonometric

| Function | Description |
|----------|-------------|
| `table_sin_arrow(t, col, result)` | Sine |
| `table_cos_arrow(t, col, result)` | Cosine |
| `table_tan_arrow(t, col, result)` | Tangent |
| `table_asin_arrow(t, col, result)` | Arc sine |
| `table_acos_arrow(t, col, result)` | Arc cosine |
| `table_atan_arrow(t, col, result)` | Arc tangent |
| `table_atan2_arrow(t, col1, col2, result)` | Two-argument arc tangent |

### Comparison

| Function | Description |
|----------|-------------|
| `table_equal_arrow(t, col1, col2, result)` | Equal (==) |
| `table_not_equal_arrow(t, col1, col2, result)` | Not equal (!=) |
| `table_less_arrow(t, col1, col2, result)` | Less than (<) |
| `table_less_equal_arrow(t, col1, col2, result)` | Less or equal (<=) |
| `table_greater_arrow(t, col1, col2, result)` | Greater than (>) |
| `table_greater_equal_arrow(t, col1, col2, result)` | Greater or equal (>=) |

### Null/Value Checks

| Function | Description |
|----------|-------------|
| `table_is_null_arrow(t, col, result)` | Check if null |
| `table_is_valid_arrow(t, col, result)` | Check if not null |
| `table_is_nan_arrow(t, col, result)` | Check if NaN |
| `table_is_finite_arrow(t, col, result)` | Check if finite |
| `table_is_in_arrow(t, col, values, result)` | Check if in set |

### Logical

| Function | Description |
|----------|-------------|
| `table_and_arrow(t, col1, col2, result)` | Logical AND |
| `table_or_arrow(t, col1, col2, result)` | Logical OR |
| `table_xor_arrow(t, col1, col2, result)` | Logical XOR |
| `table_invert_arrow(t, col, result)` | Logical NOT |

### String

| Function | Description |
|----------|-------------|
| `table_upper_arrow(t, col, result)` | Uppercase |
| `table_lower_arrow(t, col, result)` | Lowercase |
| `table_capitalize_arrow(t, col, result)` | Capitalize |
| `table_title_arrow(t, col, result)` | Title case |
| `table_strip_arrow(t, col, result)` | Strip whitespace |
| `table_lstrip_arrow(t, col, result)` | Strip left |
| `table_rstrip_arrow(t, col, result)` | Strip right |
| `table_length_arrow(t, col, result)` | String length |
| `table_starts_with_arrow(t, col, pattern, result)` | Starts with |
| `table_ends_with_arrow(t, col, pattern, result)` | Ends with |
| `table_contains_arrow(t, col, pattern, result)` | Contains |
| `table_replace_substr_arrow(t, col, pattern, replacement, result)` | Replace substring |
| `table_split_arrow(t, col, pattern, result)` | Split string |

### Aggregates

| Function | Description |
|----------|-------------|
| `table_summarize_arrow(t, group_by, aggs)` | Group by + aggregate |
| `table_sum_arrow(t, col)` | Sum (scalar) |
| `table_mean_arrow(t, col)` | Mean (scalar) |
| `table_min_arrow(t, col)` | Min (scalar) |
| `table_max_arrow(t, col)` | Max (scalar) |
| `table_count_arrow(t, col)` | Count non-null (scalar) |
| `table_count_distinct_arrow(t, col)` | Count distinct (scalar) |
| `table_stddev_arrow(t, col)` | Std deviation (scalar) |
| `table_variance_arrow(t, col)` | Variance (scalar) |
| `table_first_arrow(t, col)` | First value (scalar) |
| `table_last_arrow(t, col)` | Last value (scalar) |
| `table_any_arrow(t, col)` | Any true (scalar) |
| `table_all_arrow(t, col)` | All true (scalar) |

### Cumulative

| Function | Description |
|----------|-------------|
| `table_cumsum_arrow(t, col, result)` | Cumulative sum |
| `table_cumprod_arrow(t, col, result)` | Cumulative product |
| `table_cummin_arrow(t, col, result)` | Cumulative min |
| `table_cummax_arrow(t, col, result)` | Cumulative max |
| `table_cummean_arrow(t, col, result)` | Cumulative mean |
| `table_diff_arrow(t, col, result, n=1)` | Difference from previous |

### Selection/Filtering

| Function | Description |
|----------|-------------|
| `table_if_else_arrow(t, cond, true_val, false_val, result)` | Conditional selection |
| `table_coalesce_arrow(t, cols, result)` | First non-null |
| `table_fill_null_arrow(t, col, value, result)` | Fill nulls with value |
| `table_fill_null_forward_arrow(t, col, result)` | Forward fill nulls |
| `table_fill_null_backward_arrow(t, col, result)` | Backward fill nulls |
| `table_filter_arrow(t, mask)` | Filter rows by boolean mask |

### Example Usage

```python
from specparser.amt import (
    table_to_arrow, table_add_arrow, table_multiply_arrow,
    table_cumsum_arrow, table_filter_arrow, table_greater_arrow
)

# Create Arrow table
t = table_to_arrow({
    "orientation": "row",
    "columns": ["price", "quantity"],
    "rows": [[100, 10], [105, 15], [95, 20], [110, 12]]
})

# Add computed columns
t = table_multiply_arrow(t, "price", "quantity", "value")
t = table_cumsum_arrow(t, "value", "cumulative_value")

# Filter rows
t = table_greater_arrow(t, "price", 100, "above_100")
filtered = table_filter_arrow(t, "above_100")
```
