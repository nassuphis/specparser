# Table Utilities

The `specparser.amt.table` module provides functions for manipulating dict-based tables.

## Table Format

Tables are represented as Python dicts with three keys:

```python
{
    "orientation": "row" | "column",
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
