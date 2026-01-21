# Plan: table_unchop and table_chop Functions

## Overview

Add two complementary functions to `table.py` for handling list-valued columns:

- `table_unchop(table, column)` - Expands rows with list values into multiple rows
- `table_chop(table, column)` - Collapses multiple rows back into list values (inverse)

## Function: table_unchop

### Purpose
"Unroll" a column containing lists into multiple rows, one per list element.

### Example

**Input table:**
```python
{
    "columns": ["a", "b", "c", "d"],
    "rows": [
        ["this", "that", [1, 2, 3, 4], "something"],
        ["foo", "bar", [5, 6], "else"]
    ]
}
```

**After `table_unchop(table, "c")` or `table_unchop(table, 2)`:**
```python
{
    "columns": ["a", "b", "c", "d"],
    "rows": [
        ["this", "that", 1, "something"],
        ["this", "that", 2, "something"],
        ["this", "that", 3, "something"],
        ["this", "that", 4, "something"],
        ["foo", "bar", 5, "else"],
        ["foo", "bar", 6, "else"]
    ]
}
```

### Signature
```python
def table_unchop(table: dict[str, Any], column: str | int) -> dict[str, Any]:
    """Expand rows by unrolling a list-valued column into multiple rows.

    Args:
        table: Dict with 'columns' and 'rows'
        column: Column name (str) or index (int) containing list values

    Returns:
        New table with expanded rows (one row per list element)

    Notes:
        - If a cell contains an empty list, that row is omitted from output
        - If a cell is not a list, it's treated as a single-element list
    """
```

### Edge Cases
1. Empty list `[]` → row is dropped (no output rows for that input row)
2. Non-list value → treated as single-element list (row passes through unchanged)
3. `None` value → treated as empty list (row dropped) OR single-element? **Decision: treat as single element**

## Function: table_chop

### Purpose
The inverse of `table_unchop` - group consecutive rows and collect values into lists.

### Example

**Input table:**
```python
{
    "columns": ["a", "b", "c", "d"],
    "rows": [
        ["this", "that", 1, "something"],
        ["this", "that", 2, "something"],
        ["this", "that", 3, "something"],
        ["foo", "bar", 5, "else"],
        ["foo", "bar", 6, "else"]
    ]
}
```

**After `table_chop(table, "c")` or `table_chop(table, 2)`:**
```python
{
    "columns": ["a", "b", "c", "d"],
    "rows": [
        ["this", "that", [1, 2, 3], "something"],
        ["foo", "bar", [5, 6], "else"]
    ]
}
```

### Signature
```python
def table_chop(table: dict[str, Any], column: str | int) -> dict[str, Any]:
    """Collapse rows by grouping on non-target columns and collecting target into lists.

    Args:
        table: Dict with 'columns' and 'rows'
        column: Column name (str) or index (int) to collect into lists

    Returns:
        New table with grouped rows (target column values collected into lists)

    Notes:
        - Groups by all columns EXCEPT the target column
        - Preserves order of first occurrence of each group
        - Values are collected in the order they appear
    """
```

### Grouping Logic
- Key = tuple of all column values EXCEPT the target column
- Collect target column values into a list for each unique key
- Preserve insertion order (use dict for seen keys)

## Implementation

### Helper: resolve column index
```python
def _resolve_column_index(table: dict[str, Any], column: str | int) -> int:
    """Convert column name or index to index, with validation."""
    if isinstance(column, int):
        if column < 0 or column >= len(table["columns"]):
            raise ValueError(f"Column index {column} out of range")
        return column
    try:
        return table["columns"].index(column)
    except ValueError:
        raise ValueError(f"Column '{column}' not found in table columns: {table['columns']}")
```

### table_unchop implementation
```python
def table_unchop(table: dict[str, Any], column: str | int) -> dict[str, Any]:
    col_idx = _resolve_column_index(table, column)

    rows = []
    for row in table["rows"]:
        cell = row[col_idx]
        # Handle non-list as single element
        if not isinstance(cell, list):
            rows.append(row[:])
        elif len(cell) == 0:
            # Empty list → skip row
            continue
        else:
            # Expand: one output row per list element
            for val in cell:
                new_row = row[:col_idx] + [val] + row[col_idx + 1:]
                rows.append(new_row)

    return {"columns": table["columns"][:], "rows": rows}
```

### table_chop implementation
```python
def table_chop(table: dict[str, Any], column: str | int) -> dict[str, Any]:
    col_idx = _resolve_column_index(table, column)

    # Group by all columns except target
    groups = {}  # key tuple -> [list of target values]
    group_first_row = {}  # key tuple -> first row (for non-target values)

    for row in table["rows"]:
        # Build key from all columns except target
        key = tuple(row[:col_idx] + row[col_idx + 1:])
        target_val = row[col_idx]

        if key not in groups:
            groups[key] = []
            group_first_row[key] = row
        groups[key].append(target_val)

    # Build output rows preserving order
    rows = []
    for key in groups:
        first_row = group_first_row[key]
        collected = groups[key]
        new_row = first_row[:col_idx] + [collected] + first_row[col_idx + 1:]
        rows.append(new_row)

    return {"columns": table["columns"][:], "rows": rows}
```

## Testing

```python
# Test unchop
t1 = {"columns": ["a", "b", "c"], "rows": [["x", [1,2,3], "y"]]}
assert table_unchop(t1, "b") == {
    "columns": ["a", "b", "c"],
    "rows": [["x", 1, "y"], ["x", 2, "y"], ["x", 3, "y"]]
}

# Test unchop with int index
assert table_unchop(t1, 1) == table_unchop(t1, "b")

# Test unchop empty list
t2 = {"columns": ["a", "b"], "rows": [["x", []], ["y", [1]]]}
assert table_unchop(t2, "b") == {"columns": ["a", "b"], "rows": [["y", 1]]}

# Test unchop non-list (passthrough)
t3 = {"columns": ["a", "b"], "rows": [["x", 5]]}
assert table_unchop(t3, "b") == {"columns": ["a", "b"], "rows": [["x", 5]]}

# Test chop
t4 = {"columns": ["a", "b", "c"], "rows": [["x", 1, "y"], ["x", 2, "y"], ["x", 3, "y"]]}
assert table_chop(t4, "b") == {
    "columns": ["a", "b", "c"],
    "rows": [["x", [1, 2, 3], "y"]]
}

# Test roundtrip: unchop then chop
t5 = {"columns": ["a", "b", "c"], "rows": [["x", [1,2], "y"], ["z", [3,4], "w"]]}
assert table_chop(table_unchop(t5, "b"), "b") == t5
```

## Files to Modify

- `src/specparser/amt/table.py` - Add both functions + helper
- `src/specparser/amt/__init__.py` - Export new functions

## Decision Points

1. **Empty list behavior in unchop**: Drop the row (chosen) vs keep row with None
2. **Non-list behavior in unchop**: Pass through unchanged (chosen) vs error
3. **Helper function**: Make `_resolve_column_index` private or export it?
   - Recommend: private (underscore prefix), but could be useful elsewhere
