# Plan: Implement Real Key-Based Table Join

## Overview

Add a true SQL-style key-based join function `table_left_join()` to the table utilities. Unlike `table_stack_cols()` which assumes row alignment, this function matches rows by key values.

## Use Cases

- Join asset classification table with pricing table by underlying
- Join straddle schedules with valuation data by straddle name
- Merge any two tables that share a common key column

## API Design

```python
def table_left_join(
    left: dict[str, Any],
    right: dict[str, Any],
    left_on: str | int,
    right_on: str | int | None = None,  # defaults to left_on
    suffixes: tuple[str, str] = ("", "_right"),
) -> dict[str, Any]:
    """
    Left join two tables on key columns.

    All rows from `left` are kept. Matching rows from `right` are appended.
    Non-matching rows get None for right columns.

    Args:
        left: Left table
        right: Right table
        left_on: Key column in left table (name or index)
        right_on: Key column in right table (defaults to left_on if same name)
        suffixes: Suffixes for overlapping column names (left_suffix, right_suffix)

    Returns:
        Row-oriented table with left columns + right columns (excluding right key)

    Raises:
        ValueError: If key column not found
    """
```

### Why Left Join Only (Initially)

- **Left join covers 90% of use cases** - keep all left rows, add matching data from right
- **Inner join** is just left join filtered to non-None results (trivial wrapper)
- **Full outer join** is more complex and rarely needed for this codebase
- Can add `table_inner_join()` and `table_full_join()` later if needed

## Algorithm

```
1. Resolve key column indices using _resolve_column_index()
2. Convert right table to row-oriented (for index building)
3. Build index: dict[key_value] -> list[row_data]  (handles duplicate keys)
4. Resolve column name conflicts with suffixes
5. Convert left table to row-oriented
6. For each left row:
   - Get key value
   - Look up in index
   - If found: append right columns (may produce multiple rows for duplicates)
   - If not found: append None for each right column
7. Return row-oriented result
```

## Implementation

### Phase 1: Add `table_left_join()` to table.py

```python
def table_left_join(
    left: dict[str, Any],
    right: dict[str, Any],
    left_on: str | int,
    right_on: str | int | None = None,
    suffixes: tuple[str, str] = ("", "_right"),
) -> dict[str, Any]:
    """Left join two tables on key columns."""
    # Resolve key columns
    left_key_idx = _resolve_column_index(left, left_on)
    if right_on is None:
        right_on = left["columns"][left_key_idx]  # use same column name
    right_key_idx = _resolve_column_index(right, right_on)

    # Convert to row-oriented for processing
    left_rows = table_to_rows(left)
    right_rows = table_to_rows(right)

    # Build index from right table: key -> list of rows
    right_index: dict[Any, list[list]] = {}
    for row in right_rows["rows"]:
        key = row[right_key_idx]
        if key not in right_index:
            right_index[key] = []
        right_index[key].append(row)

    # Determine output columns (handle name conflicts)
    left_cols = left_rows["columns"]
    right_cols = [c for i, c in enumerate(right_rows["columns"]) if i != right_key_idx]

    out_columns = list(left_cols)
    left_suffix, right_suffix = suffixes
    for rc in right_cols:
        if rc in out_columns:
            # Rename conflicting columns
            out_columns = [c + left_suffix if c == rc else c for c in out_columns]
            out_columns.append(rc + right_suffix)
        else:
            out_columns.append(rc)

    # Number of right columns to add (excluding key)
    n_right_cols = len(right_rows["columns"]) - 1
    none_row = [None] * n_right_cols

    # Perform join
    out_rows = []
    for left_row in left_rows["rows"]:
        key = left_row[left_key_idx]
        matches = right_index.get(key)

        if matches:
            for right_row in matches:
                # Exclude key column from right row
                right_data = [v for i, v in enumerate(right_row) if i != right_key_idx]
                out_rows.append(list(left_row) + right_data)
        else:
            out_rows.append(list(left_row) + none_row)

    return {"orientation": "row", "columns": out_columns, "rows": out_rows}
```

### Phase 2: Add `table_inner_join()` wrapper

```python
def table_inner_join(
    left: dict[str, Any],
    right: dict[str, Any],
    left_on: str | int,
    right_on: str | int | None = None,
    suffixes: tuple[str, str] = ("", "_right"),
) -> dict[str, Any]:
    """Inner join two tables on key columns (only matching rows)."""
    result = table_left_join(left, right, left_on, right_on, suffixes)

    # Filter out rows with None values from right table
    n_left_cols = len(table_to_rows(left)["columns"])
    filtered_rows = [
        row for row in result["rows"]
        if row[n_left_cols] is not None  # check first right column
    ]

    return {"orientation": "row", "columns": result["columns"], "rows": filtered_rows}
```

### Phase 3: Export from `__init__.py`

Add to `__all__` and `_LAZY_IMPORTS`:
- `table_left_join`
- `table_inner_join`

### Phase 4: Add tests to `tests/test_amt.py`

```python
def test_table_left_join_basic():
    left = {"orientation": "row", "columns": ["id", "name"],
            "rows": [[1, "a"], [2, "b"], [3, "c"]]}
    right = {"orientation": "row", "columns": ["id", "value"],
             "rows": [[1, 100], [2, 200]]}
    result = table_left_join(left, right, "id")
    assert result["columns"] == ["id", "name", "value"]
    assert result["rows"] == [[1, "a", 100], [2, "b", 200], [3, "c", None]]

def test_table_left_join_duplicate_keys():
    left = {"orientation": "row", "columns": ["id", "x"],
            "rows": [[1, "a"]]}
    right = {"orientation": "row", "columns": ["id", "y"],
             "rows": [[1, "p"], [1, "q"]]}
    result = table_left_join(left, right, "id")
    assert len(result["rows"]) == 2  # 1 left row * 2 right matches
    assert result["rows"] == [[1, "a", "p"], [1, "a", "q"]]

def test_table_left_join_column_conflict():
    left = {"orientation": "row", "columns": ["id", "value"],
            "rows": [[1, "left"]]}
    right = {"orientation": "row", "columns": ["id", "value"],
             "rows": [[1, "right"]]}
    result = table_left_join(left, right, "id", suffixes=("_l", "_r"))
    assert result["columns"] == ["id", "value_l", "value_r"]
    assert result["rows"] == [[1, "left", "right"]]

def test_table_left_join_different_key_names():
    left = {"orientation": "row", "columns": ["asset", "price"],
            "rows": [["AAPL", 100]]}
    right = {"orientation": "row", "columns": ["ticker", "volume"],
             "rows": [["AAPL", 1000]]}
    result = table_left_join(left, right, "asset", "ticker")
    assert result["columns"] == ["asset", "price", "volume"]
    assert result["rows"] == [["AAPL", 100, 1000]]

def test_table_inner_join():
    left = {"orientation": "row", "columns": ["id", "name"],
            "rows": [[1, "a"], [2, "b"], [3, "c"]]}
    right = {"orientation": "row", "columns": ["id", "value"],
             "rows": [[1, 100], [3, 300]]}
    result = table_inner_join(left, right, "id")
    assert len(result["rows"]) == 2  # only matching rows
    assert result["rows"] == [[1, "a", 100], [3, "c", 300]]
```

---

## Files to Modify

1. **src/specparser/amt/table.py** - Add `table_left_join()` and `table_inner_join()`
2. **src/specparser/amt/__init__.py** - Export new functions
3. **tests/test_amt.py** - Add tests

---

## Verification

1. Run all tests: `uv run python -m pytest tests/test_amt.py -v`
2. Interactive test:
   ```python
   from specparser.amt import table_left_join, print_table

   assets = {"orientation": "row", "columns": ["ticker", "name"],
             "rows": [["AAPL", "Apple"], ["GOOGL", "Google"], ["MSFT", "Microsoft"]]}
   prices = {"orientation": "row", "columns": ["ticker", "price"],
             "rows": [["AAPL", 150.0], ["GOOGL", 140.0]]}

   result = table_left_join(assets, prices, "ticker")
   print_table(result)
   # ticker  name       price
   # AAPL    Apple      150.0
   # GOOGL   Google     140.0
   # MSFT    Microsoft  None
   ```

---

## Edge Cases Handled

- **Empty tables**: Returns empty result with correct columns
- **No matches**: All right columns are None
- **Duplicate keys in right**: Produces multiple output rows per left row
- **Column name conflicts**: Resolved with suffixes
- **Different key column names**: Supported via `left_on` and `right_on`
- **Column-oriented input**: Converted internally, output is row-oriented

## Future Extensions (Not in This Plan)

- `table_full_join()` - all rows from both tables
- `table_cross_join()` - cartesian product
- Multi-column keys (join on multiple columns)
