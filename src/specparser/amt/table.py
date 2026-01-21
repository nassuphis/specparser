# -------------------------------------
# Table utilities - dict-based table operations
# -------------------------------------
"""
Table manipulation utilities for dict-based tables.

Tables are represented as dicts with 'columns' and 'rows' keys:
    {"columns": ["col1", "col2"], "rows": [[val1, val2], ...]}

This module provides functions for:
- Column operations: select, add, drop, extract
- Row operations: bind, unique
- Value operations: replace
- Multi-table operations: join
"""
from typing import Any


def table_column(table: dict[str, Any], colname: str) -> list[Any]:
    """Extract a single column from a table as a list.

    Args:
        table: Dict with 'columns' and 'rows'
        colname: Name of the column to extract

    Returns:
        List of values from that column

    Raises:
        ValueError: If column name not found
    """
    try:
        idx = table["columns"].index(colname)
    except ValueError:
        raise ValueError(f"Column '{colname}' not found in table columns: {table['columns']}")
    return [row[idx] for row in table["rows"]]


def table_select_columns(table: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """
    Select and reorder columns from a table.

    Args:
        table: Dict with 'columns' and 'rows'
        columns: List of column names in desired order

    Returns:
        New table with only the specified columns in the given order

    Raises:
        ValueError: If a column name is not found in the table
    """
    old_cols = table["columns"]
    indices = []
    for col in columns:
        try:
            indices.append(old_cols.index(col))
        except ValueError:
            raise ValueError(f"Column '{col}' not found in table columns: {old_cols}")

    rows = [[row[i] for i in indices] for row in table["rows"]]
    return {"columns": columns, "rows": rows}


def table_add_column(
    table: dict[str, Any],
    colname: str,
    value: Any = None,
    position: int | None = None
) -> dict[str, Any]:
    """
    Add a new column to a table with a constant value.

    Args:
        table: Dict with 'columns' and 'rows'
        colname: Name of the new column
        value: Value to fill in for all rows (default None)
        position: Index to insert the column (default: append at end)

    Returns:
        New table with the added column
    """
    columns = table["columns"][:]
    if position is None:
        columns.append(colname)
        rows = [row[:] + [value] for row in table["rows"]]
    else:
        columns.insert(position, colname)
        rows = [row[:position] + [value] + row[position:] for row in table["rows"]]
    return {"columns": columns, "rows": rows}


def table_drop_columns(table: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """
    Remove specified columns from a table.

    Args:
        table: Dict with 'columns' and 'rows'
        columns: List of column names to remove

    Returns:
        New table without the specified columns
    """
    drop_set = set(columns)
    keep_indices = [i for i, col in enumerate(table["columns"]) if col not in drop_set]
    new_columns = [table["columns"][i] for i in keep_indices]
    rows = [[row[i] for i in keep_indices] for row in table["rows"]]
    return {"columns": new_columns, "rows": rows}


def table_replace_value(
    table: dict[str, Any],
    colname: str,
    old_value: Any,
    new_value: Any
) -> dict[str, Any]:
    """
    Replace occurrences of a value in a specific column.

    Args:
        table: Dict with 'columns' and 'rows'
        colname: Name of the column to modify
        old_value: Value to replace
        new_value: Replacement value

    Returns:
        New table with values replaced

    Raises:
        ValueError: If column name not found
    """
    try:
        col_idx = table["columns"].index(colname)
    except ValueError:
        raise ValueError(f"Column '{colname}' not found in table columns: {table['columns']}")

    rows = []
    for row in table["rows"]:
        new_row = row[:]
        if new_row[col_idx] == old_value:
            new_row[col_idx] = new_value
        rows.append(new_row)
    return {"columns": table["columns"], "rows": rows}


def table_bind_rows(*tables: dict[str, Any]) -> dict[str, Any]:
    """
    Concatenate rows from multiple tables with the same columns.

    Args:
        *tables: Tables to bind (each has 'columns' and 'rows')

    Returns:
        New table with all rows concatenated

    Raises:
        ValueError: If tables have different columns
    """
    if not tables:
        return {"columns": [], "rows": []}

    first = tables[0]
    columns = first["columns"]

    # Verify all tables have same columns
    for i, tbl in enumerate(tables[1:], start=2):
        if tbl["columns"] != columns:
            raise ValueError(f"Table {i} columns {tbl['columns']} != first table columns {columns}")

    # Concatenate all rows
    rows = []
    for tbl in tables:
        rows.extend(tbl["rows"])

    return {"columns": columns, "rows": rows}


def table_unique_rows(table: dict[str, Any]) -> dict[str, Any]:
    """
    Remove duplicate rows from a table.

    Uses tuple(row) as key to dedupe, preserving the last occurrence of each row.

    Args:
        table: Dict with 'columns' and 'rows'

    Returns:
        New table with only unique rows
    """
    seen = {tuple(row): row for row in table["rows"]}
    return {"columns": table["columns"], "rows": list(seen.values())}


def table_head(table: dict[str, Any], n: int = 10) -> dict[str, Any]:
    """
    Return the first n rows of a table.

    Args:
        table: Dict with 'columns' and 'rows'
        n: Number of rows to return (default 10)

    Returns:
        New table with only the first n rows
    """
    return {"columns": table["columns"], "rows": table["rows"][:n]}


def table_sample(table: dict[str, Any], n: int = 10) -> dict[str, Any]:
    """
    Return a random sample of n rows from a table (without replacement).

    If the table has fewer than n rows, returns all rows.

    Args:
        table: Dict with 'columns' and 'rows'
        n: Number of rows to sample (default 10)

    Returns:
        New table with sampled rows
    """
    import random
    rows = table["rows"]
    if len(rows) <= n:
        return {"columns": table["columns"], "rows": rows[:]}
    sampled = random.sample(rows, n)
    return {"columns": table["columns"], "rows": sampled}


def table_join(*tables: dict[str, Any], key_col: int = 0) -> dict[str, Any]:
    """
    Join multiple tables by combining their columns.

    Takes the key column from the first table, then appends all non-key columns
    from each table. All tables must have the same number of rows.

    Args:
        *tables: Tables to join (each has 'columns' and 'rows')
        key_col: Index of the key column (default 0, typically 'asset')

    Returns:
        Joined table with combined columns and rows
    """
    if not tables:
        return {"columns": [], "rows": []}

    first = tables[0]
    n_rows = len(first["rows"])

    # Start with key column from first table
    columns = [first["columns"][key_col]]

    # Add non-key columns from each table
    for tbl in tables:
        for i, col in enumerate(tbl["columns"]):
            if i != key_col:
                columns.append(col)

    # Build rows: key value + non-key values from each table
    rows = []
    for row_idx in range(n_rows):
        row = [first["rows"][row_idx][key_col]]
        for tbl in tables:
            for i, val in enumerate(tbl["rows"][row_idx]):
                if i != key_col:
                    row.append(val)
        rows.append(row)

    return {"columns": columns, "rows": rows}


def _format_value(v: Any) -> str:
    """Format a value for table output.

    Numbers (float or numeric strings) are formatted to 3 significant figures.
    Other values are converted to strings as-is.
    """
    if isinstance(v, float):
        if v == 0:
            return "0"
        return f"{v:.3g}"
    if isinstance(v, str):
        # Try to parse as float and format if numeric
        try:
            f = float(v)
            if f == 0:
                return "0"
            return f"{f:.3g}"
        except ValueError:
            pass
    return str(v)


def format_table(table: dict[str, Any]) -> str:
    """Format a table dict as a tab-separated string with header."""
    lines = []

    # Header
    lines.append("\t".join(str(c) for c in table["columns"]))

    # Rows
    for row in table["rows"]:
        lines.append("\t".join(_format_value(v) for v in row))

    return "\n".join(lines)


def print_table(table: dict[str, Any]) -> None:
    """Print a table with header and rows to stdout."""
    # Header
    print("\t".join(str(c) for c in table["columns"]))

    # Rows
    for row in table["rows"]:
        print("\t".join(_format_value(v) for v in row))


def _resolve_column_index(table: dict[str, Any], column: str | int) -> int:
    """Convert column name or index to index, with validation.

    Args:
        table: Dict with 'columns' and 'rows'
        column: Column name (str) or index (int)

    Returns:
        Column index

    Raises:
        ValueError: If column not found or index out of range
    """
    if isinstance(column, int):
        if column < 0 or column >= len(table["columns"]):
            raise ValueError(f"Column index {column} out of range")
        return column
    try:
        return table["columns"].index(column)
    except ValueError:
        raise ValueError(f"Column '{column}' not found in table columns: {table['columns']}")


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
