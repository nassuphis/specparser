# -------------------------------------
# Table utilities - dict-based table operations
# -------------------------------------
"""
Table manipulation utilities for dict-based tables.

Tables are represented as dicts with 'columns', 'rows', and 'orientation' keys:
    Row-oriented: {"orientation": "row", "columns": ["col1", "col2"], "rows": [[val1, val2], ...]}
    Column-oriented: {"orientation": "column", "columns": ["col1", "col2"], "rows": [[col1_vals], [col2_vals]]}
    Arrow-oriented: {"orientation": "arrow", "columns": ["col1", "col2"], "rows": [pa.Array, pa.Array, ...]}

Arrow-oriented tables are backed by PyArrow arrays for fast, vectorized operations on large datasets.
The "rows" field contains a list of PyArrow Arrays (one per column), mirroring column-oriented structure.

This module provides functions for:
- Orientation conversion: to_columns, to_rows, to_arrow
- Column operations: select, add, drop, extract
- Row operations: bind, unique, head, sample
- Value operations: replace
- Multi-table operations: stack_cols, left_join, inner_join
"""
from __future__ import annotations
from typing import Any, Literal, TYPE_CHECKING
import itertools

# PyArrow is optional - only required for arrow-oriented tables
# Use guarded import so module works without pyarrow installed
pa = None
pc = None

def _import_pyarrow():
    """Import pyarrow lazily, raising a clear error if not installed."""
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

# For type checking only - doesn't require runtime import
if TYPE_CHECKING:
    import pyarrow as pa


# -------------------------------------
# Orientation helpers
# -------------------------------------

def _is_column_oriented(table: dict[str, Any]) -> bool:
    """Check if table is column-oriented."""
    return table.get("orientation") == "column"


def _is_arrow(table: dict[str, Any]) -> bool:
    """Check if table is arrow-oriented (PyArrow-backed)."""
    return table.get("orientation") == "arrow"


def _transpose_rows_to_cols(rows: list[list], n_cols: int) -> list[list]:
    """
    Transpose row-oriented data to column-oriented without zip(*rows) splat.

    The standard zip(*rows) approach splats all rows as arguments, which
    creates O(n_rows) function arguments - huge overhead for millions of rows.
    This approach pre-allocates column lists and iterates once.

    Args:
        rows: List of row lists
        n_cols: Number of columns

    Returns:
        List of column lists
    """
    if not rows:
        return [[] for _ in range(n_cols)]

    # Pre-allocate column lists
    cols = [[] for _ in range(n_cols)]
    for row in rows:
        for i, val in enumerate(row):
            cols[i].append(val)
    return cols


def _transpose_cols_to_rows(cols: list[list]) -> list[list]:
    """
    Transpose column-oriented data to row-oriented without zip(*cols) splat.

    Args:
        cols: List of column lists

    Returns:
        List of row lists
    """
    if not cols or not cols[0]:
        return []

    n_rows = len(cols[0])
    n_cols = len(cols)
    rows = []
    for i in range(n_rows):
        rows.append([cols[j][i] for j in range(n_cols)])
    return rows


def table_orientation(table: dict[str, Any]) -> Literal["row", "column", "arrow"]:
    """
    Get the orientation of a table.

    Args:
        table: Table dict

    Returns:
        One of "row", "column", or "arrow"
    """
    orientation = table.get("orientation", "row")
    if orientation not in ("row", "column", "arrow"):
        raise ValueError(f"Unsupported orientation: {orientation}")
    return orientation


def table_nrows(table: dict[str, Any]) -> int:
    """
    Get the number of rows in a table (works for all orientations).

    Args:
        table: Table dict

    Returns:
        Number of data rows in the table
    """
    orientation = table_orientation(table)
    if orientation == "row":
        return len(table["rows"])
    else:
        # column or arrow: rows contains columns, length is in first column
        if not table["rows"]:
            return 0
        first_col = table["rows"][0]
        if orientation == "arrow":
            return len(first_col)  # pa.Array supports len()
        return len(first_col)  # Python list


def table_validate(table: dict[str, Any], strict: bool = False) -> None:
    """
    Validate table structure.

    Checks:
    - Required keys: orientation, columns, rows
    - Orientation is valid: row, column, arrow
    - len(rows) == len(columns) for column/arrow orientations
    - All columns have equal length
    - Arrow: each element in rows is pa.Array or pa.ChunkedArray

    Args:
        table: Table dict to validate
        strict: If True, performs additional type checks

    Raises:
        ValueError: If table structure is invalid
    """
    # Check required keys
    for key in ("columns", "rows"):
        if key not in table:
            raise ValueError(f"Table missing required key: '{key}'")

    # Validate orientation
    orientation = table_orientation(table)  # raises if invalid

    columns = table["columns"]
    rows = table["rows"]

    if orientation == "row":
        # Row-oriented: rows is list of row lists
        if strict:
            n_cols = len(columns)
            for i, row in enumerate(rows):
                if len(row) != n_cols:
                    raise ValueError(
                        f"Row {i} has {len(row)} values, expected {n_cols} columns"
                    )
    else:
        # Column or Arrow: rows is list of columns
        if len(rows) != len(columns):
            raise ValueError(
                f"Number of data columns ({len(rows)}) does not match "
                f"column names ({len(columns)})"
            )

        if rows:
            # Check all columns have same length
            first_len = len(rows[0])
            for i, col in enumerate(rows[1:], start=1):
                if len(col) != first_len:
                    raise ValueError(
                        f"Column {i} ({columns[i]}) has {len(col)} values, "
                        f"expected {first_len}"
                    )

            # Arrow-specific checks
            if orientation == "arrow":
                _pa, _ = _import_pyarrow()
                for i, col in enumerate(rows):
                    if not isinstance(col, (_pa.Array, _pa.ChunkedArray)):
                        raise ValueError(
                            f"Column {i} ({columns[i]}) is not a PyArrow Array, "
                            f"got {type(col).__name__}"
                        )


def table_to_columns(table: dict[str, Any]) -> dict[str, Any]:
    """
    Convert any table to column-oriented (Python lists).

    Args:
        table: Table in any orientation (row, column, or arrow)

    Returns:
        Column-oriented table with Python list columns
    """
    if _is_column_oriented(table):
        return table

    if _is_arrow(table):
        # Arrow -> column: convert each pa.Array to Python list
        return {
            "orientation": "column",
            "columns": table["columns"][:],
            "rows": [col.to_pylist() for col in table["rows"]],
        }

    # Row -> column: use optimized transpose to avoid zip(*rows) overhead
    rows = table["rows"]
    n_cols = len(table["columns"])
    cols = _transpose_rows_to_cols(rows, n_cols)
    return {"orientation": "column", "columns": table["columns"][:], "rows": cols}


def table_to_rows(table: dict[str, Any]) -> dict[str, Any]:
    """
    Convert any table to row-oriented (Python lists).

    Args:
        table: Table in any orientation (row, column, or arrow)

    Returns:
        Row-oriented table with Python list rows
    """
    orientation = table_orientation(table)

    if orientation == "row":
        return table

    if orientation == "arrow":
        # Arrow -> row: convert to Python lists, then transpose
        cols = [col.to_pylist() for col in table["rows"]]
        rows = _transpose_cols_to_rows(cols)
        return {"orientation": "row", "columns": table["columns"][:], "rows": rows}

    # Column -> row: use optimized transpose
    cols = table["rows"]
    rows = _transpose_cols_to_rows(cols)
    return {"orientation": "row", "columns": table["columns"][:], "rows": rows}


def table_to_arrow(table: dict[str, Any]) -> dict[str, Any]:
    """
    Convert any table to arrow-oriented (PyArrow arrays).

    Args:
        table: Table in any orientation (row, column, or arrow)

    Returns:
        Arrow-oriented table with pa.Array columns
    """
    if _is_arrow(table):
        return table

    _pa, _ = _import_pyarrow()

    if _is_column_oriented(table):
        # Column -> arrow: direct conversion
        return {
            "orientation": "arrow",
            "columns": table["columns"][:],
            "rows": [_pa.array(col) for col in table["rows"]],
        }

    # Row -> arrow: transpose directly without intermediate dict
    rows = table["rows"]
    n_cols = len(table["columns"])
    cols = _transpose_rows_to_cols(rows, n_cols)

    return {
        "orientation": "arrow",
        "columns": table["columns"][:],
        "rows": [_pa.array(col) for col in cols],
    }


def table_to_jsonable(
    table: dict[str, Any],
    orientation: Literal["row", "column"] = "row",
) -> dict[str, Any]:
    """
    Convert any table to a JSON-serializable dict.

    Handles special types:
    - Arrow arrays -> Python lists
    - datetime/timestamp -> ISO8601 strings
    - Decimal -> float
    - bytes -> base64 string

    Args:
        table: Table in any orientation
        orientation: Target orientation for output ("row" or "column")

    Returns:
        JSON-serializable dict table
    """
    import base64
    from datetime import date, datetime
    from decimal import Decimal

    def convert_value(v: Any) -> Any:
        """Convert a single value to JSON-serializable form."""
        if v is None:
            return None
        if isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, date):
            return v.isoformat()
        if isinstance(v, Decimal):
            return float(v)
        if isinstance(v, bytes):
            return base64.b64encode(v).decode("ascii")
        # For any other type, try str conversion
        return str(v)

    # First convert to target orientation (as Python lists)
    if orientation == "row":
        result = table_to_rows(table)
        # Convert each value in each row
        converted_rows = [
            [convert_value(v) for v in row]
            for row in result["rows"]
        ]
        return {
            "orientation": "row",
            "columns": result["columns"],
            "rows": converted_rows,
        }
    else:
        result = table_to_columns(table)
        # Convert each value in each column
        converted_cols = [
            [convert_value(v) for v in col]
            for col in result["rows"]
        ]
        return {
            "orientation": "column",
            "columns": result["columns"],
            "rows": converted_cols,
        }


# -------------------------------------
# Column operations
# -------------------------------------

def table_column(table: dict[str, Any], colname: str) -> list[Any] | pa.Array:
    """Extract a single column from a table.

    Args:
        table: Dict with 'columns' and 'rows'
        colname: Name of the column to extract

    Returns:
        For arrow tables: pa.Array (zero-copy)
        For row/column tables: Python list of values

    Raises:
        ValueError: If column name not found
    """
    try:
        idx = table["columns"].index(colname)
    except ValueError:
        raise ValueError(f"Column '{colname}' not found in table columns: {table['columns']}")

    if _is_arrow(table):
        return table["rows"][idx]  # O(1) - return Arrow array reference (zero-copy)
    if _is_column_oriented(table):
        return table["rows"][idx][:]  # O(1) - direct index, copy list
    return [row[idx] for row in table["rows"]]  # O(n) - iterate rows


def table_select_columns(table: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """
    Select and reorder columns from a table.

    Args:
        table: Dict with 'columns' and 'rows'
        columns: List of column names in desired order

    Returns:
        New table with only the specified columns in the given order (preserves orientation)

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

    if _is_arrow(table):
        # Arrow fast-path: O(k) where k is number of selected columns
        new_cols = [table["rows"][i] for i in indices]  # reference, not copy
        return {"orientation": "arrow", "columns": list(columns), "rows": new_cols}

    if _is_column_oriented(table):
        new_cols = [table["rows"][i][:] for i in indices]
        return {"orientation": "column", "columns": list(columns), "rows": new_cols}
    rows = [[row[i] for i in indices] for row in table["rows"]]
    return {"orientation": "row", "columns": list(columns), "rows": rows}


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
        New table with the added column (preserves orientation)
    """
    columns = table["columns"][:]

    if _is_arrow(table):
        # Arrow fast-path: create constant array without Python list allocation
        _pa, _ = _import_pyarrow()
        n_rows = table_nrows(table)
        if value is None:
            new_col = _pa.nulls(n_rows)
        else:
            # Use pa.repeat for efficient broadcast - avoids [value] * n_rows
            new_col = _pa.repeat(_pa.scalar(value), n_rows)
        new_cols = list(table["rows"])  # shallow copy
        if position is None:
            columns.append(colname)
            new_cols.append(new_col)
        else:
            columns.insert(position, colname)
            new_cols.insert(position, new_col)
        return {"orientation": "arrow", "columns": columns, "rows": new_cols}

    if _is_column_oriented(table):
        n_rows = len(table["rows"][0]) if table["rows"] and table["rows"][0] else 0
        new_col = [value] * n_rows
        new_cols = [col[:] for col in table["rows"]]
        if position is None:
            columns.append(colname)
            new_cols.append(new_col)
        else:
            columns.insert(position, colname)
            new_cols.insert(position, new_col)
        return {"orientation": "column", "columns": columns, "rows": new_cols}
    # Row-oriented
    if position is None:
        columns.append(colname)
        rows = [row[:] + [value] for row in table["rows"]]
    else:
        columns.insert(position, colname)
        rows = [row[:position] + [value] + row[position:] for row in table["rows"]]
    return {"orientation": "row", "columns": columns, "rows": rows}


def table_drop_columns(table: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """
    Remove specified columns from a table.

    Args:
        table: Dict with 'columns' and 'rows'
        columns: List of column names to remove

    Returns:
        New table without the specified columns (preserves orientation)
    """
    drop_set = set(columns)
    keep_indices = [i for i, col in enumerate(table["columns"]) if col not in drop_set]
    new_columns = [table["columns"][i] for i in keep_indices]

    if _is_arrow(table):
        # Arrow fast-path: O(k) where k is number of kept columns
        new_cols = [table["rows"][i] for i in keep_indices]  # reference, not copy
        return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}

    if _is_column_oriented(table):
        new_cols = [table["rows"][i][:] for i in keep_indices]
        return {"orientation": "column", "columns": new_columns, "rows": new_cols}
    rows = [[row[i] for i in keep_indices] for row in table["rows"]]
    return {"orientation": "row", "columns": new_columns, "rows": rows}


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
        New table with values replaced (preserves orientation)

    Raises:
        ValueError: If column name not found
    """
    try:
        col_idx = table["columns"].index(colname)
    except ValueError:
        raise ValueError(f"Column '{colname}' not found in table columns: {table['columns']}")

    if _is_arrow(table):
        # Arrow fast-path: vectorized replace using pc.if_else
        _pa, _pc = _import_pyarrow()
        col = table["rows"][col_idx]
        # Special case: pc.equal(null, null) returns null, not True
        # Use is_null for null comparison
        if old_value is None:
            mask = _pc.is_null(col)
        else:
            mask = _pc.equal(col, _pa.scalar(old_value, type=col.type))
        new_col = _pc.if_else(mask, _pa.scalar(new_value, type=col.type), col)
        new_cols = list(table["rows"])  # shallow copy
        new_cols[col_idx] = new_col
        return {"orientation": "arrow", "columns": table["columns"][:], "rows": new_cols}

    if _is_column_oriented(table):
        new_cols = [col[:] for col in table["rows"]]
        target_col = new_cols[col_idx]
        for i, val in enumerate(target_col):
            if val == old_value:
                target_col[i] = new_value
        return {"orientation": "column", "columns": table["columns"][:], "rows": new_cols}
    rows = []
    for row in table["rows"]:
        new_row = row[:]
        if new_row[col_idx] == old_value:
            new_row[col_idx] = new_value
        rows.append(new_row)
    return {"orientation": "row", "columns": table["columns"][:], "rows": rows}


def table_bind_rows(*tables: dict[str, Any]) -> dict[str, Any]:
    """
    Concatenate rows from multiple tables with the same columns.

    Args:
        *tables: Tables to bind (each has 'columns' and 'rows')

    Returns:
        New table with all rows concatenated.
        - If ANY table is arrow-oriented, all are converted to arrow and result is arrow.
        - Otherwise matches first table's orientation.

    Raises:
        ValueError: If tables have different columns
    """
    if not tables:
        return {"orientation": "row", "columns": [], "rows": []}

    first = tables[0]
    columns = first["columns"]

    # Verify all tables have same columns
    for i, tbl in enumerate(tables[1:], start=2):
        if tbl["columns"] != columns:
            raise ValueError(f"Table {i} columns {tbl['columns']} != first table columns {columns}")

    # Arrow fast-path: if ANY table is arrow, convert all to arrow
    if any(_is_arrow(t) for t in tables):
        _pa, _ = _import_pyarrow()
        arrow_tables = [table_to_arrow(t) for t in tables]
        # Use pa.concat_tables which is optimized for this operation
        pa_tables = [
            _pa.table({col: arr for col, arr in zip(columns, t["rows"])})
            for t in arrow_tables
        ]
        combined = _pa.concat_tables(pa_tables)
        out_cols = [combined.column(c).combine_chunks() for c in columns]
        return {"orientation": "arrow", "columns": columns[:], "rows": out_cols}

    first_is_column = _is_column_oriented(first)

    # Convert all to same orientation as first, then bind
    if first_is_column:
        # Column-oriented: extend each column list
        n_cols = len(columns)
        out_cols = [[] for _ in range(n_cols)]
        for tbl in tables:
            tbl_cols = tbl if _is_column_oriented(tbl) else table_to_columns(tbl)
            for i in range(n_cols):
                out_cols[i].extend(tbl_cols["rows"][i])
        return {"orientation": "column", "columns": columns[:], "rows": out_cols}
    # Row-oriented: extend rows list
    rows = []
    for tbl in tables:
        tbl_rows = tbl if not _is_column_oriented(tbl) else table_to_rows(tbl)
        rows.extend(tbl_rows["rows"])
    return {"orientation": "row", "columns": columns[:], "rows": rows}


def table_unique_rows(table: dict[str, Any]) -> dict[str, Any]:
    """
    Remove duplicate rows from a table.

    Note: Behavior differs by orientation:
    - Row/column: Uses dict with tuple(row) as key, preserving last occurrence.
      Order follows iteration order of Python dict (insertion order).
    - Arrow: Uses PyArrow's group_by aggregation for deduplication.
      Row order is NOT guaranteed and may differ from input order.

    Args:
        table: Dict with 'columns' and 'rows'

    Returns:
        For arrow input: arrow-oriented table with unique rows (order unspecified)
        For row/column input: row-oriented table with unique rows (last occurrence)
    """
    if _is_arrow(table):
        # Arrow fast-path: use pa.Table.group_by to dedupe
        # Build pa.Table and use distinct
        _pa, _ = _import_pyarrow()
        pa_table = _pa.table({
            col: arr for col, arr in zip(table["columns"], table["rows"])
        })
        # Use group_by with all columns to get unique rows
        # PyArrow doesn't have a direct "distinct" but we can use this workaround
        result = pa_table.group_by(table["columns"]).aggregate([])
        out_columns = result.column_names
        out_arrays = [result.column(c).combine_chunks() for c in out_columns]
        return {"orientation": "arrow", "columns": out_columns, "rows": out_arrays}

    tbl = table if not _is_column_oriented(table) else table_to_rows(table)
    seen = {tuple(row): row for row in tbl["rows"]}
    return {"orientation": "row", "columns": table["columns"][:], "rows": list(seen.values())}


def table_head(table: dict[str, Any], n: int = 10) -> dict[str, Any]:
    """
    Return the first n rows of a table.

    Args:
        table: Dict with 'columns' and 'rows'
        n: Number of rows to return (default 10)

    Returns:
        New table with only the first n rows (preserves orientation)
    """
    if _is_arrow(table):
        # Arrow fast-path: O(k) slicing, no O(n_rows) work
        new_cols = [col.slice(0, n) for col in table["rows"]]
        return {"orientation": "arrow", "columns": table["columns"][:], "rows": new_cols}

    if _is_column_oriented(table):
        new_cols = [col[:n] for col in table["rows"]]
        return {"orientation": "column", "columns": table["columns"][:], "rows": new_cols}
    return {"orientation": "row", "columns": table["columns"][:], "rows": table["rows"][:n]}


def table_sample(table: dict[str, Any], n: int = 10) -> dict[str, Any]:
    """
    Return a random sample of n rows from a table (without replacement).

    If the table has fewer than n rows, returns all rows.

    Args:
        table: Dict with 'columns' and 'rows'
        n: Number of rows to sample (default 10)

    Returns:
        New table with sampled rows (preserves orientation for arrow, row-oriented otherwise)
    """
    import random

    if _is_arrow(table):
        # Arrow fast-path: vectorized random selection
        _pa, _pc = _import_pyarrow()
        nrows = table_nrows(table)
        if nrows <= n:
            return table
        indices = _pa.array(random.sample(range(nrows), n))
        new_cols = [_pc.take(col, indices) for col in table["rows"]]
        return {"orientation": "arrow", "columns": table["columns"][:], "rows": new_cols}

    tbl = table if not _is_column_oriented(table) else table_to_rows(table)
    rows = tbl["rows"]
    if len(rows) <= n:
        return {"orientation": "row", "columns": table["columns"][:], "rows": [row[:] for row in rows]}
    sampled = random.sample(rows, n)
    return {"orientation": "row", "columns": table["columns"][:], "rows": sampled}


def table_stack_cols(*tables: dict[str, Any], key_col: int = 0, copy_data: bool = True) -> dict[str, Any]:
    """
    Stack columns from multiple tables side-by-side.

    Takes the key column from the first table, then appends all non-key columns
    from each table. All tables must have the same number of rows.

    This is NOT a SQL-style join - it assumes tables are already row-aligned.
    For true key-based joins, use table_left_join or table_inner_join.

    Args:
        *tables: Tables to stack (each has 'columns' and 'rows')
        key_col: Index of the key column (default 0, typically 'asset')
        copy_data: If True, copy column data; if False, reference existing lists
                   (ignored for arrow tables - arrow arrays are immutable)

    Returns:
        If ANY table is arrow: arrow-oriented table
        Otherwise: column-oriented table

    Raises:
        ValueError: If tables have different row counts
    """
    if not tables:
        return {"orientation": "column", "columns": [], "rows": []}

    # Arrow fast-path: if ANY table is arrow, convert all to arrow
    if any(_is_arrow(t) for t in tables):
        arrow_tables = [table_to_arrow(t) for t in tables]
        first = arrow_tables[0]

        # Row count validation
        n_rows = table_nrows(first)
        for i, t in enumerate(arrow_tables[1:], start=2):
            t_rows = table_nrows(t)
            if t_rows != n_rows:
                raise ValueError(f"Table {i} has {t_rows} rows; expected {n_rows}")

        # Build output columns list - O(k) operations
        out_columns = [first["columns"][key_col]]
        out_cols = [first["rows"][key_col]]  # key column (immutable, no copy needed)

        # Non-key columns from all tables
        for t in arrow_tables:
            for c_i, (name, col) in enumerate(zip(t["columns"], t["rows"])):
                if c_i == key_col:
                    continue
                out_columns.append(name)
                out_cols.append(col)

        return {"orientation": "arrow", "columns": out_columns, "rows": out_cols}

    # Convert all to column-oriented (O(k) if already columnar)
    col_tables = [t if _is_column_oriented(t) else table_to_columns(t) for t in tables]
    first = col_tables[0]

    # Row count validation
    if not first["rows"]:
        n_rows = 0
    else:
        n_rows = len(first["rows"][0])
    for i, t in enumerate(col_tables[1:], start=2):
        t_rows = len(t["rows"][0]) if t["rows"] else 0
        if t_rows != n_rows:
            raise ValueError(f"Table {i} has {t_rows} rows; expected {n_rows}")

    # Build output columns list - O(k) operations
    out_columns = [first["columns"][key_col]]
    out_cols = []

    # Key column from first table
    key_data = first["rows"][key_col]
    out_cols.append(key_data[:] if copy_data else key_data)

    # Non-key columns from all tables
    for t in col_tables:
        for c_i, (name, col) in enumerate(zip(t["columns"], t["rows"])):
            if c_i == key_col:
                continue
            out_columns.append(name)
            out_cols.append(col[:] if copy_data else col)

    return {"orientation": "column", "columns": out_columns, "rows": out_cols}


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


MAX_FORMAT_ROWS = 100_000


def format_table(table: dict[str, Any]) -> str:
    """Format a table dict as a tab-separated string with header.

    Raises:
        ValueError: If table has more than MAX_FORMAT_ROWS rows
    """
    # Convert non-row orientations to row for formatting
    orientation = table_orientation(table)
    tbl = table if orientation == "row" else table_to_rows(table)
    n_rows = len(tbl["rows"])
    if n_rows > MAX_FORMAT_ROWS:
        raise ValueError(f"Table has {n_rows:,} rows, exceeds limit of {MAX_FORMAT_ROWS:,}")

    lines = []

    # Header
    lines.append("\t".join(str(c) for c in tbl["columns"]))

    # Rows
    for row in tbl["rows"]:
        lines.append("\t".join(_format_value(v) for v in row))

    return "\n".join(lines)


def print_table(table: dict[str, Any]) -> None:
    """Print a table with header and rows to stdout.

    Raises:
        ValueError: If table has more than MAX_FORMAT_ROWS rows
    """
    # Convert non-row orientations to row for printing
    orientation = table_orientation(table)
    tbl = table if orientation == "row" else table_to_rows(table)
    n_rows = len(tbl["rows"])
    if n_rows > MAX_FORMAT_ROWS:
        raise ValueError(f"Table has {n_rows:,} rows, exceeds limit of {MAX_FORMAT_ROWS:,}")

    # Header
    print("\t".join(str(c) for c in tbl["columns"]))

    # Rows
    for row in tbl["rows"]:
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
        For arrow input: arrow-oriented table
        For row/column input: column-oriented table

    Notes:
        - If a cell contains an empty list, that row is omitted from output
        - If a cell is not a list, it's treated as a single-element list
        - Uses extend + itertools.repeat for O(n*m) instead of O(n*k*m)
    """
    # For arrow, convert to column-oriented first (list expansion requires Python)
    if _is_arrow(table):
        col_table = table_to_columns(table)
        result = table_unchop(col_table, column)
        return table_to_arrow(result)

    # Convert to column-oriented for efficient expansion
    tbl = table if _is_column_oriented(table) else table_to_columns(table)
    col_idx = _resolve_column_index(tbl, column)
    n_cols = len(tbl["columns"])

    # Initialize output columns
    out_cols = [[] for _ in range(n_cols)]

    # Get the target column (contains lists)
    target_col = tbl["rows"][col_idx]

    # Process each "row" (iterating by index across columns)
    n_rows = len(target_col) if target_col else 0
    for row_idx in range(n_rows):
        cell = target_col[row_idx]

        # Handle non-list as single element
        if not isinstance(cell, list):
            cell = [cell]
        if len(cell) == 0:
            continue  # skip empty lists

        k = len(cell)
        for col_i in range(n_cols):
            if col_i == col_idx:
                out_cols[col_i].extend(cell)  # extend with list contents
            else:
                val = tbl["rows"][col_i][row_idx]
                out_cols[col_i].extend(itertools.repeat(val, k))  # repeat value k times

    return {"orientation": "column", "columns": tbl["columns"][:], "rows": out_cols}


def table_chop(table: dict[str, Any], column: str | int) -> dict[str, Any]:
    """Collapse rows by grouping on non-target columns and collecting target into lists.

    Args:
        table: Dict with 'columns' and 'rows'
        column: Column name (str) or index (int) to collect into lists

    Returns:
        For arrow input: arrow-oriented table (with list column)
        For row/column input: row-oriented table

    Notes:
        - Groups by all columns EXCEPT the target column
        - Preserves order of first occurrence of each group
        - Values are collected in the order they appear
    """
    # For arrow, convert to row-oriented first (grouping requires Python dict keys)
    if _is_arrow(table):
        row_table = table_to_rows(table)
        result = table_chop(row_table, column)
        # Note: result contains Python lists in the target column, convert to arrow
        # but keep lists as they are (PyArrow supports list types)
        return table_to_arrow(result)

    # Convert to row-oriented for grouping by row identity
    tbl = table if not _is_column_oriented(table) else table_to_rows(table)
    col_idx = _resolve_column_index(tbl, column)

    # Group by all columns except target
    groups = {}  # key tuple -> [list of target values]
    group_first_row = {}  # key tuple -> first row (for non-target values)

    for row in tbl["rows"]:
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

    return {"orientation": "row", "columns": tbl["columns"][:], "rows": rows}


# -------------------------------------
# Key-based joins
# -------------------------------------

def table_left_join(
    left: dict[str, Any],
    right: dict[str, Any],
    left_on: str | int,
    right_on: str | int | None = None,
    suffixes: tuple[str, str] = ("", "_right"),
) -> dict[str, Any]:
    """
    Left join two tables on key columns.

    All rows from `left` are kept. Matching rows from `right` are appended.
    Non-matching rows get None for right columns.

    Note on ordering:
    - Row/column input: Output preserves left table row order. When a left row
      matches multiple right rows, all matches appear in right table order.
    - Arrow input: Uses PyArrow's hash join. Output row order may NOT match
      the left table order and should be considered unspecified.

    Args:
        left: Left table
        right: Right table
        left_on: Key column in left table (name or index)
        right_on: Key column in right table (defaults to left_on if same name)
        suffixes: Suffixes for overlapping column names (left_suffix, right_suffix)

    Returns:
        Arrow-oriented table if either input is arrow, otherwise row-oriented.
        Contains left columns + right columns (excluding right key).

    Raises:
        ValueError: If key column not found
    """
    # Resolve key columns
    left_key_idx = _resolve_column_index(left, left_on)
    if right_on is None:
        right_on = left["columns"][left_key_idx]  # use same column name
    right_key_idx = _resolve_column_index(right, right_on)

    # Arrow fast-path: use pa.Table.join()
    if _is_arrow(left) or _is_arrow(right):
        return _arrow_left_join(left, right, left_key_idx, right_key_idx, suffixes)

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


def _arrow_left_join(
    left: dict[str, Any],
    right: dict[str, Any],
    left_key_idx: int,
    right_key_idx: int,
    suffixes: tuple[str, str],
) -> dict[str, Any]:
    """
    Arrow fast-path for left join using pa.Table.join().

    Internal function called by table_left_join when either input is arrow-oriented.
    """
    _pa, _ = _import_pyarrow()

    # Convert both to arrow
    left_arrow = table_to_arrow(left)
    right_arrow = table_to_arrow(right)

    left_key_name = left_arrow["columns"][left_key_idx]
    right_key_name = right_arrow["columns"][right_key_idx]

    # Build pa.Table objects
    left_pa = _pa.table({
        col: arr for col, arr in zip(left_arrow["columns"], left_arrow["rows"])
    })
    right_pa = _pa.table({
        col: arr for col, arr in zip(right_arrow["columns"], right_arrow["rows"])
    })

    # Drop the right key column from right table before join
    # (pa.Table.join keeps both key columns, we want to exclude right key)
    right_cols_to_keep = [
        c for i, c in enumerate(right_arrow["columns"]) if i != right_key_idx
    ]

    # Perform join
    left_suffix, right_suffix = suffixes
    result_pa = left_pa.join(
        right_pa.select([right_key_name] + right_cols_to_keep),
        keys=left_key_name,
        right_keys=right_key_name,
        join_type="left outer",
        left_suffix=left_suffix,
        right_suffix=right_suffix,
    )

    # Convert back to our dict format
    # Note: pa.Table.join may reorder rows, but that's acceptable for join semantics
    out_columns = result_pa.column_names
    out_arrays = [result_pa.column(c).combine_chunks() for c in out_columns]

    return {"orientation": "arrow", "columns": out_columns, "rows": out_arrays}


def table_inner_join(
    left: dict[str, Any],
    right: dict[str, Any],
    left_on: str | int,
    right_on: str | int | None = None,
    suffixes: tuple[str, str] = ("", "_right"),
) -> dict[str, Any]:
    """
    Inner join two tables on key columns (only matching rows).

    Note on ordering:
    - Row/column input: Output follows left table row order (filtered to matches).
    - Arrow input: Uses PyArrow's hash join. Output row order is unspecified.

    Args:
        left: Left table
        right: Right table
        left_on: Key column in left table (name or index)
        right_on: Key column in right table (defaults to left_on if same name)
        suffixes: Suffixes for overlapping column names (left_suffix, right_suffix)

    Returns:
        Arrow-oriented table if either input is arrow, otherwise row-oriented.
        Contains only rows that have matches in both tables.

    Raises:
        ValueError: If key column not found
    """
    # Resolve key columns
    left_key_idx = _resolve_column_index(left, left_on)
    if right_on is None:
        right_on = left["columns"][left_key_idx]
    right_key_idx = _resolve_column_index(right, right_on)

    # Arrow fast-path
    if _is_arrow(left) or _is_arrow(right):
        return _arrow_inner_join(left, right, left_key_idx, right_key_idx, suffixes)

    # Non-arrow path: use left join then filter
    result = table_left_join(left, right, left_on, right_on, suffixes)

    # Filter out rows with None values from right table
    n_left_cols = len(left["columns"])
    filtered_rows = [
        row for row in result["rows"]
        if row[n_left_cols] is not None  # check first right column
    ]

    return {"orientation": "row", "columns": result["columns"], "rows": filtered_rows}


def _arrow_inner_join(
    left: dict[str, Any],
    right: dict[str, Any],
    left_key_idx: int,
    right_key_idx: int,
    suffixes: tuple[str, str],
) -> dict[str, Any]:
    """
    Arrow fast-path for inner join using pa.Table.join().

    Internal function called by table_inner_join when either input is arrow-oriented.
    """
    _pa, _ = _import_pyarrow()

    # Convert both to arrow
    left_arrow = table_to_arrow(left)
    right_arrow = table_to_arrow(right)

    left_key_name = left_arrow["columns"][left_key_idx]
    right_key_name = right_arrow["columns"][right_key_idx]

    # Build pa.Table objects
    left_pa = _pa.table({
        col: arr for col, arr in zip(left_arrow["columns"], left_arrow["rows"])
    })
    right_pa = _pa.table({
        col: arr for col, arr in zip(right_arrow["columns"], right_arrow["rows"])
    })

    # Drop the right key column from right table before join
    right_cols_to_keep = [
        c for i, c in enumerate(right_arrow["columns"]) if i != right_key_idx
    ]

    # Perform inner join
    left_suffix, right_suffix = suffixes
    result_pa = left_pa.join(
        right_pa.select([right_key_name] + right_cols_to_keep),
        keys=left_key_name,
        right_keys=right_key_name,
        join_type="inner",
        left_suffix=left_suffix,
        right_suffix=right_suffix,
    )

    # Convert back to our dict format
    out_columns = result_pa.column_names
    out_arrays = [result_pa.column(c).combine_chunks() for c in out_columns]

    return {"orientation": "arrow", "columns": out_columns, "rows": out_arrays}


# -------------------------------------
# Arrow compute functions
# -------------------------------------
# These functions expose PyArrow's vectorized compute operations.
# They accept any table orientation, convert to arrow internally if needed,
# and always return arrow-oriented tables.


def _ensure_arrow(table: dict[str, Any]) -> dict[str, Any]:
    """Convert table to arrow orientation if not already."""
    if _is_arrow(table):
        return table
    return table_to_arrow(table)


def _apply_unary_arrow(
    table: dict[str, Any],
    col: str | int,
    pc_func,
    result_col: str | None = None,
) -> dict[str, Any]:
    """
    Apply a unary PyArrow compute function to a column.

    Args:
        table: Input table (any orientation)
        col: Column name or index to operate on
        pc_func: PyArrow compute function (e.g., pc.abs, pc.negate)
        result_col: Name for result column. If None, replaces input column.

    Returns:
        Arrow-oriented table with result column.
    """
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    result_arr = pc_func(arr)

    # Build output
    new_cols = list(t["rows"])  # shallow copy
    new_columns = t["columns"][:]

    if result_col is None:
        # Replace in place
        new_cols[col_idx] = result_arr
    else:
        # Add new column
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def _apply_binary_arrow(
    table: dict[str, Any],
    col: str | int,
    value_or_col: Any,
    pc_func,
    result_col: str | None = None,
) -> dict[str, Any]:
    """
    Apply a binary PyArrow compute function to a column and a value/column.

    Args:
        table: Input table (any orientation)
        col: Column name or index for first operand
        value_or_col: Second operand - either a scalar value or column name/index
        pc_func: PyArrow compute function (e.g., pc.add, pc.multiply)
        result_col: Name for result column. If None, replaces input column.

    Returns:
        Arrow-oriented table with result column.
    """
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr1 = t["rows"][col_idx]

    # Determine second operand
    if isinstance(value_or_col, str) and value_or_col in t["columns"]:
        # It's a column name
        arr2 = t["rows"][t["columns"].index(value_or_col)]
    elif isinstance(value_or_col, int) and 0 <= value_or_col < len(t["columns"]):
        # It's a column index
        arr2 = t["rows"][value_or_col]
    else:
        # It's a scalar value
        _pa, _ = _import_pyarrow()
        arr2 = _pa.scalar(value_or_col)

    result_arr = pc_func(arr1, arr2)

    # Build output
    new_cols = list(t["rows"])  # shallow copy
    new_columns = t["columns"][:]

    if result_col is None:
        # Replace in place
        new_cols[col_idx] = result_arr
    else:
        # Add new column
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


# -------------------------------------
# Arithmetic operations
# -------------------------------------

def table_add_arrow(
    table: dict[str, Any],
    col: str | int,
    value_or_col: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Add value or another column to a column."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col, value_or_col, _pc.add, result_col)


def table_subtract_arrow(
    table: dict[str, Any],
    col: str | int,
    value_or_col: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Subtract value or another column from a column."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col, value_or_col, _pc.subtract, result_col)


def table_multiply_arrow(
    table: dict[str, Any],
    col: str | int,
    value_or_col: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Multiply column by value or another column."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col, value_or_col, _pc.multiply, result_col)


def table_divide_arrow(
    table: dict[str, Any],
    col: str | int,
    value_or_col: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Divide column by value or another column."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col, value_or_col, _pc.divide, result_col)


def table_negate_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Negate column values."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.negate, result_col)


def table_abs_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Absolute value of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.abs, result_col)


def table_sign_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Sign of column (-1, 0, 1)."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.sign, result_col)


def table_power_arrow(
    table: dict[str, Any],
    col: str | int,
    exponent: float | int | str,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Raise column to power."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col, exponent, _pc.power, result_col)


def table_sqrt_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Square root of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.sqrt, result_col)


def table_exp_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Exponential (e^x) of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.exp, result_col)


def table_ln_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Natural log of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.ln, result_col)


def table_log10_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Base-10 log of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.log10, result_col)


def table_log2_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Base-2 log of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.log2, result_col)


def table_round_arrow(
    table: dict[str, Any],
    col: str | int,
    decimals: int = 0,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Round column to decimals."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    result_arr = _pc.round(arr, ndigits=decimals)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_ceil_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Ceiling of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.ceil, result_col)


def table_floor_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Floor of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.floor, result_col)


def table_trunc_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Truncate column toward zero."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.trunc, result_col)


# -------------------------------------
# Trigonometric operations
# -------------------------------------

def table_sin_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Sine of column (radians)."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.sin, result_col)


def table_cos_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Cosine of column (radians)."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.cos, result_col)


def table_tan_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Tangent of column (radians)."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.tan, result_col)


def table_asin_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Arc sine of column (returns radians)."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.asin, result_col)


def table_acos_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Arc cosine of column (returns radians)."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.acos, result_col)


def table_atan_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Arc tangent of column (returns radians)."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.atan, result_col)


def table_atan2_arrow(
    table: dict[str, Any],
    y_col: str | int,
    x_col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Arc tangent of y/x, using signs to determine quadrant (returns radians)."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, y_col, x_col, _pc.atan2, result_col)


# -------------------------------------
# Comparison operations (return boolean column)
# -------------------------------------

def table_equal_arrow(
    table: dict[str, Any],
    col: str | int,
    value_or_col: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Column == value. Returns table with boolean column."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col, value_or_col, _pc.equal, result_col)


def table_not_equal_arrow(
    table: dict[str, Any],
    col: str | int,
    value_or_col: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Column != value. Returns table with boolean column."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col, value_or_col, _pc.not_equal, result_col)


def table_less_arrow(
    table: dict[str, Any],
    col: str | int,
    value_or_col: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Column < value. Returns table with boolean column."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col, value_or_col, _pc.less, result_col)


def table_less_equal_arrow(
    table: dict[str, Any],
    col: str | int,
    value_or_col: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Column <= value. Returns table with boolean column."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col, value_or_col, _pc.less_equal, result_col)


def table_greater_arrow(
    table: dict[str, Any],
    col: str | int,
    value_or_col: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Column > value. Returns table with boolean column."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col, value_or_col, _pc.greater, result_col)


def table_greater_equal_arrow(
    table: dict[str, Any],
    col: str | int,
    value_or_col: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Column >= value. Returns table with boolean column."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col, value_or_col, _pc.greater_equal, result_col)


# -------------------------------------
# Null/Value check operations (return boolean column)
# -------------------------------------

def table_is_null_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """True where column is null."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.is_null, result_col)


def table_is_valid_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """True where column is not null (is valid)."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.is_valid, result_col)


def table_is_nan_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """True where column is NaN."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.is_nan, result_col)


def table_is_finite_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """True where column is finite (not NaN, not Inf)."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.is_finite, result_col)


def table_is_in_arrow(
    table: dict[str, Any],
    col: str | int,
    value_set: list | set,
    result_col: str | None = None,
) -> dict[str, Any]:
    """True where column value is in value_set."""
    _pa, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    # Convert value_set to arrow array if needed
    if not isinstance(value_set, _pa.Array):
        value_set = _pa.array(list(value_set))

    result_arr = _pc.is_in(arr, value_set=value_set)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


# -------------------------------------
# Logical operations (boolean columns)
# -------------------------------------

def table_and_arrow(
    table: dict[str, Any],
    col1: str | int,
    col2: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Logical AND of two boolean columns."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col1, col2, _pc.and_, result_col)


def table_or_arrow(
    table: dict[str, Any],
    col1: str | int,
    col2: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Logical OR of two boolean columns."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col1, col2, _pc.or_, result_col)


def table_xor_arrow(
    table: dict[str, Any],
    col1: str | int,
    col2: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Logical XOR of two boolean columns."""
    _, _pc = _import_pyarrow()
    return _apply_binary_arrow(table, col1, col2, _pc.xor, result_col)


def table_invert_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Logical NOT of boolean column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.invert, result_col)


# -------------------------------------
# String operations
# -------------------------------------

def table_upper_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Uppercase string column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.utf8_upper, result_col)


def table_lower_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Lowercase string column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.utf8_lower, result_col)


def table_capitalize_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Capitalize string column (first char upper, rest lower)."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.utf8_capitalize, result_col)


def table_title_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Title case string column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.utf8_title, result_col)


def table_strip_arrow(
    table: dict[str, Any],
    col: str | int,
    chars: str | None = None,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Strip characters from both ends of string column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    if chars is None:
        result_arr = _pc.utf8_trim_whitespace(arr)
    else:
        result_arr = _pc.utf8_trim(arr, characters=chars)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_lstrip_arrow(
    table: dict[str, Any],
    col: str | int,
    chars: str | None = None,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Strip characters from left side of string column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    if chars is None:
        result_arr = _pc.utf8_ltrim_whitespace(arr)
    else:
        result_arr = _pc.utf8_ltrim(arr, characters=chars)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_rstrip_arrow(
    table: dict[str, Any],
    col: str | int,
    chars: str | None = None,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Strip characters from right side of string column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    if chars is None:
        result_arr = _pc.utf8_rtrim_whitespace(arr)
    else:
        result_arr = _pc.utf8_rtrim(arr, characters=chars)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_length_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """String length of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.utf8_length, result_col)


def table_starts_with_arrow(
    table: dict[str, Any],
    col: str | int,
    pattern: str,
    result_col: str | None = None,
) -> dict[str, Any]:
    """True if string starts with pattern."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    result_arr = _pc.starts_with(arr, pattern=pattern)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_ends_with_arrow(
    table: dict[str, Any],
    col: str | int,
    pattern: str,
    result_col: str | None = None,
) -> dict[str, Any]:
    """True if string ends with pattern."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    result_arr = _pc.ends_with(arr, pattern=pattern)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_contains_arrow(
    table: dict[str, Any],
    col: str | int,
    pattern: str,
    result_col: str | None = None,
) -> dict[str, Any]:
    """True if string contains pattern (substring match)."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    result_arr = _pc.match_substring(arr, pattern=pattern)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_replace_substr_arrow(
    table: dict[str, Any],
    col: str | int,
    pattern: str,
    replacement: str,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Replace pattern with replacement in string column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    result_arr = _pc.replace_substring(arr, pattern=pattern, replacement=replacement)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_split_arrow(
    table: dict[str, Any],
    col: str | int,
    pattern: str,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Split string column into list column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    result_arr = _pc.split_pattern(arr, pattern=pattern)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


# -------------------------------------
# Aggregate functions (reduce column to scalar)
# -------------------------------------

def table_summarize_arrow(
    table: dict[str, Any],
    aggregations: dict[str, str | list[str]],
) -> dict[str, Any]:
    """
    Summarize table columns with aggregate functions.

    Args:
        table: Input table
        aggregations: Dict mapping column names to aggregation(s)
            e.g., {"price": "mean", "qty": ["sum", "count"]}

    Returns:
        Single-row arrow table with aggregated values.
        Column names: "col_agg" (e.g., "price_mean", "qty_sum")

    Supported aggregations:
        sum, mean, min, max, count, count_distinct,
        stddev, variance, first, last, any, all
    """
    _pa, _pc = _import_pyarrow()
    t = _ensure_arrow(table)

    agg_funcs = {
        "sum": _pc.sum,
        "mean": _pc.mean,
        "min": _pc.min,
        "max": _pc.max,
        "count": _pc.count,
        "count_distinct": _pc.count_distinct,
        "stddev": _pc.stddev,
        "variance": _pc.variance,
        "first": lambda arr: arr[0] if len(arr) > 0 else None,
        "last": lambda arr: arr[-1] if len(arr) > 0 else None,
        "any": _pc.any,
        "all": _pc.all,
    }

    out_columns = []
    out_values = []

    for col_name, aggs in aggregations.items():
        col_idx = _resolve_column_index(t, col_name)
        arr = t["rows"][col_idx]

        if isinstance(aggs, str):
            aggs = [aggs]

        for agg in aggs:
            if agg not in agg_funcs:
                raise ValueError(f"Unknown aggregation '{agg}'. "
                                 f"Supported: {list(agg_funcs.keys())}")
            func = agg_funcs[agg]
            result = func(arr)
            # Convert scalar to Python value
            if hasattr(result, "as_py"):
                result = result.as_py()
            out_columns.append(f"{col_name}_{agg}")
            out_values.append(result)

    return {
        "orientation": "arrow",
        "columns": out_columns,
        "rows": [_pa.array([v]) for v in out_values],
    }


def table_sum_arrow(table: dict[str, Any], col: str | int) -> float | int:
    """Sum of column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    result = _pc.sum(t["rows"][col_idx])
    return result.as_py()


def table_mean_arrow(table: dict[str, Any], col: str | int) -> float:
    """Mean of column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    result = _pc.mean(t["rows"][col_idx])
    return result.as_py()


def table_min_arrow(table: dict[str, Any], col: str | int) -> Any:
    """Minimum of column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    result = _pc.min(t["rows"][col_idx])
    return result.as_py()


def table_max_arrow(table: dict[str, Any], col: str | int) -> Any:
    """Maximum of column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    result = _pc.max(t["rows"][col_idx])
    return result.as_py()


def table_count_arrow(table: dict[str, Any], col: str | int) -> int:
    """Count non-null values in column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    result = _pc.count(t["rows"][col_idx])
    return result.as_py()


def table_count_distinct_arrow(table: dict[str, Any], col: str | int) -> int:
    """Count distinct values in column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    result = _pc.count_distinct(t["rows"][col_idx])
    return result.as_py()


def table_stddev_arrow(table: dict[str, Any], col: str | int) -> float:
    """Standard deviation of column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    result = _pc.stddev(t["rows"][col_idx])
    return result.as_py()


def table_variance_arrow(table: dict[str, Any], col: str | int) -> float:
    """Variance of column."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    result = _pc.variance(t["rows"][col_idx])
    return result.as_py()


def table_first_arrow(table: dict[str, Any], col: str | int) -> Any:
    """First non-null value in column."""
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]
    if len(arr) == 0:
        return None
    return arr[0].as_py()


def table_last_arrow(table: dict[str, Any], col: str | int) -> Any:
    """Last non-null value in column."""
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]
    if len(arr) == 0:
        return None
    return arr[-1].as_py()


def table_any_arrow(table: dict[str, Any], col: str | int) -> bool:
    """True if any value is true (boolean column)."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    result = _pc.any(t["rows"][col_idx])
    return result.as_py()


def table_all_arrow(table: dict[str, Any], col: str | int) -> bool:
    """True if all values are true (boolean column)."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    result = _pc.all(t["rows"][col_idx])
    return result.as_py()


# -------------------------------------
# Cumulative functions (running operations)
# -------------------------------------

def table_cumsum_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Cumulative sum of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.cumulative_sum, result_col)


def table_cumprod_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Cumulative product of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.cumulative_prod, result_col)


def table_cummin_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Cumulative minimum of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.cumulative_min, result_col)


def table_cummax_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Cumulative maximum of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.cumulative_max, result_col)


def table_cummean_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Cumulative mean of column."""
    _, _pc = _import_pyarrow()
    return _apply_unary_arrow(table, col, _pc.cumulative_mean, result_col)


def table_diff_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """
    Pairwise differences (like pandas diff).

    First element will be null.
    """
    _pa, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    # Compute diff: arr[i] - arr[i-1]
    # Shift array by 1 and subtract
    shifted = _pa.concat_arrays([_pa.array([None], type=arr.type), arr.slice(0, len(arr) - 1)])
    result_arr = _pc.subtract(arr, shifted)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


# -------------------------------------
# Element-wise selection
# -------------------------------------

def table_if_else_arrow(
    table: dict[str, Any],
    cond_col: str | int,
    true_val: Any,
    false_val: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """
    Choose values based on condition.

    Args:
        table: Input table
        cond_col: Boolean column name
        true_val: Column name or scalar for true values
        false_val: Column name or scalar for false values
        result_col: Result column name (default: replaces cond_col)
    """
    _pa, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    cond_idx = _resolve_column_index(t, cond_col)
    cond_arr = t["rows"][cond_idx]

    # Resolve true/false values
    def resolve_val(val):
        if isinstance(val, str) and val in t["columns"]:
            return t["rows"][t["columns"].index(val)]
        elif isinstance(val, int) and 0 <= val < len(t["columns"]):
            return t["rows"][val]
        else:
            return _pa.scalar(val)

    true_arr = resolve_val(true_val)
    false_arr = resolve_val(false_val)

    result_arr = _pc.if_else(cond_arr, true_arr, false_arr)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[cond_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_coalesce_arrow(
    table: dict[str, Any],
    cols: list[str | int],
    result_col: str | None = None,
) -> dict[str, Any]:
    """First non-null value from list of columns."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)

    # Get all column arrays
    arrays = []
    for col in cols:
        col_idx = _resolve_column_index(t, col)
        arrays.append(t["rows"][col_idx])

    result_arr = _pc.coalesce(*arrays)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        # Replace first column
        first_col_idx = _resolve_column_index(t, cols[0])
        new_cols[first_col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_fill_null_arrow(
    table: dict[str, Any],
    col: str | int,
    value: Any,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Fill null values with value."""
    _pa, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    result_arr = _pc.fill_null(arr, _pa.scalar(value))

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_fill_null_forward_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Forward-fill null values (last observation carried forward)."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    result_arr = _pc.fill_null_forward(arr)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


def table_fill_null_backward_arrow(
    table: dict[str, Any],
    col: str | int,
    result_col: str | None = None,
) -> dict[str, Any]:
    """Backward-fill null values (next observation carried backward)."""
    _, _pc = _import_pyarrow()
    t = _ensure_arrow(table)
    col_idx = _resolve_column_index(t, col)
    arr = t["rows"][col_idx]

    result_arr = _pc.fill_null_backward(arr)

    new_cols = list(t["rows"])
    new_columns = t["columns"][:]
    if result_col is None:
        new_cols[col_idx] = result_arr
    else:
        new_columns.append(result_col)
        new_cols.append(result_arr)

    return {"orientation": "arrow", "columns": new_columns, "rows": new_cols}


# -------------------------------------
# Filter (row selection)
# -------------------------------------

def table_filter_arrow(
    table: dict[str, Any],
    col_or_mask: str | int,
) -> dict[str, Any]:
    """
    Filter table rows by boolean column or mask.

    Args:
        table: Input table
        col_or_mask: Boolean column name or index

    Returns:
        Arrow table with filtered rows.
    """
    _pa, _pc = _import_pyarrow()
    t = _ensure_arrow(table)

    # Get mask array
    if isinstance(col_or_mask, _pa.Array):
        mask = col_or_mask
    else:
        col_idx = _resolve_column_index(t, col_or_mask)
        mask = t["rows"][col_idx]

    # Filter each column
    new_cols = [_pc.filter(col, mask) for col in t["rows"]]

    return {"orientation": "arrow", "columns": t["columns"][:], "rows": new_cols}
