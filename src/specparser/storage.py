# -------------------------------------
# DuckDB/Parquet storage utilities
# -------------------------------------
"""
Module for storing and querying tables using DuckDB and Parquet files.

Tables are stored as Parquet files and can be queried using SQL via DuckDB.
"""
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def table_to_parquet(table: dict[str, Any], path: str | Path) -> None:
    """
    Write a table dict to a Parquet file.

    Args:
        table: Dict with 'columns' (list) and 'rows' (list of lists)
        path: Path to the output Parquet file

    Example:
        >>> from specparser.amt import expand_live_schedules_fixed
        >>> table = expand_live_schedules_fixed("data/amt.yml", 2024, 2025)
        >>> table_to_parquet(table, "schedules.parquet")
    """
    path = Path(path)
    columns = table["columns"]
    rows = table["rows"]

    # Build column dict for pyarrow
    data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}

    # Create Arrow table and write to parquet
    arrow_table = pa.table(data)
    pq.write_table(arrow_table, str(path))


def parquet_to_table(path: str | Path) -> dict[str, Any]:
    """
    Read a Parquet file into a table dict.

    Args:
        path: Path to the Parquet file

    Returns:
        Dict with 'columns' (list) and 'rows' (list of lists)

    Example:
        >>> table = parquet_to_table("schedules.parquet")
        >>> table['columns']
        ['xpry', 'xprm', 'ntry', 'ntrm', ...]
    """
    path = Path(path)

    con = duckdb.connect()
    result = con.execute(f"SELECT * FROM '{path}'").fetchall()
    columns = [desc[0] for desc in con.execute(f"SELECT * FROM '{path}' LIMIT 0").description]
    con.close()

    return {
        "columns": columns,
        "rows": [list(row) for row in result],
    }


def query_parquet(path: str | Path, sql: str) -> dict[str, Any]:
    """
    Run a SQL query on a Parquet file.

    The table is named after the file stem (e.g., "prices.parquet" -> "prices").

    Args:
        path: Path to the Parquet file
        sql: SQL query (use file stem as table name, e.g., "SELECT * FROM prices")

    Returns:
        Dict with 'columns' (list) and 'rows' (list of lists)

    Example:
        >>> query_parquet("schedules.parquet", "SELECT DISTINCT asset FROM schedules")
        >>> query_parquet("data/prices.parquet", "DESCRIBE prices;")
    """
    path = Path(path)
    table_name = path.stem

    con = duckdb.connect()
    # Create a view named after the file stem
    con.execute(f"CREATE VIEW {table_name} AS SELECT * FROM '{path}'")
    result = con.execute(sql).fetchall()
    columns = [desc[0] for desc in con.description]
    con.close()

    return {
        "columns": columns,
        "rows": [list(row) for row in result],
    }


def table_to_duckdb(table: dict[str, Any], db_path: str | Path, table_name: str) -> None:
    """
    Write a table dict to a DuckDB database.

    Args:
        table: Dict with 'columns' (list) and 'rows' (list of lists)
        db_path: Path to the DuckDB database file
        table_name: Name of the table to create/replace

    Example:
        >>> table_to_duckdb(table, "data.duckdb", "schedules")
    """
    db_path = Path(db_path)
    columns = table["columns"]
    rows = table["rows"]

    data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}

    # Create Arrow table
    arrow_table = pa.table(data)

    con = duckdb.connect(str(db_path))
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    # Register arrow table and create table from it
    con.register("_arrow_data", arrow_table)
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _arrow_data")
    con.close()


def query_duckdb(db_path: str | Path, sql: str) -> dict[str, Any]:
    """
    Run a SQL query on a DuckDB database.

    Args:
        db_path: Path to the DuckDB database file
        sql: SQL query

    Returns:
        Dict with 'columns' (list) and 'rows' (list of lists)

    Example:
        >>> query_duckdb("data.duckdb", "SELECT DISTINCT asset FROM schedules")
    """
    db_path = Path(db_path)

    con = duckdb.connect(str(db_path))
    result = con.execute(sql).fetchall()
    columns = [desc[0] for desc in con.description]
    con.close()

    return {
        "columns": columns,
        "rows": [list(row) for row in result],
    }


def parquet_dir_to_duckdb(parquet_dir: str | Path, db_path: str | Path) -> list[str]:
    """
    Load all Parquet files from a directory into a DuckDB database.

    Each Parquet file becomes a table named after its file stem
    (e.g., "prices.parquet" -> table "prices").

    Args:
        parquet_dir: Directory containing Parquet files
        db_path: Path to the DuckDB database file

    Returns:
        List of table names that were created

    Example:
        >>> tables = parquet_dir_to_duckdb("data/parquet/", "data.duckdb")
        >>> tables
        ['prices', 'schedules', 'straddles']
    """
    parquet_dir = Path(parquet_dir)
    db_path = Path(db_path)

    con = duckdb.connect(str(db_path))
    tables = []

    for parquet_file in sorted(parquet_dir.glob("*.parquet")):
        table_name = parquet_file.stem
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM '{parquet_file}'")
        tables.append(table_name)

    con.close()
    return tables


# ============================================================
# CLI
# ============================================================

def _main() -> int:
    import argparse
    from . import amt

    p = argparse.ArgumentParser(
        description="DuckDB/Parquet storage utilities.",
    )
    p.add_argument("--amt", metavar="PATH", help="Path to AMT YAML file")
    p.add_argument("--expand", nargs=2, type=int, metavar=("START_YEAR", "END_YEAR"), help="Expand schedules into straddle strings")
    p.add_argument("--to-parquet", "-o", metavar="PATH", help="Write to Parquet file")
    p.add_argument("--to-duckdb", nargs=2, metavar=("DB_PATH", "TABLE"), help="Write to DuckDB table")
    p.add_argument("--query", "-q", metavar="SQL", help="Run SQL query on Parquet file")
    p.add_argument("--parquet", "-p", metavar="PATH", help="Parquet file to query")
    p.add_argument("--db", "-d", metavar="PATH", help="DuckDB file to query")
    p.add_argument("--load-dir", nargs=2, metavar=("PARQUET_DIR", "DB_PATH"), help="Load all Parquet files from directory into DuckDB")
    args = p.parse_args()

    # Generate table from AMT
    if args.amt and args.expand:
        start_year, end_year = args.expand
        table = amt.expand(args.amt, start_year, end_year)

        if args.to_parquet:
            table_to_parquet(table, args.to_parquet)
            print(f"Wrote {len(table['rows'])} rows to {args.to_parquet}")
        elif args.to_duckdb:
            db_path, table_name = args.to_duckdb
            table_to_duckdb(table, db_path, table_name)
            print(f"Wrote {len(table['rows'])} rows to {db_path}:{table_name}")
        else:
            amt.print_table(table)
        return 0

    # Query existing files
    if args.query:
        if args.parquet:
            table = query_parquet(args.parquet, args.query)
        elif args.db:
            table = query_duckdb(args.db, args.query)
        else:
            p.error("--query requires --parquet or --db")
            return 1
        amt.print_table(table)
        return 0

    # Load directory of Parquet files into DuckDB
    if args.load_dir:
        parquet_dir, db_path = args.load_dir
        tables = parquet_dir_to_duckdb(parquet_dir, db_path)
        print(f"Loaded {len(tables)} tables into {db_path}: {', '.join(tables)}")
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
