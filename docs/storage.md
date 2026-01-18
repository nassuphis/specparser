# Storage Module Reference

The storage module provides utilities for persisting tables to DuckDB databases and Parquet files, and querying them with SQL.

## Overview

DuckDB + Parquet is an excellent combination for analytical data:
- **DuckDB**: Fast embedded analytical database, no server required
- **Parquet**: Columnar file format with efficient compression

The module provides functions to:
1. Write tables to Parquet files
2. Write tables to DuckDB databases
3. Query Parquet files with SQL
4. Query DuckDB databases with SQL

---

## Dependencies

```toml
# In pyproject.toml
dependencies = [
    "duckdb>=1.0",
    "pyarrow>=15.0",
]
```

---

## Core Functions

### Parquet Operations

#### `table_to_parquet(table, path)`

Write a table dict to a Parquet file.

```python
from specparser.storage import table_to_parquet
from specparser.amt import expand_schedules_fixed

# Generate table
table = expand_schedules_fixed("data/amt.yml", 2024, 2025)

# Write to Parquet
table_to_parquet(table, "schedules.parquet")
```

**Parameters:**
- `table`: Dict with `columns` (list) and `rows` (list of lists)
- `path`: Output Parquet file path

#### `parquet_to_table(path)`

Read a Parquet file into a table dict.

```python
from specparser.storage import parquet_to_table

table = parquet_to_table("schedules.parquet")
print(table['columns'])
# ['xpry', 'xprm', 'ntry', 'ntrm', ...]
print(f"Rows: {len(table['rows'])}")
```

**Returns:** Dict with `columns` (list) and `rows` (list of lists)

#### `query_parquet(path, sql)`

Run a SQL query on a Parquet file.

The table is available as `data` in the SQL query.

```python
from specparser.storage import query_parquet

# Get distinct assets
result = query_parquet("schedules.parquet", "SELECT DISTINCT asset FROM data ORDER BY asset")
print(result['columns'])  # ['asset']
for row in result['rows'][:5]:
    print(row[0])

# Aggregate query
result = query_parquet("schedules.parquet", """
    SELECT asset, COUNT(*) as cnt
    FROM data
    GROUP BY asset
    ORDER BY cnt DESC
    LIMIT 10
""")
```

**Parameters:**
- `path`: Parquet file path
- `sql`: SQL query (use `data` as the table name)

**Returns:** Dict with `columns` (list) and `rows` (list of lists)

---

### DuckDB Operations

#### `table_to_duckdb(table, db_path, table_name)`

Write a table dict to a DuckDB database.

```python
from specparser.storage import table_to_duckdb
from specparser.amt import expand_schedules_fixed

table = expand_schedules_fixed("data/amt.yml", 2024, 2025)
table_to_duckdb(table, "data.duckdb", "schedules")
```

**Parameters:**
- `table`: Dict with `columns` (list) and `rows` (list of lists)
- `db_path`: DuckDB database file path
- `table_name`: Name of the table to create/replace

**Note:** If the table already exists, it will be dropped and recreated.

#### `query_duckdb(db_path, sql)`

Run a SQL query on a DuckDB database.

```python
from specparser.storage import query_duckdb

# Query the schedules table
result = query_duckdb("data.duckdb", "SELECT * FROM schedules WHERE asset = 'AAPL US Equity'")

# Join multiple tables
result = query_duckdb("data.duckdb", """
    SELECT s.asset, s.xpry, s.xprm, p.straddle
    FROM schedules s
    JOIN packed p ON s.asset = p.asset AND s.xpry = p.xpry AND s.xprm = p.xprm
""")
```

**Parameters:**
- `db_path`: DuckDB database file path
- `sql`: SQL query

**Returns:** Dict with `columns` (list) and `rows` (list of lists)

---

## CLI Usage

The storage module provides a CLI for common operations.

### Writing Tables

```bash
# Write expanded schedules to Parquet
uv run python -m specparser.storage --amt data/amt.yml --expand 2024 2025 --to-parquet schedules.parquet

# Write packed straddles to Parquet
uv run python -m specparser.storage --amt data/amt.yml --expand 2024 2025 --pack --to-parquet straddles.parquet

# Write to DuckDB database
uv run python -m specparser.storage --amt data/amt.yml --expand 2024 2025 --to-duckdb data.duckdb schedules

# Write packed straddles to DuckDB
uv run python -m specparser.storage --amt data/amt.yml --expand 2024 2025 --pack --to-duckdb data.duckdb straddles
```

### Querying Files

```bash
# Query Parquet file
uv run python -m specparser.storage --parquet schedules.parquet --query "SELECT DISTINCT asset FROM data ORDER BY asset LIMIT 10"

# Query with aggregation
uv run python -m specparser.storage --parquet schedules.parquet --query "SELECT asset, COUNT(*) as cnt FROM data GROUP BY asset ORDER BY cnt DESC LIMIT 5"

# Query DuckDB database
uv run python -m specparser.storage --db data.duckdb --query "SELECT * FROM schedules WHERE xpry = 2024 AND xprm = 1 LIMIT 10"
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--amt PATH` | Path to AMT YAML file |
| `--expand START END` | Expand schedules for year range |
| `--pack` | Pack into straddle format (use with `--expand`) |
| `--to-parquet PATH` | Write output to Parquet file |
| `--to-duckdb DB TABLE` | Write output to DuckDB table |
| `--parquet PATH` | Parquet file to query |
| `--db PATH` | DuckDB file to query |
| `--query SQL` | SQL query to run |

---

## Examples

### Create and Query a Database

```python
from specparser.amt import expand_schedules_fixed, pack_straddle
from specparser.storage import table_to_duckdb, table_to_parquet, query_duckdb

# Generate data
expanded = expand_schedules_fixed("data/amt.yml", 2024, 2025)
packed = pack_straddle(expanded)

# Store in DuckDB
table_to_duckdb(expanded, "portfolio.duckdb", "schedules")
table_to_duckdb(packed, "portfolio.duckdb", "straddles")

# Also save as Parquet for external tools
table_to_parquet(expanded, "schedules.parquet")
table_to_parquet(packed, "straddles.parquet")

# Query: Monthly schedule counts by asset
result = query_duckdb("portfolio.duckdb", """
    SELECT asset, xpry, xprm, COUNT(*) as entries
    FROM schedules
    WHERE xpry = 2024
    GROUP BY asset, xpry, xprm
    ORDER BY asset, xprm
""")

for row in result['rows'][:10]:
    print(row)
```

### Filter and Export

```python
from specparser.storage import query_parquet, table_to_parquet

# Filter to specific asset and year
result = query_parquet("schedules.parquet", """
    SELECT * FROM data
    WHERE asset = 'AAPL US Equity'
    AND xpry = 2024
""")

# Export filtered data
table_to_parquet(result, "aapl_2024.parquet")
```

### Analyze Straddles

```python
from specparser.storage import query_parquet

# Parse straddle strings with SQL
result = query_parquet("straddles.parquet", """
    SELECT
        asset,
        SPLIT_PART(straddle, '|', 2) as entry_date,
        SPLIT_PART(straddle, '|', 3) as expiry_date,
        SPLIT_PART(straddle, '|', 4) as entry_code,
        SPLIT_PART(straddle, '|', 8) as weight
    FROM data
    LIMIT 10
""")

for row in result['rows']:
    print(row)
```

### Working with External Tools

Parquet files can be read by many tools:

```python
# With pandas
import pandas as pd
df = pd.read_parquet("schedules.parquet")

# With polars
import polars as pl
df = pl.read_parquet("schedules.parquet")

# With DuckDB directly
import duckdb
con = duckdb.connect()
result = con.execute("SELECT * FROM 'schedules.parquet' LIMIT 10").fetchdf()
```

---

## SQL Tips for DuckDB

DuckDB supports a rich SQL dialect with many useful features:

```sql
-- Date/time functions
SELECT xpry, xprm, MAKE_DATE(xpry, xprm, 1) as first_of_month FROM data;

-- String functions
SELECT asset, UPPER(asset) as upper_asset FROM data;

-- Window functions
SELECT asset, xpry, xprm,
       ROW_NUMBER() OVER (PARTITION BY asset ORDER BY xpry, xprm) as seq
FROM data;

-- Aggregation with FILTER
SELECT asset,
       COUNT(*) FILTER (WHERE ntrc = 'N') as near_count,
       COUNT(*) FILTER (WHERE ntrc = 'F') as far_count
FROM data
GROUP BY asset;

-- Create views
CREATE VIEW jan_2024 AS SELECT * FROM schedules WHERE xpry = 2024 AND xprm = 1;
SELECT * FROM jan_2024;

-- Export query results
COPY (SELECT * FROM data WHERE asset LIKE 'A%') TO 'a_assets.csv' (HEADER);
```

---

## Performance Tips

1. **Use Parquet for large datasets**: Parquet files are compressed and columnar, making them efficient for analytical queries.

2. **Create DuckDB tables for repeated queries**: If you're running many queries on the same data, load it into a DuckDB database first.

3. **Use column projections**: Only select the columns you need to reduce I/O.

4. **Partition large datasets**: For very large datasets, consider partitioning by year or asset.

```python
# Write partitioned Parquet (via DuckDB)
import duckdb
con = duckdb.connect()
con.execute("""
    COPY (SELECT * FROM 'schedules.parquet')
    TO 'partitioned' (FORMAT PARQUET, PARTITION_BY (xpry))
""")
```

---

## File Sizes

Typical file sizes for schedule data:

| Format | 1 year | 5 years | 10 years |
|--------|--------|---------|----------|
| Parquet | ~500KB | ~2MB | ~4MB |
| DuckDB | ~1MB | ~4MB | ~8MB |
| CSV | ~5MB | ~25MB | ~50MB |

Parquet achieves ~10x compression over CSV for this type of data.
