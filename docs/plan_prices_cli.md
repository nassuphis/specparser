# Plan: Add Prices CLI Commands

## Problem

Need CLI commands to investigate the prices parquet file directly:
1. `--prices-last REGEX` - show last date for each ticker/field combination matching a regex
2. `--prices-query SQL` - run arbitrary SQL and output as table

## Parquet Schema

The prices parquet file has schema: `(ticker, date, field, value)`

## Commands

### 1. `--prices-last REGEX`

Shows the last date for every ticker/field combination where ticker matches the regex.

**Usage:**
```bash
uv run python -m specparser.amt.tickers data/amt.yml --prices-last "USD.*"
uv run python -m specparser.amt.tickers data/amt.yml --prices-last "LA.*Comdty"
```

**Output:**
```
ticker              field       last_date
USD5Y5Y Curncy      PX_LAST     2026-01-07
USD_fsw0m_5_5       PX_LAST     2024-09-30
...
```

**SQL:**
```sql
SELECT ticker, field, MAX(date) AS last_date
FROM prices
WHERE regexp_matches(ticker, 'REGEX')
GROUP BY ticker, field
ORDER BY ticker, field
```

### 2. `--prices-query SQL`

Run arbitrary SQL query against the prices parquet file. The table is exposed as `prices`.

**Usage:**
```bash
uv run python -m specparser.amt.tickers data/amt.yml --prices-query "SELECT * FROM prices WHERE ticker = 'USD5Y5Y Curncy' ORDER BY date DESC LIMIT 10"
```

**Output:**
Standard table output using `print_table`.

## Implementation

### 1. Add CLI Arguments

In `_main()` after existing arguments:

```python
p.add_argument("--prices-last", metavar="REGEX",
               help="Show last date for each ticker/field matching regex")

p.add_argument("--prices-query", metavar="SQL",
               help="Run arbitrary SQL query against prices parquet (table: prices)")
```

### 2. Add Helper Functions

```python
def prices_last(prices_parquet: str | Path, pattern: str) -> dict[str, Any]:
    """Get last date for each ticker/field matching regex pattern.

    Args:
        prices_parquet: Path to prices parquet file
        pattern: Regex pattern to match tickers

    Returns:
        Table with columns: [ticker, field, last_date]
    """
    con = duckdb.connect()
    table_name = Path(prices_parquet).stem
    con.execute(f"CREATE VIEW {table_name} AS SELECT * FROM '{prices_parquet}'")

    query = f"""
        SELECT ticker, field, MAX(date) AS last_date
        FROM {table_name}
        WHERE regexp_matches(ticker, '{pattern}')
        GROUP BY ticker, field
        ORDER BY ticker, field
    """
    result = con.execute(query).fetchall()
    con.close()

    rows = [[str(ticker), str(field), str(last_date)] for ticker, field, last_date in result]
    return {"columns": ["ticker", "field", "last_date"], "rows": rows}


def prices_query(prices_parquet: str | Path, sql: str) -> dict[str, Any]:
    """Run arbitrary SQL query against prices parquet.

    The parquet file is exposed as table 'prices'.

    Args:
        prices_parquet: Path to prices parquet file
        sql: SQL query to execute

    Returns:
        Table with query results
    """
    con = duckdb.connect()
    con.execute(f"CREATE VIEW prices AS SELECT * FROM '{prices_parquet}'")

    result = con.execute(sql)
    columns = [desc[0] for desc in result.description]
    rows = [[str(v) for v in row] for row in result.fetchall()]
    con.close()

    return {"columns": columns, "rows": rows}
```

### 3. Add CLI Handlers

In `_main()` command handling section:

```python
elif args.prices_last:
    table = prices_last(args.prices, args.prices_last)
    print_table(table)

elif args.prices_query:
    table = prices_query(args.prices, args.prices_query)
    print_table(table)
```

## Files to Modify

1. **`src/specparser/amt/tickers.py`**:
   - Add `prices_last(prices_parquet, pattern)` function
   - Add `prices_query(prices_parquet, sql)` function
   - Add `--prices-last` and `--prices-query` CLI arguments
   - Add CLI handlers for new arguments

## Edge Cases

1. **No matches for regex**: Return empty table with headers
2. **Invalid regex**: DuckDB will raise error, let it propagate
3. **Invalid SQL**: DuckDB will raise error, let it propagate
4. **Empty parquet file**: Return empty table

## Testing

### Manual CLI Tests

```bash
# Test prices-last with broad pattern
uv run python -m specparser.amt.tickers data/amt.yml --prices-last ".*"

# Test prices-last with specific pattern
uv run python -m specparser.amt.tickers data/amt.yml --prices-last "USD.*"

# Test prices-query basic
uv run python -m specparser.amt.tickers data/amt.yml --prices-query "SELECT COUNT(*) as cnt FROM prices"

# Test prices-query with filter
uv run python -m specparser.amt.tickers data/amt.yml --prices-query "SELECT * FROM prices WHERE ticker = 'USD5Y5Y Curncy' LIMIT 5"

# Test prices-query with aggregation
uv run python -m specparser.amt.tickers data/amt.yml --prices-query "SELECT ticker, MIN(date) as first, MAX(date) as last FROM prices GROUP BY ticker ORDER BY ticker LIMIT 10"
```

## Notes

- The `--prices` argument already exists and defaults to `data/prices.parquet`
- Both commands use the existing `--prices` path
- Output uses `print_table` for consistent formatting with 3 significant figures
- DuckDB's `regexp_matches` is used for regex filtering (not SQL LIKE)
