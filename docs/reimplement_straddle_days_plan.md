# Plan: Reimplement `straddle_days` as `get_straddle_days`

## Summary of Changes

1. **Rename**: `straddle_days` → `get_straddle_days` (reinforces it's a getter, not a finder)
2. **Drop old format**: No backward compatibility with `['name', 'value']` format
3. **Same signature as `asset_straddle_tickers`**: Calls it internally for consistency
4. **Clean table output**: One column per param (vol, hedge, hedge1, etc.)

## Decisions

- **Q1: Backward compatibility?** No - old format was bad, new format is clean
- **Q2: Parquet schema?** Confirmed: `(ticker, date, field, value)` with date as `date32[day]`
- **Q3: Validate same asset/straddle?** Yes - it's one straddle getting its days fetched

## Function Signature

```python
def get_straddle_days(
    path: str | Path,
    underlying: str,
    year: int,
    month: int,
    prices_parquet: str | Path,
    chain_csv: str | Path | None = None,
    i: int = 0
) -> dict[str, Any]:
```

Same as `asset_straddle_tickers` plus `prices_parquet` parameter. Calls `asset_straddle_tickers` internally.

## Input (from `asset_straddle_tickers`)

```python
{
    "columns": ["asset", "straddle", "param", "source", "ticker", "field"],
    "rows": [
        ["CL Comdty", "|2024-05|2024-06|N|0|OVERRIDE|15|33.3|", "vol", "BBG", "CL1 Comdty", "VOL_FIELD"],
        ["CL Comdty", "|2024-05|2024-06|N|0|OVERRIDE|15|33.3|", "hedge", "BBG", "CLN24 Comdty", "PX_LAST"],
        ["CL Comdty", "|2024-05|2024-06|N|0|OVERRIDE|15|33.3|", "hedge1", "BBG", "CLQ24 Comdty", "PX_LAST"],
    ]
}
```

## Output Format

```python
{
    "columns": ["asset", "straddle", "date", "vol", "hedge", "hedge1"],
    "rows": [
        ["CL Comdty", "|2024-05|2024-06|...|", "2024-05-01", "25.3", "78.50", "79.20"],
        ["CL Comdty", "|2024-05|2024-06|...|", "2024-05-02", "24.8", "77.90", "78.80"],
        # ... ~60-91 rows depending on ntrc (N=~60 days, F=~91 days)
        ["CL Comdty", "|2024-05|2024-06|...|", "2024-06-30", "none", "none", "none"],
    ]
}
```

## Date Range Logic

From straddle format `|ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|`:
- Date range: **first day of entry month** to **last day of expiry month**
- ntrc="N": ~60 days (2 months)
- ntrc="F": ~91 days (3 months)

Example: `|2024-05|2024-06|N|...|` → 2024-05-01 to 2024-06-30

## Implementation Steps

### 1. Function Signature and Call `asset_straddle_tickers`
```python
def get_straddle_days(
    path: str | Path,
    underlying: str,
    year: int,
    month: int,
    prices_parquet: str | Path,
    chain_csv: str | Path | None = None,
    i: int = 0
) -> dict[str, Any]:
    """Get daily prices for a straddle from entry to expiry month.

    Columns: ['asset', 'straddle', 'date', <param1>, <param2>, ...]
    Where params are 'vol', 'hedge', 'hedge1', etc.

    Args:
        path: Path to AMT YAML file
        underlying: Asset underlying value
        year: Expiry year
        month: Expiry month
        prices_parquet: Path to prices parquet file
        chain_csv: Optional CSV for futures ticker lookup
        i: Straddle selector index (i % len(straddles))

    Returns:
        Table with one row per day, columns for each param's price
    """
    # Get ticker table from asset_straddle_tickers
    table = asset_straddle_tickers(path, underlying, year, month, chain_csv, i)

    if not table["rows"]:
        return {"columns": ["asset", "straddle", "date"], "rows": []}
```

### 2. Extract Data from Input Table
```python
    # All rows have same asset and straddle
    asset = table["rows"][0][0]
    straddle = table["rows"][0][1]

    # Parse straddle to get date range
    entry_year, entry_month = schedules.ntry(straddle), schedules.ntrm(straddle)
    expiry_year, expiry_month = schedules.xpry(straddle), schedules.xprm(straddle)
```

### 3. Build Ticker Map from Input Rows
```python
    # Map param -> (ticker, field) from input rows
    # Input columns: ["asset", "straddle", "param", "source", "ticker", "field"]
    param_idx = table["columns"].index("param")
    ticker_idx = table["columns"].index("ticker")
    field_idx = table["columns"].index("field")

    ticker_map = {}  # param -> (ticker, field)
    params_ordered = []  # preserve order for output columns
    for row in table["rows"]:
        param = row[param_idx]
        if param not in ticker_map:
            params_ordered.append(param)
            ticker_map[param] = (row[ticker_idx], row[field_idx])
```

### 4. Generate Date Range
```python
    import calendar
    from datetime import date

    # Generate all dates from entry month to expiry month (inclusive)
    dates = []
    current_year, current_month = entry_year, entry_month
    while (current_year, current_month) <= (expiry_year, expiry_month):
        _, num_days = calendar.monthrange(current_year, current_month)
        for day in range(1, num_days + 1):
            dates.append(date(current_year, current_month, day))
        # Advance to next month
        if current_month == 12:
            current_year += 1
            current_month = 1
        else:
            current_month += 1
```

### 5. Query Prices from Parquet
```python
    import duckdb

    # Build list of (ticker, field) pairs to query
    ticker_field_pairs = list(ticker_map.values())

    start_date = dates[0].isoformat()
    end_date = dates[-1].isoformat()

    con = duckdb.connect()
    con.execute(f"CREATE VIEW prices AS SELECT * FROM '{prices_parquet}'")

    # Query all prices in one go
    # Parquet schema: ticker, date, field, value
    conditions = " OR ".join(
        f"(ticker = '{t}' AND field = '{f}')" for t, f in ticker_field_pairs
    )
    query = f"""
        SELECT ticker, field, date, value
        FROM prices
        WHERE ({conditions})
        AND date >= '{start_date}'
        AND date <= '{end_date}'
    """
    result = con.execute(query).fetchall()
    con.close()

    # Organize: (ticker, field) -> {date_str -> value}
    prices = {}
    for ticker, field, dt, value in result:
        key = (ticker, field)
        if key not in prices:
            prices[key] = {}
        prices[key][str(dt)] = str(value)
```

### 6. Build Output Table
```python
    # Output columns: asset, straddle, date, <params...>
    out_columns = ["asset", "straddle", "date"] + params_ordered

    out_rows = []
    for dt in dates:
        date_str = dt.isoformat()
        row = [asset, straddle, date_str]
        for param in params_ordered:
            ticker, field = ticker_map[param]
            value = prices.get((ticker, field), {}).get(date_str, "none")
            row.append(value)
        out_rows.append(row)

    return {"columns": out_columns, "rows": out_rows}
```

## Edge Cases to Handle

1. **Empty input table**: Return empty table with base columns `["asset", "straddle", "date"]`
2. **Missing prices**: Use "none" as placeholder
3. **CV source**: Already handled by `_filter_straddle_tickers` (swaps ticker/field)
4. **calc source**: May have empty ticker/field - use "none" for all dates
5. **Year boundary**: Date range may span Dec to Jan (e.g., `|2024-12|2025-01|`)

## Testing Considerations

1. Test with ntrc="N" (~60 days, 2 months)
2. Test with ntrc="F" (~91 days, 3 months)
3. Test with missing prices in parquet
4. Test with multiple hedges (hedge, hedge1, hedge2)
5. Test date range spanning year boundary (Dec 2024 to Jan 2025)
6. Test empty ticker table (no rows from `asset_straddle_tickers`)

## Summary

| Aspect | Old `straddle_days` | New `get_straddle_days` |
|--------|---------------------|-------------------------|
| Function name | `straddle_days` | `get_straddle_days` |
| Input | Table with `['name', 'value']` | Same args as `asset_straddle_tickers` + `prices_parquet` |
| Internal | Parses spec strings | Calls `asset_straddle_tickers` |
| Output format | Mixed rows, tab-separated | Clean table: `['asset', 'straddle', 'date', 'vol', 'hedge', ...]` |
| Date range | Entry month only | Entry month to expiry month (full period) |
| Missing values | "none" | "none" |

## Exports to Add

In `__init__.py`:
- Add `"get_straddle_days"` to `__all__`
- Add lazy import: `"get_straddle_days": (".tickers", "get_straddle_days")`

## Old Code to Keep
- Keep the old `straddle_days` function without exports (make sure to delete them) for reference

## Old Code to Remove
- Remove `_split_ticker_specstr` if no longer used elsewhere
