# Plan: Load All Prices Into Memory

## Problem

Current architecture:
- Each straddle calls `get_straddle_days()` which queries DuckDB
- DuckDB query overhead per straddle: ~10-20ms (query parsing, planning, execution)
- 178,000 straddles × 15ms = ~45 minutes just in query overhead

## Proposed Solution

Load entire prices parquet into a flat dict at backtest start. All price lookups become O(1) dict lookups.

## Data Characteristics

- Prices parquet schema: `ticker, date, field, value`
- Estimated size for 20-year backtest:
  - ~100 unique tickers × 5 fields × 7,300 days = ~3.6M entries
  - ~60 bytes per entry (key + value) = ~220 MB in memory
- This is acceptable for a backtest workstation

## Implementation

### 1. Add `load_all_prices()` function to `tickers.py`

```python
_PRICES_DICT: dict[str, str] | None = None

def load_all_prices(
    prices_parquet: str | Path,
    start_date: str | None = None,
    end_date: str | None = None
) -> dict[str, str]:
    """Load all prices into a flat dict with composite key.

    Args:
        prices_parquet: Path to parquet file
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Dict mapping "ticker|field|date" -> value
    """
    global _PRICES_DICT
    if _PRICES_DICT is not None:
        return _PRICES_DICT

    import duckdb
    con = duckdb.connect()

    # Build query with optional date filters
    query = f"SELECT ticker, field, date, value FROM '{prices_parquet}'"
    conditions = []
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    if end_date:
        conditions.append(f"date <= '{end_date}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    result = con.execute(query).fetchall()
    con.close()

    # Build flat dict with composite string key
    _PRICES_DICT = {}
    for ticker, field, dt, value in result:
        key = f"{ticker}|{field}|{dt}"
        _PRICES_DICT[key] = str(value)

    return _PRICES_DICT


def get_price(prices_dict: dict[str, str], ticker: str, field: str, date_str: str) -> str:
    """Look up a price from the preloaded dict.

    Returns "none" if not found.
    """
    key = f"{ticker}|{field}|{date_str}"
    return prices_dict.get(key, "none")


def clear_prices_dict() -> None:
    """Clear the global prices dict."""
    global _PRICES_DICT
    _PRICES_DICT = None
```

### 2. Modify `get_straddle_days()` to accept preloaded prices

Add optional parameter to use preloaded dict instead of DuckDB:

```python
def get_straddle_days(
    path: str | Path,
    underlying: str,
    year: int,
    month: int,
    prices_parquet: str | Path,
    chain_csv: str | Path | None = None,
    i: int = 0,
    prices_dict: dict[str, str] | None = None  # NEW: preloaded prices
) -> dict[str, Any]:
```

Replace the DuckDB query section (~lines 1127-1149) with:

```python
if prices_dict is not None:
    # Use preloaded dict - O(1) lookups
    prices = {}
    for param, (ticker, field) in normalized_ticker_map.items():
        prices[(ticker, field)] = {}
        for dt in dates:
            date_str = dt.isoformat()
            value = get_price(prices_dict, ticker, field, date_str)
            if value != "none":
                prices[(ticker, field)][date_str] = value
else:
    # Fallback to DuckDB query (existing code)
    con = _get_prices_connection(prices_parquet)
    # ... existing query code ...
```

### 3. Modify `get_straddle_valuation()` similarly

Pass through the `prices_dict` parameter.

### 4. Update `backtest.py`

```python
def init_worker(amt, prices_path, chain, memoize, prices_dict):
    """Initialize worker process."""
    global _worker_amt, _worker_prices, _worker_chain, _worker_memoize, _worker_prices_dict
    _worker_amt = amt
    _worker_prices = prices_path
    _worker_chain = chain
    _worker_memoize = memoize
    _worker_prices_dict = prices_dict  # Shared dict (read-only)

    # Set memoization state
    tickers_module.set_memoize_enabled(memoize)
    schedules_module.set_memoize_enabled(memoize)

    # Warm up other caches
    loader.load_amt(amt)


def process_straddle(args_tuple):
    asset, year, month, i, task_id, total = args_tuple
    try:
        val_table = get_straddle_valuation(
            _worker_amt, asset, year, month,
            _worker_prices, _worker_chain, i,
            prices_dict=_worker_prices_dict  # Pass preloaded dict
        )
        # ... rest unchanged ...


def main():
    # ... argument parsing ...

    # Compute date range for price loading
    # Entry can be 1 month before expiry, so start from (start_year-1, 12)
    start_date = f"{args.start_year - 1}-01-01"
    end_date = f"{args.end_year}-12-31"

    if args.verbose:
        print(f"Loading prices from {start_date} to {end_date}...", file=sys.stderr)

    prices_dict = load_all_prices(args.prices, start_date, end_date)

    if args.verbose:
        print(f"Loaded {len(prices_dict):,} price entries", file=sys.stderr)

    # ... rest of main ...

    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(args.amt, args.prices, args.chain, memoize, prices_dict)
    ) as pool:
```

## Multiprocessing Considerations

**Key insight**: On macOS with `spawn`, each worker gets a copy of `prices_dict` when the Pool is created. This means:

1. Main process loads dict (~220 MB)
2. Each worker inherits a copy via `initargs`
3. Total memory: ~220 MB × (num_workers + 1)

With 16 workers: ~3.5 GB total. This is acceptable.

**Alternative (if memory is tight)**: Use `multiprocessing.shared_memory` to share the dict across workers. More complex but reduces memory to ~220 MB total.

## Expected Performance Improvement

| Component | Before | After |
|-----------|--------|-------|
| DuckDB query per straddle | ~15ms | 0ms |
| Price lookups per straddle | N/A | ~0.1ms (dict lookups) |
| Startup time | ~0s | ~5-10s (load all prices) |
| **Total for 178k straddles** | ~45 min | ~5 min |

## Migration Path

1. Add `load_all_prices()` and `get_price()` functions
2. Add `prices_dict` parameter to `get_straddle_days()` and `get_straddle_valuation()`
3. Keep backward compatibility (existing code still works with `prices_dict=None`)
4. Update `backtest.py` to use preloaded prices
5. `valuation.py` and `straddle_explain.py` can continue using DuckDB for single-straddle queries

## Files to Modify

1. `src/specparser/amt/tickers.py`
   - Add `load_all_prices()`, `get_price()`, `clear_prices_dict()`
   - Modify `get_straddle_days()` to accept `prices_dict`
   - Modify `get_straddle_valuation()` to accept `prices_dict`

2. `scripts/backtest.py`
   - Load prices at startup
   - Pass to worker initializer
   - Pass to `get_straddle_valuation()` calls

## Risks

1. **Memory usage**: ~3.5 GB with 16 workers. Monitor and reduce workers if needed.
2. **Startup time**: ~5-10s to load prices. Acceptable for backtests.
3. **Date range**: Must ensure loaded date range covers all straddles. Using `start_year - 1` for entry month buffer.

## Testing

```bash
# Test single straddle still works (uses DuckDB)
uv run python scripts/valuation.py "LA Comdty" 2024 3 0

# Test backtest with new approach
time uv run python scripts/backtest.py '^LA Comdty' 2020 2025 --verbose

# Compare with old approach (if we add a flag)
time uv run python scripts/backtest.py '^LA Comdty' 2020 2025 --verbose --no-preload
```
