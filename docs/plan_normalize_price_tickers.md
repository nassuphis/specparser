# Plan: Normalize Tickers for Price Lookup

## Problem

When querying prices from the parquet file in `get_straddle_days`, hedge tickers return "none" because:

1. `asset_straddle_tickers` calls `get_tickers_ym` which calls `_tschma_dict_bbgfc_ym`
2. `_tschma_dict_bbgfc_ym` converts normalized tickers to actual BBG tickers via `fut_norm2act`:
   - Normalized: `LAF2025 Comdty`
   - Actual: `LA F25 Comdty`
3. The prices parquet file stores data with **normalized** ticker names
4. `get_straddle_days` queries with **actual** ticker names → no match → "none"

Example output showing the problem:
```
asset     straddle                             date       vol   hedge
LA Comdty |2025-01|2025-02|N|0|OVERRIDE||33.3| 2025-01-01 16.07 none
```
Vol works (not a futures ticker), hedge fails (is a futures ticker that was converted).

## Solution

Create `fut_act2norm(csv_path, ticker)` - the inverse of `fut_norm2act`:
- Input: actual BBG ticker (e.g., "LA F25 Comdty")
- Output: normalized ticker (e.g., "LAF2025 Comdty") or None if not found

Then in `get_straddle_days`, before querying prices, normalize all tickers using this function.

## Implementation

### 1. Add Reverse Cache

The existing cache structure:
```python
_NORMALIZED_CACHE: dict[str, dict[str, str]] = {}
# csv_path -> {normalized -> actual}
```

Add a reverse cache:
```python
_ACTUAL_CACHE: dict[str, dict[str, str]] = {}
# csv_path -> {actual -> normalized}
```

### 2. Create `fut_act2norm` Function

```python
def fut_act2norm(csv_path: str | Path, ticker: str) -> str | None:
    """
    Convert an actual BBG futures ticker to the normalized ticker.

    This is the inverse of fut_norm2act. Uses the same CSV lookup table
    to map actual BBG tickers (e.g., "LA F25 Comdty") back to normalized
    tickers (e.g., "LAF2025 Comdty").

    The CSV is loaded once and cached for subsequent lookups.

    Args:
        csv_path: Path to the CSV file with normalized_future,actual_future columns
        ticker: The actual BBG futures ticker to look up

    Returns:
        The normalized ticker if found, or None if not found
    """
    csv_path = str(Path(csv_path).resolve())

    # Load and cache the reverse mapping if not already cached
    if csv_path not in _ACTUAL_CACHE:
        mapping = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = row.get("normalized_future", "")
                actual = row.get("actual_future", "")
                if normalized and actual:
                    mapping[actual] = normalized
        _ACTUAL_CACHE[csv_path] = mapping

    return _ACTUAL_CACHE[csv_path].get(ticker)
```

### 3. Update `clear_normalized_cache`

```python
def clear_normalized_cache() -> None:
    """Clear the normalized to actual futures cache."""
    _NORMALIZED_CACHE.clear()
    _ACTUAL_CACHE.clear()
```

### 4. Update `get_straddle_days`

In `get_straddle_days`, after building `ticker_map` and before querying prices:

```python
    # Normalize tickers for price lookup (prices DB uses normalized tickers)
    # Only applies when chain_csv is provided
    normalized_ticker_map = {}  # param -> (normalized_ticker, field)
    if chain_csv is not None:
        for param, (ticker, field) in ticker_map.items():
            normalized = fut_act2norm(chain_csv, ticker)
            if normalized is not None:
                normalized_ticker_map[param] = (normalized, field)
            else:
                # Ticker wasn't converted (not a futures ticker), use as-is
                normalized_ticker_map[param] = (ticker, field)
    else:
        normalized_ticker_map = ticker_map

    # Build list of (ticker, field) pairs to query using normalized tickers
    ticker_field_pairs = list(normalized_ticker_map.values())
```

Then when building output rows, use `normalized_ticker_map` for price lookups:

```python
    for dt in dates:
        date_str = dt.isoformat()
        row = [asset, straddle, date_str]
        for param in params_ordered:
            ticker, field = normalized_ticker_map[param]  # Use normalized for lookup
            value = prices.get((ticker, field), {}).get(date_str, "none")
            row.append(value)
        out_rows.append(row)
```

## Why This Works

1. The CSV contains 1-to-1 mapping between normalized and actual tickers
2. Loading the CSV once and caching both directions is efficient
3. Non-futures tickers (vol, etc.) won't be in the CSV, so `fut_act2norm` returns None and we use the original ticker
4. This is the minimal change - only affects price lookups, not the ticker table output

## Files to Modify

1. **`src/specparser/amt/tickers.py`**:
   - Add `_ACTUAL_CACHE` variable
   - Add `fut_act2norm` function
   - Update `clear_normalized_cache` to clear both caches
   - Update `get_straddle_days` to normalize tickers before price lookup

2. **`src/specparser/amt/__init__.py`**:
   - Add `fut_act2norm` to `__all__`
   - Add lazy import for `fut_act2norm`

## Testing Considerations

1. Test `fut_act2norm` returns correct normalized ticker
2. Test `fut_act2norm` returns None for unknown ticker
3. Test `clear_normalized_cache` clears both caches
4. Test `get_straddle_days` with futures hedge (should now return values)
5. Test `get_straddle_days` with non-futures ticker (should still work)

## Expected Result After Fix

```
asset     straddle                             date       vol   hedge
LA Comdty |2025-01|2025-02|N|0|OVERRIDE||33.3| 2025-01-01 16.07 2581.5
LA Comdty |2025-01|2025-02|N|0|OVERRIDE||33.3| 2025-01-02 16.15 2590.0
LA Comdty |2025-01|2025-02|N|0|OVERRIDE||33.3| 2025-01-03 16.51 2614.0
...
```
