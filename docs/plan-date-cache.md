# Plan: Cache Days-in-Month Computation

## Problem

In `get_straddle_days()`, the date generation loop runs for every straddle:

```python
# Current code (tickers.py:1336-1348)
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

This code is called:
- **44,460 times** for a 5-year backtest (all assets)
- **177,840 times** for a 20-year backtest

Each call:
1. Calls `calendar.monthrange()` for each month in range
2. Creates `date` objects for every day
3. Appends to a list one-by-one

The same (year, month) combinations are computed over and over:
- 2024-01 days are computed for every straddle that spans January 2024
- With 741 assets × ~2 month overlap = ~1,500 redundant computations per month

## Proposed Solution

### 1. Create `get_days_ym(year, month)` function

```python
def get_days_ym(year: int, month: int) -> list[date]:
    """Return list of all days in a given year/month."""
    from datetime import date
    _, num_days = calendar.monthrange(year, month)
    return [date(year, month, day) for day in range(1, num_days + 1)]
```

### 2. Add a cache for month days

```python
_DAYS_YM_CACHE: dict[tuple[int, int], list[date]] = {}

def get_days_ym(year: int, month: int) -> list[date]:
    """Return cached list of all days in a given year/month."""
    key = (year, month)
    if key in _DAYS_YM_CACHE:
        return _DAYS_YM_CACHE[key]

    from datetime import date
    _, num_days = calendar.monthrange(year, month)
    days = [date(year, month, day) for day in range(1, num_days + 1)]
    _DAYS_YM_CACHE[key] = days
    return days
```

### 3. Refactor date generation in `get_straddle_days()`

Replace the current loop with:

```python
# Generate all dates from entry month to expiry month (inclusive)
dates = []
current_year, current_month = entry_year, entry_month
while (current_year, current_month) <= (expiry_year, expiry_month):
    dates.extend(get_days_ym(current_year, current_month))
    # Advance to next month
    if current_month == 12:
        current_year += 1
        current_month = 1
    else:
        current_month += 1
```

## Cache Characteristics

- **Size**: ~240 entries for a 20-year backtest (12 months × 20 years)
- **Memory**: ~240 × 31 × ~48 bytes/date ≈ 360KB (negligible)
- **Hit rate**: Very high - same months accessed by many straddles
- **No eviction needed**: Cache is tiny, can keep forever

## Expected Impact

### Before
- `calendar.monthrange()` called: 44,460 straddles × ~2 months = ~89,000 times
- `date()` constructor called: ~89,000 × ~30 days = ~2.7M times
- List appends: ~2.7M times

### After
- `calendar.monthrange()` called: ~240 times (once per unique month)
- `date()` constructor called: ~7,200 times (240 months × 30 days)
- Cache lookups: ~89,000 times (O(1) dict lookup)
- List extends: ~89,000 times (faster than append loop)

### Estimated Speedup

The date generation is a small fraction of overall time, but:
- Removes ~2.7M object constructions
- Removes ~2.7M list appends
- Replaces with ~89,000 dict lookups + list extends

Rough estimate: **5-10% speedup** for the date generation portion.

Given that "Process straddles" is 66% of total time, and date generation is maybe 5% of that, overall impact might be **1-3%** total time reduction.

## Implementation

### Files to modify:
- `src/specparser/amt/tickers.py`

### Changes:
1. Add `_DAYS_YM_CACHE` dict near other caches
2. Add `get_days_ym(year, month)` function
3. Add `clear_days_cache()` function (for consistency)
4. Refactor `get_straddle_days()` to use `get_days_ym()`

### Testing:
```bash
# Verify output unchanged
uv run python scripts/backtest_fast.py '^LA Comdty' 2024 2024 > before.tsv
# ... make changes ...
uv run python scripts/backtest_fast.py '^LA Comdty' 2024 2024 > after.tsv
diff before.tsv after.tsv

# Timing comparison
uv run python scripts/backtest_fast.py '.' 2020 2024 --timing > /dev/null
```

## Risks

- **Low risk**: Simple cache, no complex invalidation needed
- **Memory**: Negligible (360KB max)
- **Thread safety**: Each worker process has its own cache (same as existing caches)

## Decision Points

1. Should we pre-populate the cache at worker init time? (Probably not worth the complexity)
2. Should this cache respect `_MEMOIZE_ENABLED`? (Probably yes, for consistency)
