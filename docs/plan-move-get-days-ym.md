# Plan: Move get_days_ym from tickers.py to schedules.py

## Overview

Move the `get_days_ym()` function and its caching infrastructure from `tickers.py` to `schedules.py`, since it's a date function and dates logically belong in schedules.

## Current State

### In tickers.py (lines 180-219):
```python
_DAYS_YM_CACHE: dict[tuple[int, int], list] = {}

def get_days_ym(year: int, month: int) -> list:
    """Return cached list of all days in a given year/month."""
    key = (year, month)
    if _MEMOIZE_ENABLED and key in _DAYS_YM_CACHE:
        return _DAYS_YM_CACHE[key]

    _, num_days = calendar.monthrange(year, month)
    days = [datetime.date(year, month, day) for day in range(1, num_days + 1)]

    if _MEMOIZE_ENABLED:
        _DAYS_YM_CACHE[key] = days
    return days

def clear_days_cache() -> None:
    """Clear the days-in-month cache."""
    _DAYS_YM_CACHE.clear()
```

### tickers.py cache clearing (line 180-184):
```python
def clear_ticker_caches() -> None:
    """Clear all ticker-related caches."""
    _TSCHEMAS_CACHE.clear()
    _TICKERS_YM_CACHE.clear()
    _DAYS_YM_CACHE.clear()  # <-- this line needs to change
```

### schedules.py already has:
- `_MEMOIZE_ENABLED` flag (line 21)
- `set_memoize_enabled()` function (line 23)
- `clear_schedule_caches()` function (line 28)
- `year_month_days(straddle)` function (line 339) - different signature, returns strings

## Changes Required

### 1. Add to schedules.py

Add after `_EXPAND_YM_CACHE` section (around line 332):

```python
# -------------------------------------
# Days-in-month cache
# -------------------------------------

_DAYS_YM_CACHE: dict[tuple[int, int], list] = {}


def get_days_ym(year: int, month: int) -> list:
    """Return cached list of all days in a given year/month.

    Args:
        year: The year (e.g., 2024)
        month: The month (1-12)

    Returns:
        List of date objects for every day in that month
    """
    key = (year, month)
    if _MEMOIZE_ENABLED and key in _DAYS_YM_CACHE:
        return _DAYS_YM_CACHE[key]

    _, num_days = calendar.monthrange(year, month)
    days = [datetime.date(year, month, day) for day in range(1, num_days + 1)]

    if _MEMOIZE_ENABLED:
        _DAYS_YM_CACHE[key] = days
    return days


def clear_days_cache() -> None:
    """Clear the days-in-month cache."""
    _DAYS_YM_CACHE.clear()
```

### 2. Update clear_schedule_caches() in schedules.py

Change from:
```python
def clear_schedule_caches() -> None:
    """Clear all schedule-related caches."""
    _SCHEDULE_CACHE.clear()
    _EXPAND_YM_CACHE.clear()
```

To:
```python
def clear_schedule_caches() -> None:
    """Clear all schedule-related caches."""
    _SCHEDULE_CACHE.clear()
    _EXPAND_YM_CACHE.clear()
    _DAYS_YM_CACHE.clear()
```

### 3. Update tickers.py imports

Add import at top:
```python
from . import schedules
```

(Already exists at line 19)

### 4. Remove from tickers.py

Delete:
- `_DAYS_YM_CACHE` declaration (line 191)
- `get_days_ym()` function (lines 194-214)
- `clear_days_cache()` function (lines 217-219)

### 5. Update tickers.py clear_ticker_caches()

Change from:
```python
def clear_ticker_caches() -> None:
    """Clear all ticker-related caches."""
    _TSCHEMAS_CACHE.clear()
    _TICKERS_YM_CACHE.clear()
    _DAYS_YM_CACHE.clear()
```

To:
```python
def clear_ticker_caches() -> None:
    """Clear all ticker-related caches."""
    _TSCHEMAS_CACHE.clear()
    _TICKERS_YM_CACHE.clear()
```

### 6. Update tickers.py usage

Change line 1376 from:
```python
dates.extend(get_days_ym(current_year, current_month))
```

To:
```python
dates.extend(schedules.get_days_ym(current_year, current_month))
```

### 7. Update __init__.py exports

Add to `__all__` list (schedules section):
```python
"get_days_ym",
"clear_days_cache",
```

Add to `_LAZY_IMPORTS` dict:
```python
"get_days_ym": (".schedules", "get_days_ym"),
"clear_days_cache": (".schedules", "clear_days_cache"),
```

### 8. Update tests

If any tests directly reference `tickers.get_days_ym` or `tickers.clear_days_cache`, update to use `schedules.get_days_ym` or import from `specparser.amt`.

## Files to Modify

1. `src/specparser/amt/schedules.py` - Add function + cache
2. `src/specparser/amt/tickers.py` - Remove function + cache, update usage
3. `src/specparser/amt/__init__.py` - Export new functions

## Testing

After changes, verify:
```bash
uv run python -c "
from specparser.amt import get_days_ym, clear_days_cache
import datetime

# Test basic functionality
days = get_days_ym(2024, 1)
assert len(days) == 31
assert days[0] == datetime.date(2024, 1, 1)
print('get_days_ym works')

# Test cache clearing
clear_days_cache()
print('clear_days_cache works')

# Test via schedules module directly
from specparser.amt import schedules
days2 = schedules.get_days_ym(2024, 2)
assert len(days2) == 29  # leap year
print('schedules.get_days_ym works')
"
```

Also run existing tests to ensure nothing breaks.
