# Plan: OVERRIDE Expiry Rule

## Problem

Some assets have expiry dates that don't follow a simple rule like "3rd Friday" or "10th business day". For these assets, the expiry dates are stored in a CSV file (`data/overrides.csv`) and need to be looked up by asset and month.

## Overrides File Format

The file `data/overrides.csv` contains:
```csv
ticker,expiry
0R Comdty,2006-02-10
0R Comdty,2006-03-10
...
XB Comdty,2026-02-24
```

- `ticker`: The asset identifier (matches `Underlying` in AMT)
- `expiry`: ISO date (YYYY-MM-DD) of the expiry for that month

Each asset has multiple rows, one per expiry date. To find the expiry for a given month, match the asset and the year-month portion of the date.

## Current Code

In `_anchor_day()` (tickers.py line ~785):
```python
def _anchor_day(xprc: str, xprv: str, year: int, month: int) -> str | None:
    # Handles F, R, W, BD codes
    # Returns anchor date or None
```

In `_compute_actions()` (tickers.py line ~958):
```python
if xprc in ["F", "R", "W", "BD"]:
    # Entry logic
    entry_anchor = _anchor_day(xprc, xprv, ntry, ntrm)
    ...
    # Expiry logic
    expiry_anchor = _anchor_day(xprc, xprv, xpry, xprm)
    ...
```

## Proposed Changes

### 1. Add override file loader with caching

```python
_OVERRIDE_CACHE: dict[str, dict[str, str]] | None = None

def _load_overrides(path: str | Path = "data/overrides.csv") -> dict[str, str]:
    """Load and cache override expiry dates.

    Returns:
        Dict mapping (ticker, "YYYY-MM") -> "YYYY-MM-DD"
    """
    global _OVERRIDE_CACHE
    if _OVERRIDE_CACHE is not None:
        return _OVERRIDE_CACHE

    _OVERRIDE_CACHE = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"]
            expiry = row["expiry"]  # "YYYY-MM-DD"
            year_month = expiry[:7]  # "YYYY-MM"
            key = (ticker, year_month)
            _OVERRIDE_CACHE[key] = expiry

    return _OVERRIDE_CACHE
```

### 2. Add override lookup function

```python
def _override_expiry(underlying: str, year: int, month: int,
                     overrides_path: str | Path = "data/overrides.csv") -> str | None:
    """Look up override expiry date for an asset/month.

    Args:
        underlying: Asset identifier (e.g., "0R Comdty")
        year: Expiry year
        month: Expiry month (1-12)
        overrides_path: Path to overrides CSV

    Returns:
        Expiry date string "YYYY-MM-DD" or None if not found
    """
    overrides = _load_overrides(overrides_path)
    key = (underlying, f"{year}-{month:02d}")
    return overrides.get(key)
```

### 3. Extend `_anchor_day()` to support OVERRIDE

```python
def _anchor_day(xprc: str, xprv: str, year: int, month: int,
                underlying: str | None = None,
                overrides_path: str | Path | None = None) -> str | None:
    """Calculate the anchor day for a given month.

    Args:
        xprc: Code ("F", "R", "W", "BD", or "OVERRIDE")
        xprv: Value (Nth occurrence, ignored for OVERRIDE)
        year: Year
        month: Month (1-12)
        underlying: Asset name (required for OVERRIDE)
        overrides_path: Path to overrides CSV (for OVERRIDE)

    Returns:
        Date string "YYYY-MM-DD" or None
    """
    if xprc == "OVERRIDE":
        if underlying is None:
            return None
        return _override_expiry(underlying, year, month,
                                overrides_path or "data/overrides.csv")

    # ... existing F/R/W/BD logic ...
```

### 4. Update `_compute_actions()` to pass underlying

The existing call:
```python
entry_anchor = _anchor_day(xprc, xprv, ntry, ntrm)
expiry_anchor = _anchor_day(xprc, xprv, xpry, xprm)
```

Becomes:
```python
entry_anchor = _anchor_day(xprc, xprv, ntry, ntrm, underlying, overrides_path)
expiry_anchor = _anchor_day(xprc, xprv, xpry, xprm, underlying, overrides_path)
```

This requires:
1. Adding `underlying` parameter to `_compute_actions()`
2. Passing it from `get_straddle_days()` which already has access to `asset`
3. Adding optional `overrides_path` parameter (defaults to "data/overrides.csv")

### 5. Update straddle_explain.py

Add support for displaying OVERRIDE anchor in the explanation output:
```python
if xprc == "OVERRIDE":
    print(f"  Entry anchor:  Override lookup for {ntry_year}-{ntry_month:02d} = {entry_anchor or 'NOT FOUND'}")
    print(f"  Expiry anchor: Override lookup for {xpry_year}-{xpry_month:02d} = {expiry_anchor or 'NOT FOUND'}")
```

## Key Behaviors

1. **OVERRIDE code**: When `xprc == "OVERRIDE"`, look up the expiry date from overrides.csv instead of computing it
2. **xprv ignored**: For OVERRIDE, the xprv value is not used (the expiry is directly looked up)
3. **Entry still uses calendar offset**: Entry = anchor + ntrv calendar days (same as F/R/W/BD)
4. **Caching**: Override file is loaded once and cached for performance
5. **Missing override**: If no override found for asset/month, anchor returns None (no ntry/xpry action)

## Files to Modify

1. **`src/specparser/amt/tickers.py`**:
   - Add `_OVERRIDE_CACHE` global
   - Add `_load_overrides()` function
   - Add `_override_expiry()` function
   - Modify `_anchor_day()` to handle OVERRIDE
   - Modify `_compute_actions()` to pass `underlying` and `overrides_path`
   - Update callers of `_compute_actions()` to pass underlying

2. **`scripts/straddle_explain.py`**:
   - Update ANCHOR CALCULATION section to handle OVERRIDE display

## Testing

### Manual Tests

```bash
# Test with an asset that uses OVERRIDE (if one exists in AMT)
uv run python scripts/straddle_explain.py 'ASSET_WITH_OVERRIDE' 2024 3 0

# Verify it shows:
#   Expiry anchor: Override lookup for 2024-03 = 2024-03-XX
```

### Unit Tests

Add to `tests/test_amt.py`:

1. **`test_load_overrides`**: Verify cache loads correctly
2. **`test_override_expiry_found`**: Lookup returns correct date
3. **`test_override_expiry_not_found`**: Lookup returns None for missing entry
4. **`test_anchor_day_override`**: `_anchor_day("OVERRIDE", ...)` returns override date
5. **`test_compute_actions_override`**: Entry/expiry computed correctly with OVERRIDE

## Example

Straddle: `|2024-01|2024-03|F|10|OVERRIDE|0|100|`

- Entry month: 2024-01
- Expiry month: 2024-03
- Entry code: F (but uses xprc=OVERRIDE for expiry month)
- Wait - this doesn't make sense. Let me reconsider...

Actually, looking at the straddle format: `|ntry_month|xpry_month|ntrc|ntrv|xprc|xprv|mult|`

The entry anchor uses `xprc` and `xprv` applied to entry month. So if `xprc=OVERRIDE`:
- Entry anchor = lookup(underlying, entry_month)
- Expiry anchor = lookup(underlying, expiry_month)

Both use the same code, which is correct.

Example with OVERRIDE:
```
Straddle: |2024-01|2024-03|N|0|OVERRIDE|0|100|

1. Entry month = 2024-01
2. Entry anchor = overrides.csv lookup for underlying + 2024-01 = 2024-01-19 (example)
3. Entry target = 2024-01-19 + 0 = 2024-01-19
4. Entry = first good day at or after 2024-01-19

5. Expiry month = 2024-03
6. Expiry anchor = overrides.csv lookup for underlying + 2024-03 = 2024-03-15 (example)
7. Expiry = first good day at or after 2024-03-15
```

## Edge Cases

1. **Override not found**: If no entry in overrides.csv for asset/month, anchor returns None, no action set
2. **Empty overrides file**: All lookups return None
3. **Invalid date in overrides**: Should be handled by validation (out of scope for now)
4. **Mixed codes**: ntrc could be "N" while xprc is "OVERRIDE" - this is valid, ntrc just selects Near vs Far vol
