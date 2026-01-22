# Plan: Fix CV Source Field Case Sensitivity Bug

## Problem

When running `get_straddle_days` for assets with CV (Constant Value) source for vol, the vol column returns "none" for all dates even though the data exists in the prices parquet file.

**Example:**
```bash
uv run python -m specparser.amt.tickers data/amt.yml --asset-days 'IBOXHYSE Curncy' 2025 3 0
```

Output shows `vol=none` for all dates, which means:
1. No day is ever considered a "good day" (requires vol + hedge to be valid)
2. No action triggers (ntry or xpry) can fire
3. The W3 (3rd Wednesday) expiry rule never triggers

## Root Cause

**Case sensitivity mismatch** between the code and the parquet file.

### In `_filter_straddle_tickers` ([tickers.py:509](src/specparser/amt/tickers.py#L509)):
```python
if row_source == "CV":
    new_row[ticker_idx] = row_field
    new_row[field_idx] = "NONE"  # <-- UPPERCASE
```

### In the prices parquet file:
```sql
SELECT DISTINCT field FROM prices WHERE ticker LIKE '%CVOL%';
-- Returns: 'none'  (lowercase)
```

### Evidence:
```python
# Query with field='NONE' returns 0 rows
# Query with field='none' returns 3812 rows
```

## Solution

Change the field value from `"NONE"` to `"none"` to match the parquet file.

### Change in `_filter_straddle_tickers`:
```python
# Before (line 509)
new_row[field_idx] = "NONE"

# After
new_row[field_idx] = "none"
```

## Why This Happens

The CV source is a special case where:
1. The original Vol row has `ticker=""` and `field="CREDIT.CVOL...."` (the actual lookup key)
2. The filter swaps them: `ticker="CREDIT.CVOL...."` and `field="NONE"` (placeholder)
3. But the parquet uses lowercase `"none"` as the placeholder field

## Testing

### Verification Query
```python
import duckdb
con = duckdb.connect()
con.execute("CREATE VIEW p AS SELECT * FROM 'data/prices.parquet'")

# Uppercase NONE: 0 results
con.execute("SELECT COUNT(*) FROM p WHERE field = 'NONE'").fetchone()

# Lowercase none: many results
con.execute("SELECT COUNT(*) FROM p WHERE field = 'none'").fetchone()
```

### After Fix
```bash
uv run python -m specparser.amt.tickers data/amt.yml --asset-days 'IBOXHYSE Curncy' 2025 3 0
```

Expected:
- Vol column should show values like `27.4369` instead of `none`
- `xpry` action should trigger on 2025-03-19 (3rd Wednesday of March)

## Impact

This affects all assets with CV source for vol, which includes credit/rate assets like:
- IBOXHYSE Curncy
- Other credit index assets

## Files to Modify

1. **`src/specparser/amt/tickers.py`** (line 509):
   - Change `"NONE"` to `"none"`

## Implementation Checklist

- [ ] Change `"NONE"` to `"none"` in `_filter_straddle_tickers`
- [ ] Run existing tests to ensure no regressions
- [ ] Verify IBOXHYSE Curncy output shows vol values
- [ ] Verify xpry action triggers on 3rd Wednesday (2025-03-19)
