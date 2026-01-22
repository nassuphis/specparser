# Plan: Add Model Column to get_straddle_days

## Problem

The `get_straddle_days` output needs a `model` column containing the asset's `Valuation.Model` value.

## Current Output

```python
{
    "columns": ["asset", "straddle", "date", "vol", "hedge", "hedge1", "action"],
    "rows": [
        ["CL Comdty", "|2024-05|2024-06|...|", "2024-05-01", "25.3", "78.50", "79.20", "-"],
        ...
    ]
}
```

## Desired Output

```python
{
    "columns": ["asset", "straddle", "date", "vol", "hedge", "hedge1", "action", "model"],
    "rows": [
        ["CL Comdty", "|2024-05|2024-06|...|", "2024-05-01", "25.3", "78.50", "79.20", "-", "BS"],
        ...
    ]
}
```

## Implementation

### 1. Import `get_asset` from loader

At the top of `get_straddle_days`, we already have access to `path` and `underlying` (the asset).

```python
from . import loader
```

### 2. Get the model value

After getting the `asset` variable from the table rows:

```python
# Get model from Valuation.Model
asset_data = loader.get_asset(path, underlying)
if asset_data is not None:
    valuation = asset_data.get("Valuation", {})
    model = valuation.get("Model", "") if isinstance(valuation, dict) else ""
else:
    model = ""
```

### 3. Add model column to output

After adding the action column (lines 1021-1024), add the model column:

```python
# Add model column (same value for all rows)
for row in out_rows:
    row.append(model)
out_columns.append("model")
```

## Files to Modify

1. **`src/specparser/amt/tickers.py`**:
   - Import `loader` module (if not already)
   - In `get_straddle_days`: get model value via `loader.get_asset(path, underlying)`
   - Add model column to output rows and columns

## Column Position

The `model` column will be added after `action`, making the final column order:
- `asset`, `straddle`, `date`, `<params...>`, `action`, `model`

## Testing

After implementing, verify with:
```bash
uv run python -m specparser.amt.tickers data/amt.yml --asset-days 'IBOXHYSE Curncy' 2025 3 0
```

Expected: A `model` column appears with the Valuation.Model value for IBOXHYSE Curncy.

## Edge Cases

1. **Asset not found**: If `get_asset` returns `None`, use empty string `""`
2. **No Valuation key**: If asset has no `Valuation` dict, use empty string `""`
3. **No Model key**: If Valuation has no `Model` key, use empty string `""`

All handled by the safe `.get()` chain with defaults.
