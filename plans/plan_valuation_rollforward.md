# Plan: Roll Forward Market Data in Valuation

## Problem

Currently, `get_straddle_valuation` returns "-" for mv on days where market data is missing (e.g., weekends where hedge="none"). The user wants all days from ntry to xpry to have an mv value by rolling forward market data from the previous good day.

## Current Behavior

```
date        vol   hedge  action  mv
2025-02-19  25.6  292    ntry    16.5
2025-02-20  25.8  292    -       16.4
2025-02-21  27.4  303    -       19.5
2025-02-22  27.4  none   -       -      <- weekend, no mv
2025-02-23  27.4  none   -       -      <- weekend, no mv
2025-02-24  27.4  305    -       19.6
```

## Desired Behavior

Roll forward market data: use the most recent good day's data, replacing with any new data that exists.

```
date        vol   hedge  action  mv
2025-02-19  25.6  292    ntry    16.5
2025-02-20  25.8  292    -       16.4
2025-02-21  27.4  303    -       19.5
2025-02-22  27.4  none   -       19.5   <- uses 2025-02-21 data (vol=27.4, hedge=303)
2025-02-23  27.4  none   -       19.5   <- uses 2025-02-21 data
2025-02-24  27.4  305    -       19.6   <- new data available
```

## Key Insight

The entry day (ntry) is always a "good day" by construction (it's selected based on having valid vol and hedge data). So there's always valid data to start rolling forward from.

## Implementation

### 1. Modify `get_straddle_valuation` to track rolled-forward data

Instead of just dictionarizing each row and calling the model, maintain a "last good" dict that gets updated with any non-missing values from each row.

```python
def get_straddle_valuation(...):
    # ... existing code to get base table, find ntry_idx, xpry_idx ...

    # Initialize rolled-forward data from ntry row
    rolled_data = dict(zip(columns, rows[ntry_idx]))

    # Compute mv for each row
    for idx, row in enumerate(rows):
        if idx < ntry_idx or idx > xpry_idx:
            row.append("-")
        else:
            # Update rolled_data with any non-missing values from current row
            row_dict = dict(zip(columns, row))
            for key, value in row_dict.items():
                if value != "none" and value != "-":
                    rolled_data[key] = value

            # Call model with rolled-forward data
            mv = model_fn(rolled_data)
            row.append(mv)

    columns.append("mv")
    return {"columns": columns, "rows": rows}
```

### 2. Fields to roll forward

The model uses these fields from the row:
- `hedge` - current underlying price (S)
- `strike` - strike price (X) - already constant from ntry
- `vol` - current implied vol
- `date` - current date - should NOT be rolled forward
- `expiry` - expiry date - already constant from ntry

Fields that should be rolled forward:
- `hedge`, `hedge1`, `hedge2`, ... (market prices)
- `vol` (volatility)

Fields that should NOT be rolled forward (use current row's value):
- `date` (must be current date for time-to-expiry calculation)
- `expiry` (already constant)
- `strike`, `strike1`, ... (already constant)
- `strike_vol` (already constant)
- `action`, `model`, `asset`, `straddle` (metadata)

### 3. Refined implementation

```python
# Fields that should be rolled forward
ROLLFORWARD_FIELDS = {"vol", "hedge", "hedge1", "hedge2", "hedge3"}

def get_straddle_valuation(...):
    # ... existing code ...

    # Initialize rolled-forward data from ntry row
    rolled_data = {}
    ntry_row_dict = dict(zip(columns, rows[ntry_idx]))
    for key in ROLLFORWARD_FIELDS:
        if key in ntry_row_dict:
            rolled_data[key] = ntry_row_dict[key]

    # Compute mv for each row
    for idx, row in enumerate(rows):
        if idx < ntry_idx or idx > xpry_idx:
            row.append("-")
        else:
            # Update rolled_data with any non-missing market values
            row_dict = dict(zip(columns, row))
            for key in ROLLFORWARD_FIELDS:
                if key in row_dict and row_dict[key] != "none":
                    rolled_data[key] = row_dict[key]

            # Build model input: current row data + rolled-forward market data
            model_input = row_dict.copy()
            model_input.update(rolled_data)

            mv = model_fn(model_input)
            row.append(mv)

    columns.append("mv")
    return {"columns": columns, "rows": rows}
```

### 4. Dynamic field detection

Instead of hardcoding `ROLLFORWARD_FIELDS`, detect them dynamically:

```python
def _get_rollforward_fields(columns: list[str]) -> set[str]:
    """Get fields that should be rolled forward (vol and hedge columns)."""
    fields = set()
    for col in columns:
        if col == "vol":
            fields.add(col)
        elif col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
            fields.add(col)
    return fields
```

## Edge Cases

1. **ntry row has missing data**: By construction, ntry is a "good day" with valid vol and hedge, so this shouldn't happen. If it does, the model will return "-" as before.

2. **All days between ntry and xpry have missing data**: The ntry data gets rolled forward to all of them.

3. **Partial data**: If only vol is missing but hedge exists, roll forward vol but use current hedge.

4. **xpry day**: Still returns "-" because t=0 (inadequate input), regardless of data availability.

## Files to Modify

1. **`src/specparser/amt/tickers.py`**:
   - Add `_get_rollforward_fields(columns)` helper function
   - Modify `get_straddle_valuation()` to:
     - Initialize rolled-forward data from ntry row
     - Update with non-missing values each day
     - Pass rolled-forward data to model

## Testing

### Manual CLI Tests

```bash
# Verify weekends now have mv values
uv run python -m specparser.amt.tickers data/amt.yml --asset-valuation 'IBOXHYSE Curncy' 2025 3 0 | grep "2025-02-2[2-4]"
```

Expected: All three rows (2025-02-22, 2025-02-23, 2025-02-24) should have mv values.

### Unit Tests

Add to `tests/test_amt.py`:

1. **`test_valuation_rollforward_weekend`**: Verify mv is computed for weekend rows using previous day's data
2. **`test_valuation_rollforward_partial_data`**: When only hedge is missing, vol is used from current row
3. **`test_valuation_rollforward_multiple_missing`**: Multiple consecutive missing days all get values
4. **`test_valuation_rollforward_updates_when_data_returns`**: When data becomes available again, it's used instead of rolled-forward data

## Summary

The change is localized to `get_straddle_valuation()`. The model functions remain unchanged - they still receive a row dict and compute mv. The difference is that the row dict now contains rolled-forward market data for missing values instead of "none".
