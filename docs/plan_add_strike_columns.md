# Plan: Add Strike Columns to get_straddle_days

## Problem

The `get_straddle_days` output needs additional columns to capture the "choice" made at entry:
1. `strike_vol` - the vol value at the ntry action row
2. `strike`, `strike1`, `strike2`, ... - the hedge values at the ntry action row (one per hedge column)

These are **new columns** - existing columns (`vol`, `hedge`, etc.) are not modified.

The strike columns represent the market values at the moment of entry. Before ntry, no choice has been made yet. After xpry, the choice no longer matters. So:
- `strike_vol` shows `-` before ntry and after xpry
- `strike` columns show the constant hedge value from ntry for all rows (since hedge choice is locked at entry)

## Current Output

```
asset  straddle  date        vol      hedge    hedge1   action  model
...    ...       2025-02-18  25.5386  290.603  4.112    -       CDS_ES
...    ...       2025-02-19  25.5774  291.72   4.0951   ntry    CDS_ES
...    ...       2025-02-20  25.8131  292.425  4.0663   -       CDS_ES
...
...    ...       2025-03-19  27.4369  337.128  3.7409   xpry    CDS_ES
...    ...       2025-03-20  27.4369  342.808  3.723    -       CDS_ES
```

## Desired Output

```
asset  straddle  date        vol      hedge    hedge1   action  model   strike_vol  strike   strike1
...    ...       2025-02-18  25.5386  290.603  4.112    -       CDS_ES  -           -        -
...    ...       2025-02-19  25.5774  291.72   4.0951   ntry    CDS_ES  25.5774     291.72   4.0951
...    ...       2025-02-20  25.8131  292.425  4.0663   -       CDS_ES  25.5774     291.72   4.0951
...
...    ...       2025-03-19  27.4369  337.128  3.7409   xpry    CDS_ES  25.5774     291.72   4.0951
...    ...       2025-03-20  27.4369  342.808  3.723    -       CDS_ES  -           -        -
```

Note: `vol`, `hedge`, `hedge1` columns are **unchanged** - they show market data as-is.

## Key Behaviors

1. **strike_vol**: Vol value captured at ntry, shown from ntry to xpry (inclusive), `-` outside that range
2. **strike columns**: Hedge values captured at ntry, shown from ntry to xpry (inclusive), `-` outside that range
3. **Existing columns unchanged**: `vol`, `hedge`, etc. remain as market data
4. **No ntry found**: All strike columns show `-` for all rows
5. **No xpry found**: Strike columns show values from ntry onward (no end cutoff)

## Implementation

### 1. Find ntry and xpry row indices

After computing actions, find the indices:

```python
# Find ntry and xpry row indices
ntry_idx = None
xpry_idx = None
for i, action in enumerate(actions):
    if action == "ntry":
        ntry_idx = i
    elif action == "xpry":
        xpry_idx = i
```

### 2. Find vol and hedge column indices

```python
# Find vol column index in out_columns
vol_col_idx = out_columns.index("vol") if "vol" in out_columns else None

# Find hedge column indices and names
hedge_col_info = []  # list of (index, name) tuples
for i, col in enumerate(out_columns):
    if col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
        hedge_col_info.append((i, col))
```

### 3. Extract strike values at ntry

```python
# Get strike values from ntry row
if ntry_idx is not None and vol_col_idx is not None:
    strike_vol_value = out_rows[ntry_idx][vol_col_idx]
    strike_values = [out_rows[ntry_idx][idx] for idx, _ in hedge_col_info]
else:
    strike_vol_value = "-"
    strike_values = ["-"] * len(hedge_col_info)
```

### 4. Add strike columns (NEW columns only)

```python
# Add strike_vol column
for i, row in enumerate(out_rows):
    # Show value only from ntry to xpry (inclusive)
    in_range = (ntry_idx is not None and i >= ntry_idx and
                (xpry_idx is None or i <= xpry_idx))
    row.append(strike_vol_value if in_range else "-")
out_columns.append("strike_vol")

# Add strike columns (one per hedge)
for j, (hedge_idx, hedge_name) in enumerate(hedge_col_info):
    strike_col_name = "strike" if j == 0 else f"strike{j}"
    for i, row in enumerate(out_rows):
        in_range = (ntry_idx is not None and i >= ntry_idx and
                    (xpry_idx is None or i <= xpry_idx))
        row.append(strike_values[j] if in_range else "-")
    out_columns.append(strike_col_name)
```

### 5. Column order

Final column order:
```
asset, straddle, date, vol, hedge, hedge1, ..., action, model, strike_vol, strike, strike1, ...
```

## Edge Cases

1. **No ntry action**: All strike columns are `-` for all rows
2. **No xpry action**: Strike columns show values from ntry to end of data
3. **ntry and xpry on same row**: Only that single row has strike values
4. **Multiple hedges**: Create matching number of strike columns

## Files to Modify

1. **`src/specparser/amt/tickers.py`**:
   - In `get_straddle_days()`: Add logic after model column to:
     - Find ntry/xpry indices from actions list
     - Find vol and hedge column indices
     - Extract strike values from ntry row
     - Add strike_vol and strike columns

## Testing

### Unit Tests

Add tests to `tests/test_amt.py` in a new `TestStrikeColumns` class:

1. **`test_strike_columns_basic`**: Verify strike_vol and strike columns are added with correct values from ntry row
2. **`test_strike_columns_before_ntry`**: Verify all strike columns show `-` before ntry
3. **`test_strike_columns_after_xpry`**: Verify all strike columns show `-` after xpry
4. **`test_strike_columns_no_ntry`**: When no ntry action exists, all strike columns are `-` for all rows
5. **`test_strike_columns_no_xpry`**: When no xpry action exists, strike values continue to end of data
6. **`test_strike_columns_multiple_hedges`**: Verify correct number of strike columns (one per hedge)
7. **`test_strike_columns_vol_unchanged`**: Verify vol column is NOT modified (still shows market data)

### CLI Verification

After implementing, verify with:
```bash
uv run python -m specparser.amt.tickers data/amt.yml --asset-days 'IBOXHYSE Curncy' 2025 3 0
```

Expected:
- `strike_vol` shows vol at ntry from ntry to xpry, `-` elsewhere
- `strike` and `strike1` show hedge values at ntry from ntry to xpry, `-` elsewhere
- `vol`, `hedge`, `hedge1` columns remain unchanged (market data)

### Test Data Requirements

The unit tests will need mock data with:
- Rows before ntry (to verify `-`)
- An ntry row with known vol/hedge values
- Rows between ntry and xpry (to verify strike values propagate)
- An xpry row
- Rows after xpry (to verify `-`)
