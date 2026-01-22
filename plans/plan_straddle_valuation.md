# Plan: Add get_straddle_valuation Function

## Problem

Need a function that computes the mark-to-market value (mv) of a straddle for each day, using the appropriate pricing model based on the asset's `Valuation.Model` setting.

## Function Signature

```python
def get_straddle_valuation(
    path: str | Path,
    underlying: str,
    year: int,
    month: int,
    prices_parquet: str | Path,
    chain_csv: str | Path | None = None,
    i: int = 0
) -> dict[str, Any]:
```

Same signature as `get_straddle_days`.

## Output

Returns the same table as `get_straddle_days` with an additional `mv` column:

```
asset  straddle  date        vol   hedge  action  model  strike_vol  strike  expiry      mv
...    ...       2025-02-18  25.5  290.6  -       ES     -           -       -           -
...    ...       2025-02-19  25.6  291.7  ntry    ES     25.6        291.7   2025-03-19  0.0
...    ...       2025-02-20  25.8  292.4  -       ES     25.6        291.7   2025-03-19  1.234
...
...    ...       2025-03-19  27.4  337.1  xpry    ES     25.6        291.7   2025-03-19  45.67
...    ...       2025-03-20  27.4  342.8  -       ES     -           -       -           -
```

## Logic

1. Call `get_straddle_days()` to get the base table
2. Check if both `ntry` and `xpry` actions exist in the action column
   - If no: return table with `mv` column all "-"
3. Get model from first row's `model` column
4. Dispatch to appropriate model function based on model name
5. For each row:
   - If row index < ntry_idx or row index > xpry_idx: mv = "-"
   - Else: call model function with dictionarized row, get mv value

## Model Dispatch

```python
MODEL_DISPATCH = {
    "ES": model_ES,
    "NS": model_NS,
    "BS": model_BS,
    # ... other models
}

def get_model_function(model_name: str):
    return MODEL_DISPATCH.get(model_name, model_default)
```

## Model Functions

### Common Interface

All model functions receive a dict with the row values:

```python
def model_ES(row: dict[str, Any]) -> str:
    """Compute mv for ES (European Straddle) model.

    Args:
        row: Dict with keys from columns: date, vol, hedge, strike_vol, strike, expiry, etc.

    Returns:
        String value for mv column (number as string, or "-" on error)
    """
```

### model_ES Implementation

Translating from R code:

```r
ES<-function(S,X,t,v,mult=1,root="")
{
  tv    <- ( v / 100 ) * sqrt( as.integer( t ) / 365 )
  d1    <- log( S / X ) / tv + 0.5 * tv
  d2    <- d1 - tv
  N_d1  <- 2 * pnorm( d1 )
  N_d2  <- 2 * pnorm( d2 )
  mv    <- S * N_d1 - X * N_d2 + X - S
```

Where:
- `S` = current hedge price (`hedge` column)
- `X` = strike price (`strike` column, captured at ntry)
- `t` = days to expiry (expiry date - current date)
- `v` = current vol (`vol` column)

Python implementation:

```python
import math
from scipy.stats import norm

def model_ES(row: dict[str, Any]) -> str:
    """European Straddle pricing model.

    Formula: mv = S * N(d1) * 2 - X * N(d2) * 2 + X - S

    Inputs from row:
        - hedge: current underlying price (S)
        - strike: strike price captured at entry (X)
        - vol: current implied vol in percent
        - date: current date
        - expiry: expiry date

    Returns "-" for any inadequate inputs (missing, non-numeric, invalid dates,
    t <= 0, zero/negative prices or vol, etc.)
    """
    try:
        S = float(row["hedge"])
        X = float(row["strike"])
        v = float(row["vol"])

        # Validate positive values
        if S <= 0 or X <= 0 or v <= 0:
            return "-"

        # Calculate days to expiry
        from datetime import date as date_type
        current_date = date_type.fromisoformat(row["date"])
        expiry_date = date_type.fromisoformat(row["expiry"])
        t = (expiry_date - current_date).days

        if t <= 0:
            return "-"  # At or past expiry - inadequate input

        # Total volatility
        tv = (v / 100) * math.sqrt(t / 365)

        d1 = math.log(S / X) / tv + 0.5 * tv
        d2 = d1 - tv

        N_d1 = 2 * _norm_cdf(d1)
        N_d2 = 2 * _norm_cdf(d2)

        mv = S * N_d1 - X * N_d2 + X - S

        return str(mv)
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return "-"
```

### Other Model Functions (Placeholder)

```python
def model_NS(row: dict[str, Any]) -> str:
    """Normal Straddle model - placeholder."""
    return "-"

def model_BS(row: dict[str, Any]) -> str:
    """Black-Scholes model - placeholder."""
    return "-"

def model_default(row: dict[str, Any]) -> str:
    """Default model for unknown model names."""
    return "-"
```

## Implementation Steps

### 1. Add Model Functions

Add before `get_straddle_valuation`:

```python
# -------------------------------------
# Valuation Models
# -------------------------------------

def model_ES(row: dict[str, Any]) -> str:
    """European Straddle pricing model."""
    # ... implementation

def model_NS(row: dict[str, Any]) -> str:
    return "-"

def model_BS(row: dict[str, Any]) -> str:
    return "-"

def model_default(row: dict[str, Any]) -> str:
    return "-"

MODEL_DISPATCH = {
    "ES": model_ES,
    "NS": model_NS,
    "BS": model_BS,
    "CDS_ES": model_ES,  # CDS_ES uses ES model
}
```

### 2. Add get_straddle_valuation Function

```python
def get_straddle_valuation(
    path: str | Path,
    underlying: str,
    year: int,
    month: int,
    prices_parquet: str | Path,
    chain_csv: str | Path | None = None,
    i: int = 0
) -> dict[str, Any]:
    """Get straddle valuation with mv column.

    Calls get_straddle_days and adds mv (mark-to-market value) column
    computed using the asset's valuation model.

    Args:
        Same as get_straddle_days

    Returns:
        Table with additional mv column
    """
    # Get base table
    table = get_straddle_days(path, underlying, year, month, prices_parquet, chain_csv, i)

    columns = table["columns"]
    rows = table["rows"]

    # Find action column and check for ntry/xpry
    if "action" not in columns:
        # No action column, add mv as all "-"
        for row in rows:
            row.append("-")
        columns.append("mv")
        return {"columns": columns, "rows": rows}

    action_idx = columns.index("action")

    ntry_idx = None
    xpry_idx = None
    for i, row in enumerate(rows):
        if row[action_idx] == "ntry":
            ntry_idx = i
        elif row[action_idx] == "xpry":
            xpry_idx = i

    if ntry_idx is None or xpry_idx is None:
        # Missing ntry or xpry, add mv as all "-"
        for row in rows:
            row.append("-")
        columns.append("mv")
        return {"columns": columns, "rows": rows}

    # Get model from first row
    model_idx = columns.index("model") if "model" in columns else None
    if model_idx is not None and rows:
        model_name = rows[0][model_idx]
    else:
        model_name = ""

    # Get model function
    model_fn = MODEL_DISPATCH.get(model_name, model_default)

    # Compute mv for each row
    for i, row in enumerate(rows):
        if i < ntry_idx or i > xpry_idx:
            row.append("-")
        else:
            # Dictionarize row and call model
            row_dict = dict(zip(columns, row))
            mv = model_fn(row_dict)
            row.append(mv)

    columns.append("mv")
    return {"columns": columns, "rows": rows}
```

### 3. Add CLI Argument

```python
p.add_argument("--asset-valuation", nargs=4, type=str, metavar=("UNDERLYING", "YEAR", "MONTH", "NDX"),
               help="Get straddle valuation with mv column.")
```

### 4. Add CLI Handler

```python
elif args.asset_valuation:
    underlying, year, month, i = args.asset_valuation
    table = get_straddle_valuation(args.path, underlying, int(year), int(month), args.prices, args.chain_csv, int(i))
    loader.print_table(table)
```

## Dependencies

Need to add scipy for `norm.cdf`:

```bash
uv add scipy
```

Or use a pure Python implementation of the normal CDF (math.erf based):

```python
def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
```

Using `math.erf` avoids the scipy dependency.

## Files to Modify

1. **`src/specparser/amt/tickers.py`**:
   - Add model functions (model_ES, model_NS, model_BS, model_default)
   - Add MODEL_DISPATCH dict
   - Add `get_straddle_valuation()` function
   - Add `--asset-valuation` CLI argument and handler

## Edge Cases

1. **No ntry action**: Return table with mv all "-"
2. **No xpry action**: Return table with mv all "-"
3. **Invalid values in row**: Model returns "-" (missing keys, non-numeric values, etc.)
4. **t <= 0 (at or past expiry)**: Model returns "-"
5. **Unknown model**: Use model_default which returns "-"
6. **Missing columns**: Handle gracefully, return "-"
7. **Zero or negative prices**: Model returns "-"
8. **Zero volatility**: Model returns "-"

All models (including ES) return "-" for any inadequate input. The ES model does not attempt to compute intrinsic values or handle edge cases specially - if inputs are not valid for the formula, it returns "-".

## Testing

### Manual CLI Tests

```bash
# Test with ES model asset
uv run python -m specparser.amt.tickers data/amt.yml --asset-valuation 'LA Comdty' 2025 2 0

# Test with unknown model
uv run python -m specparser.amt.tickers data/amt.yml --asset-valuation 'SOME_ASSET' 2025 2 0
```

### Unit Tests

Add to `tests/test_amt.py`:

1. `test_model_ES_basic`: Test ES formula with known inputs
2. `test_model_ES_at_expiry`: Test t=0 case (intrinsic value)
3. `test_model_ES_invalid_inputs`: Test error handling
4. `test_get_straddle_valuation_adds_mv_column`: Verify mv column added
5. `test_get_straddle_valuation_no_ntry`: mv all "-" when no ntry
6. `test_get_straddle_valuation_unknown_model`: Uses default model

## Notes

- The ES model is a European-style straddle pricing formula
- `pnorm` in R is equivalent to `norm.cdf` in Python (standard normal CDF)
- Using `math.erf` avoids adding scipy as a dependency
- Model dispatch allows easy addition of new models later
