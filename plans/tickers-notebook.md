# Plan: Create tickers.ipynb Notebook

## Overview

Create a comprehensive Jupyter notebook documenting the `specparser.amt.tickers` module, which handles:
- Extracting ticker schemas from AMT asset definitions
- Expanding futures ticker specs to actual tickers
- Handling split tickers (ticker transitions over time)
- Converting between normalized and actual Bloomberg tickers
- Getting tickers for specific year/month periods
- Computing straddle tickers with filtering rules

## Target Audience

Users who need to:
- Extract all tickers needed for an asset (market, vol, hedge)
- Understand ticker schemas and their structure
- Expand futures specs to actual contract tickers
- Handle ticker transitions (split tickers)
- Get tickers for specific trading periods

## Notebook Structure

### Section 1: Introduction and Concepts
- What is a ticker schema? (template for extracting tickers from AMT assets)
- The three ticker types: Market, Vol, Hedge
- Ticker sources: BBG, BBGfc (futures spec), CV (CitiVelocity), calc, nBBG
- Split ticker format: `ticker1:YYYY-MM:ticker2`

### Section 2: Ticker Schemas
```python
from specparser.amt import get_tschemas, find_tschemas, print_table

# Get tickers for a single asset
tickers = get_tschemas("data/amt.yml", "CL Comdty")
print_table(tickers)

# Find tickers for multiple assets
tickers = find_tschemas("data/amt.yml", pattern="^CL|^GC", live_only=True)
print_table(tickers)
```

**Functions covered:**
- `get_tschemas(path, underlying)` - Get ticker schema for one asset
- `find_tschemas(path, pattern, live_only)` - Find ticker schemas by regex

**Schema columns:** `asset`, `cls`, `type`, `param`, `source`, `ticker`, `field`

### Section 3: Understanding Ticker Types

**Market Tickers:**
- Source: BBG
- Used for: Market price data
- Example: `AAPL US Equity` with field `PX_LAST`

**Vol Tickers:**
- Sources: BBG, CV (CitiVelocity), BBG_LMEVOL
- Params: Near, Far (for near/far month vol)
- Example: `30DAY_IMPVOL_100.0%MNY_DF` field

**Hedge Tickers:**
- Sources: BBG (nonfut), BBGfc (futures), calc (computed), cds
- Params: hedge, hedge1, hedge2, hedge3
- BBGfc contains spec strings that need expansion

### Section 4: Futures Ticker Expansion

```python
from specparser.amt import fut_spec2ticker

# Convert futures spec to actual ticker
spec = "generic:CL1 Comdty,fut_code:CL,fut_month_map:GHJKMNQUVXZF,min_year_offset:0,market_code:Comdty"
ticker = fut_spec2ticker(spec, 2024, 6)  # June 2024
print(ticker)  # "CLN2024 Comdty"
```

**Functions covered:**
- `fut_spec2ticker(spec, year, month)` - Convert spec to ticker

**Month map explanation:**
- 12-char string mapping months to contract codes
- F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec

### Section 5: Normalized vs Actual Tickers

```python
from specparser.amt import fut_norm2act, fut_act2norm, clear_normalized_cache

# Normalized -> Actual (requires chain CSV)
actual = fut_norm2act("data/chain.csv", "CLN2024 Comdty")
print(actual)  # "CL N24 Comdty" (actual BBG ticker)

# Actual -> Normalized (reverse lookup)
normalized = fut_act2norm("data/chain.csv", "CL N24 Comdty")
print(normalized)  # "CLN2024 Comdty"

# Clear cache if needed
clear_normalized_cache()
```

**Functions covered:**
- `fut_norm2act(csv_path, ticker)` - Normalized to actual
- `fut_act2norm(csv_path, ticker)` - Actual to normalized
- `clear_normalized_cache()` - Clear lookup cache

### Section 6: Split Tickers (Ticker Transitions)

```python
from specparser.amt import _split_ticker

# Split ticker format: ticker1:YYYY-MM:ticker2
# Before the date, use ticker1; after the date, use ticker2
result = _split_ticker("USSWAP5 CMPN Curncy:2023-06:USOSFR5 Curncy", "hedge")
# Returns: [
#   ("USSWAP5 CMPN Curncy", "hedge<2023-06"),
#   ("USOSFR5 Curncy", "hedge>2023-06")
# ]
```

**Use case:** When Bloomberg changes ticker symbols (e.g., LIBOR to SOFR transition)

### Section 7: Tickers for Specific Periods

```python
from specparser.amt import find_tickers, find_tickers_ym

# Get tickers for a specific year/month
tickers = find_tickers_ym("data/amt.yml", "^CL", True, 2024, 6, "data/chain.csv")
print_table(tickers)

# Get unique tickers across a year range
tickers = find_tickers("data/amt.yml", "^CL", live_only=True,
                       start_year=2024, end_year=2024, chain_csv="data/chain.csv")
print_table(tickers)
```

**Functions covered:**
- `find_tickers_ym(path, pattern, live_only, year, month, chain_csv)` - One month
- `find_tickers(path, pattern, live_only, start_year, end_year, chain_csv)` - Year range

### Section 8: Straddle Tickers

```python
from specparser.amt import asset_straddle_tickers

# Get tickers for a specific straddle
tickers = asset_straddle_tickers(
    asset="CL Comdty",
    year=2024,
    month=6,
    i=0,  # First straddle component
    amt_path="data/amt.yml",
    chain_path="data/chain.csv"
)
print_table(tickers)
```

**Filtering rules:**
- Market tickers: excluded
- Vol/Near: kept only if straddle ntrc == "N", param → "vol"
- Vol/Far: kept only if straddle ntrc == "F", param → "vol"
- Hedge: always kept

**Functions covered:**
- `asset_straddle_tickers(asset, year, month, i, amt_path, chain_path)`

### Section 9: Practical Examples

**Example 1: Analyze Ticker Sources**
```python
# Count tickers by source across asset classes
for pattern, name in [("Comdty$", "Commodities"), ("Index$", "Indices")]:
    tickers = find_tschemas("data/amt.yml", pattern, live_only=True)
    sources = Counter(row[4] for row in tickers["rows"])
    print(f"{name}: {dict(sources)}")
```

**Example 2: Find Assets with Split Tickers**
```python
# Find assets that have ticker transitions
tickers = find_tschemas("data/amt.yml", ".", live_only=True)
split_assets = set()
for row in tickers["rows"]:
    if ":" in row[5] and "-" in row[5]:  # ticker column
        split_assets.add(row[0])
print(f"Assets with split tickers: {split_assets}")
```

**Example 3: Futures Contract Calendar**
```python
# Generate futures tickers for a year
spec = "fut_code:CL,fut_month_map:GHJKMNQUVXZF,market_code:Comdty"
for month in range(1, 13):
    ticker = fut_spec2ticker(spec, 2024, month)
    print(f"Month {month:2d}: {ticker}")
```

### Section 10: Summary

| Function | Description |
|----------|-------------|
| `get_tschemas(path, underlying)` | Get ticker schemas for one asset |
| `find_tschemas(path, pattern, live_only)` | Find ticker schemas by pattern |
| `fut_spec2ticker(spec, year, month)` | Convert futures spec to ticker |
| `fut_norm2act(csv, ticker)` | Normalized to actual ticker |
| `fut_act2norm(csv, ticker)` | Actual to normalized ticker |
| `_split_ticker(ticker, param)` | Expand split ticker format |
| `find_tickers_ym(...)` | Get tickers for one month |
| `find_tickers(...)` | Get unique tickers for year range |
| `asset_straddle_tickers(...)` | Get filtered tickers for straddle |

---

## Files to Create

1. **notebooks/tickers.ipynb** - The main notebook

## Implementation Notes

- Use sample data from `data/amt.yml`
- Note: Some functions require `data/chain.csv` for futures lookup - handle gracefully if missing
- Show both `show()` helper and `print_table()` output
- Explain the param constraints (`<YYYY-MM`, `>YYYY-MM`, `XYYYY-MM`)

## Cell Count Estimate

~45-50 cells total:
- ~6 markdown intro/concept cells
- ~6 ticker schema cells
- ~6 ticker type explanation cells
- ~6 futures expansion cells
- ~6 normalized/actual conversion cells
- ~4 split ticker cells
- ~6 period-specific ticker cells
- ~6 straddle ticker cells
- ~6 practical example cells
- ~2 summary cells

---

## Verification

After creating the notebook:
1. Run all cells to verify no errors
2. Handle missing chain.csv gracefully (use try/except or conditional logic)
