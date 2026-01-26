# Asset Straddle Tickers Reference

This module provides functions to extract ticker information needed for straddle pricing from AMT asset definitions.

## Overview

When pricing straddles, you need specific tickers for:
- **Volatility (vol)**: Implied volatility data for option pricing
- **Hedge**: The underlying hedge instrument (futures, spot, CDS, etc.)

This module extracts these tickers based on the asset configuration and the straddle's entry code (Near or Far).

## Key Concepts

### Entry Code (ntrc)

The entry code determines which volatility field to use:
- **N (Near)**: Use near-month volatility (`Vol.Near` field)
- **F (Far)**: Use far-month volatility (`Vol.Far` field)

### Year-Month String (strym)

A `YYYY-MM` formatted string representing the straddle's expiry month, e.g., `"2024-06"`.

### Vol Sources

| Source | Description | Output |
|--------|-------------|--------|
| `BBG` | Bloomberg standard vol | Uses `Vol.Ticker` with `Vol.Near` or `Vol.Far` field |
| `CV` | CitiVelocity | Uses `Vol.Near` ticker directly |
| `BBG_LMEVOL` | LME volatility | Constructs futures ticker with `R` qualifier |

### Hedge Sources

| Source | Description | Output |
|--------|-------------|--------|
| `fut` | Futures contract | Constructs ticker from `fut_code`, `fut_month_map`, etc. |
| `nonfut` | Non-futures hedge | Uses `Hedge.Ticker` directly (handles split tickers) |
| `cds` | Credit default swap | Uses `Hedge.hedge` and `Hedge.hedge1` |
| `calc` | Calculated swap | Generates synthetic ticker names for swap calculations |

## Functions

### `get_asset_straddle_tickers(asset, strym, ntrc, amt_path)`

Get tickers for a single asset and straddle configuration.

**Parameters:**
- `asset`: Asset underlying (e.g., `"CL Comdty"`)
- `strym`: Year-month string (e.g., `"2024-06"`)
- `ntrc`: Entry code (`"N"` or `"F"`)
- `amt_path`: Path to AMT YAML file

**Returns:**
A table with columns `["name", "ticker", "field"]`:
- `name`: Ticker role (`vol`, `hedge`, `hedge1`, etc.)
- `ticker`: The ticker symbol
- `field`: The data field to fetch

**Example:**
```python
from specparser.amt.asset_straddle_tickers import get_asset_straddle_tickers

tickers = get_asset_straddle_tickers("CL Comdty", "2024-06", "N", "data/amt.yml")
# Returns:
# name    ticker                field
# vol     CL1 Comdty            1ST_MTH_IMPVOL_100.0%MNY_DF
# hedge   CLN2024 Comdty        PX_LAST
```

### `find_assets_straddles_tickers(pattern, ntry, xpry, amt_path)`

Find all unique tickers needed for assets matching a pattern over a date range.

**Parameters:**
- `pattern`: Regex pattern to match asset underlyings
- `ntry`: Start year-month (e.g., `"2024-01"`)
- `xpry`: End year-month (e.g., `"2024-12"`)
- `amt_path`: Path to AMT YAML file

**Returns:**
A table with unique `(ticker, field)` combinations needed to price all matching straddles.

**Example:**
```python
from specparser.amt.asset_straddle_tickers import find_assets_straddles_tickers

# Get all unique tickers for commodities in H1 2024
tickers = find_assets_straddles_tickers("Comdty$", "2024-01", "2024-06", "data/amt.yml")
print(f"Need {len(tickers['rows'])} unique tickers")
```

### `split_ticker(ticker, year, month)`

Handle split tickers (ticker transitions at a specific date).

**Format:** `ticker1:YYYY-MM:ticker2`

Returns `ticker1` if before the date, `ticker2` otherwise.

**Example:**
```python
from specparser.amt.asset_straddle_tickers import split_ticker

# LIBOR to SOFR transition
ticker = "USSWAP5 CMPN Curncy:2023-06:USOSFR5 Curncy"

split_ticker(ticker, 2023, 1)  # → "USSWAP5 CMPN Curncy"
split_ticker(ticker, 2023, 6)  # → "USOSFR5 Curncy"
split_ticker(ticker, 2024, 1)  # → "USOSFR5 Curncy"
```

### `make_fut_ticker(fut_code, fut_month_map, min_year_offset, market_code, qualifier, year, month)`

Construct a futures ticker from components.

**Parameters:**
- `fut_code`: Futures code (e.g., `"CL"`)
- `fut_month_map`: 12-char map of month codes (e.g., `"FGHJKMNQUVXZ"`)
- `min_year_offset`: Minimum year offset for contract
- `market_code`: Market suffix (e.g., `"Comdty"`)
- `qualifier`: Optional qualifier (e.g., `"R"` for LME vol)
- `year`, `month`: Target year and month

**Example:**
```python
from specparser.amt.asset_straddle_tickers import make_fut_ticker

ticker = make_fut_ticker(
    fut_code="CL",
    fut_month_map="GHJKMNQUVXZF",
    min_year_offset=0,
    market_code="Comdty",
    qualifier="",
    year=2024,
    month=6
)
# → "CLN2024 Comdty"
```

### Cache Management

```python
from specparser.amt.asset_straddle_tickers import (
    set_memoize_enabled,  # Enable/disable caching
    clear_ticker_caches,  # Clear the ticker cache
)

# Disable caching for debugging
set_memoize_enabled(False)

# Clear cache after AMT file changes
clear_ticker_caches()
```

## CLI Usage

```bash
# Get tickers for a single asset
uv run python -m specparser.amt.asset_straddle_tickers data/amt.yml \
    --get "CL Comdty" --ym 2024-06 --ntrc N

# Get tickers using far-month vol
uv run python -m specparser.amt.asset_straddle_tickers data/amt.yml \
    --get "CL Comdty" --ym 2024-06 --ntrc F

# Find all unique tickers for commodities in 2024
uv run python -m specparser.amt.asset_straddle_tickers data/amt.yml \
    --find "Comdty$" --start 2024-01 --end 2024-12

# Find tickers for all live assets in H1 2024
uv run python -m specparser.amt.asset_straddle_tickers data/amt.yml \
    --find "." --start 2024-01 --end 2024-06
```

## Output Columns

| Column | Description |
|--------|-------------|
| `name` | Ticker role: `vol`, `hedge`, `hedge1`, `hedge2`, `hedge3`, `hedge4` |
| `ticker` | The ticker symbol to query |
| `field` | The data field to fetch (or empty for calculated tickers) |

## Caching Behavior

The module caches results based on a smart cache key:
- For assets with date-dependent tickers (futures, split tickers, LME vol), the cache key includes the year-month and entry code
- For assets with static tickers, the cache key is just the asset name

This optimizes performance while ensuring correctness for date-varying tickers.
