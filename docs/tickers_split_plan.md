# Plan: Split tickers.py into prices.py and valuation.py

## Overview

Split the monolithic `tickers.py` (2000 lines) into focused modules:

| Module | Purpose | Lines |
|--------|---------|-------|
| `tickers.py` | Ticker extraction & transformation | ~550 |
| `prices.py` | Price fetching, caching, DB access | ~400 |
| `valuation.py` | Actions, models, PnL calculations | ~650 |

## Design Principle

**Actions belong with valuation.** The action computation (entry/expiry triggers) is fundamentally part of the valuation pipeline:

```
tickers.py          prices.py              valuation.py
     │                   │                      │
filter_tickers() ──────► get_prices() ────────► actions()
                              │                      │
                              │                      ▼
                              │              get_straddle_actions()
                              │                      │
                              │                      ▼
                              │              get_straddle_valuation()
```

---

## Module 1: `prices.py` (Price Data Access)

All functions related to fetching, caching, and looking up price data.

### Functions to Move

```python
# Global caches
_PRICES_DICT: dict[str, str] | None = None
_DUCKDB_CACHE: dict[str, "duckdb.DuckDBPyConnection"] = {}

# DuckDB connection management
def _get_prices_connection(prices_parquet)                   # line 841
def _clear_prices_cache()                                    # line 861
def clear_prices_connection_cache()                          # line 871

# Prices dict cache management
def load_all_prices(prices_parquet, start_date, end_date)    # line 770
def set_prices_dict(prices_dict)                             # line 813
def get_price(prices_dict, ticker, field, date_str)          # line 819
def clear_prices_dict()                                      # line 828

# Query-based price access
def prices_last(prices_parquet, pattern) -> dict             # line 652
def prices_query(prices_parquet, sql) -> dict                # line 680

# Straddle price building (used by valuation)
def _build_ticker_map(ticker_table, chain_csv)               # line 1212
def _lookup_straddle_prices(dates, ticker_map, prices_parquet) # line 1254
def _build_prices_table(asset, straddle, dates, params, ticker_map, prices) # line 1318

# High-level price fetching (composes tickers + price lookup)
def get_prices(underlying, year, month, i, path, chain_csv, prices_parquet) # line 1519
```

### Dependencies
- `duckdb`
- `chain.fut_act2norm()` (for ticker normalization)
- `tickers.filter_tickers()` (for get_prices)
- `schedules.straddle_days()` (for get_prices)

---

## Module 2: `valuation.py` (Actions & Valuation)

All functions related to computing actions (entry/expiry triggers), pricing models, and PnL.

### Functions to Move

```python
# Global caches
_OVERRIDE_CACHE: dict[tuple[str, str], str] | None = None

# Math helpers
def _norm_cdf(x)                                             # line 560

# Pricing models
def model_ES(row) -> dict                                    # line 570
def model_NS(row) -> dict                                    # line 629
def model_BS(row) -> dict                                    # line 634
def model_default(row) -> dict                               # line 639
MODEL_DISPATCH = {...}                                       # line 644

# Override expiry handling
def _load_overrides(path)                                    # line 888
def clear_override_cache()                                   # line 917
def _override_expiry(underlying, year, month, overrides_path) # line 927

# Date/anchor computation helpers
def _add_calendar_days(date_str, days)                       # line 703
def _last_good_day_in_month(rows, vol_idx, hedge_indices, date_idx, year, month) # line 718
def _anchor_day(xprc, xprv, year, month, underlying, overrides_path) # line 949
def _nth_good_day_after(rows, vol_idx, hedge_indices, date_idx, anchor_date, n, month_limit) # line 1025

# Action computation
def _compute_actions(rows, columns, ntrc, ntrv, xprc, xprv, ...) # line 1104

# Table building helpers for actions/strikes
def _find_action_indices(table)                              # line 1354
def _add_action_column(table, straddle, underlying, overrides_csv) # line 1379
def _add_model_column(table, underlying, path)               # line 1417
def _add_strike_columns(table, ntry_idx, xpry_idx)           # line 1445

# Roll-forward helpers for valuation
def _get_rollforward_fields(columns)                         # line 1654

# PUBLIC APIS - Actions & Valuation
def actions(prices_table, path, overrides_csv)               # line 1568
def get_straddle_actions(underlying, year, month, i, path, ...) # line 1605
def get_straddle_valuation(underlying, year, month, i, path, ...) # line 1665
```

### Dependencies
- `math`
- `calendar`
- `csv`
- `datetime`
- `schedules` (for ntrc, ntrv, xprc, xprv, ntry, ntrm, xpry, xprm)
- `loader.get_asset()` (for model lookup)
- `prices.get_prices()` (for get_straddle_actions)

---

## Module 3: `tickers.py` (Ticker Extraction Only)

Pure ticker extraction and transformation - no prices, no valuation.

### Functions to Keep

```python
# Global caches
_TSCHEMAS_CACHE: dict[tuple[str, str], dict] = {}
_TICKERS_YM_CACHE: dict[tuple, dict] = {}
_MEMOIZE_ENABLED: bool = True

# Cache management
def set_memoize_enabled(enabled)                             # line 30
def clear_ticker_caches()                                    # line 36

# Ticker parsing helpers
def _split_ticker(ticker, param)                             # line 53
def _parse_date_constraint(param, xpry, xprm)                # line 64

# Ticker schema handlers
def _market_tickers(market, underlying, cls)                 # line 108
def _vol_tickers(vol, underlying, cls)                       # line 121
def _hedge_nonfut(hedge, underlying, cls)                    # line 139
def _hedge_cds(hedge, underlying, cls)                       # line 146
def _hedge_fut(hedge, underlying, cls)                       # line 156
def _hedge_calc(hedge, underlying, cls)                      # line 163
def _hedge_default(hedge, underlying, cls, source)           # line 177
_HEDGE_HANDLERS = {...}                                      # line 185
_TSCHEMA_COLUMNS = [...]                                     # line 193

# Ticker schema API
def get_tschemas(path, underlying)                           # line 195
def find_tschemas(path, pattern, live_only)                  # line 263

# Ticker computation
def fut_spec2ticker(spec, year, month)                       # line 276
def _tschema_dict_bbgfc_ym(tschema_dict, year, month, chain_csv) # line 305
def _tschema_dict_expand_bbgfc(tschema_dict, start_year, end_year, chain_csv) # line 331
def _tschema_dict_expand_split(ticker_dict)                  # line 345

# Ticker API
def get_tickers_ym(path, asset, year, month, chain_csv)      # line 364
def find_tickers_ym(path, pattern, live_only, year, month, chain_csv) # line 401
def find_tickers(path, pattern, live_only, start_year, end_year, chain_csv) # line 416

# Straddle ticker filtering
def _filter_straddle_tickers(rows, columns, ntrc)            # line 448
def filter_tickers(asset, year, month, i, amt_path, chain_path) # line 502

# CLI (remains here, imports from prices & valuation)
def _main()                                                  # line 1843
```

### Dependencies
- `re`
- `loader`
- `schedules`
- `chain`

---

## Import Structure After Split

### `prices.py`
```python
from pathlib import Path
from typing import Any
import duckdb

from . import chain
from . import tickers   # for filter_tickers (used by get_prices)
from . import schedules # for straddle_days (used by get_prices)
```

### `valuation.py`
```python
import math
import csv
import calendar
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from . import loader
from . import schedules
from . import prices  # for get_prices (used by get_straddle_actions)
```

### `tickers.py`
```python
import re
from pathlib import Path
from typing import Any

from . import loader
from . import schedules
from . import chain
# NO import of prices or valuation (they depend on tickers, not vice versa)
```

---

## Dependency Graph

```
                  loader, schedules, chain (base modules)
                            │
                            ▼
                       tickers.py
                    (ticker extraction)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
         prices.py                   valuation.py
    (price data access)          (actions & valuation)
              │                           │
              └───────────┬───────────────┘
                          │
                     valuation.py
                  (get_straddle_actions)
                  (get_straddle_valuation)
```

**No circular dependencies:**
- `tickers.py` depends only on base modules
- `prices.py` depends on `tickers.py` + base
- `valuation.py` depends on `prices.py` + base

---

## Public API Changes

### From `__init__.py`

**After split:**

```python
# From prices.py
"prices_last",
"prices_query",
"load_all_prices",
"set_prices_dict",
"get_price",
"clear_prices_dict",
"clear_prices_connection_cache",
"get_prices",  # NEW location

# From valuation.py
"actions",             # NEW location
"get_straddle_actions", # NEW location
"get_straddle_valuation",
"clear_override_cache",
"model_ES",
"model_NS",
"model_BS",
"MODEL_DISPATCH",

# From tickers.py
"get_tschemas",
"find_tschemas",
"get_tickers_ym",
"find_tickers_ym",
"find_tickers",
"filter_tickers",
"clear_ticker_caches",
"set_memoize_enabled",
"fut_spec2ticker",
```

---

## Migration Steps

### Step 1: Create `prices.py`
1. Create new file
2. Move caches: `_PRICES_DICT`, `_DUCKDB_CACHE`
3. Move DB functions: `_get_prices_connection`, `_clear_prices_cache`, `clear_prices_connection_cache`
4. Move dict functions: `load_all_prices`, `set_prices_dict`, `get_price`, `clear_prices_dict`
5. Move query functions: `prices_last`, `prices_query`
6. Move straddle price helpers: `_build_ticker_map`, `_lookup_straddle_prices`, `_build_prices_table`
7. Move `get_prices()` - imports `tickers.filter_tickers`, `schedules.straddle_days`

### Step 2: Create `valuation.py`
1. Create new file
2. Move cache: `_OVERRIDE_CACHE`
3. Move math: `_norm_cdf`
4. Move models: `model_ES`, `model_NS`, `model_BS`, `model_default`, `MODEL_DISPATCH`
5. Move override functions: `_load_overrides`, `clear_override_cache`, `_override_expiry`
6. Move date helpers: `_add_calendar_days`, `_last_good_day_in_month`, `_anchor_day`, `_nth_good_day_after`
7. Move action functions: `_compute_actions`, `_find_action_indices`, `_add_action_column`, `_add_model_column`, `_add_strike_columns`
8. Move rollforward: `_get_rollforward_fields`
9. Move public APIs: `actions()`, `get_straddle_actions()`, `get_straddle_valuation()`

### Step 3: Update `tickers.py`
1. Remove all moved functions
2. Keep only ticker extraction functions
3. Update CLI `_main()` to import from `prices` and `valuation`

### Step 4: Update `__init__.py`
1. Add lazy imports for `prices.py` exports
2. Add lazy imports for `valuation.py` exports
3. Update/remove tickers lazy imports

### Step 5: Run Tests
```bash
uv run pytest tests/ -v
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/specparser/amt/prices.py` | Create (~400 lines) |
| `src/specparser/amt/valuation.py` | Create (~650 lines) |
| `src/specparser/amt/tickers.py` | Modify (remove ~1000 lines) |
| `src/specparser/amt/__init__.py` | Update exports |

---

## Verification

```bash
# Run all tests
uv run pytest tests/ -v

# Quick smoke test
uv run python -c "
from specparser.amt import (
    # tickers (pure extraction)
    get_tickers_ym, filter_tickers, fut_spec2ticker,
    # prices (data access)
    load_all_prices, get_price, get_prices,
    # valuation (actions & models)
    actions, get_straddle_actions, get_straddle_valuation, model_ES
)
print('All imports work!')
"
```
