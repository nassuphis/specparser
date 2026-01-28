# Plan: Refactor `tickers.py` to Use `asset_straddle_tickers.py` as Core

## Overview

Simplify `tickers.py` by removing the intermediate "ticker schema" (tschema) layer and having it call functions from `asset_straddle_tickers.py` instead. The `asset_straddle_tickers.py` module remains the single source of truth for ticker computation.

Also fix indirect imports: import table functions directly from `table.py`, not through `loader`.

## Current State Analysis

### Problem with `tickers.py` (Schema Approach)
- Uses a 7-column intermediate schema: `['asset', 'cls', 'type', 'param', 'source', 'ticker', 'field']`
- Complex pipeline: `get_tschemas()` → `get_tickers_ym()` → `filter_tickers()`
- Convoluted expansion logic for futures (`BBGfc`) and split tickers
- Separate date constraint parsing (`_parse_date_constraint`)
- Two cache systems: `_TSCHEMAS_CACHE` and `_TICKERS_YM_CACHE`
- Uses indirect imports: `loader.table_column()` instead of `table.table_column()`

### Advantages of `asset_straddle_tickers.py` (Direct Approach)
- Simple 3-column output: `['name', 'ticker', 'field']`
- Direct computation from asset data without intermediate structures
- **Smart caching via `asset_straddle_ticker_key()`** - only includes `strym|ntrc` in key when actually time-dependent:
  ```
  Time-dependent (key includes ym+ntrc):
  - BBG_LMEVOL vol source (futures-based vol)
  - fut hedge source (futures)
  - nonfut with split ticker (contains ":")
  - cds with split tickers

  Time-independent (key is just asset):
  - Everything else -> computed once, reused for all months
  ```
- Clean handling of all source types in one place

## Refactoring Strategy

**Keep `asset_straddle_tickers.py` as the core. Make `tickers.py` call into it.**

- `asset_straddle_tickers.py` stays intact (and grows if needed)
- `tickers.py` loses the tschema machinery, calls `get_asset_straddle_tickers()` instead
- Front-facing API (`filter_tickers()`) signature unchanged
- Fix indirect imports: use `from .table import ...` directly

---

## Implementation Plan

### Step 1: Fix indirect imports in `tickers.py`

Change:
```python
from . import loader
# ... later ...
loader.table_column(...)
loader.table_drop_columns(...)
loader.table_add_column(...)
loader.table_unique_rows(...)
loader.bind_rows(...)
```

To:
```python
from . import loader
from .table import table_column, table_drop_columns, table_add_column, table_unique_rows, table_bind_rows
# ... later ...
table_column(...)
table_drop_columns(...)
table_add_column(...)
table_unique_rows(...)
table_bind_rows(...)
```

Current indirect usages in `tickers.py`:
- Line 264: `loader.table_column(assets_table, "asset")`
- Line 433: `loader.table_unique_rows(loader.bind_rows(tables,table))`
- Line 532: `loader.table_column(...)`
- Line 544: `loader.table_drop_columns(...)`
- Line 546: `loader.table_add_column(...)`

### Step 2: Simplify `filter_tickers()` in `tickers.py`

Replace the current implementation that goes through `get_tickers_ym()` with a call to `asset_straddle_tickers.get_asset_straddle_tickers()`:

```python
from .table import table_column

def filter_tickers(
    asset: str,
    year: int,
    month: int,
    i: int,
    amt_path: str | Path,
    chain_path: str | Path | None = None,
) -> dict[str, Any]:
    """Get tickers for an asset's straddle."""

    # Validate asset exists
    if loader.get_asset(amt_path, asset) is None:
        raise ValueError(f"Asset '{asset}' not found")

    # Get straddle info
    straddles = table_column(schedules.get_expand_ym(amt_path, asset, year, month), "straddle")
    if len(straddles) < 1:
        raise ValueError(f"'{asset}' has no straddles in {year}-{month:02d}")
    straddle = straddles[i % len(straddles)]

    xpry, xprm = schedules.xpry(straddle), schedules.xprm(straddle)
    ntrc = schedules.ntrc(straddle)
    strym = f"{xpry}-{xprm:02d}"

    # Call into asset_straddle_tickers (the source of truth)
    ticker_table = asset_straddle_tickers.get_asset_straddle_tickers(asset, strym, ntrc, amt_path)

    # Transform output to expected format
    # Input columns: ['name', 'ticker', 'field']
    # Output columns: ['asset', 'straddle', 'param', 'source', 'ticker', 'field']
    result = {
        "orientation": "row",
        "columns": ["asset", "straddle", "param", "source", "ticker", "field"],
        "rows": []
    }

    for row in ticker_table["rows"]:
        name, ticker, field = row
        source = _infer_source(field)
        result["rows"].append([asset, straddle, name, source, ticker, field])

    return result


def _infer_source(field: str) -> str:
    """Infer source from field pattern."""
    if field == "none":  # CV vol
        return "CV"
    if field == "":  # calc tickers
        return "calc"
    return "BBG"  # Default for everything with a field
```

### Step 3: Remove deprecated code from `tickers.py`

Delete the tschema machinery that is no longer needed:

| Function/Variable | Status |
|-------------------|--------|
| `_TSCHEMAS_CACHE` | Delete |
| `_TICKERS_YM_CACHE` | Delete |
| `get_tschemas()` | Delete |
| `find_tschemas()` | Delete |
| `get_tickers_ym()` | Delete |
| `find_tickers_ym()` | Delete |
| `find_tickers()` | Delete |
| `_split_ticker()` | Delete (exists in `asset_straddle_tickers.py`) |
| `_parse_date_constraint()` | Delete |
| `fut_spec2ticker()` | Delete (use `make_fut_ticker` from `asset_straddle_tickers`) |
| `_market_tickers()` | Delete |
| `_vol_tickers()` | Delete |
| `_hedge_*()` functions | Delete |
| `_HEDGE_HANDLERS` | Delete |
| `_TSCHEMA_COLUMNS` | Delete |
| `_tschema_dict_*()` functions | Delete |
| `_filter_straddle_tickers()` | Delete |

### Step 4: Keep these in `tickers.py`

| Function | Reason |
|----------|--------|
| `filter_tickers()` | Main API used by `prices.py` |
| `clear_ticker_caches()` | Delegates to `asset_straddle_tickers.clear_ticker_caches()` |
| `set_memoize_enabled()` | Delegates to `asset_straddle_tickers.set_memoize_enabled()` |
| CLI (`_main()`) | Entry point - simplify to use new approach |

### Step 5: Update `__init__.py` exports

Remove tschema-related exports:
- ~~`get_tschemas`~~
- ~~`find_tschemas`~~
- ~~`get_tickers_ym`~~
- ~~`find_tickers_ym`~~
- ~~`find_tickers`~~

Keep:
- `filter_tickers`

Add (from `asset_straddle_tickers`):
- `get_asset_straddle_tickers`
- `find_assets_straddles_tickers`

### Step 6: Update CLI in `tickers.py`

Simplify CLI options:
- `--asset-tickers` stays (calls `filter_tickers`)
- Remove `--asset-tschemas`, `--find-tschemas`, `--find-tickers`, `--find-tickers-ym`, `--get-tickers-ym`
- Add `--straddle-tickers` to directly call `get_asset_straddle_tickers`

---

## Files to Modify

| File | Changes |
|------|---------|
| [tickers.py](src/specparser/amt/tickers.py) | Fix indirect imports, remove tschema machinery, simplify `filter_tickers()` to call `asset_straddle_tickers` |
| [asset_straddle_tickers.py](src/specparser/amt/asset_straddle_tickers.py) | **Keep as-is** (source of truth) |
| [__init__.py](src/specparser/amt/__init__.py) | Update exports |

---

## Output Column Mapping

`get_asset_straddle_tickers()` output:
```
['name', 'ticker', 'field']
```

`filter_tickers()` output (unchanged):
```
['asset', 'straddle', 'param', 'source', 'ticker', 'field']
```

Mapping in `filter_tickers()`:
- `name` → `param`
- `ticker` → `ticker`
- `field` → `field`
- `source` → inferred from `field` pattern
- `asset`, `straddle` → added by wrapper

---

## Key Benefits

1. **Single source of truth** - `asset_straddle_tickers.py` owns ticker computation
2. **Smarter caching** - `asset_straddle_ticker_key()` avoids recomputing time-independent tickers
3. **Simpler `tickers.py`** - ~400 lines reduced to ~100 lines
4. **No intermediate schema** - Direct path from asset → tickers
5. **Preserved API** - `filter_tickers()` signature unchanged, `prices.py` unaffected
6. **Direct imports** - Import from source module, not through re-exports

---

## Verification

```bash
# 1. Run existing tests
uv run pytest tests/ -v -k ticker

# 2. Test filter_tickers output matches expected format
uv run python -m specparser.amt.tickers data/amt.yml --asset-tickers "CL Comdty" 2024 6 0

# 3. Test prices integration (uses filter_tickers)
uv run python -c "
from specparser.amt import tickers
t = tickers.filter_tickers('CL Comdty', 2024, 6, 0, 'data/amt.yml')
print('Columns:', t['columns'])
for row in t['rows']:
    print(row)
"

# 4. Compare old vs new output (before deleting old code)
uv run python -c "
from specparser.amt import tickers
from specparser.amt import asset_straddle_tickers as ast

# Old way
old = tickers.filter_tickers('CL Comdty', 2024, 6, 0, 'data/amt.yml')

# New way (direct)
new = ast.get_asset_straddle_tickers('CL Comdty', '2024-06', 'N', 'data/amt.yml')

print('Old:', old)
print('New:', new)
"
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| CLI commands break | Keep working commands, remove only tschema-specific ones |
| Missing edge cases | Compare output of old vs new before deleting old code |
| Cache behavior change | `asset_straddle_tickers` caching is preserved exactly |
