# Plan: Update Documentation After tickers.py Split

## Overview

Update all documentation to reflect the new module structure:
- `tickers.py` → pure ticker extraction
- `prices.py` → price data access
- `valuation.py` → actions & valuation

Also create new notebooks for the new modules.

---

## Module Dependency Clarification

**Corrected dependency graph:**

```
                  loader
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
 schedules ────► tickers      table.py
      │             │
      └──────┬──────┘
             │
             ▼
         prices.py
             │
             ▼
       valuation.py
```

**Key facts:**
- `schedules.py` imports `loader` (for `get_asset`, `table_column`, etc.)
- `tickers.py` imports `loader`, `schedules`, `chain`
- `prices.py` imports `tickers`, `schedules`, `chain`
- `valuation.py` imports `prices`, `schedules`, `loader`
- `table.py` is a utility module used by many others

---

## Documentation Files to Update

### 1. `docs/modules.md` (HIGH PRIORITY)

Update the AMT module section to reflect the split:

**Changes:**
- Add `prices.py` and `valuation.py` to package structure diagram
- Update dependency graph
- Update key exports list (organize by module)
- Update function descriptions

**Current:**
```
├── amt/
│   ├── loader.py        # Loading, caching, asset queries
│   ├── tickers.py       # Ticker extraction and expansion
│   ├── schedules.py     # Schedule expansion and straddle building
```

**New:**
```
├── amt/
│   ├── loader.py        # Loading, caching, asset queries
│   ├── schedules.py     # Schedule expansion and straddle building
│   ├── tickers.py       # Ticker extraction and transformation (pure)
│   ├── prices.py        # Price fetching, caching, DB access
│   ├── valuation.py     # Actions, models, PnL calculations
│   ├── table.py         # Table manipulation utilities
│   ├── chain.py         # Futures ticker normalization
```

### 2. `docs/amt.md` (HIGH PRIORITY)

Update the AMT reference:

**Changes:**
- Add sections for prices.py and valuation.py functions
- Reorganize exports by module
- Update CLI documentation

### 3. `docs/functions.md` (LOW PRIORITY)

This file documents DSL functions, not AMT. No changes needed.

### 4. `docs/tickers_split_plan.md`

This is the plan file - mark as **COMPLETED** or archive.

---

## Notebooks to Update (ALL)

All existing notebooks need review and standardization:
- Must have **Table of Contents** (markdown cell with section links)
- Must have **Benchmarks** section with timing tests
- Must use `show_table` from `specparser.amt` (not local definitions)

### 1. `notebooks/tickers.ipynb` (HIGH PRIORITY)

The notebook currently demonstrates functions that are now split across modules.

**Current Issues:**
- Has local `show_table()` definition (line ~cell 3) - needs to use import
- References `get_prices`, `actions`, `get_straddle_actions`, `get_straddle_valuation` which are now in `prices`/`valuation` modules
- Needs TOC update to reflect module split

**Changes:**
- Remove local `show_table()` definition, import from `specparser.amt`
- Update imports to use new module locations
- Update Section 8 ("Getting Prices for Straddles") to note that `get_prices` is now in `prices` module
- Note that `actions`, `get_straddle_actions`, `get_straddle_valuation` are now in `valuation` module
- Update Summary section to organize functions by module
- Add note about module split in introduction

### 2. `notebooks/schedules.ipynb` (LOW PRIORITY - already good)

**Current Status:** ✓ Good
- Uses `show_table` from `specparser.amt` correctly
- Has Table of Contents
- Has Benchmarks section

**Minor Changes:**
- Add note about loader dependency in introduction

### 3. `notebooks/loader.ipynb` (LOW PRIORITY - already good)

**Current Status:** ✓ Good
- Uses `show_table` from `specparser.amt` correctly

**Changes:**
- Verify has TOC (add if missing)
- Verify has Benchmarks section (add if missing)

### 4. `notebooks/table.ipynb` (LOW PRIORITY - already good)

**Current Status:** ✓ Good
- Uses `show_table` from `specparser.amt` correctly
- Has Table of Contents
- Has Benchmarks section

**Changes:**
- No changes needed

### 5. `notebooks/asset_straddle_tickers.ipynb` (MEDIUM PRIORITY)

**Current Status:** Partial
- Uses `show_table` from `specparser.amt` correctly
- Has Benchmarks section

**Changes:**
- Add Table of Contents if missing
- Update any references to functions now in `prices`/`valuation` modules

### 6. `notebooks/strings.ipynb` (LOW PRIORITY)

**Current Status:** Review needed

**Changes:**
- Verify uses `show_table` from `specparser.amt` (add if needed)
- Add Table of Contents if missing
- Add Benchmarks section if missing

### 7. `notebooks/numpy_strings.ipynb` (LOW PRIORITY)

**Current Status:** Review needed

**Changes:**
- Verify uses `show_table` from `specparser.amt` (add if needed)
- Add Table of Contents if missing
- Add Benchmarks section if missing

### 8. `notebooks/string_benchmarks.ipynb` (LOW PRIORITY)

**Current Status:** Review needed (likely has benchmarks already)

**Changes:**
- Verify uses `show_table` from `specparser.amt` (add if needed)
- Add Table of Contents if missing

### 9. `notebooks/test.ipynb` (SKIP or CLEAN UP)

**Current Status:** Scratch notebook with local function definitions

**Options:**
- Option A: Delete (if truly temporary)
- Option B: Clean up and standardize
- Recommend: Leave as-is (scratch notebooks don't need standardization)

---

## New Notebooks to Create

### 1. `notebooks/prices.ipynb` (NEW)

**Structure:**
```markdown
# Prices Module

This notebook demonstrates the price data access utilities from `specparser.amt.prices`.

## Table of Contents
1. Setup
2. Loading Prices (`load_all_prices`)
3. Price Dict Access (`get_price`, `set_prices_dict`)
4. Query-Based Access (`prices_last`, `prices_query`)
5. DuckDB Connection Management
6. Getting Prices for Straddles (`get_prices`)
7. Performance Benchmarks
8. Summary

## Functions Covered
- `load_all_prices(prices_parquet, start_date, end_date)`
- `set_prices_dict(prices_dict)`
- `get_price(prices_dict, ticker, field, date_str)`
- `clear_prices_dict()`
- `clear_prices_connection_cache()`
- `prices_last(prices_parquet, pattern)`
- `prices_query(prices_parquet, sql)`
- `get_prices(underlying, year, month, i, path, chain_csv, prices_parquet)`
```

### 2. `notebooks/valuation.ipynb` (NEW)

**Structure:**
```markdown
# Valuation Module

This notebook demonstrates the valuation utilities from `specparser.amt.valuation`.

## Table of Contents
1. Setup
2. Understanding the Valuation Pipeline
3. Actions (`actions`, `get_straddle_actions`)
4. Valuation Models (`model_ES`, `model_NS`, `model_BS`)
5. Override Expiry Handling
6. Strike Columns and Entry/Expiry Triggers
7. Full Valuation (`get_straddle_valuation`)
8. PnL Calculations
9. Performance Benchmarks
10. Summary

## Functions Covered
- `actions(prices_table, path, overrides_csv)`
- `get_straddle_actions(underlying, year, month, i, ...)`
- `get_straddle_valuation(underlying, year, month, i, ...)`
- `clear_override_cache()`
- `model_ES(row)`, `model_NS(row)`, `model_BS(row)`
- `MODEL_DISPATCH`
```

---

## Notebook Requirements

All notebooks should have:

1. **Table of Contents** at the top (markdown cell with links)
2. **Setup cell** with imports
3. **Use `show_table()` from `specparser.amt`** for displaying tables
4. **Benchmark section** at the end with timing tests
5. **Summary section** with function reference table

### Standard Setup Cell Template

```python
# Setup
import pandas as pd
from specparser.amt import (
    # Table display
    show_table,
    table_to_rows,
    # ... module-specific imports
)

# Data paths
AMT_PATH = "../data/amt.yml"
PRICES_PATH = "../data/prices.parquet"
CHAIN_PATH = None  # "../data/futs.csv"
OVERRIDE_PATH = "../data/overrides.csv"
```

### Standard Benchmark Template

```python
import timeit

# Warmup
_ = function_to_test(args)

# Benchmark
times = timeit.repeat(
    lambda: function_to_test(args),
    repeat=3,
    number=1
)

print(f"Function benchmark:")
print(f"  Min: {min(times)*1000:.1f}ms")
print(f"  Avg: {sum(times)/len(times)*1000:.1f}ms")
```

---

## Implementation Steps

### Step 1: Update `docs/modules.md`
1. Update package structure diagram
2. Add prices.py and valuation.py descriptions
3. Update AMT key exports (organize by module)
4. Update dependency graph

### Step 2: Update `docs/amt.md`
1. Add "Price Functions" section
2. Add "Valuation Functions" section
3. Reorganize existing sections
4. Update CLI documentation

### Step 3: Update `notebooks/tickers.ipynb` (HIGH PRIORITY)
1. Remove local `show_table()` definition
2. Add `show_table` to imports from `specparser.amt`
3. Update Section 8 with module note (prices/valuation)
4. Update Summary section (organize by module)
5. Ensure TOC is present and complete
6. Ensure Benchmarks section is present

### Step 4: Review and update remaining notebooks
For each notebook (`schedules.ipynb`, `loader.ipynb`, `table.ipynb`, `asset_straddle_tickers.ipynb`, `strings.ipynb`, `numpy_strings.ipynb`, `string_benchmarks.ipynb`):
1. Check for Table of Contents - add if missing
2. Check for Benchmarks section - add if missing
3. Check `show_table` usage - ensure imported from `specparser.amt`
4. Update any stale imports/references from module split

### Step 5: Create `notebooks/prices.ipynb`
1. Create notebook structure
2. Add table of contents
3. Add all sections with code examples
4. Add benchmark section
5. Add summary

### Step 6: Create `notebooks/valuation.ipynb`
1. Create notebook structure
2. Add table of contents
3. Add all sections with code examples
4. Add benchmark section
5. Add summary

### Step 7: Verify all notebooks run
```bash
# Execute all notebooks to verify they work
uv run jupyter nbconvert --execute --inplace notebooks/tickers.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/schedules.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/loader.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/table.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/asset_straddle_tickers.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/strings.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/numpy_strings.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/string_benchmarks.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/prices.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/valuation.ipynb
```

---

## Files Summary

| File | Action | Priority |
|------|--------|----------|
| `docs/modules.md` | Update | HIGH |
| `docs/amt.md` | Update | HIGH |
| `docs/tickers_split_plan.md` | Mark completed | LOW |
| `notebooks/tickers.ipynb` | Update (local show_table, imports) | HIGH |
| `notebooks/asset_straddle_tickers.ipynb` | Review/update (TOC) | MEDIUM |
| `notebooks/schedules.ipynb` | Review (add loader note) | LOW |
| `notebooks/loader.ipynb` | Review (TOC, benchmarks) | LOW |
| `notebooks/table.ipynb` | Review (likely no changes) | LOW |
| `notebooks/strings.ipynb` | Review (TOC, benchmarks) | LOW |
| `notebooks/numpy_strings.ipynb` | Review (TOC, benchmarks) | LOW |
| `notebooks/string_benchmarks.ipynb` | Review (TOC) | LOW |
| `notebooks/test.ipynb` | Skip (scratch notebook) | SKIP |
| `notebooks/prices.ipynb` | Create | HIGH |
| `notebooks/valuation.ipynb` | Create | HIGH |

---

## Verification

```bash
# Check all notebooks execute
uv run pytest notebooks/ --nbmake -v

# Or manually run each:
uv run jupyter execute notebooks/tickers.ipynb
uv run jupyter execute notebooks/schedules.ipynb
uv run jupyter execute notebooks/loader.ipynb
uv run jupyter execute notebooks/table.ipynb
uv run jupyter execute notebooks/asset_straddle_tickers.ipynb
uv run jupyter execute notebooks/strings.ipynb
uv run jupyter execute notebooks/numpy_strings.ipynb
uv run jupyter execute notebooks/string_benchmarks.ipynb
uv run jupyter execute notebooks/prices.ipynb
uv run jupyter execute notebooks/valuation.ipynb
```

---

## Notebook Status Summary

| Notebook | TOC | Benchmarks | show_table | Module Split Impact |
|----------|-----|------------|------------|---------------------|
| `tickers.ipynb` | ✓ | ✓ | LOCAL (fix) | HIGH (prices/valuation refs) |
| `schedules.ipynb` | ✓ | ✓ | ✓ import | None |
| `loader.ipynb` | ? | ? | ✓ import | None |
| `table.ipynb` | ✓ | ✓ | ✓ import | None |
| `asset_straddle_tickers.ipynb` | ? | ✓ | ✓ import | Check refs |
| `strings.ipynb` | ? | ? | ? | None |
| `numpy_strings.ipynb` | ? | ? | ? | None |
| `string_benchmarks.ipynb` | ? | ✓ | ? | None |
| `test.ipynb` | - | - | LOCAL | Skip (scratch) |

Legend: ✓ = present, ? = needs verification, LOCAL = needs fix
