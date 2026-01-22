# Plan: loader.ipynb - Interactive Examples for AMT Loader Functions

## Goals

Create an interactive Jupyter notebook demonstrating all public functions from `specparser.amt.loader`. The notebook should be:
- **Self-contained**: Create test AMT data within the notebook (no external files needed)
- **Educational**: Clear explanations of what each function does
- **Interactive**: Users can modify examples and see results
- **Visual**: Use pandas DataFrames for pretty table display

---

## Notebook Structure

### 1. Introduction & Setup
- Brief overview of AMT file format (YAML with `amt` section for assets)
- Import statements
- Helper function to display tables as pandas DataFrames
- **Create temporary test AMT file** for all examples

### 2. File Loading & Caching
- **`load_amt(path)`** - Load and cache YAML file
- **`clear_cache()`** - Clear the cache

### 3. Value Access
- **`get_value(path, key_path, default)`** - Get value by dot-separated path
- **`get_aum(path)`** - Get AUM value
- **`get_leverage(path)`** - Get leverage value

### 4. Asset Queries
- **`get_asset(path, underlying)`** - Get single asset by Underlying
- **`find_assets(path, pattern, live_only)`** - Find assets by regex pattern
- **`assets(path, live_only, pattern)`** - List all assets
- **`cached_assets(path)`** - List assets from cache
- **`_iter_assets(path, live_only, pattern)`** - Iterator over assets

### 5. Asset Classification
- **`asset_class(path, live_only, pattern)`** - Assets with class/source info

### 6. Embedded Tables
- **`get_table(path, key_path)`** - Get embedded table from AMT

### 7. Rule Matching
- **`_compile_rules(table)`** - Compile regex rules from table
- **`_match_rules(rules, field_values, default)`** - Match against rules
- **`asset_table(path, table_name, default, live_only, pattern)`** - Evaluate rule table against assets

### 8. Asset Grouping
- **`asset_group(path, live_only, pattern)`** - Assets with group/subgroup/liquidity

### 9. Cleanup
- Remove temporary test file

---

## Test AMT Data

Create a comprehensive test file covering all features:

```python
test_amt_content = """
backtest:
  aum: 1000000.0
  leverage: 2.5

amt:
  Apple:
    Underlying: "AAPL US Equity"
    Class: "Equity"
    WeightCap: 0.10
    Vol:
      Source: "BBG"
    Hedge:
      Source: "BBG"
    Valuation:
      Model: "BS"

  Google:
    Underlying: "GOOGL US Equity"
    Class: "Equity"
    WeightCap: 0.08
    Vol:
      Source: "BBG"
    Hedge:
      Source: "BBG"
    Valuation:
      Model: "BS"

  CrudeOil:
    Underlying: "CL1 Comdty"
    Class: "Commodity"
    WeightCap: 0.05
    Vol:
      Source: "Internal"
    Hedge:
      Source: "BBG"
    Valuation:
      Model: "Bachelier"

  Gold:
    Underlying: "GC1 Comdty"
    Class: "Commodity"
    WeightCap: 0.0  # Not live (WeightCap = 0)
    Vol:
      Source: "BBG"
    Hedge:
      Source: "BBG"

  TenYear:
    Underlying: "TY1 Comdty"
    Class: "Rate"
    WeightCap: 0.15
    Vol:
      Source: "Internal"
    Hedge:
      Source: "Internal"
    Valuation:
      Model: "Bachelier"

group_table:
  Columns: [field, rgx, value]
  Rows:
    - [Class, "^Equity$", "equities"]
    - [Class, "^Commodity$", "commodities"]
    - [Class, "^Rate$", "rates"]
    - [Class, ".*", "other"]

subgroup_table:
  Columns: [field, rgx, value]
  Rows:
    - [Underlying, ".*Equity$", "stocks"]
    - [Underlying, ".*Comdty$", "futures"]
    - [Underlying, ".*", ""]

liquidity_table:
  Columns: [field, rgx, value]
  Rows:
    - [Class, "^Equity$", "high"]
    - [Class, "^Commodity$", "medium"]
    - [Class, ".*", "low"]

limit_overrides:
  Columns: [field, rgx, value]
  Rows:
    - [Underlying, "^CL1", "0.03"]
    - [Underlying, ".*", ""]
"""
```

This test data includes:
- 5 assets (4 live, 1 not live)
- 3 asset classes (Equity, Commodity, Rate)
- Various Vol/Hedge sources
- 4 rule tables for grouping

---

## Cell Structure for Each Function

Each function section:
1. **Markdown cell**: Function signature, description, parameters
2. **Code cell**: Call function with example
3. **Code cell**: Display result (show() for tables, print for values)
4. **Markdown cell** (optional): Notes, edge cases

---

## Sections Detail

### 1. Introduction (~4 cells)
- Markdown: Overview of AMT format
- Code: Imports + show() helper
- Code: Create temp file with test data
- Code: Show the test data structure

### 2. Loading (~4 cells)
- `load_amt()` - load and show keys
- `clear_cache()` - demonstrate caching behavior

### 3. Value Access (~5 cells)
- `get_value()` - get backtest.aum, nested paths
- `get_value()` - with default for missing key
- `get_aum()` and `get_leverage()` - convenience functions

### 4. Asset Queries (~8 cells)
- `get_asset()` - get single asset, show structure
- `get_asset()` - not found case
- `find_assets()` - regex pattern matching
- `find_assets()` - with live_only=True
- `assets()` - all assets
- `assets()` - live only
- `cached_assets()` - from cache
- `_iter_assets()` - show iterator usage

### 5. Asset Classification (~3 cells)
- `asset_class()` - all assets with class info
- `asset_class()` - live only

### 6. Embedded Tables (~4 cells)
- `get_table()` - get group_table
- `get_table()` - error case (not found)
- Show table structure with Columns/Rows

### 7. Rule Matching (~5 cells)
- `_compile_rules()` - show compiled rules
- `_match_rules()` - match examples
- `asset_table()` - evaluate group_table
- `asset_table()` - evaluate limit_overrides

### 8. Asset Grouping (~3 cells)
- `asset_group()` - combined grouping
- Show result with all columns

### 9. Cleanup (~2 cells)
- Remove temp file
- Clear cache

---

## Estimated Size

- ~40-45 cells total
- Notebook should execute in <1 second
- No external dependencies beyond pandas, yaml, tempfile

---

## Functions to Cover

| Function | Public | Section |
|----------|--------|---------|
| `load_amt` | Yes | Loading |
| `clear_cache` | Yes | Loading |
| `get_value` | Yes | Value Access |
| `get_aum` | Yes | Value Access |
| `get_leverage` | Yes | Value Access |
| `get_asset` | Yes | Asset Queries |
| `find_assets` | Yes | Asset Queries |
| `assets` | Yes | Asset Queries |
| `cached_assets` | Yes | Asset Queries |
| `_iter_assets` | Semi-public | Asset Queries |
| `asset_class` | Yes | Classification |
| `get_table` | Yes | Embedded Tables |
| `_compile_rules` | Semi-public | Rule Matching |
| `_match_rules` | Semi-public | Rule Matching |
| `asset_table` | Yes | Rule Matching |
| `asset_group` | Yes | Grouping |

**Note:** Table utility functions (table_column, table_join, etc.) are covered in table.ipynb, not here.

---

## Error Cases to Show

1. `get_asset()` - asset not found returns None
2. `get_value()` - key not found returns default
3. `get_table()` - key path not found raises ValueError
4. `find_assets()` - no matches returns empty table

---

## Files to Create

1. **loader.ipynb** - The notebook in project root
