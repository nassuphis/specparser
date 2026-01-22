# AMT Module Reference

The AMT (Asset Management Table) module provides utilities for processing AMT YAML files containing asset definitions, expiry schedules, and related configuration.

## Overview

AMT files are YAML configuration files that define:
- Assets with their underlying instruments and weight caps
- Expiry schedules for options/derivatives
- Risk multiplier tables and other configuration

The module provides functions to:
1. Load and query AMT data
2. Extract and expand schedules across time periods
3. Transform expiry codes into computed values
4. Pack schedule data into compact straddle strings

---

## Core Functions

### Loading and Caching

#### `load_amt(path)`

Load an AMT YAML file with caching.

```python
from specparser.amt import load_amt

data = load_amt("data/amt.yml")
# Returns full parsed YAML as dict
# Subsequent calls return cached data
```

#### `clear_cache()`

Clear the AMT file cache.

```python
from specparser.amt import clear_cache
clear_cache()
```

---

### Value Extraction

#### `get_value(path, key_path, default=None)`

Get a value from an AMT file by its dot-separated key path.

```python
from specparser.amt import get_value

aum = get_value("data/amt.yml", "backtest.aum")
# Returns: 800.0

leverage = get_value("data/amt.yml", "backtest.leverage")
# Returns: 20.0

# With default value
window = get_value("data/amt.yml", "backtest.signal_window", default=252)
```

**Parameters:**
- `path`: Path to the AMT YAML file
- `key_path`: Dot-separated path to the value (e.g., "backtest.aum")
- `default`: Default value if the key is not found

**Returns:** The value at the key path, or default if not found

#### `get_aum(path)`

Get the AUM (Assets Under Management) value from an AMT file.

```python
from specparser.amt import get_aum

aum = get_aum("data/amt.yml")
# Returns: 800.0
```

#### `get_leverage(path)`

Get the leverage value from an AMT file.

```python
from specparser.amt import get_leverage

leverage = get_leverage("data/amt.yml")
# Returns: 20.0
```

---

### Asset Queries

#### `get_asset(path, name)`

Get asset data by its YAML key.

```python
from specparser.amt import get_asset

asset = get_asset("data/amt.yml", "LA Comdty OLD")
# Returns: {'Underlying': 'LA Comdty', 'WeightCap': 0.05, ...}
```

#### `find_assets(path, pattern)`

Find all assets with Underlying values matching a regex pattern.

```python
from specparser.amt import find_assets

table = find_assets("data/amt.yml", "^LA.*")
# Columns: ['asset']
```

#### `cached_assets(path)`

List all asset Underlying values from the cache.

```python
from specparser.amt import cached_assets

table = cached_assets("data/amt.yml")
# Columns: ['asset']
```

---

### Table Functions

Tables are returned as dicts with `columns` (list) and `rows` (list of lists).

#### `get_table(path, key_path)`

Get an embedded table from an AMT file by its key path.

```python
from specparser.amt import get_table

table = get_table("data/amt.yml", "group_risk_multiplier_table")
# Returns: {'columns': ['group', 'multiplier'], 'rows': [...]}
```

#### `format_table(table)`

Format a table as tab-separated string with header.

```python
from specparser.amt import format_table

table = {'columns': ['a', 'b'], 'rows': [[1, 2], [3, 4]]}
print(format_table(table))
# a	b
# 1	2
# 3	4
```

#### `print_table(table)`

Print a table to stdout.

```python
from specparser.amt import print_table
print_table(table)
```

---

### Asset Tables

#### `all_assets(path)`

Get all assets with their Underlying values.

```python
from specparser.amt import all_assets

table = all_assets("data/amt.yml")
# Columns: ['asset']
```

#### `live_assets(path)`

Get all live assets (WeightCap > 0).

```python
from specparser.amt import live_assets

table = live_assets("data/amt.yml")
# Columns: ['asset']
```

#### `live_class(path)`

Get all live assets with their class and source information.

```python
from specparser.amt import live_class

table = live_class("data/amt.yml")
# Columns: ['asset', 'cls', 'volsrc', 'hdgsrc', 'model']
```

**Column descriptions:**
| Column | Description |
|--------|-------------|
| `asset` | Underlying asset name |
| `cls` | Asset class (e.g., 'Commodity', 'Rate', 'FX') |
| `volsrc` | Volatility data source (from Vol.Source) |
| `hdgsrc` | Hedge instrument source (from Hedge.Source) |
| `model` | Valuation model (from Valuation.Model) |

#### `live_table(path, table_name, default="")`

Get all live assets with values from any rule table.

```python
from specparser.amt import live_table

# Get group assignments
table = live_table("data/amt.yml", "group_table")
# Columns: ['asset', 'group']

# Get limit overrides
table = live_table("data/amt.yml", "limit_overrides")
# Columns: ['asset', 'limit_overrides']

# Get liquidity factors
table = live_table("data/amt.yml", "liquidity_table")
# Columns: ['asset', 'liquidity']
```

**Parameters:**
- `path`: Path to the AMT YAML file
- `table_name`: Name of the rule table (e.g., "group_table", "limit_overrides")
- `default`: Default value if no rule matches (default: "")

**Returns:** Table with columns `['asset', '<column_name>']` where column_name is the table_name with `_table` suffix removed (if present).

#### `live_group(path)`

Get all live assets with their group, subgroup, liquidity, and limit override.

Group, subgroup, liquidity, and limit override are determined by matching each asset against the rules in the `group_table`, `subgroup_table`, `liquidity_table`, and `limit_overrides` from the AMT file. Rules are evaluated in order, and the first matching rule determines the value.

```python
from specparser.amt import live_group

table = live_group("data/amt.yml")
# Columns: ['asset', 'grp', 'sgrp', 'lqdty', 'lmtovr']
```

**Column descriptions:**
| Column | Description |
|--------|-------------|
| `asset` | Underlying asset name |
| `grp` | Group assignment (e.g., 'rates', 'equity', 'fx', 'commodity', 'stonks') |
| `sgrp` | Subgroup assignment (e.g., 'EAST', 'WEST', 'UA', 'UE', 'EE') |
| `lqdty` | Liquidity factor (e.g., 1, 0.75, 0.5) |
| `lmtovr` | Limit override value |

**Rule table structure:**

The `group_table`, `subgroup_table`, `liquidity_table`, and `limit_overrides` in the AMT YAML file define matching rules:

```yaml
group_table:
    Columns: [field, rgx, value]
    Rows:
    - [Underlying, '^(LQD).*$', 'rates']
    - [Class,      '^(Rate|Swap)$', 'rates']
    - [Class,      '^Equity$', 'equity']
    - [Class,      '^Currency$', 'fx']
    - [Class,      '^Commodity$', 'commodity']
    - [Class,      '^SingleStock$', 'stonks']
    - [Underlying, '^.*$', 'error']

subgroup_table:
    Columns: [field, rgx, value]
    Rows:
    - [Underlying, '^AS51 Index.*$', 'EAST']
    - [Underlying, '^SPY US Equity.*$', 'WEST']
    # ...

liquidity_table:
    Columns: [field, rgx, value]
    Rows:
    - [Underlying, '^(USDINR|USDIDR).*$', 0.75]
    - [Underlying, '^(EMB|HYG).*$', 0.5]
    - [Underlying, '^.*$', 1]

limit_overrides:
    Columns: [field, rgx, value]
    Rows:
    - [Class, '^SingleStock$', 7.5]
    - [Underlying, '^CL Comdty.*$', 623.0]
    # ...
```

Each rule specifies:
- `field`: Which asset field to check ('Underlying' or 'Class')
- `rgx`: Regex pattern to match against the field value
- `value`: Value to assign if the pattern matches

---

### Ticker Functions

#### `get_tschemas(path, underlying)`

Get all tickers for an asset by its Underlying value.

```python
from specparser.amt import get_tschemas

table = get_tschemas("data/amt.yml", "LA Comdty")
# Columns: ['asset', 'cls', 'type', 'param', 'source', 'ticker', 'field']
```

**Column descriptions:**
| Column | Description |
|--------|-------------|
| `asset` | Underlying asset name |
| `cls` | Asset class (e.g., 'Commodity', 'Rate', 'FX') |
| `type` | Ticker type: 'Market', 'Vol', or 'Hedge' |
| `param` | Parameter (e.g., 'Near', 'Far', 'hedge', 'calc') |
| `source` | Data source ('BBG', 'BBGfc', 'calc', etc.) |
| `ticker` | The ticker symbol or spec string |
| `field` | Bloomberg field (e.g., 'PX_LAST', 'Near', 'Far') |

#### `live_tickers(path, start_year=None, end_year=None, chain_csv=None)`

Get all tickers for all live assets.

```python
from specparser.amt import live_tickers

# Basic usage - no expansion
table = live_tickers("data/amt.yml")

# With BBGfc expansion to monthly tickers
table = live_tickers("data/amt.yml", 2024, 2025)

# With normalized to actual ticker lookup
table = live_tickers("data/amt.yml", 2024, 2025, "data/current_bbg_chain_data.csv")
```

**Parameters:**
- `path`: Path to the AMT YAML file
- `start_year`: Optional start year for BBGfc expansion
- `end_year`: Optional end year for BBGfc expansion
- `chain_csv`: Optional path to CSV with `normalized_future,actual_future` columns

When `start_year` and `end_year` are provided, BBGfc rows are expanded into monthly tickers. When `chain_csv` is provided, normalized tickers are converted to actual BBG tickers.

#### `asset_straddle(path, underlying, straddle, chain_csv=None)`

Build a straddle info table with asset metadata and relevant tickers.

```python
from specparser.amt import asset_straddle

table = asset_straddle(
    "data/amt.yml",
    "C Comdty",
    "|2023-12|2024-01|N|0|OVERRIDE||33.3|"
)
# Columns: ['name', 'value']
```

**Output rows:**
| Name | Description |
|------|-------------|
| `asset` | The underlying asset name |
| `straddle` | The packed straddle string |
| `valuation` | Comma-delimited name=value pairs from Valuation dict |
| `vol` | Vol ticker as `source:ticker:field` (Near or Far based on ntrc) |
| `hedge` | Hedge ticker as `source:ticker:field` |
| `hedge1` | Additional hedge ticker (if present) |
| `calc` | Calculated hedge spec (if present) |

**Straddle format:** `|ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|`

The `ntrc` field (position 2) determines which Vol ticker to use:
- `"N"` (Near): Uses Vol.Near field
- `"F"` (Far): Uses Vol.Far field

**Example output:**
```
name       value
asset      C Comdty
straddle   |2023-12|2024-01|N|0|OVERRIDE||33.3|
valuation  Model=CDS_ES,tenor=5,S=px,X=strike,t=expiry_date - date,v=vol
vol        BBG:CL1 Comdty:Near
hedge      BBG:CL F24 Comdty:PX_LAST
```

---

### Schedule Functions

#### `get_schedule(path, underlying)`

Get the expiry schedule for an asset by its Underlying value.

Values like 'a', 'b', 'c', 'd' are automatically fixed to computed day numbers.

```python
from specparser.amt import get_schedule

table = get_schedule("data/amt.yml", "LA Comdty")
# Columns: ['schcnt', 'schid', 'asset', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']
```

#### `find_schedules(path, pattern)`

Find assets matching a regex pattern and return their schedule components.

```python
from specparser.amt import find_schedules

table = find_schedules("data/amt.yml", "^LA.*")
# Columns: ['schcnt', 'schid', 'asset', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']
```

**Column descriptions:**
| Column | Description |
|--------|-------------|
| `schcnt` | Schedule count (total components in schedule) |
| `schid` | Schedule ID (1-based component index) |
| `asset` | Underlying asset name |
| `ntrc` | Entry code (e.g., 'N' for Near, 'F' for Far) |
| `ntrv` | Entry value (computed day number) |
| `xprc` | Expiry code (e.g., 'OVERRIDE', 'BD', 'F') |
| `xprv` | Expiry value (computed day number) |
| `wgt` | Weight percentage |

**Value fixing:**

Values 'a', 'b', 'c', 'd' in schedule entries are automatically fixed to computed day numbers using:
```
value = (schedule_id - 1) * day_stride + day_offset
where:
    day_offset = asset_hash % 5 + 1
    day_stride = 20 / (schedule_count + 1)
    asset_hash = MD5(underlying)[:8] % 1000000
```

This spreads entries across the month to avoid clustering, with deterministic results based on the asset's underlying name.

---

### Schedule Expansion

#### `find_straddle_yrs(path, start_year, end_year, pattern, live_only)`

Expand all live schedules across a year/month range into straddle strings.

Computes the cartesian product of:
- years: start_year to end_year (inclusive)
- months: 1 to 12
- rows from schedules() (already expanded by schedule component)

Each row is packed into a pipe-delimited straddle string.

```python
from specparser.amt import find_straddle_yrs

table = find_straddle_yrs("data/amt.yml", 2024, 2025)
# Columns: ['asset', 'straddle']
```

**Straddle format:** `|ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|`

Example: `|2023-12|2024-01|N|0|OVERRIDE||33.3|`

**Entry date calculation:**
- `ntrc = "N"` (Near): Entry is 1 month before expiry
- `ntrc = "F"` (Far): Entry is 2 months before expiry

#### `find_straddle_ym(path, year, month, pattern, live_only)`

Expand all live schedules for a specific year/month into straddle strings.

```python
from specparser.amt import find_straddle_ym

table = find_straddle_ym("data/amt.yml", 2024, 6)
# Columns: ['asset', 'straddle']
```

#### `get_straddle_yrs(path, underlying, start_year, end_year)`

Expand a single asset's schedule across a year range into straddle strings.

```python
from specparser.amt import get_straddle_yrs

table = get_straddle_yrs("data/amt.yml", "LA Comdty", 2024, 2025)
# Columns: ['asset', 'straddle']
```

#### `get_expand_ym(path, underlying, year, month)`

Expand a single asset's schedule for a specific year/month into straddle strings.

```python
from specparser.amt import get_expand_ym

table = get_expand_ym("data/amt.yml", "LA Comdty", 2024, 6)
# Columns: ['asset', 'straddle']
```

#### `count_straddle_days(straddle)`

Count the number of calendar days in a straddle period.

```python
from specparser.amt import count_straddle_days

# Count days in a straddle spanning Jan-Mar 2024
n = count_straddle_days("|2024-01|2024-03|1|1|1|1|1|")
# Returns: 91 (Jan 1 through Mar 31)
```

**Parameters:**
- `straddle`: Straddle string in format `|ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|`

**Returns:** Integer count of days from entry month start to expiry month end (inclusive)

#### `straddle_days(straddle)`

Get all calendar days in a straddle period.

```python
from specparser.amt import straddle_days

# Get all days in a straddle
days = straddle_days("|2024-01|2024-03|1|1|1|1|1|")
# Returns: [date(2024, 1, 1), date(2024, 1, 2), ..., date(2024, 3, 31)]
```

**Parameters:**
- `straddle`: Straddle string in format `|ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|`

**Returns:** List of `datetime.date` objects for all days from entry month start to expiry month end

**Note:** Results are cached for performance when memoization is enabled.

#### `count_straddles_days(straddles)`

Count the total number of days across all straddles in a table.

```python
from specparser.amt import find_straddle_yrs, count_straddles_days

straddles = find_straddle_yrs("data/amt.yml", 2024, 2024, "LA Comdty")
total_days = count_straddles_days(straddles)
# Returns: total count of days across all straddles
```

**Parameters:**
- `straddles`: Table dict with `columns` containing "straddle" and `rows`

**Returns:** Integer total count of days across all straddles

#### `find_straddle_days(path, start_year, end_year, pattern=".", live_only=True)`

Expand straddles to daily rows for all matching assets.

```python
from specparser.amt import find_straddle_days

table = find_straddle_days("data/amt.yml", 2024, 2024, "LA Comdty")
# Returns column-oriented table:
# {
#     "orientation": "column",
#     "columns": ["asset", "straddle", "date"],
#     "rows": [asset_col, straddle_col, date_col]
# }
```

**Parameters:**
- `path`: Path to AMT YAML file
- `start_year`: Starting year
- `end_year`: Ending year
- `pattern`: Regex pattern to match Underlying values (default: "." matches all)
- `live_only`: If True, only include assets with `Live: true`

**Returns:** Column-oriented table dict with columns: asset, straddle, date

---

## CLI Usage

The AMT module has two CLIs:
- `python -m specparser.amt` - Asset and ticker operations
- `python -m specparser.amt.schedules` - Schedule operations

### Main CLI (specparser.amt)

```bash
# Get asset by name
uv run python -m specparser.amt data/amt.yml --get "LA Comdty OLD"

# Find assets by underlying
uv run python -m specparser.amt data/amt.yml --find "LA Comdty"

# Get embedded table
uv run python -m specparser.amt data/amt.yml --table group_risk_multiplier_table

# List all asset names
uv run python -m specparser.amt data/amt.yml --list

# List all assets with weight caps
uv run python -m specparser.amt data/amt.yml --all

# List live assets only
uv run python -m specparser.amt data/amt.yml --live

# List live assets with class and source info
uv run python -m specparser.amt data/amt.yml --class

# List live assets with group assignment
uv run python -m specparser.amt data/amt.yml --group

# List live assets with values from any rule table
uv run python -m specparser.amt data/amt.yml --live-table group_table
uv run python -m specparser.amt data/amt.yml --live-table limit_overrides
uv run python -m specparser.amt data/amt.yml --live-table liquidity_table

# Get value by key path
uv run python -m specparser.amt data/amt.yml --value backtest.aum
uv run python -m specparser.amt data/amt.yml --value backtest.reference_asset

# Get AUM
uv run python -m specparser.amt data/amt.yml --aum

# Get leverage
uv run python -m specparser.amt data/amt.yml --leverage

# Get tickers for a specific asset
uv run python -m specparser.amt data/amt.yml --asset-tickers "LA Comdty"

# Get all tickers for live assets
uv run python -m specparser.amt data/amt.yml --live-tickers

# Get all tickers with BBGfc expansion
uv run python -m specparser.amt data/amt.yml --live-tickers 2024 2025

# Get all tickers with normalized to actual lookup
uv run python -m specparser.amt data/amt.yml --live-tickers 2024 2025 --chain-csv data/current_bbg_chain_data.csv

# Compute futures ticker from spec
uv run python -m specparser.amt data/amt.yml --fut "generic:LA1 Comdty,fut_code:LA,fut_month_map:FGHJKMNQUVXZ,min_year_offset:0,market_code:Comdty" 2024 7

# Get straddle info with tickers
uv run python -m specparser.amt data/amt.yml --straddle "C Comdty" "|2023-12|2024-01|N|0|OVERRIDE||33.3|"
```

### Schedule CLI (specparser.amt.schedules)

```bash
# Get schedule for a single asset
uv run python -m specparser.amt.schedules data/amt.yml --get "LA Comdty"

# Find schedules matching regex pattern
uv run python -m specparser.amt.schedules data/amt.yml --find "^LA.*"

# List schedules for all live assets
uv run python -m specparser.amt.schedules data/amt.yml --live

# Expand live schedules across year range
uv run python -m specparser.amt.schedules data/amt.yml --expand 2024 2025

# Expand live schedules for specific year/month
uv run python -m specparser.amt.schedules data/amt.yml --expand-ym 2024 6

# Expand single asset's schedule across year range
uv run python -m specparser.amt.schedules data/amt.yml --get-expand "LA Comdty" 2024 2025

# Expand single asset's schedule for specific year/month
uv run python -m specparser.amt.schedules data/amt.yml --get-expand-ym "LA Comdty" 2024 6
```

---

## AMT YAML Structure

Expected structure of an AMT YAML file:

```yaml
amt:
  "Asset Name":
    Underlying: "Ticker Symbol"
    WeightCap: 0.05
    Options: "schedule_name"
    Description: "Human readable description"
    # ... other asset-specific fields

expiry_schedules:
  schedule_name:
    - "N0_OVERRIDE_33.3"    # Near entry, 0 days, OVERRIDE expiry, 33.3% weight
    - "N5_OVERRIDE_33.3"    # Near entry, 5 days, OVERRIDE expiry, 33.3% weight
    - "F10_OVERRIDE_12.5"   # Far entry, 10 days, OVERRIDE expiry, 12.5% weight
    - "F15_OVERRIDE_12.5"   # Far entry, 15 days, OVERRIDE expiry, 12.5% weight

group_risk_multiplier_table:
  Columns: [group, multiplier]
  Rows:
    - [rates, 1.0]
    - [equities, 1.5]
```

**Schedule component format:** `{entry}_{expiry}_{weight}`

Where:
- `entry`: Entry code + optional value (e.g., `N0`, `F5`, `Na`, `Fb`)
- `expiry`: Expiry code + optional value (e.g., `OVERRIDE`, `BD15`, `F3`)
- `weight`: Weight percentage (e.g., `33.3`, `12.5`)

---

## Examples

### Get all schedules for 2024-2025 as straddle strings

```python
from specparser.amt import find_straddle_yrs

table = find_straddle_yrs("data/amt.yml", 2024, 2025)
# Columns: ['asset', 'straddle']

for row in table['rows'][:5]:
    asset, straddle = row
    print(f"{asset}: {straddle}")
```

### Query specific asset's schedule

```python
from specparser.amt import get_schedule

table = get_schedule("data/amt.yml", "AAPL US Equity")
if table['rows']:
    print("Schedule for AAPL US Equity:")
    for row in table['rows']:
        print(f"  {row}")
```

### Expand schedules for specific assets

```python
from specparser.amt import find_straddle_yrs

# Expand only commodities matching pattern
table = find_straddle_yrs("data/amt.yml", 2024, 2024, pattern="^CL.*")

for row in table['rows'][:5]:
    asset, straddle = row
    print(f"{asset}: {straddle}")
```

### Parse straddle strings

```python
from specparser.amt import find_straddle_yrs

table = find_straddle_yrs("data/amt.yml", 2024, 2024)

# Straddle format: |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|
for row in table['rows'][:5]:
    asset, straddle = row
    parts = straddle.split('|')
    # parts[0] is empty (leading |)
    entry_date = parts[1]   # ntry-ntrm
    expiry_date = parts[2]  # xpry-xprm
    entry_code = parts[3]   # ntrc (N or F)
    weight = parts[7]       # wgt
    print(f"{asset}: entry={entry_date}, expiry={expiry_date}, code={entry_code}, weight={weight}")
```
