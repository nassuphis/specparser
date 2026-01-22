# Plan: Create schedules.ipynb Notebook

## Overview

Create a comprehensive Jupyter notebook documenting the `specparser.amt.schedules` module, which handles:
- Reading expiry schedules from AMT YAML files
- Expanding schedules across year/month ranges
- Converting schedules to straddle strings
- Parsing straddle strings back to components
- Computing straddle day ranges

## Target Audience

Users who need to:
- Understand how options expiry schedules work in the AMT system
- Expand schedules to specific trading periods
- Parse and manipulate straddle strings
- Compute trading day ranges for straddles

## Notebook Structure

### Section 1: Introduction and Concepts
- What is a schedule? (expiry timing patterns for options strategies)
- Schedule entry format: `{entry}_{expiry}_{weight}` (e.g., `N0_F1_25`)
- Entry codes: N (near month), F (far month) with day offsets
- Expiry codes: F1-F4 (futures), BD (business day), OVERRIDE, etc.
- Straddle string format: `|ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|`

### Section 2: Loading Schedules
```python
from specparser.amt import get_schedule, find_schedules, print_table

# Get schedule for a single asset
schedule = get_schedule("data/amt.yml", "CL Comdty")
print_table(schedule)

# Find schedules for assets matching pattern
schedules = find_schedules("data/amt.yml", pattern="^CL|^GC", live_only=True)
print_table(schedules)
```

**Functions covered:**
- `get_schedule(path, underlying)` - Get schedule for one asset
- `find_schedules(path, pattern, live_only)` - Find schedules by regex
- Schedule table columns: `schcnt`, `schid`, `asset`, `ntrc`, `ntrv`, `xprc`, `xprv`, `wgt`

### Section 3: Understanding Schedule Components

Explain the columns in a schedule table:
- `schcnt`: Number of schedule components (e.g., 4 for quarterly)
- `schid`: Schedule component ID (1 to schcnt)
- `asset`: The underlying asset
- `ntrc`: Entry code (N=near, F=far)
- `ntrv`: Entry value (day offset)
- `xprc`: Expiry code (F1, BD, OVERRIDE, etc.)
- `xprv`: Expiry value (specific day or offset)
- `wgt`: Weight percentage for this component

Show example schedule interpretations:
```
N0_F1_25  -> Enter at near month (0 days offset), exit at 1st futures expiry, 25% weight
F10_F3_12.5 -> Enter 10 days into far month, exit at 3rd futures expiry, 12.5% weight
```

### Section 4: Expanding Schedules to Straddles

```python
from specparser.amt import (
    get_expand_ym,
    find_straddle_ym,
    get_straddle_yrs,
    find_straddle_yrs,
)

# Expand single asset for one month
straddles = get_expand_ym("data/amt.yml", "CL Comdty", 2024, 6)
print_table(straddles)

# Expand all live assets for one month
straddles = find_straddle_ym("data/amt.yml", 2024, 6, pattern=".", live_only=True)
print_table(straddles)

# Expand single asset across year range
straddles = get_straddle_yrs("data/amt.yml", "CL Comdty", 2024, 2025)
print_table(straddles)

# Expand all assets across year range
straddles = find_straddle_yrs("data/amt.yml", 2024, 2025, pattern=".", live_only=True)
print_table(straddles)
```

**Functions covered:**
- `get_expand_ym(path, underlying, year, month)` - Single asset, single month
- `find_straddle_ym(path, year, month, pattern, live_only)` - Multiple assets, single month
- `get_straddle_yrs(path, underlying, start_year, end_year)` - Single asset, year range
- `find_straddle_yrs(path, start_year, end_year, pattern, live_only)` - Multiple assets, year range

### Section 5: Straddle String Format

Explain the straddle format:
```
|2023-12|2024-01|N|0|OVERRIDE|15|33.3|
 ^       ^       ^ ^  ^       ^  ^
 |       |       | |  |       |  weight
 |       |       | |  |       expiry value
 |       |       | |  expiry code
 |       |       | entry value
 |       |       entry code
 |       expiry (year-month)
 entry (year-month)
```

### Section 6: Parsing Straddle Strings

```python
from specparser.amt import ntr, ntry, ntrm, xpr, xpry, xprm, ntrc, ntrv, xprc, xprv, wgt

straddle = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"

# Extract components
print(f"Entry date: {ntr(straddle)}")      # "2023-12"
print(f"Entry year: {ntry(straddle)}")     # 2023
print(f"Entry month: {ntrm(straddle)}")    # 12
print(f"Expiry date: {xpr(straddle)}")     # "2024-01"
print(f"Expiry year: {xpry(straddle)}")    # 2024
print(f"Expiry month: {xprm(straddle)}")   # 1
print(f"Entry code: {ntrc(straddle)}")     # "N"
print(f"Entry value: {ntrv(straddle)}")    # "0"
print(f"Expiry code: {xprc(straddle)}")    # "OVERRIDE"
print(f"Expiry value: {xprv(straddle)}")   # "15"
print(f"Weight: {wgt(straddle)}")          # "33.3"
```

**Functions covered:**
- `ntr(s)` - Entry date string (YYYY-MM)
- `ntry(s)` - Entry year (int)
- `ntrm(s)` - Entry month (int)
- `xpr(s)` - Expiry date string (YYYY-MM)
- `xpry(s)` - Expiry year (int)
- `xprm(s)` - Expiry month (int)
- `ntrc(s)` - Entry code
- `ntrv(s)` - Entry value
- `xprc(s)` - Expiry code
- `xprv(s)` - Expiry value
- `wgt(s)` - Weight

### Section 7: Computing Straddle Days

```python
from specparser.amt import (
    straddle_days,
    count_straddle_days,
    count_straddles_days,
    find_straddle_days,
    get_days_ym,
)

# Days in a month (utility)
days = get_days_ym(2024, 6)
print(f"June 2024 has {len(days)} days: {days[0]} to {days[-1]}")

# Days in a straddle
straddle = "|2024-01|2024-03|N|0|F1|0|100|"
days = straddle_days(straddle)
print(f"Straddle spans {len(days)} days: {days[0]} to {days[-1]}")

# Count days in a straddle
n = count_straddle_days(straddle)
print(f"Day count: {n}")

# Count total days across a straddles table
straddles = find_straddle_yrs("data/amt.yml", 2024, 2024, pattern="^CL", live_only=False)
total = count_straddles_days(straddles)
print(f"Total straddle-days: {total}")

# Expand straddles to individual days (column-oriented for efficiency)
days_table = find_straddle_days("data/amt.yml", 2024, 2024, pattern="^CL", live_only=False)
print(f"Columns: {days_table['columns']}")
print(f"Row count: {len(days_table['rows'][0])}")  # column-oriented
```

**Functions covered:**
- `get_days_ym(year, month)` - List of dates in a month
- `straddle_days(straddle)` - List of dates in straddle period
- `count_straddle_days(straddle)` - Number of days in straddle
- `count_straddles_days(straddles)` - Total days across table
- `find_straddle_days(path, start_year, end_year, pattern, live_only)` - Expand to day-level table

### Section 8: Practical Examples

**Example 1: Portfolio Calendar View**
```python
# Get all straddles for 2024
straddles = find_straddle_yrs("data/amt.yml", 2024, 2024, live_only=True)

# Group by month
from specparser.amt import table_column, table_chop
# ... show calendar distribution
```

**Example 2: Compute Position Weights Over Time**
```python
# Expand to day-level detail
days = find_straddle_days("data/amt.yml", 2024, 2024, pattern="^SPX", live_only=True)

# Process with pandas
import pandas as pd
df = pd.DataFrame({
    "asset": days["rows"][0],
    "straddle": days["rows"][1],
    "date": days["rows"][2],
})
# ... analyze weight over time
```

**Example 3: Schedule Analysis**
```python
# Compare schedule patterns across asset classes
for pattern, name in [("^CL|^GC", "Commodities"), ("^SPX|^NDX", "Equity"), ("^EUR|^USD", "FX")]:
    schedules = find_schedules("data/amt.yml", pattern, live_only=True)
    print(f"{name}: {len(schedules['rows'])} schedule components")
```

### Section 9: Summary Table

| Function | Description |
|----------|-------------|
| `get_schedule` | Get schedule for one asset |
| `find_schedules` | Find schedules by pattern |
| `get_expand_ym` | Expand asset to straddles (one month) |
| `find_straddle_ym` | Expand assets to straddles (one month) |
| `get_straddle_yrs` | Expand asset to straddles (year range) |
| `find_straddle_yrs` | Expand assets to straddles (year range) |
| `ntr`, `ntry`, `ntrm` | Parse entry date from straddle |
| `xpr`, `xpry`, `xprm` | Parse expiry date from straddle |
| `ntrc`, `ntrv` | Parse entry code/value from straddle |
| `xprc`, `xprv` | Parse expiry code/value from straddle |
| `wgt` | Parse weight from straddle |
| `get_days_ym` | Get all days in a month |
| `straddle_days` | Get all days in a straddle |
| `count_straddle_days` | Count days in a straddle |
| `count_straddles_days` | Count total days in straddles table |
| `find_straddle_days` | Expand straddles to day-level table |

---

## Files to Create

1. **notebooks/schedules.ipynb** - The main notebook

## Implementation Notes

- Use sample data from `data/amt.yml` (actual AMT file)
- Show both `show()` helper for pandas rendering and `print_table()` for CLI output
- Include error handling examples (asset not found, invalid straddle format)
- Performance notes: caching is automatic (memoization enabled by default)
- Note that `find_straddle_days` returns column-oriented for efficiency

## Cell Count Estimate

~45-50 cells total:
- ~8 markdown intro/concept cells
- ~6 loading schedule cells
- ~6 schedule component explanation cells
- ~8 expansion function cells
- ~4 straddle format explanation cells
- ~8 parsing function cells
- ~8 days computation cells
- ~6 practical example cells
- ~2 summary cells

---

## Verification

After creating the notebook:
1. Run all cells to verify no errors
2. Optionally generate PDF: `uv run python -m nbconvert --to webpdf --execute notebooks/schedules.ipynb --output-dir=notebooks/`
