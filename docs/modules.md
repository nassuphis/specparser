# Module Reference

This document describes the package structure and the role of each module.

## Package Structure

```
src/specparser/
├── expander.py          # DSL core: scanner, parser, expansion
├── expander_state.py    # Shared state: DICT, NAMES, RNG
├── init_funcs.py        # Init-time functions (!{...})
├── expand_funcs.py      # Expand-time functions (${...})
├── render_funcs.py      # Render-time functions (#{...})
├── chain.py             # Key:value chain parsing
├── chain_state.py       # Chain parser state: NAMES, FUNCS
├── files.py             # File reading utilities
├── dates.py             # Exchange calendar utilities
├── slots.py             # Slot management for images
├── image2spec.py        # Image metadata (spec embedding)
├── amt.py               # AMT YAML processing and schedule expansion
└── storage.py           # DuckDB/Parquet storage utilities
```

---

## Core Modules

### expander.py

The main DSL engine. Contains:

- **Segment types**: `Lit`, `Dim`, `Ref`, `Init`
- **Scanner**: `scan_segments(spec)` - tokenizes spec string
- **Parser**: `split_list_items()`, `expand_list_choices()`
- **Expansion**: `expand(spec)` - full cartesian expansion
- **Macros**: `macro()`, `macro_init()`

**Key exports:**
```python
from specparser.expander import expand, DICT, NAMES, RNG, set_dict
```

**CLI:**
```bash
python -m specparser.expander --selftest
python -m specparser.expander "[a,b]{1:3}" --expand
python -m specparser.expander "[a,b,c]" --choices
```

### expander_state.py

Shared mutable state used across all function modules.

**Contents:**
- `DICT: dict[str, str]` - term dictionary for `@{...}` lookups
- `NAMES: dict[str, object]` - constants for expressions
- `RNG: random.Random` - single source of randomness

**Key exports:**
```python
from specparser.expander_state import DICT, NAMES, RNG
from specparser.expander_state import set_dict, get_dict, clear_dict
from specparser.expander_state import set_name, get_name, get_names
from specparser.expander_state import seed, get_rng
```

### init_funcs.py

Functions available in `!{...}` expressions (run once at setup).

**Registry:** `INIT_FUNCS`

**Functions:** `seed`, `const`, `new`, `add`, `load`

### expand_funcs.py

Functions available in `${...}` expressions (run at expansion time, return lists).

**Registry:** `FUNCS`

**Functions:** `range`, `rint`, `rfloat`, `key`, `value`, `seq`, `txt`, `lines`, `lines2`, `specf`, `img`, `ocr`, `ocrs`, `spec`, `rot`, `swap`, `free`, `slots`, `used_slots`, `free_slots`, `slots2jpegs`, `slots2specs`, `first`, `bizdays`

### render_funcs.py

Functions available in `#{...}` expressions (run per row at render time, return scalars).

**Registry:** `REF_FUNCS`

**Functions:** `choose`, `rval`, `rint`, `rfloat`, `rfloat3`, `key`, `value`, `num`, `i`, `str`, `zfill`, `fmt`, `at`, `wat`, `line`, `rline`, `rline2`, `r2line`, `square`, `lerp`, `first`

**Also exports:**
```python
from specparser.render_funcs import set_render_ctx, reset_render_ctx, get_render_ctx, with_render_ctx
```

---

## Chain Modules

### chain.py

Parse and manipulate colon/comma-separated key:value chains (spec format).

**Key exports:**
```python
from specparser.chain import split_chain, concat_chain, parse_chain

# Parse chain format
d = split_chain("key1:value1,key2:value2")
# → {'key1': ['value1'], 'key2': ['value2']}

# Rebuild chain string
s = concat_chain(d)
# → 'key1:value1,key2:value2'

# Parse with scalar evaluation
specs = parse_chain("op:1:max(2,3),foo:sin(pi)")
# → [('op', (1+0j, 3+0j)), ('foo', (0j,))]
```

**CLI:**
```bash
python -m specparser.chain --selftest
python -m specparser.chain --spec "op:1:2,foo:sin(pi)"
python -m specparser.chain --spec "op:x" --const x=42
```

### chain_state.py

Shared state for the chain parser.

**Contents:**
- `NAMES: dict` - constants for scalar expressions (pi, tau, e, inf, nan, j, i, zero, one)
- `FUNCS: dict` - functions for scalar expressions (sin, cos, tan, sqrt, log, exp, abs, min, max, lerp, rint, rfloat, slotmax, slotmin, slots)
- `ALLOWED_OPS: dict` - whitelisted operators for simpleeval

**Key exports:**
```python
from specparser.chain_state import NAMES, FUNCS, ALLOWED_OPS
from specparser.chain_state import set_const, get_const
```

---

## Utility Modules

### files.py

File reading utilities with caching and random line selection.

**Key exports:**
```python
from specparser.files import (
    all_lines,       # All lines from file
    get_line,        # Specific line (0-based)
    get_lines,       # N random lines
    get_random_line, # One random line
    get_lines_paired,# N pairs of random lines
    set_seed,        # Set RNG seed
    clear_cache,     # Clear file cache
)
```

**CLI:**
```bash
python -m specparser.files myfile.txt          # List all lines
python -m specparser.files myfile.txt -l 5     # Get line 5
python -m specparser.files myfile.txt -r 3     # 3 random lines
python -m specparser.files myfile.txt -c       # Count lines
```

### dates.py

Exchange calendar utilities using `exchange_calendars` package.

**Key exports:**
```python
from specparser.dates import (
    good_days,         # Trading days in month
    holidays,          # All non-trading days
    weekday_holidays,  # Weekday-only holidays
    calendars,         # Available calendar codes
)

good_days("2024-01")  # NYSE trading days in Jan 2024
good_days("2024-01", "XLON")  # London exchange
```

**CLI:**
```bash
python -m specparser.dates 2024-01              # Trading days
python -m specparser.dates 2024-01 -o           # All non-trading days
python -m specparser.dates 2024-01 -w           # Weekday holidays only
python -m specparser.dates 2024-01 -c XLON      # London exchange
python -m specparser.dates -l                   # List all calendars
```

### slots.py

Slot management for organizing generated images. Slots are numbered file suffixes.

**Key exports:**
```python
from specparser.slots import (
    used_files,       # List files matching schema
    used_slots,       # List used slot numbers
    max_slot,         # Highest used slot
    first_free_slot,  # Next available slot
    free_slots,       # N free slots
    slots2jpegs,      # Convert slots to jpeg paths
    slots2specs,      # Convert slots to spec paths
)

# Schema is "directory/stem" - matches files like "directory/stem_00001.jpg"
used_slots("output/image")  # → [1, 2, 5, 7]
first_free_slot("output/image")  # → 3
free_slots("output/image", 4)  # → [3, 4, 6, 8]
```

### image2spec.py

Read and write spec strings in JPEG EXIF metadata.

**Key exports:**
```python
from specparser.image2spec import (
    spec2image,         # Write spec to image metadata
    read_spec_exiftool, # Read spec from image metadata
)

spec2image("image.jpg", "key1:value1,key2:value2")
spec = read_spec_exiftool("image.jpg")
```

**Requires:** `exiftool` command-line tool

---

## Dependency Graph

```
expander_state.py  ← (no dependencies, base state)
       ↑
   init_funcs.py   ← expander_state
       ↑
  expand_funcs.py  ← expander_state, chain, image2spec, slots, dates, files
       ↑
  render_funcs.py  ← expander_state, slots, files
       ↑
    expander.py    ← expander_state, init_funcs, expand_funcs, render_funcs


chain_state.py     ← slots (no circular deps, base state for chain)
       ↑
    chain.py       ← chain_state
```

Utility modules (`files`, `dates`, `slots`, `image2spec`) have minimal dependencies and can be used standalone.

The `chain` module has its own state (`chain_state.py`) separate from the expander's state (`expander_state.py`). Both are independent subsystems.

---

## AMT Module

### amt.py

Process AMT (Asset Management Table) YAML files for expiry schedules.

**Key exports:**
```python
from specparser.amt import (
    load_amt, clear_cache,           # Loading and caching
    get_value, get_aum, get_leverage,# Value extraction
    get_asset, find_by_underlying,   # Asset queries
    list_assets, get_schedule,       # Asset and schedule lookups
    get_table, format_table,         # Table utilities
    assets, live_assets,             # Asset tables
    live_class, live_table, live_group,  # Asset class, generic table, and group info
    asset_tickers, live_tickers,     # Ticker extraction
    asset_straddle,                  # Straddle info with tickers
    live_schedules, fix_expiry,      # Schedule processing
    expand_schedules,                # Schedule expansion (raw)
    expand_schedules_fixed,          # Schedule expansion (with fix_expiry)
    pack_straddle,                   # Pack into straddle strings
)
```

**CLI:**
```bash
uv run python -m specparser.amt data/amt.yml --get "Asset Name"
uv run python -m specparser.amt data/amt.yml --expand 2024 2025
uv run python -m specparser.amt data/amt.yml --pack 2024 2025
uv run python -m specparser.amt data/amt.yml --aum
uv run python -m specparser.amt data/amt.yml --leverage
uv run python -m specparser.amt data/amt.yml --value backtest.aum
uv run python -m specparser.amt data/amt.yml --group
```

See [AMT Reference](amt.md) for detailed documentation.

---

## Storage Module

### storage.py

DuckDB and Parquet storage utilities for persisting tables.

**Key exports:**
```python
from specparser.storage import (
    table_to_parquet,    # Write table to Parquet file
    parquet_to_table,    # Read Parquet file into table
    query_parquet,       # Run SQL query on Parquet file
    table_to_duckdb,     # Write table to DuckDB database
    query_duckdb,        # Run SQL query on DuckDB database
)
```

**CLI:**
```bash
uv run python -m specparser.storage --amt data/amt.yml --expand 2024 2025 --to-parquet schedules.parquet
uv run python -m specparser.storage --parquet schedules.parquet --query "SELECT DISTINCT asset FROM data"
```

See [Storage Reference](storage.md) for detailed documentation.
