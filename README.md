# specparser

A DSL (Domain Specific Language) for generating parametric combinations and expanding specifications. Designed for creating variations of specifications with support for ranges, choices, references, and image metadata embedding.

## Installation

```bash
pip install specparser
```

Requires Python >= 3.13

### External Dependencies

- `exiftool` - for reading/writing image metadata

## Quick Start

```python
from specparser.expander import expand

# Cartesian product of lists
expand("[a,b][1,2]")
# → ['a1', 'a2', 'b1', 'b2']

# Numeric ranges with zero-padding
expand("file_{01:03}.jpg")
# → ['file_01.jpg', 'file_02.jpg', 'file_03.jpg']

# Ranges with step
expand("value_{0:10_2}")
# → ['value_0', 'value_2', 'value_4', 'value_6', 'value_8', 'value_10']

# References to dimension values
expand("[x,y]{1:2}::#{d1}-#{d2}")
# → ['x1::x-1', 'x2::x-2', 'y1::y-1', 'y2::y-2']
```

## Documentation

| Document | Description |
|----------|-------------|
| [DSL Syntax](docs/dsl-syntax.md) | Complete syntax reference for the spec language |
| [Functions](docs/functions.md) | Available functions for init, expand, and render phases |
| [Modules](docs/modules.md) | Package structure and module API reference |
| [AMT Reference](docs/amt.md) | AMT YAML processing and schedule expansion |
| [Storage Reference](docs/storage.md) | DuckDB/Parquet storage utilities |

## Spec Language

The spec language is a DSL for generating parametric combinations. Specs are processed in three phases:

1. **Init** (`!{...}`) - One-time setup (set seed, define constants, load data)
2. **Expand** (`${...}`, `[...]`, `{...}`, `@{...}`) - Generate cartesian product of all dimensions
3. **Render** (`#{...}`) - Evaluate per-row expressions with access to dimension values

### Pattern Reference

| Pattern | Phase | Description |
|---------|-------|-------------|
| `[a,b,c]` | Expand | Cartesian product dimension |
| `>[a,b,c]` | Expand | Progressive (accumulating) list |
| `{1:10}` | Expand | Numeric range (inclusive) |
| `{0:1_0.1}` | Expand | Range with explicit step |
| `{0:1\|5}` | Expand | Linspace (5 evenly-spaced samples) |
| `{01:05}` | Expand | Zero-padded range |
| `${expr}` | Expand | Expression evaluation |
| `@{regex}` | Expand | Dict key selector (regex match) |
| `#{expr}` | Render | Per-row reference |
| `!{expr}` | Init | One-time setup |

### Examples

```python
from specparser.expander import expand

# Lists create dimensions - cartesian product is computed
expand("[a,b][1,2]")
# → ['a1', 'a2', 'b1', 'b2']

# Numeric ranges with zero-padding
expand("file_{01:03}.jpg")
# → ['file_01.jpg', 'file_02.jpg', 'file_03.jpg']

# Ranges with step or linspace
expand("{0:10_2}")      # step of 2 → ['0', '2', '4', '6', '8', '10']
expand("{0:1|5}")       # 5 samples → ['0', '0.25', '0.5', '0.75', '1']

# References access dimension values at render time
expand("[x,y]{1:2}::#{d1}-#{d2}-#{row}")
# → ['x1::x-1-1', 'x2::x-2-2', 'y1::y-1-3', 'y2::y-2-4']

# Built-in references: row, nrows, d1..dN (dimension values)
expand("[a,b,c] row #{row} of #{nrows}")
# → ['a row 1 of 3', 'b row 2 of 3', 'c row 3 of 3']

# Expressions evaluated at expand time
expand("${2**10}")      # → ['1024']
expand("${[1,2,3]}")    # list becomes dimension → ['1', '2', '3']

# Init expressions run once before expansion
expand("!{seed(42)}${rint(1,100)}")  # reproducible random
```

### Render-Time Functions

Render functions (`#{...}`) execute once per output row:

```python
# Random selection per row
expand("[1,2,3] #{choose('red','green','blue')}")

# Linear interpolation based on row position
expand("{1:5} #{lerp(0,100)}")
# → ['1 0.0', '2 25.0', '3 50.0', '4 75.0', '5 100.0']

# File access
expand("[1,2,3] #{rline('words.txt')}")  # random line per row

# Date calculations (exchange calendars)
expand("[2024][1,2,3] #{expiry(d1,d2,'F3')}")  # 3rd Friday of each month
expand("[2024][6] #{entry(d1,d2,'BD15','N')}")  # entry date 1 month before
```

### Macros

Simple text substitutions loaded from a file. Macros are expanded in a **single pass from first line to last**, substituting each macro into the text in order:

```python
from specparser.expander import macro, macro_init

# macros.txt:
# @SPEC=@PATH@/img_@ID@.jpg   ← top-level, substituted first
# @PATH@=@BASE@/output        ← substituted second, replaces @PATH@ left by first
# @BASE@=/data                ← substituted last, replaces @BASE@ left by second
# @ID@=001

macro_init("macros.txt")
macro("@SPEC@")
# → '/data/output/img_001.jpg'
```

**Important**: Top-level macros go first, component macros go later. Each line's macro is substituted into the result of previous substitutions.

See [DSL Syntax](docs/dsl-syntax.md) for complete documentation.

## Package Structure

```
specparser/
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
├── image2spec.py        # Image metadata handling
├── amt.py               # AMT YAML processing and schedule expansion
└── storage.py           # DuckDB/Parquet storage utilities
```

See [Modules](docs/modules.md) for detailed API documentation.

## CLI Usage

```bash
# Run self-test
python -m specparser.expander --selftest

# Expand a spec
python -m specparser.expander "[a,b]{1:3}" --expand

# List trading days
python -m specparser.dates 2024-01

# Read file lines
python -m specparser.files myfile.txt -r 5
```

## Testing

```bash
# Install dev dependencies
pip install specparser[dev]

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_expander.py
```

## Dependencies

- `numpy` - numeric operations
- `pyvips` - image manipulation
- `simpleeval` - safe expression evaluation
- `exchange_calendars` - trading day calculations
- `pyyaml` - YAML file parsing
- `duckdb` - embedded analytical database
- `pyarrow` - Parquet file support

## License

MIT
