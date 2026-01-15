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

## Syntax Overview

| Pattern | Phase | Description |
|---------|-------|-------------|
| `[a,b,c]` | Expand | Cartesian product dimension |
| `>[a,b,c]` | Expand | Progressive (accumulating) list |
| `{1:10}` | Expand | Numeric range |
| `{0:1_0.1}` | Expand | Range with step |
| `{0:1\|5}` | Expand | Linspace (5 samples) |
| `${expr}` | Expand | Expression evaluation |
| `@{regex}` | Expand | Dict key selector |
| `#{expr}` | Render | Per-row reference |
| `!{expr}` | Init | One-time setup |

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
└── image2spec.py        # Image metadata handling
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

## License

MIT
