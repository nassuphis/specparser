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

# Simple list expansion (cartesian product)
expand("[a,b][1,2]")
# → ['a1', 'a2', 'b1', 'b2']

# Numeric ranges with zero-padding
expand("file_{01:03}.jpg")
# → ['file_01.jpg', 'file_02.jpg', 'file_03.jpg']

# Ranges with step
expand("value_{0:10:2}")
# → ['value_0', 'value_2', 'value_4', 'value_6', 'value_8', 'value_10']
```

## Spec Syntax

| Pattern | Description | Example |
|---------|-------------|---------|
| `[a,b,c]` | Cartesian list | `[x,y][1,2]` → `x1, x2, y1, y2` |
| `>[a,b,c]` | Progressive list | `>[a,b,c]` → `a`, `a,b`, `a,b,c` |
| `{start:end}` | Numeric range | `{1:3}` → `1, 2, 3` |
| `{start:end:step}` | Range with step | `{0:10:2}` → `0, 2, 4, 6, 8, 10` |
| `{start:end:count}` | Linspace (floats) | `{0:1:5}` → `0, 0.25, 0.5, 0.75, 1` |
| `@{regex}` | Dict selector | `@{/^A/}` matches keys starting with 'A' |
| `${expr}` | Expression eval | `${2+2}` → `4` |
| `#{n}` | Reference | `#{1}` references the 1st dimension |
| `!{expr}` | Initialization | `!{seed(42)}` seeds RNG |

## Modules

### expander

Core expansion engine with full DSL support:

```python
from specparser.expander import expand, expand_list_choices, progressive_choices

# Full cartesian expansion
specs = expand("prefix_[a,b]_{1:3}_suffix")

# List-only expansion (union, not cartesian)
expand_list_choices("[a,b,c]")  # → ['a', 'b', 'c']

# Progressive accumulation
progressive_choices(">[x,y,z]")  # → ['x', 'x,y', 'x,y,z']
```

### slots

File slot management for organizing generated images:

```python
from specparser.slots import used_slots, first_free_slot, free_slots

# Find used slot numbers for a schema
used = used_slots("output")  # finds output_00001.jpg, output_00002.jpg, etc.

# Get next available slot
next_slot = first_free_slot("output")

# Reserve N consecutive slots
slots = free_slots("output", 10)
```

### image2spec

Embed and extract specs from JPEG metadata:

```python
from specparser.image2spec import spec2image, read_spec_exiftool

# Embed spec in image metadata
spec2image("path/to/image.jpg", "my_spec_string")

# Read spec back from image
spec = read_spec_exiftool("path/to/image.jpg")
```

### chain

Parse and manipulate colon/comma-separated key:value chains:

```python
from specparser.chain import split_chain, concat_chain

# Parse chain format
d = split_chain("key1:value1,key2:value2")
# → {'key1': 'value1', 'key2': 'value2'}

# Rebuild chain string
s = concat_chain(d)
# → 'key1:value1,key2:value2'
```

## Advanced Features

### References

Reference earlier dimensions in specs:

```python
expand("[x,y]{1:2}::#{d1}-#{d2}")
# → ['x1::x-1', 'x2::x-2', 'y1::y-1', 'y2::y-2']
```

### Random Functions

Use render-time random selection:

```python
from specparser.expander import expand, DICT

DICT['colors'] = ['red', 'green', 'blue']
expand("color_#{choose(colors)}")  # randomly picks a color at render time
```

### Custom Functions

Register custom expansion and render-time functions:

```python
from specparser.expander import FUNCS, REF_FUNCS

# Expansion-time function
FUNCS['double'] = lambda x: [str(int(x) * 2)]

# Render-time function
REF_FUNCS['upper'] = lambda ctx, s: s.upper()
```

## Dependencies

- `numpy` - numeric operations
- `pyvips` - image manipulation
- `simpleeval` - safe expression evaluation

## License

MIT
