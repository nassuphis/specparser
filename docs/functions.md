# Function Reference

Functions are available in three contexts, each with its own registry:

| Context | Syntax | Registry | When Executed |
|---------|--------|----------|---------------|
| Init | `!{...}` | `INIT_FUNCS` | Once, before expansion |
| Expand | `${...}` | `FUNCS` | During expansion (returns choices) |
| Render | `#{...}` | `REF_FUNCS` | Per row, during rendering |

---

## Init Functions (`!{...}`)

These run once at the start of expansion, before any dimensions are processed.

| Function | Signature | Description |
|----------|-----------|-------------|
| `seed` | `seed(n=None)` | Set RNG seed. `None` or `"auto"` reseeds from entropy |
| `const` | `const(name, value)` | Define a constant available in expressions |
| `new` | `new(**kwargs)` | Replace DICT with key=value pairs |
| `add` | `add(**kwargs)` | Add key=value pairs to DICT |
| `load` | `load(path, mode="new", start=1, strip=True, skip_empty=False)` | Load file lines into DICT |

### Examples

```python
# Deterministic random
expand("!{seed(42)}#{rint(100)}")

# Define constants
expand("!{const('scale', 2.5)}${scale * 10}")
# → ['25.0']

# Load file into DICT (line numbers as keys)
expand("!{load('words.txt')}@{1}")  # Get line 1
```

---

## Expand Functions (`${...}`)

These are evaluated during expansion and return lists of choices that participate in the cartesian product.

### Built-ins

| Function | Signature | Description |
|----------|-----------|-------------|
| `range` | `range(start, stop, step=1)` | Python range |

### Random

| Function | Signature | Description |
|----------|-----------|-------------|
| `rint` | `rint(N)` | Random int from 1 to N |
| `rfloat` | `rfloat(a, b)` | Random float between a and b |

### Dictionary

| Function | Signature | Description |
|----------|-----------|-------------|
| `key` | `key(pattern)` | Keys from DICT matching regex |
| `value` | `value(pattern)` | Values from DICT where key matches regex |

### Sequences

| Function | Signature | Description |
|----------|-----------|-------------|
| `seq` | `seq(n)` | n values from 0.0 to 1.0 (linspace) |

### File Reading

| Function | Signature | Description |
|----------|-----------|-------------|
| `txt` | `txt(filename)` | All lines from file |
| `lines` | `lines(filename, count)` | count random lines from file |
| `lines2` | `lines2(filename, count, delim=":")` | count pairs of random lines, joined |

### Specs from Files

| Function | Signature | Description |
|----------|-----------|-------------|
| `specf` | `specf(specfile, slots)` | Read slots from specfile, assign to free slots |

### Specs from Images

| Function | Signature | Description |
|----------|-----------|-------------|
| `img` | `img(schema, slots)` | Read specs from images, assign to free slots |
| `ocr` | `ocr(imagefile)` | OCR image to spec, assign to free slot |
| `ocrs` | `ocrs(schema, slots)` | OCR multiple images, assign to free slots |

### Spec Manipulation

| Function | Signature | Description |
|----------|-----------|-------------|
| `spec` | `spec(specstr)` | Assign spec string to free slot |
| `rot` | `rot(specs, rotation)` | Set rotation on specs |
| `swap` | `swap(specs, key, old, new)` | Replace in spec key |
| `free` | `free(specs)` | Assign specs to free slots |

### Slot Management

| Function | Signature | Description |
|----------|-----------|-------------|
| `slots` | `slots(count)` | Get count free slot numbers |
| `used_slots` | `used_slots(schema)` | List used slot numbers |
| `free_slots` | `free_slots(schema, count)` | Get count free slots for schema |
| `slots2jpegs` | `slots2jpegs(schema, slots)` | Convert slots to jpeg paths |
| `slots2specs` | `slots2specs(schema, slots)` | Convert slots to spec paths |
| `first` | `first(schema)` | First free slot number |

### Dates

| Function | Signature | Description |
|----------|-----------|-------------|
| `bizdays` | `bizdays(month, calendar="XNYS")` | Trading days in month (YYYY-MM) |

### Examples

```python
# Generate sequence
expand("${seq(5)}")
# → ['0.0', '0.25', '0.5', '0.75', '1.0']

# Read from file
expand("${lines('words.txt', 3)}")
# → [3 random lines]

# Trading days
expand("${bizdays('2024-01')}")
# → ['2024-01-02', '2024-01-03', ...]
```

---

## Render Functions (`#{...}`)

These are evaluated per row during rendering. They return scalar values (strings).

### Built-in Variables

| Name | Description |
|------|-------------|
| `row` | Current row number (1-based) |
| `nrows` | Total number of rows |
| `d1`, `d2`, ... | Selected dimension values for current row |
| `dim1`, `dim2`, ... | Full dimension series (indexable by row) |
| `choices1`, `choices2`, ... | Full choice lists |
| `ndims` | Number of dimensions |

### Random Selection

| Function | Signature | Description |
|----------|-----------|-------------|
| `choose` | `choose(*args)` | Random choice from arguments |
| `rval` | `rval()` | Random value from DICT |
| `rint` | `rint(N)` | Random int from 1 to N |
| `rfloat` | `rfloat(a, b)` | Random float between a and b |
| `rfloat3` | `rfloat3(a, b)` | Random float, rounded to 3 decimals |

### Dictionary

| Function | Signature | Description |
|----------|-----------|-------------|
| `key` | `key(pattern, exclude=None)` | Random key matching pattern |
| `value` | `value(pattern, exclude=None)` | Random value where key matches |

### Type Conversion

| Function | Signature | Description |
|----------|-----------|-------------|
| `num` | `num(x)` | Convert to float |
| `i` | `i(x)` | Convert to int |
| `str` | `str(x)` | Convert to string |

### Formatting

| Function | Signature | Description |
|----------|-----------|-------------|
| `zfill` | `zfill(x, width)` | Zero-pad to width |
| `fmt` | `fmt(template, *args)` | String format |

### Indexing

| Function | Signature | Description |
|----------|-----------|-------------|
| `at` | `at(seq, idx)` | Element at index |
| `wat` | `wat(seq, idx)` | Element at index with wrap-around |

### File Access

| Function | Signature | Description |
|----------|-----------|-------------|
| `line` | `line(filename, lineno)` | Specific line (0-based) |
| `rline` | `rline(filename)` | Random line |
| `rline2` | `rline2(filename, delim=":")` | Two random lines joined |
| `r2line` | `r2line(file1, file2, delim=":")` | Random line from each file joined |

### Interpolation

| Function | Signature | Description |
|----------|-----------|-------------|
| `lerp` | `lerp(start, end)` | Linear interpolation based on row position |
| `square` | `square(start, end)` | Generate square coordinates interpolated |

### Date Calculations

| Function | Signature | Description |
|----------|-----------|-------------|
| `expiry` | `expiry(year, month, descriptor, calendar="XNYS")` | Expiry date for descriptor |
| `entry` | `entry(year, month, descriptor, near_far, calendar="XNYS")` | Entry date (1-2 months before expiry) |

**Expiry descriptors:**

| Pattern | Description |
|---------|-------------|
| `BD{n}` | Nth business day (BD1, BD5, BD15) |
| `LBD` | Last business day |
| `LBD{n}` | Nth-to-last business day |
| `F{n}` | Nth Friday (F3 = 3rd Friday) |
| `W{n}` | Nth Wednesday |
| `T{n}` | Nth Thursday |
| `M{n}` | Nth Monday |
| `MFBD{n}` | Modified following Nth business day |
| `MF{W}{n}` | Modified following Nth weekday |

**Entry near_far values:**
- `"N"` (Near): Entry is 1 month before expiry
- `"F"` (Far): Entry is 2 months before expiry

### Utilities

| Function | Signature | Description |
|----------|-----------|-------------|
| `first` | `first(schema)` | First free slot number |

### Examples

```python
# Random per row
expand("[a,b,c] #{choose('red','green','blue')}")
# → ['a green', 'b red', 'c blue']  (random each row)

# Interpolation across rows
expand("{1:5} value=#{lerp(0,100)}")
# → ['1 value=0', '2 value=25.0', '3 value=50.0', '4 value=75.0', '5 value=100']

# Access dimension values
expand("[x,y]{1:2} -> #{d1}#{d2}")
# → ['x1 -> x1', 'x2 -> x2', 'y1 -> y1', 'y2 -> y2']

# Format with zero-padding
expand("{1:3} file_#{zfill(d1, 4)}.jpg")
# → ['1 file_0001.jpg', '2 file_0002.jpg', '3 file_0003.jpg']

# Date calculations
expand("[2024][1,2,3] expiry=#{expiry(d1, d2, 'F3')}")
# → ['20241 expiry=2024-01-19', ...]  (3rd Friday of each month)

expand("[2024][6] entry=#{entry(d1, d2, 'F3', 'N')}")
# → ['20246 entry=2024-05-17']  (entry date 1 month before June expiry)
```

---

## Adding Custom Functions

You can register your own functions:

```python
from specparser.expand_funcs import FUNCS
from specparser.render_funcs import REF_FUNCS
from specparser.init_funcs import INIT_FUNCS

# Expand-time function (returns list)
FUNCS['mylist'] = lambda n: [f"item_{i}" for i in range(n)]

# Render-time function (returns scalar)
REF_FUNCS['upper'] = lambda s: s.upper()

# Init function
INIT_FUNCS['setup'] = lambda: print("Setting up...")
```
