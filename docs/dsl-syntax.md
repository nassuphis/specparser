# DSL Syntax Reference

The specparser DSL enables parametric expansion of specifications. Specs are expanded in two phases:

1. **Expansion time** - Cartesian product of all dimensions is computed
2. **Render time** - Each row is rendered with references resolved

## Syntax Overview

| Pattern | Name | Phase | Description |
|---------|------|-------|-------------|
| `[a,b,c]` | List | Expand | Cartesian product dimension |
| `>[a,b,c]` | Progressive List | Expand | Accumulating dimension |
| `{start:end}` | Range | Expand | Numeric range |
| `${expr}` | Expression | Expand | Evaluated expression |
| `@{regex}` | Dict Selector | Expand | Match keys in DICT |
| `#{expr}` | Reference | Render | Per-row evaluation |
| `!{expr}` | Init | Setup | One-time initialization |

---

## Lists `[...]`

Square brackets define a dimension. The cartesian product of all dimensions is computed.

```python
expand("[a,b][1,2]")
# → ['a1', 'a2', 'b1', 'b2']

expand("prefix_[x,y]_[1,2,3]_suffix")
# → ['prefix_x_1_suffix', 'prefix_x_2_suffix', 'prefix_x_3_suffix',
#    'prefix_y_1_suffix', 'prefix_y_2_suffix', 'prefix_y_3_suffix']
```

### Nested Lists

Lists can be nested inside list items for union expansion:

```python
expand_list_choices("a[b,c]d,e")
# → ['abd', 'acd', 'e']
```

---

## Progressive Lists `>[...]`

Progressive lists accumulate items instead of selecting one:

```python
expand(">[a,b,c]")
# → ['a', 'a,b', 'a,b,c']
```

Useful for building cumulative specifications.

---

## Numeric Ranges `{...}`

Curly braces define numeric ranges with several formats:

### Step Mode (default step = 1)

```python
expand("{1:5}")
# → ['1', '2', '3', '4', '5']

expand("{5:1}")  # descending
# → ['5', '4', '3', '2', '1']
```

### Step Mode with Explicit Step

Use underscore `_` to specify step magnitude:

```python
expand("{0:10_2}")
# → ['0', '2', '4', '6', '8', '10']

expand("{0:1_0.25}")
# → ['0', '0.25', '0.5', '0.75', '1']
```

### Linspace Mode

Use pipe `|` to specify number of samples (inclusive):

```python
expand("{0:1|5}")
# → ['0', '0.25', '0.5', '0.75', '1']

expand("{0:100|3}")
# → ['0', '50', '100']
```

### Zero-Padding

Leading zeros in endpoints enable padding:

```python
expand("{01:05}")
# → ['01', '02', '03', '04', '05']

expand("{001:100}")
# → ['001', '002', ..., '100']
```

---

## Expressions `${...}`

Dollar-brace expressions are evaluated at expansion time. They can return scalars or lists:

```python
expand("value_${2+3}")
# → ['value_5']

expand("item_${[1,2,3]}")
# → ['item_1', 'item_2', 'item_3']

expand("${range(1,4)}")
# → ['1', '2', '3']
```

Expressions have access to:
- Math operators: `+`, `-`, `*`, `/`, `**`, `//`, `%`
- Constants: `pi`
- All functions in `FUNCS` (see [functions.md](functions.md))

---

## Dict Selectors `@{...}`

At-brace selectors match keys in the global `DICT` and return their values:

```python
from specparser.expander import DICT

DICT.clear()
DICT.update({"color_red": "#ff0000", "color_blue": "#0000ff", "size": "10"})

expand("@{color_.*}")
# → ['#ff0000', '#0000ff']  (values where key matches regex)
```

The pattern uses Python regex with `fullmatch` semantics.

---

## References `#{...}`

Hash-brace references are evaluated at render time (once per output row). They have access to:

- `row` - Current row number (1-based)
- `nrows` - Total number of rows
- `d1`, `d2`, ... - Selected dimension values for current row
- `dim1`, `dim2`, ... - Full dimension series (indexable)
- `choices1`, `choices2`, ... - Full choice lists
- All functions in `REF_FUNCS` (see [functions.md](functions.md))

```python
expand("[a,b]{1:2}::#{d1}-#{d2}-#{row}")
# → ['a1::a-1-1', 'a2::a-2-2', 'b1::b-1-3', 'b2::b-2-4']

expand("[x,y] row=#{row} of #{nrows}")
# → ['x row=1 of 2', 'y row=2 of 2']
```

### Render-Time Functions

```python
expand("[1,2,3] #{choose('red','green','blue')}")
# → ['1 red', '2 blue', '3 green']  (random selection per row)

expand("{1:5} #{lerp(0,100)}")
# → ['1 0', '2 25.0', '3 50.0', '4 75.0', '5 100']
```

---

## Initialization `!{...}`

Exclamation-brace expressions run once before expansion:

```python
expand("!{seed(42)}[a,b,c]")  # Set deterministic seed

expand("!{const('myval', 100)}${myval}")  # Define constant
# → ['100']

expand("!{load('words.txt')}@{.*}")  # Load file into DICT
```

Available init functions:
- `seed(n)` - Set random seed
- `const(name, value)` - Define a constant
- `new(k1=v1, k2=v2)` - Replace DICT
- `add(k1=v1, k2=v2)` - Update DICT
- `load(path)` - Load file lines into DICT

---

## Macros

Macros are simple text replacements defined in `data/macros.txt`:

```
@OUTDIR=/path/to/output
@PREFIX=my_prefix
```

Used in specs:

```python
expand("@OUTDIR/@PREFIX_{1:3}.jpg")
# → ['/path/to/output/my_prefix_1.jpg', ...]
```

Macro keys must match pattern `@[A-Z0-9@]+`.

---

## Combining Patterns

Patterns can be freely combined:

```python
# Range inside list
expand("[{1:3},x,y]")
# → ['1', '2', '3', 'x', 'y']

# Expression inside list
expand("[${range(1,4)},100]")
# → ['1', '2', '3', '100']

# Multiple dimensions with references
expand("!{seed(0)}[a,b][1,2]_#{row}_#{choose('x','y')}")
# → ['a1_1_x', 'a2_2_y', 'b1_3_x', 'b2_4_y']
```

---

## Escaping

Currently there is no escape mechanism. If you need literal `[`, `{`, `$`, `@`, `#`, or `!` followed by `{`, structure your spec to avoid ambiguity or use macros.
