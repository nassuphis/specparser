# specparser.amt.strings - Fast String/Matrix Operations

This document describes the `specparser.amt.strings` module for high-performance string generation using uint8 byte matrices.

## Key Insight

Instead of working with Python strings or date objects, work directly with **raw bytes** (uint8 arrays). The output is a matrix of ASCII characters that can be zero-copy converted to NumPy/Arrow string arrays.

## Core Concepts

### Fast Strings

All strings are "fast strings" - fixed-width strings encoded as uint8 matrices:
- A string of 8 characters → 8-column uint8 matrix row
- Multiple strings → multiple rows, same column width

### Conversion Utilities

```python
# Python strings → uint8 matrix
mat = strs2u8mat(["hello", "world"])  # shape (2, 5)

# uint8 matrix → numpy fixed-width strings (zero-copy!)
strings = u8mat2S(mat)  # dtype='|S5'

# Separator helper for cartesian products
pipe = sep("|")  # returns np.array([[ord("|")]], dtype=np.uint8)
```

The `u8mat2S()` function uses `view()` to reinterpret bytes as strings **without copying data**.

---

## Cartesian Product

### `cartesian_product(matrices)`

General N-way cartesian product of uint8 byte matrices.

```python
@njit
def cartesian_product(matrices: tuple) -> np.ndarray:
    """
    Cartesian product of N fixed-width byte matrices.

    Args:
        matrices: tuple of uint8 matrices

    Returns:
        uint8 matrix with shape (product of row counts, sum of widths)
    """
```

**Example - basic usage:**
```python
v1 = strs2u8mat(["AA", "BB"])      # (2, 2)
v2 = strs2u8mat(["X", "Y", "Z"])   # (3, 1)

result = cartesian_product((v1, v2))  # (6, 3)
# ["AAX", "AAY", "AAZ", "BBX", "BBY", "BBZ"]
```

**Example - with separators:**
```python
# Separators are just (1,1) matrices - use sep() helper
result = cartesian_product((v1, sep("|"), v2))  # (6, 4)
# ["AA|X", "AA|Y", "AA|Z", "BB|X", "BB|Y", "BB|Z"]

# Three-way with separators
assets = strs2u8mat(["CL", "GC"])
entry_yms = strs2u8mat(["2024-01", "2024-02"])
expiry_yms = strs2u8mat(["2024-03"])

result = cartesian_product((assets, sep("|"), entry_yms, sep("|"), expiry_yms))
# ["CL|2024-01|2024-03", "CL|2024-02|2024-03", "GC|2024-01|2024-03", "GC|2024-02|2024-03"]
```

### Algorithm: Odometer vs Stride

Two approaches were considered for iterating through indices. Both are O(n) per output row since both loop over n matrices to copy data - the difference is the constant factor in index computation.

**Stride approach:**
```python
strides = make_strides(sizes)  # precompute once
for p in range(total_rows):
    for i in range(n):
        row_idx = (p // strides[i]) % sizes[i]  # n divisions + n modulos per row
```

**Odometer approach (chosen):**
```python
indices = np.zeros(n, dtype=np.int64)
for p in range(total_rows):
    # use indices[i] directly, then increment like odometer:
    for i in range(n - 1, -1, -1):
        indices[i] += 1
        if indices[i] < sizes[i]: break
        indices[i] = 0
```

**Benchmark results:**
| Matrices | Rows | Odometer | Stride | Ratio |
|----------|------|----------|--------|-------|
| 2 (100×50) | 5,000 | 0.020 ms | 0.033 ms | 1.6× slower |
| 3 (100×50×20) | 100,000 | 0.671 ms | 1.051 ms | 1.6× slower |

**Why odometer wins:**

Index computation cost per row:
- **Stride**: 2n expensive ops (n divisions + n modulos, ~20-30 cycles each)
- **Odometer**: ~2 cheap ops (1 increment + 1 compare, ~1 cycle each); carry cascades average out to ~1 extra op per row (harmonic series)

Odometer has the better constant factor.

### Alternative: NumPy repeat/tile

A pure NumPy approach using `np.repeat` and `np.tile` to build index arrays:

**2 vectors** (A length na, B length nb):
```python
ia = np.repeat(np.arange(na), nb)   # 0,0,0,...,1,1,1,...
ib = np.tile(np.arange(nb), na)     # 0,1,2,...,0,1,2,...

A_out = A[ia]
B_out = B[ib]
```

**3 vectors**:
```python
ia = np.repeat(np.arange(na), nb * nc)
ib = np.tile(np.repeat(np.arange(nb), nc), na)
ic = np.tile(np.arange(nc), na * nb)
```

**N vectors** (general formula for dimension k with sizes `[s0, s1, ..., s{n-1}]`):
```python
left  = np.prod(sizes[k+1:], dtype=np.int64)  # repeat block size
right = np.prod(sizes[:k],   dtype=np.int64)  # tile count

ik = np.tile(np.repeat(np.arange(sizes[k]), left), right)
```

**Implementation:** `cartesian_product_np(matrices)` - same interface as `cartesian_product()`.

**Benchmark results:**
| Test | Rows | Numba | NumPy | Ratio |
|------|------|-------|-------|-------|
| 2-way (50×120) | 6,000 | 0.02 ms | 0.15 ms | 7× slower |
| 3-way (50×120×20) | 120,000 | 0.8 ms | 4.4 ms | 5.5× slower |

**When NumPy version is a good idea:**
- When output size is moderate and you're OK allocating the index arrays
- When you want simple code without Numba dependency
- For quick prototyping

**When it's not a good idea:**
- If the output is huge: index arrays can dominate memory (n int64 arrays of size total_rows)
- If you need uint8 output anyway: Numba two-pass fill avoids the big intermediate index arrays
- For our use case with millions of rows, the Numba odometer approach is preferred

### Parallel: `cartesian_product_par(matrices)`

Multi-threaded version using Numba's `prange`. Uses stride-based index computation since `prange` requires each iteration to be independent (no shared mutable state like the odometer's indices array).

```python
@njit(parallel=True)
def cartesian_product_par(matrices: tuple) -> np.ndarray:
    """
    Parallel cartesian product of N fixed-width byte matrices.
    Uses stride-based index computation for independent iterations.
    """
    # ... compute strides once ...
    for p in prange(total_rows):
        for i in range(n):
            row_idx = (p // strides[i]) % sizes[i]  # independent computation
            # copy data...
```

**Trade-off:**
- Stride is ~1.6× slower per row than odometer (n divs + n mods vs ~2 ops)
- But with multiple cores, parallelism outweighs the per-row cost

**When to use each:**
| Function | Best For |
|----------|----------|
| `cartesian_product` | Small to medium outputs, single-threaded context |
| `cartesian_product_par` | Large outputs (100k+ rows), multi-core systems |
| `cartesian_product_np` | Quick prototyping, no Numba dependency needed |

---

## Calendar Generation

### `make_calendar_from_ranges(src)`

Generate date strings from year-month ranges using a two-pass algorithm.

```python
@njit
def make_calendar_from_ranges(src):
    """
    Args:
        src: int matrix shape (R, 4) with columns [starty, startm, endy, endm]

    Returns:
        src_idx: int64 vector - which source row each output row came from
        cal: uint8 matrix (N, 10) - "YYYY-MM-DD" as bytes
    """
```

**Two-pass algorithm:**
1. Pass 1: Count total days needed → single allocation
2. Pass 2: Fill the pre-allocated buffer

No dynamic resizing, no list appends, no Python object creation.

**Performance:** ~35ms for 7.2M date strings (239k ranges × ~30 days each).

---

## Unfurl Operations

"Unfurl" expands a matrix by duplicating rows according to a specification.

### `unfurl(mat, counts)`

Duplicate rows based on an integer count vector.

```python
mat = strs2u8mat(["AA", "BB", "CC"])
counts = np.array([2, 1, 3], dtype=np.uint8)

out, src_idx = unfurl(mat, counts)
# out: ["AA", "AA", "BB", "CC", "CC", "CC"]
# src_idx: [0, 0, 1, 2, 2, 2]
```

### `unfurl_by_spec(mat, spec)`

Unfurl with per-expansion data injection. The spec matrix encodes both counts and data items.

```python
# spec layout: [count, item0, item1, ...]
# item_width = data_cols // max_count

mat = strs2u8mat(["STR1", "STR2"])
spec = np.array([
    [2, ord('N'), ord('F'), 0, 0],           # 2 expansions: N, F
    [4, ord('N'), ord('N'), ord('F'), ord('F')],  # 4 expansions
], dtype=np.uint8)

out, src_idx = unfurl_by_spec(mat, spec)
# out: ["STR1N", "STR1F", "STR2N", "STR2N", "STR2F", "STR2F"]
```

### `unfurl_concat(mat, values, src_idx)`

Concatenate mat rows with pre-expanded values using src_idx mapping.

```python
# Useful for combining with calendar output
assets = strs2u8mat(["CL", "GC"])
src_idx, dates = make_calendar_from_ranges(ranges)
result = unfurl_concat(assets, dates, src_idx)
```

### Separator variants

All unfurl functions have `_sep` variants that insert a separator byte:
- `unfurl_by_spec_sep(mat, spec, sep)`
- `unfurl_concat_sep(mat, values, src_idx, sep)`

---

## Performance Characteristics

### Why This Is Fast

1. **No Python objects in inner loop**: Everything is uint8/int64 arrays
2. **No allocations in inner loop**: Output is pre-sized in pass 1
3. **Numba JIT compilation**: Inner loops compile to native code
4. **Cache-friendly**: Sequential writes to contiguous memory
5. **Zero-copy output**: `view()` doesn't copy, just reinterprets

### Benchmark: Cartesian Product

```
50 assets × 120 yearmonths = 6,000 rows
  cartesian_product:  0.03 ms
  Python itertools:   0.22 ms

50 × 120 × 120 = 720,000 rows (three-way)
  cartesian_product:  5.8 ms
  Python itertools:  50.9 ms
```

---

## Function Summary

| Function | Description |
|----------|-------------|
| `strs2u8mat(strings)` | Python strings → uint8 matrix |
| `u8mat2S(mat)` | uint8 matrix → numpy S strings (zero-copy) |
| `sep(char)` | Create (1,1) separator matrix for cartesian_product |
| `cartesian_product(matrices)` | N-way cartesian product of byte matrices (Numba, sequential) |
| `cartesian_product_par(matrices)` | N-way cartesian product (Numba, parallel) |
| `cartesian_product_np(matrices)` | N-way cartesian product (pure NumPy, no Numba) |
| `make_calendar_from_ranges(src)` | Generate date strings from year-month ranges |
| `unfurl(mat, counts)` | Duplicate rows by count vector |
| `unfurl_by_spec(mat, spec)` | Unfurl with per-expansion data injection |
| `unfurl_by_spec_sep(mat, spec, sep)` | Same with separator |
| `unfurl_concat(mat, values, src_idx)` | Concatenate with pre-expanded values |
| `unfurl_concat_sep(mat, values, src_idx, sep)` | Same with separator |

---

## Related Code

- `tests/test_strings.py`: Tests for all functions
- `notebooks/strings.ipynb`: Examples and benchmarks
- `docs/matrix_unfurl.md`: Detailed unfurl design document
