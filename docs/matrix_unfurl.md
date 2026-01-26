# Plan: Matrix Unfurl Operations

## Overview

"Unfurl" expands a matrix by duplicating rows according to a specification. This is a general-purpose primitive for row replication with optional per-expansion data injection.

**Key distinction from `make_calendar_from_ranges`:** The calendar function takes a *description* of data (year/month ranges) and generates both the expansion counts AND the output values internally. Unfurl is more general - it takes actual data and a separate specification for how to expand it.

## Context: Fast Strings

All strings in this system are "fast strings" - fixed-width strings encoded as uint8 spread over matrix columns:
- A string of 8 characters → 8-column uint8 matrix row
- Multiple strings → multiple rows, same column width

## Proposed Functions

---

### Level 1: `unfurl_by_counts`

The simplest case - duplicate rows based on an integer count vector.

```
Input:
  mat: uint8 matrix shape (R, W)
  counts: int64 vector length R

Output:
  out: uint8 matrix shape (sum(counts), W)
  src_idx: int64 vector length sum(counts) - which input row each output row came from

Example:
  mat = [["AA"],      counts = [2,
         ["BB"],                1,
         ["CC"]]                3]

  out = [["AA"],      src_idx = [0,
         ["AA"],                 0,
         ["BB"],                 1,
         ["CC"],                 2,
         ["CC"],                 2,
         ["CC"]]                 2]
```

**Algorithm:**
```python
@njit
def unfurl_by_counts(mat, counts):
    R, W = mat.shape
    total = 0
    for r in range(R):
        total += counts[r]

    out = np.empty((total, W), dtype=np.uint8)
    src_idx = np.empty(total, dtype=np.int64)

    p = 0
    for r in range(R):
        for _ in range(counts[r]):
            for k in range(W):
                out[p, k] = mat[r, k]
            src_idx[p] = r
            p += 1

    return out, src_idx
```

---

### Level 2: `unfurl_by_spec` (Unfurl with Embedded Data)

Expand rows AND inject per-expansion data from a spec matrix.

The spec matrix encodes both:
1. **Column 0**: The dupe count for each input row
2. **Columns 1+**: The data values to inject into each expanded row

The data item width is inferred: `item_width = (spec_cols - 1) / max_dupe_count`

```
Input:
  mat: uint8 matrix shape (R, W)
  spec: uint8 matrix shape (R, 1 + max_count * item_width)
        - Column 0: dupe count (as uint8, max 255)
        - Columns 1+: data items packed sequentially

Output:
  out: uint8 matrix shape (sum(counts), W + item_width)
  src_idx: int64 vector length sum(counts)
```

#### Example 1: Single-byte data items

```
mat (R=2, W=2):           spec (R=2, cols=5):
  [["AA"],                  [2, N, F, -, -],    # dupe=2, items=[N, F]
   ["BB"]]                  [4, N, N, F, F]]    # dupe=4, items=[N, N, F, F]

Spec layout: [count, item0, item1, item2, item3]
max_count = 4, data_cols = 4, item_width = 4/4 = 1 byte

Output (6 rows, W + item_width = 3 cols):
  out = [["AAN"],     src_idx = [0,
         ["AAF"],                0,
         ["BBN"],                1,
         ["BBN"],                1,
         ["BBF"],                1,
         ["BBF"]]                1]
```

#### Example 2: Two-byte data items

```
mat (R=2, W=2):           spec (R=2, cols=9):
  [["AA"],                  [2, N,N, F,F, -,-, -,-],   # dupe=2, items=["NN","FF"]
   ["BB"]]                  [4, N,N, N,N, F,F, F,F]]   # dupe=4, items=["NN","NN","FF","FF"]

Spec layout: [count, item0_b0, item0_b1, item1_b0, item1_b1, ...]
max_count = 4, data_cols = 8, item_width = 8/4 = 2 bytes

Output (6 rows, W + item_width = 4 cols):
  out = [["AANN"],    src_idx = [0,
         ["AAFF"],               0,
         ["BBNN"],               1,
         ["BBNN"],               1,
         ["BBFF"],               1,
         ["BBFF"]]               1]
```

**Algorithm:**
```python
@njit
def unfurl_by_spec(mat, spec, max_count):
    """
    Unfurl matrix with per-expansion data injection.

    Args:
        mat: uint8 (R, W) - input data
        spec: uint8 (R, 1 + max_count * item_width) - counts and data
        max_count: int - maximum dupe count (needed to compute item_width)

    Returns:
        out: uint8 (sum(counts), W + item_width)
        src_idx: int64 (sum(counts),)
    """
    R, W = mat.shape
    spec_cols = spec.shape[1]
    data_cols = spec_cols - 1
    item_width = data_cols // max_count

    # Pass 1: count total rows
    total = 0
    for r in range(R):
        total += spec[r, 0]  # count is in column 0

    out = np.empty((total, W + item_width), dtype=np.uint8)
    src_idx = np.empty(total, dtype=np.int64)

    # Pass 2: fill
    p = 0
    for r in range(R):
        count = spec[r, 0]
        for i in range(count):
            # Copy mat row
            for k in range(W):
                out[p, k] = mat[r, k]
            # Copy data item (item i starts at column 1 + i * item_width)
            item_start = 1 + i * item_width
            for k in range(item_width):
                out[p, W + k] = spec[r, item_start + k]
            src_idx[p] = r
            p += 1

    return out, src_idx
```

---

### Level 3: `unfurl_concat`

Given pre-computed values and src_idx (like from `make_calendar_from_ranges`), just concatenate.

```
Input:
  mat: uint8 matrix shape (R, W1)
  values: uint8 matrix shape (N, W2)
  src_idx: int64 vector length N - which row of mat each value row corresponds to

Output:
  out: uint8 matrix shape (N, W1 + W2)
```

This is for when the expansion has already been computed elsewhere (like calendar generation).

**Algorithm:**
```python
@njit
def unfurl_concat(mat, values, src_idx):
    N, W2 = values.shape
    _, W1 = mat.shape
    out = np.empty((N, W1 + W2), dtype=np.uint8)

    for p in range(N):
        r = src_idx[p]
        for k in range(W1):
            out[p, k] = mat[r, k]
        for k in range(W2):
            out[p, W1 + k] = values[p, k]

    return out
```

---

## Spec Matrix Design

The spec matrix is a clever encoding that packs:
1. Row-specific expansion count
2. Row-specific data values for each expansion

### Layout

```
spec shape: (R, 1 + max_count * item_width)

Column 0:                    dupe count (uint8, 0-255)
Columns 1 to item_width:     data for expansion 0
Columns item_width+1 to 2*item_width: data for expansion 1
...
```

### Inferring item_width

Given `max_count` (the maximum dupe count across all rows):
```python
data_cols = spec.shape[1] - 1  # exclude count column
item_width = data_cols // max_count
```

This requires the caller to know `max_count`, OR we could store it in a separate parameter.

### Unused slots

When a row has fewer expansions than `max_count`, the unused data slots can contain any value (they're ignored). Convention: fill with spaces or zeros.

---

## Relationship to Other Functions

### vs `make_calendar_from_ranges`

| Aspect | `make_calendar_from_ranges` | `unfurl_by_spec` |
|--------|----------------------------|------------------|
| Input | Range descriptions (year, month) | Actual data + spec |
| Counts | Computed internally | Explicit in spec |
| Output data | Generated (dates) | Copied from spec |
| Use case | Calendar-specific | General purpose |

The calendar function is a *specialized generator*. Unfurl is a *general expander*.

### vs `cartesian_product`

Cartesian product: all rows get same expansion count = len(other matrix)
Unfurl: each row can have different expansion count

Cartesian product is a special case where `counts = [M, M, M, ...]` and all rows get the same data items.

---

## Use Cases

### 1. NTRC Expansion

Each straddle has different NTRC pattern (N=near, F=far):
- Some straddles: 2 expansions [N, F]
- Some straddles: 4 expansions [N, N, F, F]

```python
straddles = strs2u8mat(["STR001", "STR002"])
spec = np.array([
    [2, ord('N'), ord('F'), 0, 0],      # 2 expansions: N, F
    [4, ord('N'), ord('N'), ord('F'), ord('F')],  # 4 expansions
], dtype=np.uint8)

out, src_idx = unfurl_by_spec(straddles, spec, max_count=4)
# out: [["STR001N"], ["STR001F"], ["STR002N"], ["STR002N"], ["STR002F"], ["STR002F"]]
```

### 2. Month Expansion with Month Codes

Each asset expands to different months with month letter codes:
```python
assets = strs2u8mat(["CL", "GC"])
# CL trades F,G,H,J,K,M,N,Q,U,V,X,Z (12 months)
# GC trades G,J,M,Q,V,Z (6 months)

spec_CL = [12, F,G,H,J,K,M,N,Q,U,V,X,Z]  # 12 single-byte items
spec_GC = [6, G,J,M,Q,V,Z,0,0,0,0,0,0]    # 6 items, 6 unused

spec = np.array([spec_CL, spec_GC], dtype=np.uint8)
out, src_idx = unfurl_by_spec(assets, spec, max_count=12)
```

---

## Function Signatures

```python
@njit
def unfurl_by_counts(
    mat: np.ndarray,      # uint8 (R, W)
    counts: np.ndarray,   # int64 (R,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Duplicate rows according to counts.
    Returns: (out, src_idx)
    """

@njit
def unfurl_by_spec(
    mat: np.ndarray,      # uint8 (R, W)
    spec: np.ndarray,     # uint8 (R, 1 + max_count * item_width)
    max_count: int,       # maximum dupe count (to compute item_width)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unfurl with per-expansion data injection.
    Returns: (out, src_idx)
    """

@njit
def unfurl_concat(
    mat: np.ndarray,      # uint8 (R, W1)
    values: np.ndarray,   # uint8 (N, W2)
    src_idx: np.ndarray,  # int64 (N,)
) -> np.ndarray:
    """
    Concatenate mat rows with pre-expanded values.
    Returns: out
    """
```

---

## Implementation Priority

1. **`unfurl_by_counts`** - Simplest, foundation for testing
2. **`unfurl_by_spec`** - Main workhorse with embedded data
3. **`unfurl_concat`** - For combining with external generators (calendar, etc.)

---

## Design Decisions

1. **max_count**: Derived from spec during the counting pass (find max of column 0)
2. **Count dtype**: uint8 - same as spec matrix, max 255 expansions per row (sufficient for all use cases)
3. **Separator variants**: Yes, provide `_sep` versions
4. **Naming**: `unfurl`, `unfurl_by_spec`, `unfurl_by_spec_sep`, `unfurl_concat`, `unfurl_concat_sep`

## Final Function Signatures

```python
@njit
def unfurl(
    mat: np.ndarray,      # uint8 (R, W)
    counts: np.ndarray,   # uint8 (R,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Duplicate rows according to counts.
    Returns: (out, src_idx)
        out: uint8 (sum(counts), W)
        src_idx: int64 (sum(counts),)
    """

@njit
def unfurl_by_spec(
    mat: np.ndarray,      # uint8 (R, W)
    spec: np.ndarray,     # uint8 (R, 1 + data_cols)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unfurl with per-expansion data injection.
    max_count derived from max(spec[:, 0]).
    item_width = data_cols // max_count.
    Returns: (out, src_idx)
        out: uint8 (sum(counts), W + item_width)
        src_idx: int64 (sum(counts),)
    """

@njit
def unfurl_by_spec_sep(
    mat: np.ndarray,      # uint8 (R, W)
    spec: np.ndarray,     # uint8 (R, 1 + data_cols)
    sep: np.uint8,        # separator byte
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like unfurl_by_spec but with separator between mat and data.
    Returns: (out, src_idx)
        out: uint8 (sum(counts), W + 1 + item_width)
        src_idx: int64 (sum(counts),)
    """

@njit
def unfurl_concat(
    mat: np.ndarray,      # uint8 (R, W1)
    values: np.ndarray,   # uint8 (N, W2)
    src_idx: np.ndarray,  # int64 (N,)
) -> np.ndarray:
    """
    Concatenate mat rows with pre-expanded values using src_idx mapping.
    Returns: out uint8 (N, W1 + W2)
    """

@njit
def unfurl_concat_sep(
    mat: np.ndarray,      # uint8 (R, W1)
    values: np.ndarray,   # uint8 (N, W2)
    src_idx: np.ndarray,  # int64 (N,)
    sep: np.uint8,        # separator byte
) -> np.ndarray:
    """
    Like unfurl_concat but with separator between mat and values.
    Returns: out uint8 (N, W1 + 1 + W2)
    """
```
