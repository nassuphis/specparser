# Plan: Analyze Parquet vs DuckDB for Price Loading

## Context

The `load_all_prices()` and `load_prices_matrix()` functions in `prices.py` use DuckDB to load price data:

```python
con = duckdb.connect()
query = f"SELECT ticker, field, date, value FROM '{prices_parquet}'"
# ... optional date filters ...
result = con.execute(query).fetchall()
con.close()
```

The hypothesis is that **direct Parquet loading via PyArrow** could be faster when:
1. No filtering is needed (reading entire file)
2. Date filtering covers most of the data anyway

---

## Benchmark Results (COMPLETED)

### Key Finding: DuckDB is 20% faster

| Operation | DuckDB | PyArrow | Speedup |
|-----------|--------|---------|---------|
| `load_all_prices()` no filter | 2.6s | 3.2s | **0.80x (slower)** |

### Where the Time Goes

**PyArrow breakdown (3.2s total):**
| Step | Time | % of Total |
|------|------|------------|
| `pq.read_table()` | 0.105s | 3% |
| Arrow compute (key building) | 3.7s | 116%* |
| `keys.to_pylist()` | 5.0s | 156%* |
| `dict(zip())` | 1.7s | 53%* |

*Note: Percentages overlap because this breakdown was measured in separate cells.

**DuckDB breakdown (2.6s total):**
| Step | Time | % |
|------|------|---|
| Query + fetchall | 2.3s | 88% |
| Python dict loop | 0.3s | 12% |

### Conclusion

**The bottleneck is NOT file I/O** - PyArrow reads the file 25x faster than DuckDB (0.1s vs 2.3s).

**The bottleneck is Python conversion** - converting Arrow arrays to Python objects is extremely expensive:
- `to_pylist()`: 5.0s to convert 17M values
- Arrow compute operations: 3.7s for string concat
- `dict(zip())`: 1.7s to build the dict

DuckDB's fetchall() returns Python tuples directly, avoiding the Arrow→Python conversion overhead.

---

## Investigation: Why String Concat, to_pylist(), and dict(zip())?

The user asked: *"What is the PyArrow format? Why is there any string concat being used? Why to_pylist()? Why the dict(zip()) needed?"*

### Why the Current Design Uses a Python Dict

The current architecture (`prices.py` → `valuation.py`) was designed for **row-by-row lookups**:

```python
# In valuation.py (lines 2303-2327) - the "dict" price_lookup mode
for i in range(length):
    date_str = prices_module._date32_to_isoformat(int(dates[start + i]))
    key = f"{vol_ticker}|{vol_field}|{date_str}"
    value_str = prices_dict.get(key)  # O(1) lookup
    if value_str is not None and value_str != "none":
        vol_array[start + i] = float(value_str)
```

This pattern works well for:
1. **Single straddle valuation** - ~50-100 lookups per straddle, dict.get() is O(1)
2. **Interactive use** - quick lookups without loading any special data structure

But for **batch valuation** (1728 straddles × 50 days = 86,000 lookups), the Python loop overhead dominates.

### Why String Concatenation?

The dict uses **composite keys** `"ticker|field|date"` because:
1. Python dicts need hashable keys
2. A tuple `(ticker, field, date)` would work, but strings are more debuggable
3. The key format was chosen for simplicity: `f"{ticker}|{field}|{date}"`

In the PyArrow benchmark, we replicated this with Arrow compute:
```python
keys = pc.binary_join_element_wise(
    pc.binary_join_element_wise(table['ticker'], table['field'], '|'),
    pc.cast(table['date'], pa.string()),
    '|'
)
```

### Why to_pylist()?

`to_pylist()` converts a PyArrow Array to a Python list. It's required because:
1. Python's `dict()` and `zip()` work on Python iterables, not Arrow arrays
2. There's no Arrow-native "build dict from two arrays" operation
3. The final output must be a Python `dict[str, str]` for compatibility with `get_price()`

### Why dict(zip())?

```python
return dict(zip(keys.to_pylist(), values.to_pylist()))
```

This is the standard Python idiom for building a dict from two parallel lists. It's required because:
1. We need a `dict[str, str]` for the existing `get_price()` API
2. Arrow doesn't have a dict-building operation
3. The values must be accessible by key, not by position

### Clarification: These Are NOT Per-Row Operations

**Important:** `dict(zip())` and `to_pylist()` are called **once**, not per-row.

Here's the actual flow for PyArrow:

```python
# 1. Read entire file (0.1s) - returns Arrow Table with ~8.5M rows
table = pq.read_table(prices_parquet)

# 2. Build ALL keys at once using Arrow compute (3.7s)
#    Returns Arrow Array with 8.5M strings - ONE operation, vectorized
keys = pc.binary_join_element_wise(
    pc.binary_join_element_wise(table['ticker'], table['field'], '|'),
    pc.cast(table['date'], pa.string()),
    '|'
)

# 3. Convert Arrow Arrays to Python lists (5.0s)
#    Creates 8.5M + 8.5M = 17M Python string objects - TWO calls
keys_list = keys.to_pylist()
values_list = values.to_pylist()

# 4. Build dict from two lists (1.7s)
#    ONE call that iterates through 8.5M pairs
prices_dict = dict(zip(keys_list, values_list))
```

Compare to DuckDB:

```python
# 1. Query + fetchall (2.3s) - returns list of 8.5M Python tuples
result = con.execute(query).fetchall()

# 2. Python loop (0.3s) - iterates once through all rows
for ticker, field, dt, value in result:  # 8.5M iterations
    key = f"{ticker}|{field}|{dt}"
    prices_dict[key] = str(value)
```

### Why DuckDB Is Faster

The difference comes down to **how Python objects are created**:

| Approach | Object Creation | Time |
|----------|-----------------|------|
| DuckDB `fetchall()` | Creates tuples directly in C, returns Python list | 2.3s |
| Arrow `to_pylist()` | Converts Arrow arrays to Python objects one-by-one | 5.0s |

DuckDB's `fetchall()` is optimized for returning Python-native objects. Arrow's `to_pylist()` must convert from Arrow's columnar format to Python objects, which has more overhead per element.

---

## Investigation: PyArrow to NumPy/Numba Arrays

The user asked to compare `to_pylist()` with converting PyArrow arrays to Numba-compatible arrays.

### Key Finding: 42x Faster Conversion

| Path | Conversion Time | Description |
|------|-----------------|-------------|
| **PATH A: to_pylist() → dict** | 16.8s | Convert to Python objects, build dict |
| **PATH B: to_numpy() → Numba** | 0.4s | Dictionary-encode strings, extract numpy arrays |
| **Speedup** | **42x** | |

### How It Works

**PATH A (Python dict):**
```python
# 1. Load file (0.2s)
table = pq.read_table(prices_parquet)

# 2. Convert ALL columns to Python lists (9.8s) - SLOW!
tickers = table['ticker'].to_pylist()  # 8.5M Python strings
fields = table['field'].to_pylist()
dates = table['date'].to_pylist()
values = table['value'].to_pylist()

# 3. Build dict (6.8s)
for t, f, d, v in zip(tickers, fields, dates, values):
    prices_dict[f'{t}|{f}|{d}'] = str(v)
```

**PATH B (Numba arrays):**
```python
# 1. Load file (0.2s)
table = pq.read_table(prices_parquet)

# 2. Dictionary-encode strings (0.13s) - converts strings to integer indices
ticker_dict = pc.dictionary_encode(table['ticker']).combine_chunks()
field_dict = pc.dictionary_encode(table['field']).combine_chunks()

# 3. Extract numpy arrays (0.03s) - near zero-copy!
ticker_idx = ticker_dict.indices.to_numpy()  # int32[8.5M]
field_idx = field_dict.indices.to_numpy()    # int32[8.5M]
date_int32 = table['date'].to_numpy()        # datetime64 → int32
value_arr = table['value'].to_numpy()        # float64[8.5M]

# 4. Small string dictionaries (Python, but only ~4,646 + 7 entries)
ticker_strings = ticker_dict.dictionary.to_pylist()
field_strings = field_dict.dictionary.to_pylist()
```

### What `dictionary_encode()` Does

Dictionary encoding converts a column of repeated strings into two parts:
1. **A small "dictionary"** - the unique string values
2. **An integer index array** - pointing to which dictionary entry each row uses

```
BEFORE dictionary_encode():
┌─────────────────────────────────────────────────────────┐
│ ticker column: 8.5M Python string objects               │
│ ["CL1 Comdty", "CL1 Comdty", "CO1 Comdty", "CL1 Comdty",│
│  "GC1 Comdty", "CO1 Comdty", "CL1 Comdty", ...]         │
└─────────────────────────────────────────────────────────┘

AFTER dictionary_encode():
┌─────────────────────────────────────────────────────────┐
│ dictionary: 4,646 unique strings (small Python list)    │
│ ["CL1 Comdty", "CO1 Comdty", "GC1 Comdty", ...]         │
│      ↑ idx=0       ↑ idx=1       ↑ idx=2                │
├─────────────────────────────────────────────────────────┤
│ indices: 8.5M int32 values (NumPy array, Numba-ready)   │
│ [0, 0, 1, 0, 2, 1, 0, ...]                              │
│  ↑  ↑  ↑  ↑  ↑  ↑  ↑                                    │
│  │  │  │  │  │  │  └── "CL1 Comdty"                     │
│  │  │  │  │  │  └───── "CO1 Comdty"                     │
│  │  │  │  │  └──────── "GC1 Comdty"                     │
│  │  │  │  └─────────── "CL1 Comdty"                     │
│  │  │  └────────────── "CO1 Comdty"                     │
│  │  └───────────────── "CL1 Comdty"                     │
│  └──────────────────── "CL1 Comdty"                     │
└─────────────────────────────────────────────────────────┘
```

**Why this matters for Numba:**
- Numba cannot handle Python strings
- Numba CAN handle int32 arrays
- The indices array is a NumPy int32 array that Numba can use directly
- The small dictionary (4,646 strings) stays in Python but is only used for display/debugging

### Why to_numpy() Is So Fast

| Operation | Time | Notes |
|-----------|------|-------|
| `to_pylist()` for 8.5M strings | 2.3s | Creates 8.5M Python string objects |
| `to_pylist()` for 8.5M floats | 1.1s | Creates 8.5M Python float objects |
| `to_numpy()` for 8.5M floats | 0.012s | **Near zero-copy** - Arrow and NumPy share memory layout |
| `dictionary_encode()` + `indices.to_numpy()` | 0.06s | Converts strings to int32 indices |

The key insight: **Arrow's columnar format is already NumPy-compatible for numeric types**. No object creation needed.

### Numba Lookup Performance

Using sorted arrays with binary search:

| Approach | Lookups/sec | μs/lookup |
|----------|-------------|-----------|
| Python dict | 877K | 1.14 |
| Numba binary search | 4.6M | 0.22 |
| **Speedup** | **5.3x** | |

### Complete Timing Comparison

| Step | Python Dict Path | Numba Array Path |
|------|------------------|------------------|
| Load parquet | 0.2s | 0.2s |
| Convert columns | 9.8s (to_pylist) | 0.17s (dict_encode + to_numpy) |
| Build lookup structure | 6.8s (dict loop) | 0.15s (sort arrays) |
| **Total load time** | **16.8s** | **0.5s** |
| 100K lookups | 114ms | 22ms |
| **Total with lookups** | **16.9s** | **0.5s** |

### Data Structure for Numba

```python
# Instead of: prices_dict["CL1 Comdty|PX_LAST|2024-01-02"] = "72.31"

# Use parallel sorted arrays:
sorted_keys: int64[8.5M]    # composite key = (ticker_id * n_fields + field_id) * n_dates + date_offset
sorted_values: float64[8.5M]  # the price values

# Plus small lookup tables:
ticker_strings: list[str]   # 4,646 unique tickers
field_strings: list[str]    # 7 unique fields
min_date: int32             # for date offset calculation
```

The Numba kernel uses binary search on `sorted_keys` to find values in O(log n) time.

---

## The Real Insight: We Don't Need a Python Dict

The user correctly identified that if downstream functions could use **PyArrow format directly**, we could skip all the expensive conversions.

### What "PyArrow Format" Means

Instead of a Python dict:
```python
_PRICES_DICT = {"CL1|PX_LAST|2024-01-02": "72.31", ...}  # 17M entries, 392MB
```

We could store a PyArrow Table:
```python
_PRICES_ARROW = pa.Table  # 4 columns: ticker, field, date, value
                          # ~200MB, columnar, zero-copy
```

### table.py Already Supports Arrow-Oriented Tables

From [table.py:8-10](src/specparser/amt/table.py#L8-L10):
```python
# Arrow-oriented: {"orientation": "arrow", "columns": ["col1", "col2"], "rows": [pa.Array, pa.Array, ...]}
```

Arrow-oriented tables store PyArrow arrays directly, enabling:
- Zero-copy column access
- Vectorized compute operations
- Efficient joins without Python iteration

### How Downstream Code Uses Prices

There are **three price lookup patterns** in the codebase:

| Mode | Where Used | Data Source | Lookup Method |
|------|------------|-------------|---------------|
| `"dict"` | `valuation.py:2259-2327` | `_PRICES_DICT` | `prices_dict.get(key)` in Python loop |
| `"matrix"` | `valuation.py:2197-2227` | `PriceMatrix` | Numba kernel with `price_matrix[row, col]` |
| `"arrow"`/`"duckdb"` | `valuation.py:2229-2257` | Arrow table / parquet | PyArrow join or DuckDB join |

The **matrix** and **arrow/duckdb** modes already avoid the Python dict!

---

## Alternative: Arrow-Native Price Lookup

Instead of building a Python dict, we can keep prices in Arrow format and use **joins** for lookup:

```python
# Load prices once (0.1s)
_PRICES_ARROW = pq.read_table(prices_parquet)

# For batch lookup, build a request table
request = pa.table({
    'ticker': pa.array(['CL1 Comdty', 'CO1 Comdty', ...]),
    'field': pa.array(['PX_LAST', 'PX_LAST', ...]),
    'date': pa.array([date(2024, 1, 2), date(2024, 1, 3), ...]),
})

# Join to get values (vectorized, no Python iteration)
result = request.join(_PRICES_ARROW, keys=['ticker', 'field', 'date'])
```

### Already Implemented!

This is **already implemented** as `price_lookup="arrow"` in `get_straddle_backtests()`:

```python
# valuation.py:2248-2252
if price_lookup == "arrow":
    vol_array, hedge_array = _arrow_price_lookup(
        row_idx, req_dates, tickers, fields, params,
        prices_parquet, n_days
    )
```

The `_arrow_price_lookup()` function (implemented in the current plan) uses:
1. `load_prices_arrow()` - caches the Arrow table (0.1s load, then instant)
2. PyArrow join on (ticker, field, date)
3. Numpy advanced indexing to pivot results

---

## Recommended Path Forward

Given the benchmark results:

### For `load_all_prices()` - KEEP DUCKDB

The function returns a Python dict for the existing API. DuckDB is 20% faster because:
- fetchall() returns Python tuples directly (no Arrow→Python conversion)
- The Python dict loop is well-optimized

### For Batch Valuation - USE ARROW OR MATRIX MODE

The existing `price_lookup` options bypass the Python dict entirely:

| Mode | Load Time | Lookup Speed | Memory |
|------|-----------|--------------|--------|
| `"dict"` (default) | 2.6s | Slow (Python loop) | 392MB |
| `"matrix"` | 2-3s | **Fast** (Numba) | ~300MB |
| `"arrow"` | 0.1s (cached) | **Fast** (join) | ~200MB |
| `"duckdb"` | 0s (streams) | Fast (SQL join) | ~50MB |

### Decision: No Changes to load_all_prices()

The benchmark shows DuckDB is the right choice for building a Python dict.

For performance-critical batch operations, users should use:
```python
get_straddle_backtests(..., price_lookup="arrow")  # or "matrix" or "numba"
```

---

## Summary

| Question | Answer |
|----------|--------|
| Why string concat? | To build composite keys `"ticker|field|date"` for Python dict lookup |
| Why `to_pylist()`? | Arrow arrays must be converted to Python for dict/zip operations |
| Why `dict(zip())`? | Standard Python idiom to build dict from two parallel lists |
| Can we avoid all this? | **Yes!** Use `to_numpy()` instead of `to_pylist()` - 42x faster |
| Should we change `load_all_prices()`? | **Maybe** - if downstream uses Numba, skip the Python dict entirely |

### The Key Insight

The expensive conversions (`to_pylist`, `dict`, `zip`) are only needed if the downstream code requires a **Python dict**.

Two alternatives avoid this overhead:

1. **Arrow joins** (`price_lookup="arrow"`) - keep data in Arrow format, use PyArrow joins
2. **Numba arrays** - use `to_numpy()` + dictionary encoding, 42x faster than Python dict

### Performance Comparison

| Approach | Load Time | Lookup Speed | Best For |
|----------|-----------|--------------|----------|
| DuckDB → Python dict | 2.6s | 877K/sec | Single straddle, interactive use |
| PyArrow → Python dict | 16.8s | 877K/sec | Don't use this |
| PyArrow → Numba arrays | 0.5s | 4.6M/sec | Batch valuation, high throughput |
| Arrow joins | 0.1s (cached) | Vectorized | Batch valuation |

---

## Original Plan Sections (for reference)

<details>
<summary>Click to expand original benchmark plan</summary>

### Functions Affected

| Function | Location | Current Approach |
|----------|----------|------------------|
| `load_all_prices()` | prices.py:29 | DuckDB SELECT → Python loop → dict |
| `load_prices_matrix()` | prices.py:265 | DuckDB SELECT → 2-pass Python loop → PriceMatrix |

### What DuckDB Does

1. Opens/parses the Parquet file
2. Executes SQL query (with optional WHERE clause)
3. Returns Python tuples via `fetchall()`
4. Python loop iterates over tuples to build dict/matrix

### What Direct Parquet Would Do

1. PyArrow reads the Parquet file directly
2. Optional: filter using PyArrow predicates
3. Zero-copy access to columnar data
4. Vectorized operations to build dict/matrix

### Why DuckDB Won

1. **fetchall() returns Python tuples directly** - no Arrow→Python conversion
2. **Efficient tuple iteration** - Python for-loops over tuples are well-optimized
3. **Streaming execution** - DuckDB can stream results without loading entire file

### Why PyArrow Lost

1. **Arrow→Python conversion is expensive** - `to_pylist()` creates Python objects
2. **Arrow compute adds overhead** - `binary_join_element_wise` is vectorized but still costly
3. **Two-step dict building** - `zip()` then `dict()` adds overhead

</details>

---

## Files Referenced

| File | Purpose |
|------|---------|
| [prices.py](src/specparser/amt/prices.py) | `load_all_prices()`, `_PRICES_DICT`, `get_price()` |
| [valuation.py](src/specparser/amt/valuation.py) | Price lookup modes: dict, matrix, arrow, duckdb |
| [table.py](src/specparser/amt/table.py) | Arrow-oriented table support |
| [benchmark_price_loading.ipynb](notebooks/benchmark_price_loading.ipynb) | Benchmark results |
