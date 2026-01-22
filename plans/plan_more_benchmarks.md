# Plan: Comprehensive Arrow Compute Benchmarks

## Overview

Expand the Arrow Compute benchmarks section in `notebooks/table.ipynb` to cover more operations, especially string functions, and present results in clean, readable tables.

## Current State

- 5 benchmarks (multiply, uppercase, sum, cumsum, filter)
- 1,000,000 rows
- Inline print statements for each benchmark
- Summary table at the end

## Goals

1. **More benchmarks** - Cover all major function categories
2. **Clean presentation** - Factor out benchmark harness, show results in tables
3. **Consistent format** - All benchmarks use same structure

---

## Implementation

### Phase 1: Create Benchmark Harness

Add a helper cell that defines:

```python
import time

# Stores benchmark results: [(name, python_time, arrow_time), ...]
benchmark_results = []

def run_benchmark(name, python_fn, arrow_fn, iterations=1):
    """Run a benchmark and store results."""
    # Python timing
    t0 = time.perf_counter()
    for _ in range(iterations):
        python_fn()
    t_python = (time.perf_counter() - t0) / iterations

    # Arrow timing
    t0 = time.perf_counter()
    for _ in range(iterations):
        arrow_fn()
    t_arrow = (time.perf_counter() - t0) / iterations

    benchmark_results.append((name, t_python, t_arrow))

def show_benchmark_results():
    """Display benchmark results as a formatted table."""
    print("=" * 70)
    print("ARROW COMPUTE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Data size: {n_rows:,} rows")
    print()
    print(f"{'Operation':<30} {'Python':<12} {'Arrow':<12} {'Speedup':<10}")
    print("-" * 70)
    for name, t_python, t_arrow in benchmark_results:
        speedup = t_python / t_arrow if t_arrow > 0 else float('inf')
        print(f"{name:<30} {t_python:.3f}s{'':<6} {t_arrow:.3f}s{'':<6} {speedup:.1f}x")
    print("-" * 70)
```

### Phase 2: Benchmark Categories

#### 2.1 Arithmetic (4 benchmarks)
- `multiply` - scalar multiplication (existing)
- `add_columns` - add two columns together
- `power` - raise to power (expensive operation)
- `round` - rounding decimals

#### 2.2 String Operations (8 benchmarks)
- `upper` - uppercase (existing)
- `lower` - lowercase
- `length` - string length
- `strip` - strip whitespace
- `starts_with` - prefix check
- `contains` - substring search
- `replace` - substring replacement
- `split` - split by delimiter

#### 2.3 Aggregations (4 benchmarks)
- `sum` - sum column (existing)
- `mean` - average
- `min_max` - min and max
- `count_distinct` - unique count

#### 2.4 Cumulative (3 benchmarks)
- `cumsum` - cumulative sum (existing)
- `cummax` - running maximum
- `diff` - pairwise differences

#### 2.5 Filtering/Selection (3 benchmarks)
- `filter` - filter by condition (existing)
- `fill_null` - fill missing values
- `if_else` - conditional selection

**Total: 22 benchmarks** (vs current 5)

### Phase 3: Notebook Structure

Replace current benchmark cells with:

```
[Cell: Markdown] ### Arrow Compute Benchmarks
[Cell: Code] # Benchmark harness (run_benchmark, show_benchmark_results)
[Cell: Code] # Generate benchmark data (1M rows)
[Cell: Markdown] #### Arithmetic Operations
[Cell: Code] # Run arithmetic benchmarks (silent)
[Cell: Markdown] #### String Operations
[Cell: Code] # Run string benchmarks (silent)
[Cell: Markdown] #### Aggregations
[Cell: Code] # Run aggregation benchmarks (silent)
[Cell: Markdown] #### Cumulative Operations
[Cell: Code] # Run cumulative benchmarks (silent)
[Cell: Markdown] #### Filtering & Selection
[Cell: Code] # Run filter benchmarks (silent)
[Cell: Markdown] #### Results Summary
[Cell: Code] # show_benchmark_results() - displays the table
```

### Phase 4: Example Benchmark Code

```python
# String benchmarks
def python_lower(table, col):
    col_idx = table["columns"].index(col)
    return {"orientation": "row", "columns": table["columns"],
            "rows": [[v.lower() if i == col_idx else v for i, v in enumerate(row)]
                     for row in table["rows"]]}

run_benchmark("lower",
              lambda: python_lower(benchmark_data, "text"),
              lambda: table_lower_arrow(benchmark_data, "text"))

def python_length(table, col):
    col_idx = table["columns"].index(col)
    return {"orientation": "row", "columns": table["columns"] + ["len"],
            "rows": [row + [len(row[col_idx])] for row in table["rows"]]}

run_benchmark("length",
              lambda: python_length(benchmark_data, "text"),
              lambda: table_length_arrow(benchmark_data, "text", result_col="len"))

def python_contains(table, col, pattern):
    col_idx = table["columns"].index(col)
    return {"orientation": "row", "columns": table["columns"],
            "rows": [[pattern in v if i == col_idx else v for i, v in enumerate(row)]
                     for row in table["rows"]]}

run_benchmark("contains",
              lambda: python_contains(benchmark_data, "text", "999"),
              lambda: table_contains_arrow(benchmark_data, "text", "999"))
```

---

## Files to Modify

1. **notebooks/table.ipynb** - Replace benchmark section

---

## Cell-by-Cell Changes

### Remove these cells:
- `f75ppqigc5o` (Benchmark 1: multiply)
- `q49f8lvw65` (Benchmark 2: uppercase)
- `fxxapk9cwj7` (Benchmark 3: sum)
- `5zu6c9p3p7v` (Benchmark 4: cumsum)
- `x56tc1ml64l` (Benchmark 5: filter)
- `n5cmnqawbtm` (Summary)

### Add these cells (after `wdmtj0nj1r` data generation):

1. **Benchmark harness** (code cell)
2. **Arithmetic benchmarks** (markdown + code)
3. **String benchmarks** (markdown + code)
4. **Aggregation benchmarks** (markdown + code)
5. **Cumulative benchmarks** (markdown + code)
6. **Filter benchmarks** (markdown + code)
7. **Results table** (code cell calling `show_benchmark_results()`)

---

## Expected Output

```
======================================================================
ARROW COMPUTE BENCHMARK RESULTS
======================================================================
Data size: 1,000,000 rows

Operation                      Python       Arrow        Speedup
----------------------------------------------------------------------
multiply (scalar)              1.234s       0.012s       103.0x
add (columns)                  1.456s       0.015s       97.0x
power                          2.345s       0.089s       26.3x
round                          1.567s       0.023s       68.1x
upper                          3.456s       0.156s       22.1x
lower                          3.234s       0.145s       22.3x
length                         1.234s       0.034s       36.3x
strip                          2.345s       0.178s       13.2x
starts_with                    1.567s       0.045s       34.8x
contains                       2.789s       0.067s       41.6x
replace                        4.567s       0.234s       19.5x
split                          3.456s       0.289s       12.0x
sum                            0.234s       0.008s       29.3x
mean                           0.345s       0.009s       38.3x
min_max                        0.456s       0.012s       38.0x
count_distinct                 0.567s       0.089s       6.4x
cumsum                         1.234s       0.023s       53.7x
cummax                         1.345s       0.025s       53.8x
diff                           1.456s       0.034s       42.8x
filter                         0.789s       0.045s       17.5x
fill_null                      0.567s       0.023s       24.7x
if_else                        0.678s       0.034s       19.9x
----------------------------------------------------------------------
```

---

## Verification

1. Run notebook: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/table.ipynb`
2. Generate PDF: `uv run jupyter nbconvert --to pdf notebooks/table.ipynb`
3. Verify all 22 benchmarks appear in results table
4. Verify speedups are reasonable (Arrow should be faster for all)
