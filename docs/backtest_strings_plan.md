# Plan: backtest_strings.py - Fast String Processing Benchmark

## Overview

Create `scripts/backtest_strings.py` to benchmark the uint8 matrix string processing from `strings.py` against the standard Python string approach in `schedules.py`. This will measure the speedup from using Numba-accelerated byte matrix operations for straddle expansion.

## Current Approach (schedules.py)

The current `find_straddle_yrs()` uses Python string operations:

```python
def _schedule2straddle(xpry, xprm, ntrc, ntrv, xprc, xprv, wgt):
    offset = {"N": 1, "F": 2}.get(ntrc, 0)
    ntry, ntrm = _year_month_plus(xpry, xprm, offset)
    straddle = (
        f"|{ntry}-{ntrm:02d}"
        f"|{xpry}-{xprm:02d}"
        f"|{ntrc}|{ntrv}"
        f"|{xprc}|{xprv}"
        f"|{wgt}|"
    )
    return straddle

def _schedules2straddle_yrs(table, start_year, end_year):
    rows = []
    for xpry in range(start_year, end_year + 1):
        for xprm in range(1, 13):
            packed_table = _schedules2straddles(table, xpry, xprm)
            rows.extend(packed_table["rows"])
    return {"columns": ["asset", "straddle"], "rows": rows}
```

**Problems:**
- Creates Python string objects in inner loop
- List append operations with dynamic resizing
- Nested loops in pure Python

## Fast Approach (strings.py)

Use uint8 matrix operations:

1. **Pre-compute templates** for each asset's schedule (fixed parts)
2. **Generate year-months** as uint8 matrix using `make_ym_matrix()`
3. **Cartesian product** of templates × year-months using `cartesian_product()`
4. **Date offset** for entry dates using `add_months2specs_inplace_NF()`

### Key Functions from strings.py

| Function | Purpose |
|----------|---------|
| `strs2u8mat()` | Convert Python strings to uint8 matrix |
| `make_ym_matrix()` | Generate YYYY-MM range as uint8 matrix |
| `cartesian_product()` | N-way cartesian product of byte matrices |
| `add_months2specs_inplace_NF()` | Compute entry month from expiry + N/F offset |
| `u8m2s()` | Convert uint8 matrix back to strings |

---

## Implementation Plan

### 1. Template Precomputation

For each asset's schedule, extract fixed parts:
- `ntrc` (entry code: N or F)
- `ntrv` (entry value)
- `xprc` (expiry code)
- `xprv` (expiry value)
- `wgt` (weight)

```python
def precompute_templates(schedules_table):
    """
    Convert schedules table to uint8 templates.

    Returns:
        asset_mat: uint8 matrix of asset names
        template_mat: uint8 matrix of fixed straddle parts (ntrc|ntrv|xprc|xprv|wgt)
    """
```

### 2. Year-Month Generation

Generate expiry year-months for the range:

```python
from specparser.amt.strings import make_ym_matrix

# Generate 2001-01 to 2024-12 = 288 year-months
xpr_ym = make_ym_matrix((2001, 1, 2024, 12))  # shape (288, 7)
```

### 3. Cartesian Product

Combine assets × year-months:

```python
from specparser.amt.strings import cartesian_product, sep

# asset_templates: (N_assets, W_template)
# xpr_ym: (288, 7)

result = cartesian_product((
    asset_templates,  # N_assets rows
    sep(b"|"),        # separator
    xpr_ym,           # 288 year-months
))
# Result: (N_assets * 288, W_template + 1 + 7)
```

### 4. Entry Date Computation

Compute entry dates from expiry dates using N/F offset:

```python
from specparser.amt.strings import add_months2specs_inplace_NF

# ntr_ym: output matrix for entry year-months
# xpr_ym: expiry year-months (source)
# ntrc_col: column with N/F codes

add_months2specs_inplace_NF(ntr_ym, xpr_ym, ntrc_col)
```

### 5. Final Assembly

Concatenate all parts into straddle format:

```
|NTRY-MM|XPRY-MM|NTRC|NTRV|XPRC|XPRV|WGT|
```

---

## Benchmark Script Structure

```python
#!/usr/bin/env python3
"""
Benchmark fast string processing for straddle expansion.

Compares:
1. Standard Python approach (schedules.find_straddle_yrs)
2. Fast uint8 matrix approach (strings.py operations)
"""

import time
from specparser.amt import schedules, loader
from specparser.amt.strings import (
    strs2u8mat, make_ym_matrix, cartesian_product,
    add_months2specs_inplace_NF, u8m2s, sep
)

def expand_standard(amt_path, pattern, start_year, end_year):
    """Standard Python string approach."""
    return schedules.find_straddle_yrs(amt_path, start_year, end_year, pattern, True)

def expand_fast(amt_path, pattern, start_year, end_year):
    """Fast uint8 matrix approach."""
    # 1. Get schedules
    found = schedules.find_schedules(amt_path, pattern=pattern, live_only=True)

    # 2. Precompute templates
    templates = precompute_templates(found)

    # 3. Generate year-months
    xpr_ym = make_ym_matrix((start_year, 1, end_year, 12))

    # 4. Cartesian product
    expanded = cartesian_product((templates, sep(b"|"), xpr_ym))

    # 5. Compute entry dates
    # ... (implementation details)

    # 6. Convert back to table format
    return build_output_table(expanded)

def benchmark(amt_path, pattern, start_year, end_year, iterations=5):
    """Run both approaches and compare."""

    print(f"Benchmarking: pattern='{pattern}', years={start_year}-{end_year}")

    # Warmup
    expand_standard(amt_path, pattern, start_year, end_year)
    expand_fast(amt_path, pattern, start_year, end_year)

    # Standard approach
    times_std = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result_std = expand_standard(amt_path, pattern, start_year, end_year)
        times_std.append(time.perf_counter() - t0)

    # Fast approach
    times_fast = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result_fast = expand_fast(amt_path, pattern, start_year, end_year)
        times_fast.append(time.perf_counter() - t0)

    # Report
    avg_std = sum(times_std) / len(times_std)
    avg_fast = sum(times_fast) / len(times_fast)
    speedup = avg_std / avg_fast

    print(f"  Standard: {avg_std*1000:.2f}ms")
    print(f"  Fast:     {avg_fast*1000:.2f}ms")
    print(f"  Speedup:  {speedup:.1f}x")
    print(f"  Rows:     {len(result_std['rows']):,}")
```

---

## Expected Results

Based on `strings_comments.md` benchmarks:

| Operation | Standard | Fast | Speedup |
|-----------|----------|------|---------|
| Cartesian product (6k rows) | ~0.22ms | ~0.03ms | ~7x |
| Cartesian product (720k rows) | ~50ms | ~5.8ms | ~9x |
| Calendar generation (7.2M dates) | - | ~35ms | - |

For straddle expansion (e.g., 50 assets × 288 months = 14,400 straddles):
- **Expected speedup: 5-12x** depending on workload

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/backtest_strings.py` | Main benchmark script |
| `src/specparser/amt/strings_expand.py` (optional) | Fast expansion functions if complex |

---

## Verification

1. **Correctness**: Output from fast approach must match standard approach
2. **Performance**: Measure wall-clock time for both approaches
3. **Memory**: Fast approach should use less memory (no Python string objects)

```bash
# Run benchmark
uv run python scripts/backtest_strings.py '^LA Comdty' 2001 2024

# Verify correctness
uv run python scripts/backtest_strings.py '^LA Comdty' 2024 2024 --verify
```

---

## Implementation Steps

1. [ ] Create `scripts/backtest_strings.py` skeleton with CLI
2. [ ] Implement `precompute_templates()` function
3. [ ] Implement `expand_fast()` using strings.py operations
4. [ ] Add verification mode to compare outputs
5. [ ] Run benchmarks and report results
6. [ ] Document findings in comments/docstrings

---

## Notes

- The original `backtest_fast.py` referenced `precompute_templates` and `expand_fast` which were never implemented
- This plan creates those functions using the existing `strings.py` infrastructure
- Focus is on the **expansion** step, not the valuation step (which is already fast as shown in our profiling)
