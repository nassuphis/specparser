# Plan: Update backtest.py for Module Split

## Overview

Update `scripts/backtest.py` to work with the new AMT module structure:
- `tickers.py` → pure ticker extraction only
- `prices.py` → price data access (load_all_prices, set_prices_dict)
- `valuation.py` → actions & valuation (get_straddle_valuation)

Also add a `--benchmark` mode to measure performance.

---

## Current Issues

### Line 16-19: Imports

```python
from specparser.amt import loader, schedules
from specparser.amt.tickers import get_straddle_valuation, load_all_prices, set_prices_dict  # WRONG
from specparser.amt import tickers as tickers_module
from specparser.amt import schedules as schedules_module
```

**Problems:**
1. `get_straddle_valuation` is now in `valuation.py`, not `tickers.py`
2. `load_all_prices` is now in `prices.py`, not `tickers.py`
3. `set_prices_dict` is now in `prices.py`, not `tickers.py`

### Line 40-41: Memoization Control

```python
tickers_module.set_memoize_enabled(memoize)
schedules_module.set_memoize_enabled(memoize)
```

**Status:** OK - both modules still have `set_memoize_enabled()` function.

### Line 44: Set Prices Dict

```python
set_prices_dict(prices_dict)
```

**Problem:** `set_prices_dict` now in `prices.py`, import needs update.

### Line 56-57: Get Straddle Valuation

```python
val_table = get_straddle_valuation(
    asset, year, month, i, _worker_amt, _worker_chain, _worker_prices
)
```

**Problem:** `get_straddle_valuation` now in `valuation.py`, import needs update.

### Line 161: Load All Prices

```python
prices_dict = load_all_prices(args.prices, start_date, end_date)
```

**Problem:** `load_all_prices` now in `prices.py`, import needs update.

---

## Solution

### Updated Imports (Lines 16-19)

**From:**
```python
from specparser.amt import loader, schedules
from specparser.amt.tickers import get_straddle_valuation, load_all_prices, set_prices_dict
from specparser.amt import tickers as tickers_module
from specparser.amt import schedules as schedules_module
```

**To:**
```python
from specparser.amt import loader, schedules
from specparser.amt.prices import load_all_prices, set_prices_dict
from specparser.amt.valuation import get_straddle_valuation
from specparser.amt import tickers as tickers_module
from specparser.amt import schedules as schedules_module
```

**Alternative (using top-level exports):**
```python
from specparser.amt import (
    loader,
    schedules,
    load_all_prices,
    set_prices_dict,
    get_straddle_valuation,
)
from specparser.amt import tickers as tickers_module
from specparser.amt import schedules as schedules_module
```

---

## Implementation Steps

### Step 1: Update imports
Change line 17 from:
```python
from specparser.amt.tickers import get_straddle_valuation, load_all_prices, set_prices_dict
```

To:
```python
from specparser.amt.prices import load_all_prices, set_prices_dict
from specparser.amt.valuation import get_straddle_valuation
```

### Step 2: Verify no other changes needed
- Line 40-41: `tickers_module.set_memoize_enabled()` - still works
- Line 41: `schedules_module.set_memoize_enabled()` - still works
- Line 44: `set_prices_dict()` - works with new import
- Line 56-57: `get_straddle_valuation()` - works with new import
- Line 161: `load_all_prices()` - works with new import

### Step 3: Add benchmark mode

Add a `--benchmark` CLI argument that:
1. Runs the backtest multiple times (default 3 iterations)
2. Measures and reports timing statistics
3. Suppresses normal output (only shows timing)

**New argument:**
```python
parser.add_argument("--benchmark", "-b", type=int, nargs="?", const=3, default=None,
                    metavar="N",
                    help="Run N iterations and report timing (default: 3)")
```

**Benchmark logic (after argument parsing):**
```python
if args.benchmark:
    import time

    times = []
    for run in range(args.benchmark):
        start = time.perf_counter()

        # Run the backtest (capture results but don't print)
        # ... existing logic but with output suppressed ...

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Run {run + 1}/{args.benchmark}: {elapsed:.2f}s", file=sys.stderr)

    # Report statistics
    print(f"\nBenchmark Results ({args.benchmark} runs):", file=sys.stderr)
    print(f"  Min:  {min(times):.2f}s", file=sys.stderr)
    print(f"  Max:  {max(times):.2f}s", file=sys.stderr)
    print(f"  Avg:  {sum(times)/len(times):.2f}s", file=sys.stderr)
    print(f"  Total straddles: {total_straddles}", file=sys.stderr)
    print(f"  Throughput: {total_straddles/min(times):.1f} straddles/sec", file=sys.stderr)
    return 0
```

**Implementation approach:**

Extract the main processing into a helper function that can be called multiple times:

```python
def run_backtest(tasks, num_workers, amt, prices, chain, memoize, prices_dict, verbose=False):
    """Run backtest and return (columns, all_rows, completed_count)."""
    all_rows = []
    columns = None
    completed = 0

    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(amt, prices, chain, memoize, prices_dict)
    ) as pool:
        for result in pool.imap_unordered(process_straddle, tasks):
            completed += 1
            if verbose and completed % 100 == 0:
                print(f"[{completed}/{len(tasks)}] ...", file=sys.stderr)

            if not result["error"]:
                if columns is None and result["columns"]:
                    columns = result["columns"]
                all_rows.extend(result["rows"])

    return columns, all_rows, completed
```

Then in `main()`:
```python
if args.benchmark:
    import time
    times = []
    for run in range(args.benchmark):
        start = time.perf_counter()
        columns, all_rows, _ = run_backtest(
            tasks, num_workers, args.amt, args.prices, args.chain,
            memoize, prices_dict, verbose=False
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Run {run + 1}/{args.benchmark}: {elapsed:.2f}s", file=sys.stderr)

    # Report stats
    print(f"\nBenchmark ({args.benchmark} runs, {total_straddles} straddles, {num_workers} workers):", file=sys.stderr)
    print(f"  Min: {min(times):.2f}s  ({total_straddles/min(times):.1f} straddles/sec)", file=sys.stderr)
    print(f"  Avg: {sum(times)/len(times):.2f}s", file=sys.stderr)
    print(f"  Max: {max(times):.2f}s", file=sys.stderr)
    return 0
else:
    # Existing non-benchmark logic
    columns, all_rows, completed = run_backtest(...)
    # ... print table ...
```

### Step 4: Test the script
```bash
uv run python scripts/backtest.py --help
uv run python scripts/backtest.py "^LA Comdty" 2024 2024 --verbose
uv run python scripts/backtest.py "^LA Comdty" 2024 2024 --benchmark
uv run python scripts/backtest.py "^LA Comdty" 2024 2024 --benchmark 5
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/backtest.py` | Update imports, add `--benchmark` mode |

---

## Function Location Summary

| Function | Old Location | New Location |
|----------|--------------|--------------|
| `get_straddle_valuation` | `tickers.py` | `valuation.py` |
| `load_all_prices` | `tickers.py` | `prices.py` |
| `set_prices_dict` | `tickers.py` | `prices.py` |
| `set_memoize_enabled` | `tickers.py` | `tickers.py` (unchanged) |
| `set_memoize_enabled` | `schedules.py` | `schedules.py` (unchanged) |

---

## Verification

```bash
# Verify imports work
uv run python -c "
from specparser.amt.prices import load_all_prices, set_prices_dict
from specparser.amt.valuation import get_straddle_valuation
print('Imports OK')
"

# Run with --help to verify script loads
uv run python scripts/backtest.py --help

# Run a small test (if data available)
uv run python scripts/backtest.py "^LA Comdty" 2024 2024 --verbose

# Run benchmark mode (3 iterations by default)
uv run python scripts/backtest.py "^LA Comdty" 2024 2024 --benchmark

# Run benchmark with 5 iterations
uv run python scripts/backtest.py "^LA Comdty" 2024 2024 --benchmark 5

# Compare memoization vs no-memoization performance
uv run python scripts/backtest.py "^LA Comdty" 2024 2024 --benchmark --no-memoize
```
