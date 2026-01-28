# Backtest Performance Conclusions (Updated)

## Scope
This report is based on a code review of the backtest pipeline plus the new implementations in `scripts/backtest_fast.py` and `scripts/backtest_strings.py` (no new runtime profiling). It revisits the earlier conclusions about the ~29 straddles/sec throughput in `scripts/backtest.py` and evaluates how the newer variants address those issues.

## Executive summary (updated)
1. **`backtest_fast.py` removes multiprocessing overhead entirely**
   - The script is explicitly single-threaded and notes that multiprocessing is ~100x slower for this workload due to large `prices_dict` serialization.
   - This directly addresses the earlier “spawn + pickle the prices dict” bottleneck in `backtest.py`.

2. **`backtest_fast.py` still exercises the same heavy per‑straddle pipeline**
   - It still runs the full `get_straddle_valuation` path (prices lookup → actions → valuation).
   - So the core per-row Python work remains a major cost, even without multiprocessing.

3. **`backtest_strings.py` is an isolated micro-benchmark**
   - It measures straddle expansion speed only. It **does not include prices lookup, action computation, or valuation**. Any speedup here won’t translate directly to end‑to‑end backtest throughput unless expansion is the dominant cost.

4. **The dominant costs likely shift from IPC to Python loops**
   - With multiprocessing removed in `backtest_fast.py`, the main bottleneck is now **Python-level per-row loops** in prices/actions/valuation and building large row tables in memory.

## Primary cost centers in the code (unchanged)
The following functions are the hottest paths by inspection (all per straddle):

- **Price lookup and table building**
  - `src/specparser/amt/prices.py:get_prices()`
  - `src/specparser/amt/prices.py:_lookup_straddle_prices()`
  - `src/specparser/amt/prices.py:_build_prices_table()`
  - These functions do nested loops of (dates × params) in Python, building list-of-list tables.

- **Action + model + strike columns (multiple passes)**
  - `src/specparser/amt/valuation.py:_compute_actions()`
  - `src/specparser/amt/valuation.py:_add_action_column()`
  - `src/specparser/amt/valuation.py:_add_model_column()`
  - `src/specparser/amt/valuation.py:_add_strike_columns()`
  - All of these traverse the same rows multiple times.

- **Valuation loop with per-row dict creation**
  - `src/specparser/amt/valuation.py:get_straddle_valuation()`
  - The valuation loop does `dict(zip(columns, row))` per row and converts strings to floats repeatedly.

- **Straddle date expansion (pure Python)**
  - `src/specparser/amt/schedules.py:straddle_days()`
  - Creates a Python list of `datetime.date` objects per straddle, then used in Python loops.

- **Ticker filtering / expansion**
  - `src/specparser/amt/tickers.py:filter_tickers()`
  - `src/specparser/amt/tickers.py:get_tickers_ym()`
  - Repeated per straddle, with per-process caches only.

## Specific issues inflating the benchmark time (updated)
- **Benchmark loop recreates process pools**
  - `scripts/backtest.py:run_backtest()` creates a new `Pool` every call.
  - `--benchmark` runs N iterations and **recreates the Pool every time** (including re-pickling the `prices_dict`).

- **Full results are collected even in benchmark mode**
  - `scripts/backtest.py:run_backtest()` appends every row from every straddle to `all_rows`.
  - That data is serialized from workers to the parent, even though benchmark mode doesn’t print it.

- **Process-local caches reduce reuse**
  - Caches like `_TICKERS_YM_CACHE`, `_EXPAND_YM_CACHE`, `_STRADDLE_DAYS_CACHE` are not shared across workers.
  - Work for the same (asset, year, month) repeats across workers.

Additional context with the new scripts:
- **`backtest_fast.py` removes multiprocessing overhead**  
  It is explicitly single‑threaded to avoid pickling the large `prices_dict`. The heavy parts (prices/actions/valuation and Python table building) still dominate total runtime.

- **`backtest_strings.py` is not end-to-end**  
  It shows the expansion subsystem can be much faster in isolation, but this does not remove the overall backtest bottlenecks.

## Supporting evidence in the repo (updated)
- Backtest entry point and multiprocessing: `scripts/backtest.py`
- Fast expansion backtest: `scripts/backtest_fast.py`
- Expansion-only benchmark: `scripts/backtest_strings.py`
- Price lookup + table building: `src/specparser/amt/prices.py`
- Actions + valuation: `src/specparser/amt/valuation.py`
- Straddle date expansion: `src/specparser/amt/schedules.py`
- Ticker filtering and caching: `src/specparser/amt/tickers.py`
- There is already a profiling helper that times these parts: `scripts/profile_backtest.py`

## Bottom-line conclusion (updated)
The single-threaded `backtest_fast.py` correctly avoids the multiprocessing serialization cliff, so **IPC is no longer the main tax** there. The remaining throughput ceiling is now dominated by:
- **Python-level per-row transformations** in price lookup, action computation, valuation, and table building.
- **Large in‑memory row materialization** and sorting/output.

The `backtest_strings.py` improvements are still valuable, but they only move the needle if straddle expansion was a material fraction of total time.
