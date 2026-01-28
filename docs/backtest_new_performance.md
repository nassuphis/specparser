# backtest_new.py Performance Analysis

## Executive Summary

Performance profiling of `backtest_new.py` (the `numba_sorted_kernel` code path) reveals:

| Phase | Time | % | Status |
|-------|------|---|--------|
| Phase 1: Load prices (PyArrow) | 0.60s | 6% | ✓ Optimized |
| Phase 2: JIT Warmup | 1.21s | 12% | One-time cost |
| Phase 3: Find straddles + expand days | 0.46s | 5% | ✓ Optimized |
| Phase 4: Parse straddle strings | 0.14s | 1% | ✓ Fast |
| Phase 5: Resolve tickers | 0.33s | 3% | ⚠️ Can optimize |
| **Phase 6: Prepare backtest arrays** | **2.28s** | **23%** | **🔴 Main bottleneck** |
| Phase 7: Run Numba kernel | 0.26s | 3% | ✓ Optimized |
| Phase 8: Build Arrow output | 0.32s | 3% | ✓ Fast |
| **TOTAL (post-warmup)** | **~4.4s** | | |

**Key Finding:** Phase 6 (`_prepare_backtest_arrays_sorted`) is the main bottleneck, taking 2.28s (51% of post-warmup time).

---

## Detailed Phase Breakdown

### Phase 1: Load Prices (1.81s total)

| Sub-phase | Time | Notes |
|-----------|------|-------|
| Import PyArrow modules | 1.21s | One-time import |
| `load_prices_numba()` | 0.60s | Fast, pre-sorted parquet helps |

The PyArrow loader is already highly optimized (8x faster than DuckDB baseline).

### Phase 3: Find Straddles + Expand Days (0.46s)

| Sub-phase | Time | Notes |
|-----------|------|-------|
| `find_straddle_days_u8m()` | 0.34s | Numba kernel, fast |
| Compute starts/lengths | 0.05s | Vectorized numpy |
| u8m → string conversion | 0.05s | Required for downstream |

This phase is well-optimized using the u8m format and Numba kernels.

### Phase 5: Resolve Tickers (0.33s with optimization)

The current implementation in `_batch_resolve_tickers()` calls `loader.get_asset()` for each of the 231K straddles. With pre-loading unique asset data (189 assets), this can be reduced significantly.

| Current | Optimized | Savings |
|---------|-----------|---------|
| 3.26s | 0.33s | **90%** |

**Optimization:** Pre-load asset data for unique assets before the loop.

### Phase 6: Prepare Backtest Arrays (2.28s) - MAIN BOTTLENECK

Detailed sub-phase breakdown:

| Sub-phase | Time | % of Phase 6 |
|-----------|------|--------------|
| Parse straddle strings | 0.29s | 13% |
| Ticker lookup & normalization | 0.20s | 9% |
| **Anchor date computation** | **1.65s** | **72%** |
| Other (array init, etc.) | 0.14s | 6% |

**Root Cause:** `_anchor_day()` is called twice per straddle (entry + expiry), for 231K straddles = 462K calls.

Each `_anchor_day()` call:
1. Parses xprc/xprv codes
2. Creates Python `date` objects
3. Iterates over days in month to find Nth weekday/business day
4. For OVERRIDE, loads and searches overrides CSV

At ~3.6μs per call, this adds up to 1.65s for 462K calls.

### Phase 7: Run Numba Kernel (0.26s)

The `full_backtest_kernel_sorted()` Numba kernel is highly optimized:
- Binary search price lookups: O(log n)
- Parallelized across straddles
- Efficient entry/expiry detection
- Vectorized model computation and PnL

This phase processes 16M days in 0.26s - extremely fast.

---

## Optimization Opportunities

### 1. Pre-load Asset Data in `_batch_resolve_tickers()` (High Impact)

**Current:** Calls `loader.get_asset()` 231K times
**Proposed:** Pre-load once per unique asset (189 calls)

```python
# Pre-compute asset data for unique assets
unique_asset_names = set(assets)
asset_to_data = {a: loader.get_asset(amt_path, a) for a in unique_asset_names}
```

**Expected Savings:** 2.9s → 0.3s (90% reduction)

### 2. Vectorize Anchor Date Computation (High Impact)

**Current:** Python loop with `_anchor_day()` called 462K times
**Proposed:** Numba kernel to compute anchor dates

The anchor date logic is deterministic:
- Input: xprc (F/R/W/BD/OVERRIDE), xprv (int), year, month
- Output: date32 offset

This can be vectorized:

```python
@njit(parallel=True)
def compute_anchor_dates_batch(
    xprc_codes: np.ndarray,  # uint8[S] - F=0, R=1, W=2, BD=3, OVERRIDE=4
    xprv_values: np.ndarray,  # int32[S]
    years: np.ndarray,        # int32[S]
    months: np.ndarray,       # int32[S]
) -> np.ndarray:
    """Vectorized anchor date computation."""
    # Pre-computed: days_in_month, first_day_of_week, etc.
    ...
```

**Expected Savings:** 1.65s → ~0.05s (97% reduction)

### 3. Cache Anchor Dates (Medium Impact)

Many straddles share the same (xprc, xprv, year, month) combination.

```python
anchor_cache: dict[tuple[str, str, int, int], str] = {}
```

**Expected Savings:** ~50% of anchor computation time

### 4. Batch Override Lookups (Low Impact)

For OVERRIDE codes, pre-load all overrides and do batch lookups.

---

## Projected Performance After Optimizations

| Phase | Current | Optimized | Savings |
|-------|---------|-----------|---------|
| Phase 5: Resolve tickers | 3.26s | 0.33s | 2.93s |
| Phase 6: Prepare arrays | 2.28s | 0.63s | 1.65s |
| **TOTAL** | **9.74s** | **5.16s** | **47%** |

Post-optimization rate: **~45,000 straddles/sec** (vs current 23,700)

---

## Test Data Characteristics

- **Straddles:** 231,192
- **Days:** 16,010,256
- **Output rows (valid):** 5,883,650
- **Unique assets:** 189
- **Ticker map entries:** 13,894
- **Year range:** 2001-2026

---

## Files to Modify

| File | Changes |
|------|---------|
| `valuation.py` | Pre-load asset data in `_batch_resolve_tickers()` |
| `valuation.py` | Vectorize anchor date computation |
| `valuation_numba.py` | Add `compute_anchor_dates_batch()` kernel |

---

## Profiling Script

The profiling script is at `scripts/profile_backtest_new.py`. Run with:

```bash
uv run python scripts/profile_backtest_new.py '.' 2001 2026
```
