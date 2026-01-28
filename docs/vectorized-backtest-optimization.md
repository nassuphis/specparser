# Plan: Massive Speedup for Full Backtest

## Current Performance Profile

For **26,676 straddles** (pattern=".", 2022-2024):

| Phase | Time | % of Total | Description |
|-------|------|------------|-------------|
| Phase 2 | 280ms | 5.3% | Expand straddles to daily rows |
| Phase 3 | 410ms | 7.8% | Resolve tickers |
| **Phase 4** | **3372ms** | **63.7%** | **Price lookup** |
| Phase 5 | 230ms | 4.4% | Compute actions |
| Phase 6 | 373ms | 7.1% | Model computation (Numba) |
| Phase 7 | 6ms | 0.1% | PnL computation (Numba) |
| Phase 8 | 618ms | 11.7% | Output assembly |
| **Total** | **5289ms** | | |

## Full Scale Target

- **231,192 straddles** (pattern=".", 2001-2026)
- **~17 million days**
- **~34 million price lookups**
- Current extrapolated time: ~2 minutes

**Goal: Sub-10 second full backtest** (10-20x speedup)

## The Problem: Phase 4 Price Lookup (63.7%)

Current implementation does:
```python
for s in range(n_straddles):
    for i in range(length):
        date_str = _date32_to_isoformat(int(dates[idx]))  # Python call
        key = f'{ticker}|{field}|{date_str}'              # String concat
        value_str = prices_dict.get(key)                  # Dict lookup
        float_val = float(value_str)                      # String->float
```

Problems:
1. **Python loop overhead** - 34M iterations through Python interpreter
2. **String key construction** - 34M f-string concatenations
3. **Date conversion** - 17M `date.fromordinal()` calls
4. **String->float conversion** - ~25M `float()` calls

## Optimization Strategies

### Strategy 1: Restructure Price Data for Numba Access

**Key insight**: The bottleneck is Python iteration + string operations. Numba can't use Python dicts or strings. We need a **numeric-only lookup structure**.

#### 1.1 Integer-keyed Price Storage

Replace string keys with integer hash:
```python
# Current: prices_dict["LA1 Comdty|PX_LAST|2024-01-15"] = "2345.67"

# New: Build at load time
# ticker_field_to_idx: {"LA1 Comdty|PX_LAST": 0, ...}  # O(unique ticker-fields) ~4600
# date_to_col: {20240115: 0, 20240116: 1, ...}         # O(unique dates) ~6500
# price_matrix: float64[n_ticker_fields, n_dates]      # ~4600 × 6500 = 30M entries
```

Lookup becomes:
```python
@njit
def lookup_price(price_matrix, ticker_field_idx, date_col):
    return price_matrix[ticker_field_idx, date_col]  # O(1) array access
```

#### 1.2 Pre-compute Lookup Indices

Before the main loop, build:
```python
# For each straddle day, pre-compute:
vol_ticker_idx: int32[n_days]     # Index into price_matrix rows
hedge_ticker_idx: int32[n_days]   # Index into price_matrix rows
date_col_idx: int32[n_days]       # Index into price_matrix cols
```

Then the lookup is pure Numba:
```python
@njit(parallel=True)
def batch_price_lookup(price_matrix, vol_ticker_idx, hedge_ticker_idx, date_col_idx):
    n = len(date_col_idx)
    vol = np.empty(n, dtype=np.float64)
    hedge = np.empty(n, dtype=np.float64)
    for i in prange(n):
        vol[i] = price_matrix[vol_ticker_idx[i], date_col_idx[i]]
        hedge[i] = price_matrix[hedge_ticker_idx[i], date_col_idx[i]]
    return vol, hedge
```

**Expected speedup**: 10-50x for Phase 4 (from 3372ms to ~100ms)

### Strategy 2: Vectorize Date Operations

#### 2.1 Pre-build Date Index Mapping

```python
# Build once at prices load time
min_date32 = min(all_date32_values)  # e.g., 11323 (2001-01-01)
max_date32 = max(all_date32_values)  # e.g., 20820 (2026-12-31)
n_dates = max_date32 - min_date32 + 1

# Dense array: date32 -> column index (-1 if no prices for that date)
date32_to_col: int32[n_dates]  # ~9500 entries
```

Lookup:
```python
@njit
def date32_to_col_idx(date32, min_date32, date32_to_col):
    offset = date32 - min_date32
    if offset < 0 or offset >= len(date32_to_col):
        return -1
    return date32_to_col[offset]
```

### Strategy 3: Batch Ticker Resolution

Current: Resolve tickers per-straddle with Python dict lookups.

Optimization:
1. **Cache ticker indices** at the straddle level
2. Use **Numba-typed dict** for ticker->idx mapping
3. Pre-build **ticker_idx arrays** before the price lookup loop

### Strategy 4: Parallel Output Assembly

Phase 8 (618ms) does string formatting in Python. Options:
1. **Return Arrow table** directly (no Python string conversion)
2. **Parallel string formatting** using `concurrent.futures`
3. **Lazy formatting** - only format rows that are actually used

### Strategy 5: Memory-Mapped Price Matrix

For the full 2001-2026 range:
- ~4600 ticker-fields × ~6500 dates = 30M float64 = 240MB

Options:
1. **Dense matrix in RAM** (240MB is manageable)
2. **Sparse matrix** (if many missing prices)
3. **Memory-mapped file** (zero copy, OS handles caching)

## Implementation Plan

### Phase A: Price Matrix Storage (HIGHEST IMPACT)

1. **Modify `load_all_prices()`** to also build:
   - `price_matrix: np.ndarray[float64]` - shape (n_ticker_fields, n_dates)
   - `ticker_field_to_row: dict[str, int]` - ticker|field -> row index
   - `date32_to_col: np.ndarray[int32]` - date32 -> col index (dense)
   - `min_date32: int` - minimum date for offset calculation

2. **Add new function `load_prices_matrix()`**:
   ```python
   def load_prices_matrix(prices_parquet: str, start_date: str, end_date: str) -> PriceMatrix:
       """Load prices into a Numba-friendly matrix structure."""
   ```

3. **Add Numba kernel `batch_price_lookup()`**:
   ```python
   @njit(parallel=True)
   def batch_price_lookup(
       price_matrix: np.ndarray,
       vol_row_idx: np.ndarray,
       hedge_row_idx: np.ndarray,
       col_idx: np.ndarray,
   ) -> tuple[np.ndarray, np.ndarray]:
   ```

### Phase B: Pre-compute Lookup Indices

1. **Build ticker-to-row mapping** for each straddle's vol/hedge tickers
2. **Build date32-to-col indices** for all days
3. **Expand to per-day arrays** (vol_row_idx, hedge_row_idx, col_idx)

### Phase C: Integrate into get_straddle_backtests()

1. Add `price_lookup="matrix"` option
2. Replace Phase 4 loop with:
   ```python
   if price_lookup == "matrix":
       vol_row_idx, hedge_row_idx = _build_ticker_row_indices(...)
       col_idx = _build_date_col_indices(dates, min_date32, date32_to_col)
       vol_array, hedge_array = batch_price_lookup(
           price_matrix, vol_row_idx, hedge_row_idx, col_idx
       )
   ```

### Phase D: Optimize Secondary Bottlenecks

1. **Phase 8 (output)**: Return Arrow table instead of list-of-lists
2. **Phase 3 (tickers)**: Cache resolution results more aggressively
3. **Phase 2 (expand)**: Use Numba for date expansion

## Data Structures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Price Matrix Structure                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ticker_field_to_row: dict[str, int]                                        │
│  ┌─────────────────────────────────────┐                                    │
│  │ "LA1 Comdty|PX_LAST" -> 0           │                                    │
│  │ "LA1 Comdty|MID_IMPL_VOL" -> 1      │     n_ticker_fields ≈ 4600         │
│  │ "CO1 Comdty|PX_LAST" -> 2           │                                    │
│  │ ...                                 │                                    │
│  └─────────────────────────────────────┘                                    │
│                                                                             │
│  date32_to_col: int32[max_date32 - min_date32 + 1]                         │
│  ┌─────────────────────────────────────┐                                    │
│  │ [0, 1, -1, 2, 3, ...]               │     Weekends/holidays = -1         │
│  └─────────────────────────────────────┘     n_business_days ≈ 6500         │
│                                                                             │
│  price_matrix: float64[n_ticker_fields, n_business_days]                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │     date0    date1    date2    date3    ...                         │   │
│  │  ┌────────┬────────┬────────┬────────┬─────────────────────────┐   │   │
│  │  │ 2345.5 │ 2350.0 │  NaN   │ 2355.2 │ ...  │ ticker_field 0   │   │   │
│  │  │  25.3  │  24.8  │  25.1  │  25.5  │ ...  │ ticker_field 1   │   │   │
│  │  │  82.1  │  81.9  │  82.5  │  NaN   │ ...  │ ticker_field 2   │   │   │
│  │  │  ...   │  ...   │  ...   │  ...   │ ...  │ ...              │   │   │
│  │  └────────┴────────┴────────┴────────┴─────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Lookup: price_matrix[ticker_field_to_row[ticker|field], date32_to_col[date32 - min_date32]]
```

## Expected Performance

| Phase | Current | After Optimization | Speedup |
|-------|---------|-------------------|---------|
| Phase 4 (price lookup) | 3372ms | ~100ms | 30x |
| Phase 8 (output) | 618ms | ~50ms (Arrow) | 12x |
| Other phases | 1299ms | ~1000ms | 1.3x |
| **Total** | **5289ms** | **~1150ms** | **4.6x** |

For full scale (231K straddles, 17M days):
- Current: ~2 minutes
- After: ~25 seconds

## Files to Modify

| File | Changes |
|------|---------|
| `prices.py` | Add `load_prices_matrix()`, `PriceMatrix` class |
| `valuation_numba.py` | Add `batch_price_lookup()` kernel |
| `valuation.py` | Add `price_lookup="matrix"` option, integrate matrix lookup |
| `__init__.py` | Export new functions |

## Verification

```bash
# Compare outputs
uv run python -c "
from specparser.amt import get_straddle_backtests, load_all_prices, set_prices_dict

# Dict-based (baseline)
prices_dict = load_all_prices('data/prices.parquet')
set_prices_dict(prices_dict)
result_dict = get_straddle_backtests('.', 2022, 2024, 'data/amt.yml', price_lookup='dict')

# Matrix-based (new)
result_matrix = get_straddle_backtests('.', 2022, 2024, 'data/amt.yml',
                                        prices_parquet='data/prices.parquet',
                                        price_lookup='matrix')

assert len(result_dict['rows']) == len(result_matrix['rows'])
print('Outputs match!')
"
```

## Summary

The key insight is that **Python string operations and dict lookups cannot be optimized by Numba**. The solution is to:

1. **Pre-convert prices to a numeric matrix** at load time
2. **Pre-compute integer indices** for ticker/date lookups
3. **Use Numba parallel loops** for the actual lookup

This transforms 34M Python dict lookups into 34M NumPy array accesses in a parallel Numba kernel, which should be 10-50x faster.
