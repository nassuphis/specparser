# Vectorized Straddle Computation

## Performance Summary

| Phase | Time |
|-------|------|
| Load YAML + parse schedules | ~40ms |
| Load prices (8.5M rows) | ~74ms |
| Allocate & compute straddles | ~261ms |
| **Total** | **~375ms** |

The original loop-based approach took ~783ms just for the compute phase. The vectorized version is **~3x faster overall**.

## Key Techniques

### 1. Within-Group Index Generation

Generate sequential indices within groups (e.g., schedule ID within each asset):

```python
counts = aschlen[inidx]  # [2, 3, 1] - schedules per asset
si = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
# Result: [0, 1, 0, 1, 2, 0]
```

How it works:
- `cumsum(counts)` = `[2, 5, 6]` (cumulative totals)
- `cumsum - counts` = `[0, 2, 5]` (group start positions)
- `repeat(..., counts)` = `[0, 0, 2, 2, 2, 5]` (broadcast to each element)
- `arange(6) - [0, 0, 2, 2, 2, 5]` = `[0, 1, 0, 1, 2, 0]`

### 2. Cartesian Product with repeat/tile

Expand straddle arrays across all months:

```python
# sidx: asset index per straddle (length = straddles_per_month)
# ym: year-months array (length = num_months)

smidx = np.repeat(sidx, ym_len)  # each straddle repeated for all months
smym = np.tile(ym, len(sidx))    # all months tiled for each straddle
```

This creates the full (straddle × month) matrix in two operations.

### 3. Vectorized Conditionals with np.select

Replace if/elif chains with `np.select`:

```python
conditions = [
    asset_sources[smidx, 0] == HEDGE_NONFUT,
    asset_sources[smidx, 0] == HEDGE_FUT,
    asset_sources[smidx, 0] == HEDGE_CDS,
    asset_sources[smidx, 0] == HEDGE_CALC
]
choices = [
    hedge_ticker[smidx],
    hedge_fut_code[smidx] + fut_month_code + ...,
    hedge_hedge[smidx],
    hedge_ccy[smidx] + "_fsw0m_" + hedge_tenor[smidx]
]
hedge1t_vec = np.select(conditions, choices, default="")
```

### 4. Vectorized String Operations

Use `np.strings` for string manipulation:

```python
# Split strings
ntrcv, _, rest = np.strings.partition(eastmp, '_')

# Slice strings
np.strings.slice(ntrcv, 0, 1)   # first character
np.strings.slice(ntrcv, 1, 20)  # rest of string
```

### 5. Vectorized Date Arithmetic

Leap year and day-count calculation without loops:

```python
dpm = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int8)

leap = ((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))) & (month == 2)
days = np.where(leap, 29, dpm[month - 1])
```

### 6. Fancy Indexing for Matrix Lookup

Access schedule_matrix with arrays:

```python
ntrc_vec = schedule_matrix[smidx, schid_vec, 0]
```

This retrieves `schedule_matrix[asset_idx, schedule_idx, column]` for all straddles at once.

## Data Flow

```
Assets (189)
    │
    ├── aschlen: schedules per asset [4, 2, 3, ...]
    │
    ▼
Straddles per month (741)
    │
    ├── sidx: asset index for each straddle
    ├── si: schedule index within asset (0, 1, 0, 1, 2, ...)
    │
    ▼
Total straddles (222,300 = 741 × 300 months)
    │
    ├── smidx = repeat(sidx, num_months)
    ├── smym = tile(year_months, num_straddles)
    │
    ▼
Output vectors (all length 222,300)
```

## Potential Further Optimization

The remaining ~40ms in YAML loading includes a Python loop for `asset_hedge_tickers` and `asset_vol_tickers` (lines 130-151). This could potentially be vectorized or moved to Numba, but the gains would be minimal (~30ms max).

The `np.select` calls are already highly optimized. For even faster performance, the entire computation could be ported to Numba with pre-encoded integer arrays instead of strings.
