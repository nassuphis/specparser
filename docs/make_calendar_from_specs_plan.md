# Plan: `make_calendar_from_specs_par`

## Overview

Create a function that takes a uint8 matrix of straddle specs (format `|2020-01|2020-03|`) and expands them to calendar dates. This combines date parsing (`get_uint8_ym`) with calendar generation (`make_calendar_from_ranges_par`).

## Input Format

```
Straddle spec: |2020-01|2020-03|...
               ^       ^
               |       +-- expiry year-month (positions 9-15)
               +-- entry year-month (positions 1-7)
```

- Input: uint8 matrix shape `(R, W)` where `W >= 16`
- Each row contains `|YYYY-MM|YYYY-MM|...` as ASCII bytes
- Entry yearmonth at positions 1-7 (`YYYY-MM`)
- Expiry yearmonth at positions 9-15 (`YYYY-MM`)

## Output

Same as `make_calendar_from_ranges`:
- `src_idx`: int64 vector - which input row each output date came from
- `cal`: uint8 matrix `(N, 10)` - dates as `YYYY-MM-DD`

## Algorithm

Two-pass, same structure as `make_calendar_from_ranges_par`:

```python
@njit(parallel=True)
def make_calendar_from_specs_par(specs):
    """
    specs: uint8 matrix shape (R, W) with W >= 16
           Each row: |YYYY-MM|YYYY-MM|...

    Returns:
        src_idx: int64 vector length N
        cal: uint8 matrix (N, 10)
    """
    R = specs.shape[0]
    if R == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 10), dtype=np.uint8)

    # Pass 1: parse specs and compute start positions (sequential, O(R))
    src_starts = np.empty(R + 1, dtype=np.int64)
    src_starts[0] = 0

    for r in range(R):
        # Parse entry yearmonth from positions 1-7
        entry_y = _read_4digits(specs[r], 1)
        entry_m = _read_2digits(specs[r], 6)

        # Parse expiry yearmonth from positions 9-15
        expiry_y = _read_4digits(specs[r], 9)
        expiry_m = _read_2digits(specs[r], 14)

        days = days_between(entry_y, entry_m, expiry_y, expiry_m)
        src_starts[r + 1] = src_starts[r] + days

    total = src_starts[R]

    # Allocate output
    src_idx = np.empty(total, dtype=np.int64)
    cal = np.empty((total, 10), dtype=np.uint8)

    # Pass 2: fill dates (parallel)
    for r in prange(R):
        p = src_starts[r]

        # Parse again (cheap, avoids storing intermediate arrays)
        entry_y = _read_4digits(specs[r], 1)
        entry_m = _read_2digits(specs[r], 6)
        expiry_y = _read_4digits(specs[r], 9)
        expiry_m = _read_2digits(specs[r], 14)

        ym_start = entry_y * 12 + (entry_m - 1)
        ym_end = expiry_y * 12 + (expiry_m - 1)

        for ym in range(ym_start, ym_end + 1):
            year = ym // 12
            month = (ym % 12) + 1
            days = last_day_of_month(year, month)

            for d0 in range(days):
                _write_yyymmdd(cal[p], 0, year, month, d0)
                src_idx[p] = r
                p += 1

    return src_idx, cal
```

## Key Design Decisions

1. **Parse twice**: Parse in both Pass 1 and Pass 2 rather than storing intermediate arrays. Parsing is cheap (just digit arithmetic), and this avoids allocating `4 * R` int64s for year/month values.

2. **Use existing helpers**: `_read_4digits`, `_read_2digits`, `_write_yyymmdd`, `days_between`, `last_day_of_month` are already defined.

3. **Same output format**: Returns `(src_idx, cal)` just like `make_calendar_from_ranges_par` for consistency.

## Position Constants

For clarity, could define constants:

```python
# Straddle spec positions: |YYYY-MM|YYYY-MM|...
#                          0123456789...
ENTRY_YEAR_POS = 1    # "YYYY" starts at position 1
ENTRY_MONTH_POS = 6   # "MM" starts at position 6
EXPIRY_YEAR_POS = 9   # "YYYY" starts at position 9
EXPIRY_MONTH_POS = 14 # "MM" starts at position 14
```

Or just use magic numbers with comments since the format is fixed.

## Testing

```python
def test_make_calendar_from_specs_par():
    # Create specs matrix
    specs = strs2u8mat([
        "|2024-01|2024-01|extra",  # Jan 2024: 31 days
        "|2024-02|2024-02|extra",  # Feb 2024: 29 days (leap)
    ])

    src_idx, cal = make_calendar_from_specs_par(specs)

    assert len(src_idx) == 31 + 29
    assert cal.shape == (60, 10)
    assert np.all(src_idx[:31] == 0)
    assert np.all(src_idx[31:] == 1)

def test_matches_ranges_version():
    # Verify output matches make_calendar_from_ranges_par
    specs = strs2u8mat(["|2024-01|2024-03|"])
    ranges = np.array([[2024, 1, 2024, 3]], dtype=np.int64)

    idx_specs, cal_specs = make_calendar_from_specs_par(specs)
    idx_ranges, cal_ranges = make_calendar_from_ranges_par(ranges)

    np.testing.assert_array_equal(idx_specs, idx_ranges)
    np.testing.assert_array_equal(cal_specs, cal_ranges)
```

## Files to Modify

1. **`dev/strings.py`** - Add `make_calendar_from_specs_par()` after `make_calendar_from_ranges_par()`
2. **`dev/test_strings.py`** - Add tests
3. **`dev/strings.ipynb`** - Add benchmarks

## Alternative: Non-parallel version?

Could also add `make_calendar_from_specs()` (sequential) for consistency, but if the parallel version is always preferred for this use case, maybe just the `_par` version is enough.
