# Plan: Fix Entry Date Calculation

## Problem

Current behavior for F/R/W rules:
- Entry anchor = Nth weekday of entry month (using xprc, xprv)
- Then count `ntrv` **good days** after anchor

Current behavior for BD rules:
- Entry = count `ntrv + xprv` **good days** from month start
- Expiry = count `xprv` **good days** from month start

Correct behavior (unified for all codes):
- Compute anchor date based on code (F/R/W = Nth weekday, BD = Nth business day)
- Add `ntrv` **calendar days** to anchor for entry
- Find first good day at or after that date
- If past entry month or no good days found, use last good day of entry month

## Current Code

In `_compute_actions()` (lines ~994-1011):

```python
elif xprc in ["F", "R", "W"]:
    # Rule 4: Weekday entry trigger ("ntry")
    if ntry is not None and ntrm is not None:
        entry_anchor = _anchor_day(xprc, xprv, ntry, ntrm)
        if entry_anchor is not None:
            try:
                ntrv_int = int(ntrv) if ntrv else 0
                _, entry_num_days = calendar.monthrange(ntry, ntrm)
                entry_month_end = f"{ntry}-{ntrm:02d}-{entry_num_days:02d}"

                idx = _nth_good_day_after(rows, vol_idx, hedge_indices, date_idx,
                                          entry_anchor, ntrv_int, entry_month_end)
                if idx is not None:
                    actions[idx] = "ntry"
            except (ValueError, TypeError):
                pass
```

The issue is `_nth_good_day_after(anchor, ntrv_int, ...)` counts `ntrv` **good days** after anchor.

## Proposed Changes

### 1. Add helper function: `_add_calendar_days()`

```python
def _add_calendar_days(date_str: str, days: int) -> str:
    """Add calendar days to a date string.

    Args:
        date_str: ISO date string like "2024-01-19"
        days: Number of calendar days to add (can be 0)

    Returns:
        New date string
    """
    from datetime import date, timedelta
    d = date.fromisoformat(date_str)
    return (d + timedelta(days=days)).isoformat()
```

### 2. Add helper function: `_last_good_day_in_month()`

```python
def _last_good_day_in_month(
    rows: list[list],
    vol_idx: int,
    hedge_indices: list[int],
    date_idx: int,
    year: int,
    month: int
) -> int | None:
    """Find the last good day in a given month.

    Returns row index of last good day, or None if no good days exist.
    """
    month_start = f"{year}-{month:02d}-01"
    _, num_days = calendar.monthrange(year, month)
    month_end = f"{year}-{month:02d}-{num_days:02d}"

    def is_good_day(row: list) -> bool:
        if row[vol_idx] == "none":
            return False
        return all(row[idx] != "none" for idx in hedge_indices)

    last_good_idx = None
    for i, row in enumerate(rows):
        row_date = row[date_idx]
        if row_date < month_start:
            continue
        if row_date > month_end:
            break
        if is_good_day(row):
            last_good_idx = i

    return last_good_idx
```

### 3. Modify `_anchor_day()` to support BD code

```python
def _anchor_day(xprc: str, xprv: str, year: int, month: int) -> str | None:
    """
    Calculate the anchor day for a given month.

    Args:
        xprc: Code ("F" for Friday, "R" for Thursday, "W" for Wednesday, "BD" for business day)
        xprv: Value (string, the Nth occurrence)
        year: Year (int)
        month: Month (int, 1-12)

    Returns:
        Date string in ISO format (e.g., "2024-06-21"), or None if not found
    """
    WEEKDAY_MAP = {"F": 4, "R": 3, "W": 2}  # Friday, Thursday, Wednesday

    try:
        n = int(xprv)
        if n < 1:
            return None
    except (ValueError, TypeError):
        return None

    _, num_days = calendar.monthrange(year, month)

    if xprc == "BD":
        # Find Nth business day (Mon-Fri) in the month
        bd_count = 0
        for day in range(1, num_days + 1):
            from datetime import date
            d = date(year, month, day)
            if d.weekday() < 5:  # Mon=0, Fri=4
                bd_count += 1
                if bd_count == n:
                    return d.isoformat()
        return None  # Not enough business days

    elif xprc in WEEKDAY_MAP:
        target_weekday = WEEKDAY_MAP[xprc]
        weekday_dates = []
        for day in range(1, num_days + 1):
            from datetime import date
            d = date(year, month, day)
            if d.weekday() == target_weekday:
                weekday_dates.append(d)
        if n > len(weekday_dates):
            return None
        return weekday_dates[n - 1].isoformat()

    return None
```

### 4. Unify entry/expiry logic for all codes (F/R/W/BD)

The entry and expiry logic becomes the same for all codes:

```python
# Unified entry logic for F/R/W/BD
if xprc in ["F", "R", "W", "BD"]:
    # Entry trigger ("ntry")
    # Anchor = Nth weekday/business day of entry month, then add ntrv calendar days
    if ntry is not None and ntrm is not None:
        entry_anchor = _anchor_day(xprc, xprv, ntry, ntrm)
        if entry_anchor is not None:
            try:
                ntrv_int = int(ntrv) if ntrv else 0
                _, entry_num_days = calendar.monthrange(ntry, ntrm)
                entry_month_end = f"{ntry}-{ntrm:02d}-{entry_num_days:02d}"

                # Add calendar days to anchor
                target_date = _add_calendar_days(entry_anchor, ntrv_int)

                # If target is past entry month, use last good day of month
                if target_date > entry_month_end:
                    idx = _last_good_day_in_month(rows, vol_idx, hedge_indices, date_idx, ntry, ntrm)
                else:
                    # Find first good day at or after target
                    idx = _nth_good_day_after(rows, vol_idx, hedge_indices, date_idx,
                                              target_date, 0, entry_month_end)
                    # If no good day found at or after target, use last good day of month
                    if idx is None:
                        idx = _last_good_day_in_month(rows, vol_idx, hedge_indices, date_idx, ntry, ntrm)

                if idx is not None:
                    actions[idx] = "ntry"
            except (ValueError, TypeError):
                pass

    # Expiry trigger ("xpry")
    # Anchor = Nth weekday/business day of expiry month, first good day at or after
    if xpry is not None and xprm is not None:
        expiry_anchor = _anchor_day(xprc, xprv, xpry, xprm)
        if expiry_anchor is not None:
            _, expiry_num_days = calendar.monthrange(xpry, xprm)
            expiry_month_end = f"{xpry}-{xprm:02d}-{expiry_num_days:02d}"

            idx = _nth_good_day_after(rows, vol_idx, hedge_indices, date_idx,
                                      expiry_anchor, 0, expiry_month_end)
            if idx is not None:
                actions[idx] = "xpry"
```

## Key Differences

| Aspect | Old Behavior | New Behavior |
|--------|-------------|--------------|
| ntrv meaning | Count N good days after anchor | Add N calendar days to anchor |
| Finding entry | Nth good day after anchor | First good day at or after (anchor + ntrv) |
| Past month end | No entry found | Last good day of entry month |
| No good day at target | No entry found | Last good day of entry month |

## Files to Modify

1. **`src/specparser/amt/tickers.py`**:
   - Add `_add_calendar_days()` helper
   - Add `_last_good_day_in_month()` helper
   - Modify F/R/W entry logic in `_compute_actions()`

## Testing

### Manual Tests

```bash
# Test case that previously failed (10 calendar days from 3rd Friday)
uv run python -m specparser.amt.tickers data/amt.yml --asset-days 'TSLA US Equity' 2024 3 2

# Expected: ntry should appear on first good day at or after 2024-01-29
```

### Unit Tests

Add to `tests/test_amt.py`:

1. **`test_entry_calendar_days_basic`**: anchor + ntrv lands on good day → that day is ntry
2. **`test_entry_calendar_days_weekend`**: anchor + ntrv lands on weekend → first good day after is ntry
3. **`test_entry_calendar_days_past_month`**: anchor + ntrv > month end → last good day of month is ntry
4. **`test_entry_calendar_days_no_good_after_target`**: no good days after target → last good day of month is ntry
5. **`test_entry_calendar_days_no_good_in_month`**: no good days in month → no ntry action
6. **`test_add_calendar_days`**: verify helper function

## Example Calculation

Straddle: `|2024-01|2024-03|F|10|F|3|12.5|`

1. Entry month = January 2024
2. Anchor = 3rd Friday of January 2024 = 2024-01-19
3. Target = 2024-01-19 + 10 days = 2024-01-29
4. Entry month end = 2024-01-31
5. Target (2024-01-29) ≤ month end (2024-01-31) → search for first good day at or after 2024-01-29
6. From the data: 2024-01-29 has vol=42.4, hedge=191 → good day
7. ntry = 2024-01-29
