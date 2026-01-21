# Plan: Implement Action Column for `get_straddle_days`

> **Status: IMPLEMENTED** ✓
>
> All changes have been implemented and tested. See implementation notes at the end.

## Overview

Add an "action" column to the output of `get_straddle_days` that flags certain conditions being met. The action is determined by analyzing straddle parameters (`ntrc`, `ntrv`, `xprc`, `xprv`) combined with daily price data (`vol`, `hedge`).

## Current Output Structure

```python
{
    "columns": ["asset", "straddle", "date", "vol", "hedge", "hedge1", ...],
    "rows": [
        ["CL Comdty", "|2024-05|2024-06|N|0|OVERRIDE|15|33.3|", "2024-05-01", "25.3", "78.50", "79.20"],
        ["CL Comdty", "|2024-05|2024-06|N|0|OVERRIDE|15|33.3|", "2024-05-02", "24.8", "77.90", "78.80"],
        ...
    ]
}
```

## Proposed Output Structure

```python
{
    "columns": ["asset", "straddle", "date", "vol", "hedge", "hedge1", ..., "action"],
    "rows": [
        ["CL Comdty", "|...|", "2024-05-01", "25.3", "78.50", "79.20", "-"],
        ["CL Comdty", "|...|", "2024-05-02", "24.8", "77.90", "78.80", "-"],
        ...
        ["CL Comdty", "|...|", "2024-05-15", "26.1", "80.10", "81.00", "ntry"],  # Action triggered
        ...
    ]
}
```

## Action Rules

### Default
- Default value: `"-"` (no action)

### Rule 1: BD Entry Trigger (`"ntry"`)

**Condition:** `xprc == "BD"` (Business Day based expiry)

**Logic:**
1. Count consecutive days (from start) where **both** `vol` and `hedge` are not `"none"`
2. When count equals `int(ntrv) + int(xprv)`, set action to `"ntry"`
3. Only one row gets the `"ntry"` action (the trigger day)

**Example:**
- Straddle: `|2024-05|2024-06|N|5|BD|10|33.3|`
- `ntrv = "5"`, `xprv = "10"`, `xprc = "BD"`
- Trigger threshold: `5 + 10 = 15` business days with valid data
- On the 15th day with valid vol+hedge data → action = `"ntry"`

### Rule 2: BD Expiry Trigger (`"xpry"`)

**Condition:** `xprc == "BD"` (Business Day based expiry)

**Logic:**
1. Find the first day of the expiry month: `{xpry}-{xprm:02d}-01`
2. Count days (from that date onwards) where **both** `vol` and all `hedge` columns are not `"none"`
3. When count equals `int(xprv)`, set action to `"xpry"`
4. Only one row gets the `"xpry"` action (the trigger day)

**Example:**
- Straddle: `|2024-05|2024-06|N|5|BD|10|33.3|`
- `xpry = 2024`, `xprm = 6`, `xprv = "10"`, `xprc = "BD"`
- First day of expiry month: `2024-06-01`
- Trigger threshold: `10` business days with valid data from expiry month start
- On the 10th day (from June 1st) with valid vol+hedge data → action = `"xpry"`

**Key Difference from Rule 1:**
- Rule 1 ("ntry"): counts from the beginning of all data, threshold = `ntrv + xprv`
- Rule 2 ("xpry"): counts from first day of expiry month only, threshold = `xprv`

### Rule 3: Weekday Expiry Trigger (`"xpr"`)

**Condition:** `xprc in ["F", "R", "W"]` (Friday, Thursday, Wednesday based expiry)

**Weekday mapping:**
- `"F"` = Friday (weekday 4)
- `"R"` = Thursday (weekday 3)
- `"W"` = Wednesday (weekday 2)

**Logic:**
1. Parse `N = int(xprv)` - the Nth occurrence of the weekday
2. Find the Nth weekday (F/R/W) of the **expiry month** (`xpry-xprm`) → this is the `expiry_anchor`
3. If `expiry_anchor` is a "good day" (vol + all hedges valid), set action to `"xpr"` on that day
4. If `expiry_anchor` is NOT a good day, find the **next good day** after it and set action to `"xpr"`
5. If no good day is found after `expiry_anchor`, no `"xpr"` action is set

**Example:**
- Straddle: `|2024-05|2024-06|N|5|F|3|33.3|`
- `xprc = "F"` (Friday), `xprv = "3"`, `xpry = 2024`, `xprm = 6`
- Find the 3rd Friday of June 2024 → `2024-06-21` (expiry_anchor)
- If 2024-06-21 has valid vol+hedge → action = `"xpr"` on that day
- If 2024-06-21 has "none" for vol or hedge → find next good day after 2024-06-21

### Rule 4: Weekday Entry Trigger (`"ntry"`)

**Condition:** `xprc in ["F", "R", "W"]` (Friday, Thursday, Wednesday based expiry)

**Weekday mapping:** Same as Rule 3

**Logic:**
1. Parse `N = int(xprv)` - the Nth occurrence of the weekday
2. Find the Nth weekday (F/R/W) of the **entry month** (`ntry-ntrm`) → this is the `entry_anchor`
3. Count `ntrv` good days starting from `entry_anchor`:
   - If `ntrv = 0` and `entry_anchor` is a good day → action = `"ntry"` on `entry_anchor`
   - If `ntrv > 0`, count good days from `entry_anchor` onwards until we reach `ntrv` good days, then set action = `"ntry"`
4. If we can't find enough good days, no `"ntry"` action is set

**Example:**
- Straddle: `|2024-05|2024-06|N|5|F|3|33.3|`
- `xprc = "F"` (Friday), `xprv = "3"`, `ntrv = "5"`, `ntry = 2024`, `ntrm = 5`
- Find the 3rd Friday of May 2024 → `2024-05-17` (entry_anchor)
- Count 5 good days from 2024-05-17 onwards → action = `"ntry"` on the 5th good day

**Key Differences:**
| Rule | Condition | Anchor | Count From | Threshold | Action |
|------|-----------|--------|------------|-----------|--------|
| 1 (BD) | `xprc == "BD"` | First row | Beginning | `ntrv + xprv` | `"ntry"` |
| 2 (BD) | `xprc == "BD"` | First of expiry month | Expiry month start | `xprv` | `"xpry"` |
| 3 (F/R/W) | `xprc in ["F","R","W"]` | Nth weekday of expiry month | Anchor day | 0 (anchor or next good) | `"xpr"` |
| 4 (F/R/W) | `xprc in ["F","R","W"]` | Nth weekday of entry month | Anchor day | `ntrv` | `"ntry"` |

---

## Counting Semantics (CONFIRMED)

### "Good Day" Counting Rules

**Anchor = Day 0 Rule:**
- Day 0 = anchor day if anchor is good, otherwise first good day after anchor
- Day 1 = 1st good day after Day 0
- Day 2 = 1st good day after Day 1 (= 2nd good day after anchor)
- etc.

**Implications:**
- `n = 0`: action on Day 0 (anchor if good, else first good day after anchor)
- `n = 1`: action on Day 1 (first good day after Day 0)
- `n = 2`: action on Day 2 (second good day after Day 0)

### BD (Business Day) Counting

- `BD 1` = first good day of the month
- `BD 2` = second good day of the month
- `BD 0` = **invalid/meaningless** (no such thing as zeroth business day)

### Entry Value (ntrv) Counting

The entry value `ntrv` is an **offset** from the entry anchor:
- `ntrv = 0`: action on the entry anchor equivalent (Day 0)
- `ntrv = 1`: action on 1st good day after entry anchor
- `ntrv = 2`: action on 2nd good day after entry anchor

This makes sense because `ntrv` represents "how many good days to wait after the anchor" - zero waiting means act on the anchor itself.

---

## Helper Functions

To avoid code duplication and ensure consistent counting semantics, we propose two helper functions:

### `_nth_good_day(rows, columns, vol_idx, hedge_indices, month_start, n)`

Find the Nth good day starting from a month.

**Parameters:**
- `rows`: data rows
- `columns`: column names
- `vol_idx`: index of vol column
- `hedge_indices`: list of hedge column indices
- `month_start`: date string like "2024-06-01" to start counting from
- `n`: which good day (1 = first, 2 = second, etc.)

**Returns:** Row index of the Nth good day, or None if not found

**Use cases:**
- Rule 1 (BD): `_nth_good_day(rows, cols, vol_idx, hedge_indices, "2024-05-01", ntrv + xprv)`
- Rule 2 (BD): `_nth_good_day(rows, cols, vol_idx, hedge_indices, expiry_month_start, xprv)`

**Note:** For BD rules, `n >= 1` always (BD 0 is invalid).

### `_nth_good_day_after(rows, columns, vol_idx, hedge_indices, date_idx, anchor_date, n)`

Find the Nth good day after (or at) an anchor date.

**Parameters:**
- `rows`: data rows
- `columns`: column names
- `vol_idx`: index of vol column
- `hedge_indices`: list of hedge column indices
- `date_idx`: index of date column
- `anchor_date`: date string like "2024-05-17" (the anchor)
- `n`: offset from anchor (0 = anchor or first good after, 1 = first good after Day 0, etc.)

**Returns:** Row index of the target good day, or None if not found

**Special semantics for n:**
- `n = 0`: Return anchor row if anchor is good, else first good day after anchor
- `n > 0`: Return Nth good day after Day 0

**Use cases:**
- Rule 3 (F/R/W expiry): `_nth_good_day_after(..., expiry_anchor, 0)` - anchor or next good
- Rule 4 (F/R/W entry): `_nth_good_day_after(..., entry_anchor, ntrv)` - ntrv days after anchor

---

## Revised Rule Implementations

### Rule 1: BD Entry Trigger (`"ntry"`)

```python
# threshold = ntrv + xprv (must be >= 1, since BD 0 is invalid)
threshold = int(ntrv) + int(xprv)
if threshold < 1:
    return  # Invalid threshold
idx = _nth_good_day(rows, columns, vol_idx, hedge_indices, entry_month_start, threshold)
if idx is not None:
    actions[idx] = "ntry"
```

### Rule 2: BD Expiry Trigger (`"xpry"`)

```python
# threshold = xprv (must be >= 1, since BD 0 is invalid)
threshold = int(xprv)
if threshold < 1:
    return  # Invalid threshold
idx = _nth_good_day(rows, columns, vol_idx, hedge_indices, expiry_month_start, threshold)
if idx is not None:
    actions[idx] = "xpry"
```

### Rule 3: Weekday Expiry Trigger (`"xpr"`)

```python
# Find Nth weekday of expiry month
expiry_anchor = _find_nth_weekday(xpry, xprm, int(xprv), weekday_map[xprc])
# Find Day 0 (anchor if good, else first good after)
idx = _nth_good_day_after(rows, columns, vol_idx, hedge_indices, date_idx, expiry_anchor, 0)
if idx is not None:
    actions[idx] = "xpr"
```

### Rule 4: Weekday Entry Trigger (`"ntry"`)

```python
# Find Nth weekday of entry month (using xprv for N, same weekday)
entry_anchor = _find_nth_weekday(ntry, ntrm, int(xprv), weekday_map[xprc])
# Find Day ntrv (offset from anchor)
idx = _nth_good_day_after(rows, columns, vol_idx, hedge_indices, date_idx, entry_anchor, int(ntrv))
if idx is not None:
    actions[idx] = "ntry"
```

---

## Additional Helper

### `_find_nth_weekday(year, month, n, weekday)`

Find the Nth occurrence of a weekday in a month.

**Parameters:**
- `year`: year (int)
- `month`: month (int, 1-12)
- `n`: which occurrence (1 = first, 2 = second, 3 = third, etc.)
- `weekday`: Python weekday (0=Monday, 1=Tuesday, ..., 4=Friday)

**Returns:** Date string in ISO format (e.g., "2024-06-21")

**Example:**
```python
_find_nth_weekday(2024, 6, 3, 4)  # 3rd Friday of June 2024
# Returns: "2024-06-21"
```

---

## Clarifications (CONFIRMED)

1. **Rule 1 threshold validation:** Silently produce no action if threshold is invalid (BD 0 meaningless).

2. **Weekday anchor not in data:** No action if anchor date not in data rows.

3. **Rule 3 action name:** Changed to `"xpry"` (same as BD rule) for consistency.

4. **Search scope for "next good day":** Only within expiry month. If no good day found in expiry month after anchor, no action.

5. **Anchor calculation:** `xprc` and `xprv` determine the anchor day for any month. Create `_anchor_day(xprc, xprv, year, month)` helper that returns the date string for the anchor. Works for W/R/F codes only.

---

## Revised Helper Functions

### `_anchor_day(xprc, xprv, year, month)`

Calculate the anchor day for a given month based on xprc/xprv.

**Parameters:**
- `xprc`: Expiry code ("F", "R", "W")
- `xprv`: Expiry value (string, the Nth occurrence)
- `year`: Year (int)
- `month`: Month (int, 1-12)

**Returns:** Date string in ISO format (e.g., "2024-06-21"), or None if xprc not in ["F", "R", "W"]

**Weekday mapping:**
- `"F"` = Friday (weekday 4)
- `"R"` = Thursday (weekday 3)
- `"W"` = Wednesday (weekday 2)

**Example:**
```python
_anchor_day("F", "3", 2024, 6)  # 3rd Friday of June 2024
# Returns: "2024-06-21"

_anchor_day("BD", "10", 2024, 6)  # Not a weekday code
# Returns: None
```

### `_nth_good_day(rows, vol_idx, hedge_indices, date_idx, month_start, n)`

Find the Nth good day starting from a month (for BD rules).

**Parameters:**
- `rows`: data rows
- `vol_idx`: index of vol column
- `hedge_indices`: list of hedge column indices
- `date_idx`: index of date column
- `month_start`: date string like "2024-06-01" to start counting from
- `n`: which good day (1 = first, 2 = second, etc.)

**Returns:** Row index of the Nth good day, or None if not found

**Note:** For BD rules, `n >= 1` always (BD 0 is invalid).

### `_nth_good_day_after(rows, vol_idx, hedge_indices, date_idx, anchor_date, n, month_limit=None)`

Find the Nth good day after (or at) an anchor date.

**Parameters:**
- `rows`: data rows
- `vol_idx`: index of vol column
- `hedge_indices`: list of hedge column indices
- `date_idx`: index of date column
- `anchor_date`: date string like "2024-05-17" (the anchor)
- `n`: offset from anchor (0 = anchor or first good after, 1 = first good after Day 0, etc.)
- `month_limit`: Optional date string like "2024-06-30" - stop searching after this date

**Returns:** Row index of the target good day, or None if not found

**Special semantics for n:**
- `n = 0`: Return anchor row if anchor is good, else first good day after anchor (within month_limit)
- `n > 0`: Return Nth good day after Day 0 (within month_limit if specified)

---

## Final Rule Implementations

### Rule 1: BD Entry Trigger (`"ntry"`)

```python
if xprc == "BD":
    threshold = int(ntrv) + int(xprv)
    if threshold < 1:
        pass  # No action (BD 0 invalid)
    else:
        idx = _nth_good_day(rows, vol_idx, hedge_indices, date_idx, entry_month_start, threshold)
        if idx is not None:
            actions[idx] = "ntry"
```

### Rule 2: BD Expiry Trigger (`"xpry"`)

```python
if xprc == "BD":
    threshold = int(xprv)
    if threshold < 1:
        pass  # No action (BD 0 invalid)
    else:
        idx = _nth_good_day(rows, vol_idx, hedge_indices, date_idx, expiry_month_start, threshold)
        if idx is not None:
            actions[idx] = "xpry"
```

### Rule 3: Weekday Expiry Trigger (`"xpry"`)

```python
if xprc in ["F", "R", "W"]:
    expiry_anchor = _anchor_day(xprc, xprv, xpry, xprm)
    if expiry_anchor is None:
        pass  # Invalid anchor
    else:
        # Search only within expiry month
        expiry_month_end = last_day_of_month(xpry, xprm)
        idx = _nth_good_day_after(rows, vol_idx, hedge_indices, date_idx, expiry_anchor, 0, month_limit=expiry_month_end)
        if idx is not None:
            actions[idx] = "xpry"
```

### Rule 4: Weekday Entry Trigger (`"ntry"`)

```python
if xprc in ["F", "R", "W"]:
    entry_anchor = _anchor_day(xprc, xprv, ntry, ntrm)
    if entry_anchor is None:
        pass  # Invalid anchor
    else:
        # No month limit for entry - can search into expiry month if needed
        idx = _nth_good_day_after(rows, vol_idx, hedge_indices, date_idx, entry_anchor, int(ntrv))
        if idx is not None:
            actions[idx] = "ntry"
```

---

## Updated Summary Table

| Rule | Condition | Anchor | Search Scope | Threshold | Action |
|------|-----------|--------|--------------|-----------|--------|
| 1 (BD) | `xprc == "BD"` | Entry month start | All data | `ntrv + xprv` | `"ntry"` |
| 2 (BD) | `xprc == "BD"` | Expiry month start | Expiry month onwards | `xprv` | `"xpry"` |
| 3 (F/R/W) | `xprc in ["F","R","W"]` | `_anchor_day(xprc, xprv, xpry, xprm)` | **Expiry month only** | 0 (anchor or next good) | `"xpry"` |
| 4 (F/R/W) | `xprc in ["F","R","W"]` | `_anchor_day(xprc, xprv, ntry, ntrm)` | All data | `ntrv` | `"ntry"` |

---

## Final Clarifications (CONFIRMED)

1. **Search scope:** All searches are confined to their "home" month:
   - **Entry rules (Rule 1, Rule 4):** Search only within entry month (ntry-ntrm)
   - **Expiry rules (Rule 2, Rule 3):** Search only within expiry month (xpry-xprm)

2. **5th weekday edge case:** If anchor can't be found (e.g., 5th Friday doesn't exist), no action.

3. **Parameter passing:** Need to add `ntry` (entry year) and `ntrm` (entry month) to `_compute_actions` for Rule 4.

4. **BD vs F/R/W mutual exclusivity:** Use `if/elif` structure since xprc can only be one value.

5. **Helper function location:** Keep in `tickers.py` as private functions.

---

## Final Summary Table

| Rule | Condition | Anchor | Search Scope | Threshold | Action |
|------|-----------|--------|--------------|-----------|--------|
| 1 (BD) | `xprc == "BD"` | Entry month start | **Entry month only** | `ntrv + xprv` | `"ntry"` |
| 2 (BD) | `xprc == "BD"` | Expiry month start | **Expiry month only** | `xprv` | `"xpry"` |
| 3 (F/R/W) | `xprc in ["F","R","W"]` | `_anchor_day(xprc, xprv, xpry, xprm)` | **Expiry month only** | 0 (anchor or next good) | `"xpry"` |
| 4 (F/R/W) | `xprc in ["F","R","W"]` | `_anchor_day(xprc, xprv, ntry, ntrm)` | **Entry month only** | `ntrv` | `"ntry"` |

---

## Final Rule Implementations

### Rule 1: BD Entry Trigger (`"ntry"`)

```python
if xprc == "BD":
    threshold = int(ntrv) + int(xprv)
    if threshold >= 1:
        entry_month_end = last_day_of_month(ntry, ntrm)
        idx = _nth_good_day(rows, vol_idx, hedge_indices, date_idx, entry_month_start, threshold, month_limit=entry_month_end)
        if idx is not None:
            actions[idx] = "ntry"
```

### Rule 2: BD Expiry Trigger (`"xpry"`)

```python
if xprc == "BD":
    threshold = int(xprv)
    if threshold >= 1:
        expiry_month_end = last_day_of_month(xpry, xprm)
        idx = _nth_good_day(rows, vol_idx, hedge_indices, date_idx, expiry_month_start, threshold, month_limit=expiry_month_end)
        if idx is not None:
            actions[idx] = "xpry"
```

### Rule 3: Weekday Expiry Trigger (`"xpry"`)

```python
elif xprc in ["F", "R", "W"]:
    expiry_anchor = _anchor_day(xprc, xprv, xpry, xprm)
    if expiry_anchor is not None:
        expiry_month_end = last_day_of_month(xpry, xprm)
        idx = _nth_good_day_after(rows, vol_idx, hedge_indices, date_idx, expiry_anchor, 0, month_limit=expiry_month_end)
        if idx is not None:
            actions[idx] = "xpry"
```

### Rule 4: Weekday Entry Trigger (`"ntry"`)

```python
elif xprc in ["F", "R", "W"]:
    entry_anchor = _anchor_day(xprc, xprv, ntry, ntrm)
    if entry_anchor is not None:
        entry_month_end = last_day_of_month(ntry, ntrm)
        idx = _nth_good_day_after(rows, vol_idx, hedge_indices, date_idx, entry_anchor, int(ntrv), month_limit=entry_month_end)
        if idx is not None:
            actions[idx] = "ntry"
```

---

## Implementation Checklist

- [x] Add `_anchor_day(xprc, xprv, year, month)` helper
- [x] Add `_nth_good_day(rows, vol_idx, hedge_indices, date_idx, month_start, n, month_limit)` helper
- [x] Add `_nth_good_day_after(rows, vol_idx, hedge_indices, date_idx, anchor_date, n, month_limit)` helper
- [x] Add `_last_day_of_month(year, month)` helper (or use calendar.monthrange)
- [x] Update `_compute_actions` signature to add `ntry`, `ntrm` parameters
- [x] Update `get_straddle_days` to pass `entry_year`, `entry_month` to `_compute_actions`
- [x] Implement Rules 1-4 with month limits
- [x] Add tests for Rules 3 & 4
- [x] Add tests for `_anchor_day` helper
- [x] Add tests for edge cases (5th weekday, anchor not in data, etc.)

## Implementation

### 1. Create `_compute_actions` Function

```python
def _compute_actions(
    rows: list[list],
    columns: list[str],
    ntrc: str,
    ntrv: str,
    xprc: str,
    xprv: str
) -> list[str]:
    """
    Compute action values for each row in get_straddle_days output.

    Args:
        rows: Output rows from get_straddle_days (before action column added)
        columns: Column names (to find vol/hedge indices)
        ntrc: Entry code from straddle
        ntrv: Entry value from straddle
        xprc: Expiry code from straddle
        xprv: Expiry value from straddle

    Returns:
        List of action strings, one per row
    """
    actions = ["-"] * len(rows)

    # Rule 1: BD entry trigger
    if xprc == "BD":
        # Empty ntrv or xprv means no action
        if not ntrv or not xprv:
            return actions

        try:
            threshold = int(ntrv) + int(xprv)
        except (ValueError, TypeError):
            return actions  # Invalid ntrv/xprv, no action

        # Find vol index
        vol_idx = columns.index("vol") if "vol" in columns else None
        if vol_idx is None:
            return actions  # Missing vol column

        # Find all hedge column indices (hedge, hedge1, hedge2, ...)
        hedge_indices = []
        for i, col in enumerate(columns):
            if col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
                hedge_indices.append(i)

        if not hedge_indices:
            return actions  # No hedge columns

        valid_count = 0
        for i, row in enumerate(rows):
            # Check vol is valid
            if row[vol_idx] == "none":
                continue

            # Check ALL hedges are valid
            all_hedges_valid = all(row[idx] != "none" for idx in hedge_indices)
            if not all_hedges_valid:
                continue

            valid_count += 1
            if valid_count == threshold:
                actions[i] = "ntry"
                break  # Only one trigger

    return actions
```

### 2. Update `get_straddle_days`

After building `out_rows` and before returning:

```python
    # Compute action column
    ntrc_val = schedules.ntrc(straddle)
    ntrv_val = schedules.ntrv(straddle)
    xprc_val = schedules.xprc(straddle)
    xprv_val = schedules.xprv(straddle)

    actions = _compute_actions(out_rows, out_columns, ntrc_val, ntrv_val, xprc_val, xprv_val)

    # Add action column using table utilities
    result = {"columns": out_columns, "rows": out_rows}

    # Add actions to each row
    for i, action in enumerate(actions):
        result["rows"][i].append(action)
    result["columns"].append("action")

    return result
```

**Alternative using `table_add_column`:**

The `table_add_column` function adds a constant value to all rows. Since action varies per row, we need to either:
1. Manually append to each row (as shown above), or
2. Create a variant helper that accepts a list of values

Option 1 is simpler and matches the existing pattern in `get_straddle_days`.

## Files to Modify

1. **`src/specparser/amt/tickers.py`**:
   - Add `_compute_actions` function
   - Update `get_straddle_days` to compute and add action column

2. **`tests/test_amt.py`**:
   - Add tests for `_compute_actions` function
   - Add integration test for `get_straddle_days` with action column

## Test Cases

### Unit Tests for `_compute_actions`

1. **Default action (non-BD xprc)**
   ```python
   # xprc = "OVERRIDE" → all actions are "-"
   rows = [["A", "s", "2024-01-01", "10", "100"], ...]
   columns = ["asset", "straddle", "date", "vol", "hedge"]
   actions = _compute_actions(rows, columns, "N", "5", "OVERRIDE", "15")
   assert all(a == "-" for a in actions)
   ```

2. **BD trigger at exact threshold**
   ```python
   # xprc = "BD", ntrv = "2", xprv = "3" → threshold = 5
   # 5 rows with valid data → 5th row gets "ntry"
   rows = [
       ["A", "s", "2024-01-01", "10", "100"],
       ["A", "s", "2024-01-02", "11", "101"],
       ["A", "s", "2024-01-03", "12", "102"],
       ["A", "s", "2024-01-04", "13", "103"],
       ["A", "s", "2024-01-05", "14", "104"],  # 5th valid day
       ["A", "s", "2024-01-06", "15", "105"],
   ]
   columns = ["asset", "straddle", "date", "vol", "hedge"]
   actions = _compute_actions(rows, columns, "N", "2", "BD", "3")
   assert actions == ["-", "-", "-", "-", "ntry", "-"]
   ```

3. **BD with missing data delays trigger**
   ```python
   # vol or hedge = "none" doesn't count
   rows = [
       ["A", "s", "2024-01-01", "10", "100"],   # valid (1)
       ["A", "s", "2024-01-02", "none", "101"], # invalid (vol missing)
       ["A", "s", "2024-01-03", "12", "102"],   # valid (2)
       ["A", "s", "2024-01-04", "13", "none"],  # invalid (hedge missing)
       ["A", "s", "2024-01-05", "14", "104"],   # valid (3)
   ]
   columns = ["asset", "straddle", "date", "vol", "hedge"]
   actions = _compute_actions(rows, columns, "N", "1", "BD", "2")  # threshold = 3
   assert actions == ["-", "-", "-", "-", "ntry"]
   ```

4. **BD threshold never reached**
   ```python
   # Not enough valid days
   rows = [
       ["A", "s", "2024-01-01", "10", "100"],
       ["A", "s", "2024-01-02", "11", "101"],
   ]
   columns = ["asset", "straddle", "date", "vol", "hedge"]
   actions = _compute_actions(rows, columns, "N", "5", "BD", "5")  # threshold = 10
   assert actions == ["-", "-"]  # Never reached
   ```

5. **Invalid ntrv/xprv values**
   ```python
   # Non-numeric ntrv/xprv → no action
   rows = [["A", "s", "2024-01-01", "10", "100"]]
   columns = ["asset", "straddle", "date", "vol", "hedge"]
   actions = _compute_actions(rows, columns, "N", "abc", "BD", "5")
   assert actions == ["-"]
   ```

6. **Empty ntrv or xprv**
   ```python
   # Empty ntrv → no action
   rows = [["A", "s", "2024-01-01", "10", "100"]]
   columns = ["asset", "straddle", "date", "vol", "hedge"]
   actions = _compute_actions(rows, columns, "N", "", "BD", "5")
   assert actions == ["-"]

   # Empty xprv → no action
   actions = _compute_actions(rows, columns, "N", "5", "BD", "")
   assert actions == ["-"]
   ```

7. **Missing vol or hedge column**
   ```python
   # No vol column → no action
   rows = [["A", "s", "2024-01-01", "100"]]
   columns = ["asset", "straddle", "date", "hedge"]
   actions = _compute_actions(rows, columns, "N", "1", "BD", "1")
   assert actions == ["-"]
   ```

8. **Multiple hedges - all must be valid**
   ```python
   # With hedge and hedge1, BOTH must be non-"none"
   rows = [
       ["A", "s", "2024-01-01", "10", "100", "200"],   # valid (1) - all present
       ["A", "s", "2024-01-02", "11", "101", "none"],  # invalid - hedge1 missing
       ["A", "s", "2024-01-03", "12", "none", "202"],  # invalid - hedge missing
       ["A", "s", "2024-01-04", "13", "103", "203"],   # valid (2)
       ["A", "s", "2024-01-05", "14", "104", "204"],   # valid (3) - trigger!
   ]
   columns = ["asset", "straddle", "date", "vol", "hedge", "hedge1"]
   actions = _compute_actions(rows, columns, "N", "1", "BD", "2")  # threshold = 3
   assert actions == ["-", "-", "-", "-", "ntry"]
   ```

9. **Multiple hedges with hedge2**
   ```python
   # hedge, hedge1, hedge2 - all three must be valid
   rows = [
       ["A", "s", "2024-01-01", "10", "100", "200", "300"],   # valid (1)
       ["A", "s", "2024-01-02", "11", "101", "201", "none"],  # invalid - hedge2 missing
       ["A", "s", "2024-01-03", "12", "102", "202", "302"],   # valid (2) - trigger!
   ]
   columns = ["asset", "straddle", "date", "vol", "hedge", "hedge1", "hedge2"]
   actions = _compute_actions(rows, columns, "N", "1", "BD", "1")  # threshold = 2
   assert actions == ["-", "-", "ntry"]
   ```

### Integration Test

```python
def test_get_straddle_days_has_action_column(test_amt_file, chain_csv_file, prices_parquet_file):
    """Test that get_straddle_days includes action column."""
    table = get_straddle_days(test_amt_file, "CL Comdty", 2024, 6, prices_parquet_file, chain_csv_file, 0)
    assert "action" in table["columns"]
    assert table["columns"][-1] == "action"  # Last column
    # All rows have action value
    for row in table["rows"]:
        assert len(row) == len(table["columns"])
        assert row[-1] in ["-", "ntry"]  # Valid action values
```

## Future Action Rules

The `_compute_actions` function is designed to be extensible. Future rules can be added:

```python
def _compute_actions(...) -> list[str]:
    actions = ["-"] * len(rows)

    # Rule 1: BD entry trigger
    if xprc == "BD":
        # ... existing logic ...

    # Rule 2: Future rule example
    # if some_condition:
    #     # ... logic ...

    # Rule 3: Another rule
    # ...

    return actions
```

## Clarifications (Confirmed)

1. **Counting logic:** Count from the first row (row 0), skipping invalid days.

2. **Multiple hedges:** ALL hedge columns (`hedge`, `hedge1`, `hedge2`, etc.) must be valid (not "none") for a day to count.

3. **Only one "ntry" action:** Yes, only one trigger per straddle.

4. **Empty ntrv or xprv:** No action can happen if either is empty.

---

## Additional Considerations

### Hedge Column Detection

The function identifies hedge columns by pattern matching:
- `"hedge"` - exact match
- `"hedge1"`, `"hedge2"`, ... - starts with "hedge" followed by digits

This excludes any unrelated columns that might happen to start with "hedge" (unlikely but defensive).

### Performance

The action computation is O(n) where n = number of rows, with early exit when trigger is found. For typical straddles (~60-90 days), this is negligible.

### Extensibility

The `_compute_actions` function is designed to handle multiple rules. Future rules can be added as additional `if` blocks:

```python
def _compute_actions(...) -> list[str]:
    actions = ["-"] * len(rows)

    # Rule 1: BD entry trigger ("ntry")
    if xprc == "BD":
        # ... count from beginning, threshold = ntrv + xprv ...

    # Rule 2: BD expiry trigger ("xpry")
    if xprc == "BD":
        # ... count from first day of expiry month, threshold = xprv ...

    return actions
```

Note: Both rules can trigger in the same dataset - "ntry" triggers early (from start), "xpry" triggers later (from expiry month start).

### Weekends and Holidays

The current implementation counts all calendar days. The "BD" (Business Day) in `xprc` refers to the triggering condition, not filtering of days. If vol/hedge data is missing on weekends (as expected), those days simply won't count toward the threshold.

This is the correct behavior: we count days with valid data, regardless of whether they're business days.

---

## Implementation Notes

**Implemented on:** 2026-01-20
**Rule 2 added on:** 2026-01-20
**Rules 3 & 4 added on:** 2026-01-20

### Files Modified

1. **`src/specparser/amt/tickers.py`**:
   - Added `_anchor_day(xprc, xprv, year, month)` helper function for finding Nth weekday
   - Added `_nth_good_day(rows, vol_idx, hedge_indices, date_idx, month_start, n, month_limit)` helper for BD rules
   - Added `_nth_good_day_after(rows, vol_idx, hedge_indices, date_idx, anchor_date, n, month_limit)` helper for F/R/W rules
   - Updated `_compute_actions(rows, columns, ntrc, ntrv, xprc, xprv, xpry, xprm, ntry, ntrm)` with:
     - Added `ntry` (entry year) and `ntrm` (entry month) parameters
     - Refactored Rules 1 & 2 (BD) to use `_nth_good_day` helper with month limits
     - Added Rule 3 (F/R/W expiry trigger) using `_nth_good_day_after`
     - Added Rule 4 (F/R/W entry trigger) using `_nth_good_day_after`
   - Updated `get_straddle_days()` to pass entry_year and entry_month to `_compute_actions`

2. **`tests/test_amt.py`**:
   - Added imports for `_anchor_day`, `_nth_good_day`, `_nth_good_day_after`
   - Updated existing BD rule tests to include `ntry` and `ntrm` parameters
   - Added Rule 3 & 4 (F/R/W) tests to `TestComputeActions`:
     - `test_friday_expiry_trigger_on_anchor`
     - `test_friday_expiry_trigger_next_good_day`
     - `test_thursday_entry_trigger_with_offset`
     - `test_wednesday_expiry_5th_weekday_not_exist`
     - `test_friday_entry_and_expiry_different_months`
     - `test_friday_anchor_with_missing_data_delays_day0`
     - `test_bd_with_month_limit_entry`
     - `test_frw_with_month_limit_expiry`
   - Added `TestAnchorDay` class with 7 tests for `_anchor_day` helper
   - Added `TestNthGoodDay` class with 7 tests for `_nth_good_day` helper
   - Added `TestNthGoodDayAfter` class with 8 tests for `_nth_good_day_after` helper

### Verification

All 350 tests pass (197 in test_amt.py, including 29 tests for `_compute_actions` and 22 tests for helper functions).
