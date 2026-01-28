# -------------------------------------
# AMT valuation - Actions & Valuation
# -------------------------------------
"""
Valuation utilities.

Handles computing actions (entry/expiry triggers), pricing models, and PnL.
"""
import math
import csv
import calendar
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from . import loader
from . import schedules
from . import prices as prices_module
from . import strings as strings_module


# -------------------------------------
# Math helpers
# -------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# -------------------------------------
# Valuation Models
# -------------------------------------


def model_ES(row: dict[str, Any]) -> dict[str, str]:
    """European Straddle pricing model.

    Formula: mv = S * 2*N(d1) - X * 2*N(d2) + X - S
             delta = N_d1 - 1

    Inputs from row:
        - hedge: current underlying price (S)
        - strike: strike price captured at entry (X)
        - vol: current implied vol in percent
        - date: current date
        - expiry: expiry date

    Returns dict with "mv" and "delta" keys. Values are "-" for any inadequate
    inputs (missing, non-numeric, invalid dates, t < 0, zero/negative prices or vol, etc.)
    """
    try:
        from datetime import date as date_type

        S = float(row["hedge"])
        X = float(row["strike"])
        v = float(row["vol"])

        # Validate positive values
        if S <= 0 or X <= 0 or v <= 0:
            return {"mv": "-", "delta": "-"}

        # Calculate days to expiry
        current_date = date_type.fromisoformat(row["date"])
        expiry_date = date_type.fromisoformat(row["expiry"])
        t = (expiry_date - current_date).days

        if t < 0:
            return {"mv": "-", "delta": "-"}  # Past expiry - inadequate input

        if t == 0:
            # At expiry - intrinsic value, delta is +1 or -1 depending on S vs X
            mv = abs(S - X) / X
            delta = 1.0 if S >= X else -1.0
            return {"mv": str(mv), "delta": str(delta)}

        # Total volatility
        tv = (v / 100) * math.sqrt(t / 365)

        d1 = math.log(S / X) / tv + 0.5 * tv
        d2 = d1 - tv

        N_d1 = 2 * _norm_cdf(d1)
        N_d2 = 2 * _norm_cdf(d2)

        mv = S * N_d1 - X * N_d2 + X - S
        delta = N_d1 - 1

        # Return option value divided by strike, and delta
        return {"mv": str(mv / X), "delta": str(delta)}
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return {"mv": "-", "delta": "-"}


def model_NS(row: dict[str, Any]) -> dict[str, str]:
    """Normal Straddle model - placeholder."""
    return {"mv": "-", "delta": "-"}


def model_BS(row: dict[str, Any]) -> dict[str, str]:
    """Black-Scholes model - placeholder."""
    return {"mv": "-", "delta": "-"}


def model_default(row: dict[str, Any]) -> dict[str, str]:
    """Default model for unknown model names."""
    return {"mv": "-", "delta": "-"}


MODEL_DISPATCH = {
    "ES": model_ES,
    "NS": model_NS,
    "BS": model_BS,
    "CDS_ES": model_ES,  # CDS_ES uses ES model
}


# -------------------------------------
# Override expiry handling
# -------------------------------------

_OVERRIDE_CACHE: dict[tuple[str, str], str] | None = None


def _load_overrides(path: str | Path = "data/overrides.csv") -> dict[tuple[str, str], str]:
    """Load and cache override expiry dates.

    Args:
        path: Path to overrides CSV file

    Returns:
        Dict mapping (ticker, "YYYY-MM") -> "YYYY-MM-DD"
    """
    global _OVERRIDE_CACHE
    if _OVERRIDE_CACHE is not None:
        return _OVERRIDE_CACHE

    _OVERRIDE_CACHE = {}
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row["ticker"]
                expiry = row["expiry"]  # "YYYY-MM-DD"
                year_month = expiry[:7]  # "YYYY-MM"
                key = (ticker, year_month)
                _OVERRIDE_CACHE[key] = expiry
    except (FileNotFoundError, KeyError):
        pass  # Return empty cache if file not found or malformed

    return _OVERRIDE_CACHE


def clear_override_cache():
    """Clear the cached override expiry dates.

    Call this to force reloading the overrides CSV file,
    useful after modifying the file or in long-running processes.
    """
    global _OVERRIDE_CACHE
    _OVERRIDE_CACHE = None


def _override_expiry(
    underlying: str,
    year: int,
    month: int,
    overrides_path: str | Path = "data/overrides.csv"
) -> str | None:
    """Look up override expiry date for an asset/month.

    Args:
        underlying: Asset identifier (e.g., "0R Comdty")
        year: Expiry year
        month: Expiry month (1-12)
        overrides_path: Path to overrides CSV

    Returns:
        Expiry date string "YYYY-MM-DD" or None if not found
    """
    overrides = _load_overrides(overrides_path)
    key = (underlying, f"{year}-{month:02d}")
    return overrides.get(key)


# -------------------------------------
# Date/anchor computation helpers
# -------------------------------------


def _add_calendar_days(date_str: str, days: int) -> str:
    """Add calendar days to a date string.

    Args:
        date_str: ISO date string like "2024-01-19"
        days: Number of calendar days to add (can be 0)

    Returns:
        New date string
    """
    d = date.fromisoformat(date_str)
    return (d + timedelta(days=days)).isoformat()


def _last_good_day_in_month(
    rows: list[list],
    vol_idx: int,
    hedge_indices: list[int],
    date_idx: int,
    year: int,
    month: int
) -> int | None:
    """Find the last good day in a given month.

    A "good day" is one where vol and all hedge columns are not "none".

    Args:
        rows: Data rows from get_straddle_actions output
        vol_idx: Index of vol column
        hedge_indices: List of hedge column indices
        date_idx: Index of date column
        year: Year
        month: Month (1-12)

    Returns:
        Row index of last good day, or None if no good days exist.
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


def _anchor_day(
    xprc: str,
    xprv: str,
    year: int,
    month: int,
    underlying: str | None = None,
    overrides_path: str | Path | None = None
) -> str | None:
    """
    Calculate the anchor day for a given month.

    Args:
        xprc: Code ("F", "R", "W", "BD", or "OVERRIDE")
        xprv: Value (string, the Nth occurrence; ignored for OVERRIDE)
        year: Year (int)
        month: Month (int, 1-12)
        underlying: Asset name (required for OVERRIDE)
        overrides_path: Path to overrides CSV (for OVERRIDE)

    Returns:
        Date string in ISO format (e.g., "2024-06-21"), or None if:
        - xprc is not in ["F", "R", "W", "BD", "OVERRIDE"]
        - xprv is not a valid positive integer (for non-OVERRIDE)
        - The Nth weekday/business day doesn't exist in the month
        - OVERRIDE lookup fails (no entry for asset/month)
    """
    # Handle OVERRIDE first - no xprv validation needed
    if xprc == "OVERRIDE":
        if underlying is None:
            return None
        return _override_expiry(underlying, year, month,
                                overrides_path or "data/overrides.csv")

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
            d = date(year, month, day)
            if d.weekday() < 5:  # Mon=0, Fri=4
                bd_count += 1
                if bd_count == n:
                    return d.isoformat()
        return None  # Not enough business days

    elif xprc in WEEKDAY_MAP:
        target_weekday = WEEKDAY_MAP[xprc]

        # Find all occurrences of target weekday in the month
        weekday_dates = []
        for day in range(1, num_days + 1):
            d = date(year, month, day)
            if d.weekday() == target_weekday:
                weekday_dates.append(d)

        # Return Nth occurrence (1-indexed)
        if n > len(weekday_dates):
            return None  # e.g., 5th Friday doesn't exist

        return weekday_dates[n - 1].isoformat()

    return None


def _nth_good_day_after(
    rows: list[list],
    vol_idx: int,
    hedge_indices: list[int],
    date_idx: int,
    anchor_date: str,
    n: int,
    month_limit: str | None = None
) -> int | None:
    """
    Find the Nth good day after (or at) an anchor date.

    A "good day" is one where vol and all hedge columns are not "none".

    Semantics for n:
    - n = 0: Return anchor row if anchor is good, else first good day after anchor
    - n > 0: Return Nth good day after Day 0

    Args:
        rows: Data rows from get_straddle_actions output
        vol_idx: Index of vol column
        hedge_indices: List of hedge column indices
        date_idx: Index of date column
        anchor_date: Date string like "2024-05-17" (the anchor)
        n: Offset from anchor (0 = anchor or first good after, 1 = first good after Day 0, etc.)
        month_limit: Optional date string like "2024-06-30" - stop searching after this date

    Returns:
        Row index of the target good day, or None if not found
    """
    if n < 0:
        return None

    def is_good_day(row: list) -> bool:
        if row[vol_idx] == "none":
            return False
        return all(row[idx] != "none" for idx in hedge_indices)

    # First, find Day 0 (anchor if good, else first good day after anchor)
    day_0_idx = None
    for i, row in enumerate(rows):
        row_date = row[date_idx]

        # Skip rows before anchor
        if row_date < anchor_date:
            continue

        # Stop if past month_limit
        if month_limit is not None and row_date > month_limit:
            break

        if is_good_day(row):
            day_0_idx = i
            break

    if day_0_idx is None:
        return None  # No good day found at or after anchor

    if n == 0:
        return day_0_idx

    # For n > 0, count n good days after Day 0
    count = 0
    for i in range(day_0_idx + 1, len(rows)):
        row = rows[i]
        row_date = row[date_idx]

        # Stop if past month_limit
        if month_limit is not None and row_date > month_limit:
            break

        if is_good_day(row):
            count += 1
            if count == n:
                return i

    return None  # Not enough good days after Day 0


# -------------------------------------
# Action computation
# -------------------------------------


def _compute_actions(
    rows: list[list],
    columns: list[str],
    ntrc: str,
    ntrv: str,
    xprc: str,
    xprv: str,
    xpry: int | None = None,
    xprm: int | None = None,
    ntry: int | None = None,
    ntrm: int | None = None,
    underlying: str | None = None,
    overrides_path: str | Path | None = None
) -> list[str]:
    """
    Compute action values for each row in get_straddle_actions output.

    Args:
        rows: Output rows from get_straddle_actions (before action column added)
        columns: Column names (to find vol/hedge indices)
        ntrc: Entry code from straddle
        ntrv: Entry value from straddle
        xprc: Expiry code from straddle
        xprv: Expiry value from straddle
        xpry: Expiry year from straddle
        xprm: Expiry month from straddle
        ntry: Entry year from straddle
        ntrm: Entry month from straddle
        underlying: Asset name (required for OVERRIDE code)
        overrides_path: Path to overrides CSV (for OVERRIDE code)

    Returns:
        List of action strings, one per row ("-" for no action, "ntry" for entry trigger, "xpry" for expiry trigger)
    """
    actions = ["-"] * len(rows)

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

    # Find date column index
    date_idx = columns.index("date") if "date" in columns else None
    if date_idx is None:
        return actions  # Missing date column

    # Unified rules for F/R/W/BD/OVERRIDE codes
    # Entry: anchor + ntrv calendar days -> first good day at or after -> fallback to last good day
    # Expiry: anchor -> first good day at or after
    if xprc in ["F", "R", "W", "BD", "OVERRIDE"]:
        # Entry trigger ("ntry")
        if ntry is not None and ntrm is not None:
            entry_anchor = _anchor_day(xprc, xprv, ntry, ntrm, underlying, overrides_path)
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
        # Anchor -> first good day at or after
        if xpry is not None and xprm is not None:
            expiry_anchor = _anchor_day(xprc, xprv, xpry, xprm, underlying, overrides_path)
            if expiry_anchor is not None:
                _, expiry_num_days = calendar.monthrange(xpry, xprm)
                expiry_month_end = f"{xpry}-{xprm:02d}-{expiry_num_days:02d}"

                idx = _nth_good_day_after(rows, vol_idx, hedge_indices, date_idx,
                                          expiry_anchor, 0, expiry_month_end)
                if idx is not None:
                    actions[idx] = "xpry"

    return actions


# -------------------------------------
# Table building helpers
# -------------------------------------


def _find_action_indices(table: dict[str, Any]) -> tuple[int | None, int | None]:
    """Find ntry and xpry row indices from action column.

    Args:
        table: Table with 'action' column

    Returns:
        (ntry_idx, xpry_idx) - row indices or None if not found
    """
    if "action" not in table["columns"]:
        return None, None

    action_idx = table["columns"].index("action")
    ntry_idx = None
    xpry_idx = None

    for i, row in enumerate(table["rows"]):
        if row[action_idx] == "ntry":
            ntry_idx = i
        elif row[action_idx] == "xpry":
            xpry_idx = i

    return ntry_idx, xpry_idx


def _add_action_column(
    table: dict[str, Any],
    straddle: str,
    underlying: str,
    overrides_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Add action column to prices table.

    Args:
        table: Prices table with columns [asset, straddle, date, <params...>]
        straddle: Straddle string for parsing entry/expiry info
        underlying: Asset underlying for override lookups
        overrides_csv: Path to overrides CSV (for OVERRIDE code)

    Returns:
        Table with 'action' column added
    """
    ntrc_val = schedules.ntrc(straddle)
    ntrv_val = schedules.ntrv(straddle)
    xprc_val = schedules.xprc(straddle)
    xprv_val = schedules.xprv(straddle)
    entry_year, entry_month = schedules.ntry(straddle), schedules.ntrm(straddle)
    expiry_year, expiry_month = schedules.xpry(straddle), schedules.xprm(straddle)

    actions_list = _compute_actions(
        table["rows"], table["columns"],
        ntrc_val, ntrv_val, xprc_val, xprv_val,
        expiry_year, expiry_month, entry_year, entry_month,
        underlying, overrides_csv
    )

    # Add action to each row
    new_rows = [row + [action] for row, action in zip(table["rows"], actions_list)]
    new_columns = table["columns"] + ["action"]

    return {"orientation": "row", "columns": new_columns, "rows": new_rows}


def _add_model_column(
    table: dict[str, Any],
    underlying: str,
    path: str | Path,
) -> dict[str, Any]:
    """Add model column to table.

    Args:
        table: Input table
        underlying: Asset underlying
        path: Path to AMT YAML file

    Returns:
        Table with 'model' column added
    """
    asset_data = loader.get_asset(path, underlying)
    if asset_data is not None:
        valuation = asset_data.get("Valuation", {})
        model = valuation.get("Model", "") if isinstance(valuation, dict) else ""
    else:
        model = ""

    new_rows = [row + [model] for row in table["rows"]]
    new_columns = table["columns"] + ["model"]

    return {"orientation": "row", "columns": new_columns, "rows": new_rows}


def _add_strike_columns(
    table: dict[str, Any],
    ntry_idx: int | None,
    xpry_idx: int | None,
) -> dict[str, Any]:
    """Add strike_vol, strike, strike1..., and expiry columns.

    Values come from ntry row, shown only between ntry and xpry.

    Args:
        table: Table with prices, action, model columns
        ntry_idx: Row index of entry trigger (or None)
        xpry_idx: Row index of expiry trigger (or None)

    Returns:
        Table with strike and expiry columns added
    """
    columns = table["columns"]
    rows = [row[:] for row in table["rows"]]  # Copy rows

    # Find vol column index
    vol_col_idx = columns.index("vol") if "vol" in columns else None

    # Find hedge column indices
    hedge_col_indices = []
    for i, col in enumerate(columns):
        if col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
            hedge_col_indices.append(i)

    # Get strike values from ntry row
    if ntry_idx is not None and vol_col_idx is not None:
        strike_vol_value = rows[ntry_idx][vol_col_idx]
        strike_values = [rows[ntry_idx][idx] for idx in hedge_col_indices]
    else:
        strike_vol_value = "-"
        strike_values = ["-"] * len(hedge_col_indices)

    # Add strike_vol column
    new_columns = columns + ["strike_vol"]
    for i, row in enumerate(rows):
        in_range = (ntry_idx is not None and i >= ntry_idx and
                    (xpry_idx is None or i <= xpry_idx))
        row.append(strike_vol_value if in_range else "-")

    # Add strike columns (one per hedge)
    for j in range(len(hedge_col_indices)):
        strike_col_name = "strike" if j == 0 else f"strike{j}"
        for i, row in enumerate(rows):
            in_range = (ntry_idx is not None and i >= ntry_idx and
                        (xpry_idx is None or i <= xpry_idx))
            row.append(strike_values[j] if in_range else "-")
        new_columns.append(strike_col_name)

    # Add expiry column
    date_col_idx = columns.index("date") if "date" in columns else None
    if xpry_idx is not None and date_col_idx is not None:
        expiry_value = rows[xpry_idx][date_col_idx]
    else:
        expiry_value = "-"

    for i, row in enumerate(rows):
        in_range = (ntry_idx is not None and i >= ntry_idx and
                    (xpry_idx is None or i <= xpry_idx))
        row.append(expiry_value if in_range else "-")
    new_columns.append("expiry")

    return {"orientation": "row", "columns": new_columns, "rows": rows}


def _get_rollforward_fields(columns: list[str]) -> set[str]:
    """Get fields that should be rolled forward (vol and hedge columns)."""
    fields = set()
    for col in columns:
        if col == "vol":
            fields.add(col)
        elif col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
            fields.add(col)
    return fields


# -------------------------------------
# Public API: actions, get_straddle_actions, get_straddle_valuation
# -------------------------------------


def actions(
    prices_table: dict[str, Any],
    path: str | Path,
    overrides_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Add action, model, and strike columns to a prices table.

    Args:
        prices_table: Output from get_prices() - must have 'asset' and 'straddle' columns
        path: Path to AMT YAML
        overrides_csv: Path to overrides CSV (for OVERRIDE code)

    Returns:
        Table with added columns: action, model, strike_vol, strike, expiry
    """
    if not prices_table["rows"]:
        return prices_table

    # Extract asset (underlying) and straddle from first row
    asset_idx = prices_table["columns"].index("asset")
    straddle_idx = prices_table["columns"].index("straddle")
    underlying = prices_table["rows"][0][asset_idx]
    straddle = prices_table["rows"][0][straddle_idx]

    # 1. Add action column
    table = _add_action_column(prices_table, straddle, underlying, overrides_csv)

    # 2. Add model column
    table = _add_model_column(table, underlying, path)

    # 3. Add strike columns
    ntry_idx, xpry_idx = _find_action_indices(table)
    table = _add_strike_columns(table, ntry_idx, xpry_idx)

    return table


def get_straddle_actions(
    underlying: str,
    year: int,
    month: int,
    i: int,
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
    overrides_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Get daily prices for a straddle from entry to expiry month.

    Columns: ['asset', 'straddle', 'date', <param1>, <param2>, ..., 'action', 'model', 'strike_vol', 'strike', 'expiry']
    Where params are 'vol', 'hedge', 'hedge1', etc.

    This is a convenience function that composes:
    1. get_prices() - price lookup
    2. actions() - action, model, and strike columns

    For more control, use the individual functions directly.

    Uses the module-level _PRICES_DICT if set (via set_prices_dict or load_all_prices),
    otherwise falls back to DuckDB queries on prices_parquet.

    Args:
        underlying: Asset underlying value
        year: Expiry year
        month: Expiry month
        i: Straddle selector index (i % len(straddles))
        path: Path to AMT YAML file
        chain_csv: Optional CSV for futures ticker lookup
        prices_parquet: Path to prices parquet file (for DuckDB fallback)
        overrides_csv: Path to overrides CSV (for OVERRIDE code)

    Returns:
        Table with one row per day, columns for each param's price plus action/strike columns
    """
    # Get prices table
    prices_table = prices_module.get_prices(
        underlying, year, month, i, path, chain_csv, prices_parquet
    )

    if not prices_table["rows"]:
        return prices_table

    # Add actions, model, and strikes
    return actions(prices_table, path, overrides_csv)


def get_straddle_valuation(
    underlying: str,
    year: int,
    month: int,
    i: int,
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
    overrides_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Get straddle valuation with mv column.

    Calls get_straddle_actions and adds mv (mark-to-market value) column
    computed using the asset's valuation model.

    Args:
        Same as get_straddle_actions

    Returns:
        Table with additional mv column
    """
    # Get base table
    table = get_straddle_actions(underlying, year, month, i, path, chain_csv, prices_parquet, overrides_csv)

    columns = table["columns"]
    rows = table["rows"]

    # Find action column and check for ntry/xpry
    if "action" not in columns:
        # No action column, add all valuation columns as "-"
        for row in rows:
            row.extend(["-", "-", "-", "-", "-"])
        columns.extend(["mv", "delta", "opnl", "hpnl", "pnl"])
        return {"orientation": "row", "columns": columns, "rows": rows}

    action_idx = columns.index("action")

    ntry_idx = None
    xpry_idx = None
    for idx, row in enumerate(rows):
        if row[action_idx] == "ntry":
            ntry_idx = idx
        elif row[action_idx] == "xpry":
            xpry_idx = idx

    if ntry_idx is None or xpry_idx is None:
        # Missing ntry or xpry, add all valuation columns as "-"
        for row in rows:
            row.extend(["-", "-", "-", "-", "-"])
        columns.extend(["mv", "delta", "opnl", "hpnl", "pnl"])
        return {"orientation": "row", "columns": columns, "rows": rows}

    # Get model from first row
    model_idx = columns.index("model") if "model" in columns else None
    if model_idx is not None and rows:
        model_name = rows[0][model_idx]
    else:
        model_name = ""

    # Get model function
    model_fn = MODEL_DISPATCH.get(model_name, model_default)

    # Get fields to roll forward (vol and hedge columns)
    rollforward_fields = _get_rollforward_fields(columns)

    # Initialize rolled-forward data from ntry row
    rolled_data = {}
    ntry_row_dict = dict(zip(columns, rows[ntry_idx]))
    for key in rollforward_fields:
        if key in ntry_row_dict:
            rolled_data[key] = ntry_row_dict[key]

    # Get strike price for hpnl calculation (hedge at entry)
    strike_col_idx = columns.index("strike") if "strike" in columns else None
    strike_price = None
    if strike_col_idx is not None:
        try:
            strike_price = float(rows[ntry_idx][strike_col_idx])
        except (ValueError, TypeError):
            pass

    # Track previous day's values for PnL calculations
    prev_mv = None
    prev_delta = None
    prev_hedge = None

    # Compute mv, delta, opnl, hpnl, pnl for each row
    for idx, row in enumerate(rows):
        if idx < ntry_idx or idx > xpry_idx:
            row.append("-")  # mv
            row.append("-")  # delta
            row.append("-")  # opnl
            row.append("-")  # hpnl
            row.append("-")  # pnl
        else:
            # Update rolled_data with any non-missing market values
            row_dict = dict(zip(columns, row))
            for key in rollforward_fields:
                if key in row_dict and row_dict[key] != "none":
                    rolled_data[key] = row_dict[key]

            # Build model input: current row data + rolled-forward market data
            model_input = row_dict.copy()
            model_input.update(rolled_data)

            result = model_fn(model_input)
            mv_str = result["mv"]
            delta_str = result["delta"]
            row.append(mv_str)
            row.append(delta_str)

            # Get current hedge (rolled forward)
            current_hedge = None
            try:
                current_hedge = float(rolled_data.get("hedge", ""))
            except (ValueError, TypeError):
                pass

            # Compute PnL columns
            if idx == ntry_idx:
                # First day: opnl = 0, hpnl = 0, pnl = 0
                row.append("0")  # opnl
                row.append("0")  # hpnl
                row.append("0")  # pnl
            else:
                # opnl = mv[today] - mv[yesterday]
                opnl = "-"
                if mv_str != "-" and prev_mv is not None:
                    try:
                        opnl = str(float(mv_str) - prev_mv)
                    except (ValueError, TypeError):
                        pass

                # hpnl = -delta[yesterday] * (hedge[today] - hedge[yesterday]) / strike
                hpnl = "-"
                if (prev_delta is not None and current_hedge is not None and
                    prev_hedge is not None and strike_price is not None and strike_price != 0):
                    try:
                        hpnl = str(-prev_delta * (current_hedge - prev_hedge) / strike_price)
                    except (ValueError, TypeError):
                        pass

                # pnl = opnl + hpnl
                pnl = "-"
                if opnl != "-" and hpnl != "-":
                    try:
                        pnl = str(float(opnl) + float(hpnl))
                    except (ValueError, TypeError):
                        pass

                row.append(opnl)
                row.append(hpnl)
                row.append(pnl)

            # Update previous values for next iteration
            try:
                prev_mv = float(mv_str) if mv_str != "-" else None
            except (ValueError, TypeError):
                prev_mv = None
            try:
                prev_delta = float(delta_str) if delta_str != "-" else None
            except (ValueError, TypeError):
                prev_delta = None
            prev_hedge = current_hedge

    columns.append("mv")
    columns.append("delta")
    columns.append("opnl")
    columns.append("hpnl")
    columns.append("pnl")
    return {"orientation": "row", "columns": columns, "rows": rows}


# -------------------------------------
# Batch valuation (vectorized)
# -------------------------------------

import numpy as np
from numba import njit, prange
from . import valuation_numba
from . import asset_straddle_tickers
from . import chain


def _batch_resolve_tickers(
    assets: list[str],
    stryms: list[str],
    ntrcs: list[str],
    amt_path: str,
) -> dict[str, dict[str, tuple[str, str]]]:
    """Resolve tickers for all unique (asset, strym, ntrc) combinations.

    Uses asset_straddle_ticker_key() to determine the cache key for each
    combination, then resolves tickers only once per unique key.

    Args:
        assets: List of asset names
        stryms: List of strym strings (YYYY-MM format)
        ntrcs: List of ntrc codes (N or F)
        amt_path: Path to AMT YAML file

    Returns:
        Dict mapping cache_key -> {param: (ticker, field)}
    """
    result: dict[str, dict[str, tuple[str, str]]] = {}
    seen_keys: set[str] = set()

    for asset, strym, ntrc in zip(assets, stryms, ntrcs):
        # Get asset data for cache key computation
        asset_data = loader.get_asset(amt_path, asset)
        if asset_data is None:
            continue

        vol = asset_data.get("Vol")
        hedge = asset_data.get("Hedge")
        if vol is None or hedge is None:
            continue

        # Compute cache key
        cache_key = asset_straddle_tickers.asset_straddle_ticker_key(
            asset, strym, ntrc, vol, hedge
        )

        if cache_key in seen_keys:
            continue
        seen_keys.add(cache_key)

        # Get tickers for this combination
        ticker_table = asset_straddle_tickers.get_asset_straddle_tickers(
            asset, strym, ntrc, amt_path
        )

        # Build param -> (ticker, field) mapping
        param_map: dict[str, tuple[str, str]] = {}
        for row in ticker_table["rows"]:
            name, ticker, field = row
            param_map[name] = (ticker, field)

        result[cache_key] = param_map

    return result


def _compute_actions_batch_simple(
    dates: np.ndarray,
    straddle_starts: np.ndarray,
    straddle_lengths: np.ndarray,
    vol_valid: np.ndarray,
    hedge_valid: np.ndarray,
    ntry_target_offsets: np.ndarray,
    xpry_target_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find ntry and xpry offsets for each straddle.

    This is a simplified version that finds the first valid day at or after
    the target offset for both entry and expiry.

    Args:
        dates: int32[N] - date32 values for all days
        straddle_starts: int32[S] - start index for each straddle
        straddle_lengths: int32[S] - number of days per straddle
        vol_valid: bool[N] - True if vol is non-NaN
        hedge_valid: bool[N] - True if hedge is non-NaN
        ntry_target_offsets: int32[S] - target entry offset within straddle
        xpry_target_offsets: int32[S] - target expiry offset within straddle

    Returns:
        ntry_offsets: int32[S] - actual entry offset (-1 if not found)
        xpry_offsets: int32[S] - actual expiry offset (-1 if not found)
    """
    n_straddles = len(straddle_starts)
    ntry_offsets = np.full(n_straddles, -1, dtype=np.int32)
    xpry_offsets = np.full(n_straddles, -1, dtype=np.int32)

    for s in range(n_straddles):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        ntry_target = ntry_target_offsets[s]
        xpry_target = xpry_target_offsets[s]

        # Find entry: first valid day at or after ntry_target
        for i in range(ntry_target, length):
            idx = start + i
            if vol_valid[idx] and hedge_valid[idx]:
                ntry_offsets[s] = i
                break

        # Find expiry: first valid day at or after xpry_target
        for i in range(xpry_target, length):
            idx = start + i
            if vol_valid[idx] and hedge_valid[idx]:
                xpry_offsets[s] = i
                break

    return ntry_offsets, xpry_offsets


# -------------------------------------
# Vectorized price lookup (Arrow/DuckDB)
# -------------------------------------

import pyarrow as pa
import pyarrow.parquet as pq

# Global cache for Arrow table
_PRICES_ARROW: pa.Table | None = None


def load_prices_arrow(prices_parquet: str) -> pa.Table:
    """Load prices parquet into Arrow table (cached).

    The table is cached globally for repeated lookups. Use clear_prices_arrow()
    to release the memory.

    Args:
        prices_parquet: Path to prices parquet file

    Returns:
        PyArrow Table with columns: ticker, field, date, value
    """
    global _PRICES_ARROW
    if _PRICES_ARROW is None:
        _PRICES_ARROW = pq.read_table(prices_parquet)
    return _PRICES_ARROW


def clear_prices_arrow() -> None:
    """Clear the cached Arrow table to free memory."""
    global _PRICES_ARROW
    _PRICES_ARROW = None


def _build_price_request_table(
    straddle_list: list[tuple[str, str]],
    straddle_starts: np.ndarray,
    straddle_lengths: np.ndarray,
    dates: np.ndarray,
    ticker_map: dict[str, dict[str, tuple[str, str]]],
    stryms: list[str],
    ntrcs: list[str],
    amt_path: str,
    chain_csv: str | None = None,
) -> tuple[np.ndarray, list, list[str], list[str], list[str]]:
    """Build arrays for vectorized price lookup.

    Creates a "request table" with one row per (day, param) combination.
    This is used by both Arrow and DuckDB lookup functions.

    Args:
        straddle_list: List of (asset, straddle) tuples
        straddle_starts: int32[S] - start index for each straddle
        straddle_lengths: int32[S] - number of days per straddle
        dates: int32[N] - date32 values for all days
        ticker_map: Dict from _batch_resolve_tickers()
        stryms: List of strym strings per straddle
        ntrcs: List of ntrc codes per straddle
        amt_path: Path to AMT YAML file
        chain_csv: Optional CSV for futures ticker normalization

    Returns:
        row_idx: int32[M] - index into output vol/hedge arrays
        req_dates: list[M] - date objects for lookup
        tickers: list[M] - ticker strings
        fields: list[M] - field strings
        params: list[M] - param name ('vol' or 'hedge')

    Where M = total number of (day, param) pairs across all straddles
    """
    row_idx_list = []
    req_dates = []
    tickers = []
    fields = []
    params = []

    epoch = date(1970, 1, 1)

    for s, (asset, straddle) in enumerate(straddle_list):
        start = int(straddle_starts[s])
        length = int(straddle_lengths[s])
        strym = stryms[s]
        ntrc = ntrcs[s]

        # Get ticker info for this straddle
        asset_data = loader.get_asset(amt_path, asset)
        if asset_data is None:
            continue

        vol_cfg = asset_data.get("Vol")
        hedge_cfg = asset_data.get("Hedge")
        if vol_cfg is None or hedge_cfg is None:
            continue

        cache_key = asset_straddle_tickers.asset_straddle_ticker_key(
            asset, strym, ntrc, vol_cfg, hedge_cfg
        )

        if cache_key not in ticker_map:
            continue

        param_map = ticker_map[cache_key]

        # Get vol ticker
        vol_ticker, vol_field = None, None
        if "vol" in param_map:
            vol_ticker, vol_field = param_map["vol"]
            if chain_csv is not None:
                normalized = chain.fut_act2norm(chain_csv, vol_ticker)
                if normalized is not None:
                    vol_ticker = normalized

        # Get hedge ticker
        hedge_ticker, hedge_field = None, None
        if "hedge" in param_map:
            hedge_ticker, hedge_field = param_map["hedge"]
            if chain_csv is not None:
                normalized = chain.fut_act2norm(chain_csv, hedge_ticker)
                if normalized is not None:
                    hedge_ticker = normalized

        # Add requests for each day
        for i in range(length):
            idx = start + i
            dt = date.fromordinal(epoch.toordinal() + int(dates[idx]))

            if vol_ticker is not None:
                row_idx_list.append(idx)
                req_dates.append(dt)
                tickers.append(vol_ticker)
                fields.append(vol_field)
                params.append('vol')

            if hedge_ticker is not None:
                row_idx_list.append(idx)
                req_dates.append(dt)
                tickers.append(hedge_ticker)
                fields.append(hedge_field)
                params.append('hedge')

    return np.array(row_idx_list, dtype=np.int32), req_dates, tickers, fields, params


def _arrow_price_lookup(
    row_idx: np.ndarray,
    req_dates: list,
    tickers: list[str],
    fields: list[str],
    params: list[str],
    prices_parquet: str,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lookup prices using PyArrow join.

    Performs a vectorized join between the request table and the prices
    Arrow table. Much faster than Python loops for large datasets.

    Args:
        row_idx: int32[M] - index into output arrays for each request
        req_dates: list[M] - date objects for lookup
        tickers: list[M] - ticker strings
        fields: list[M] - field strings
        params: list[M] - param name ('vol' or 'hedge')
        prices_parquet: Path to prices parquet file
        n_days: Total number of days (size of output arrays)

    Returns:
        vol_array: float64[n_days] - vol values, NaN for missing
        hedge_array: float64[n_days] - hedge values, NaN for missing
    """
    # Load prices (cached)
    prices = load_prices_arrow(prices_parquet)

    # Build request table
    request = pa.table({
        'row_idx': pa.array(row_idx, type=pa.int32()),
        'date': pa.array(req_dates),
        'ticker': pa.array(tickers),
        'field': pa.array(fields),
        'param': pa.array(params),
    })

    # Join on (ticker, field, date)
    result = request.join(
        prices,
        keys=['ticker', 'field', 'date'],
        join_type='left outer'
    )

    # Extract to numpy arrays
    result_row_idx = result['row_idx'].to_numpy()
    result_params = result['param'].to_pylist()
    result_values = result['value'].to_pylist()  # Use to_pylist to handle None

    # Pivot to vol/hedge arrays
    vol_array = np.full(n_days, np.nan, dtype=np.float64)
    hedge_array = np.full(n_days, np.nan, dtype=np.float64)

    for i in range(len(result_row_idx)):
        value = result_values[i]
        if value is not None:
            idx = result_row_idx[i]
            param = result_params[i]
            try:
                float_val = float(value)
                if param == 'vol':
                    vol_array[idx] = float_val
                elif param == 'hedge':
                    hedge_array[idx] = float_val
            except (ValueError, TypeError):
                pass

    return vol_array, hedge_array


from dataclasses import dataclass


@dataclass
class BacktestArrays:
    """Container for pre-computed arrays needed by full_backtest_kernel.

    All string lookups are done once during preparation, resulting in
    pure numeric arrays that Numba can process efficiently.
    """
    # Per-straddle arrays (length S)
    vol_row_idx: np.ndarray           # int32[S] - price matrix row for vol
    hedge_row_idx: np.ndarray         # int32[S] - price matrix row for hedge
    hedge1_row_idx: np.ndarray        # int32[S] - price matrix row for hedge1 (-1 if not used)
    hedge2_row_idx: np.ndarray        # int32[S] - price matrix row for hedge2 (-1 if not used)
    hedge3_row_idx: np.ndarray        # int32[S] - price matrix row for hedge3 (-1 if not used)
    n_hedges: np.ndarray              # int8[S] - number of hedge columns required (1-4)
    ntry_anchor_date32: np.ndarray    # int32[S] - date32 of entry anchor (from override/code)
    xpry_anchor_date32: np.ndarray    # int32[S] - date32 of expiry anchor (from override/code)
    ntrv_offsets: np.ndarray          # int32[S] - calendar days to add to entry anchor
    ntry_month_end: np.ndarray        # int32[S] - date32 of entry month end (for fallback)
    xpry_month_end: np.ndarray        # int32[S] - date32 of expiry month end (for fallback)
    asset_idx: np.ndarray             # int32[S] - index into unique_assets
    straddle_idx: np.ndarray          # int32[S] - index into unique_straddles
    model_idx: np.ndarray             # int32[S] - index into unique_models

    # For Arrow dictionary encoding
    unique_assets: list[str]
    unique_straddles: list[str]
    unique_models: list[str]


@dataclass
class BacktestArraysSorted:
    """Container for pre-computed arrays needed by full_backtest_kernel_sorted.

    Similar to BacktestArrays but uses ticker/field indices for sorted array lookup
    instead of price matrix row indices.
    """
    # Per-straddle ticker/field indices (length S)
    vol_ticker_idx: np.ndarray        # int32[S] - ticker index from PricesNumba.ticker_to_idx
    vol_field_idx: np.ndarray         # int32[S] - field index from PricesNumba.field_to_idx
    hedge_ticker_idx: np.ndarray      # int32[S] - ticker index from PricesNumba.ticker_to_idx
    hedge_field_idx: np.ndarray       # int32[S] - field index from PricesNumba.field_to_idx

    # Additional hedge ticker/field indices (for CDS and calc assets)
    hedge1_ticker_idx: np.ndarray     # int32[S] - ticker index for hedge1 (-1 if not used)
    hedge1_field_idx: np.ndarray      # int32[S] - field index for hedge1
    hedge2_ticker_idx: np.ndarray     # int32[S] - ticker index for hedge2 (-1 if not used)
    hedge2_field_idx: np.ndarray      # int32[S] - field index for hedge2
    hedge3_ticker_idx: np.ndarray     # int32[S] - ticker index for hedge3 (-1 if not used)
    hedge3_field_idx: np.ndarray      # int32[S] - field index for hedge3
    n_hedges: np.ndarray              # int8[S] - number of hedge columns required (1-4)

    # Entry/expiry anchor dates (length S)
    ntry_anchor_date32: np.ndarray    # int32[S] - date32 of entry anchor (from override/code)
    xpry_anchor_date32: np.ndarray    # int32[S] - date32 of expiry anchor (from override/code)
    ntrv_offsets: np.ndarray          # int32[S] - calendar days to add to entry anchor
    ntry_month_end: np.ndarray        # int32[S] - date32 of entry month end (for fallback)
    xpry_month_end: np.ndarray        # int32[S] - date32 of expiry month end (for fallback)

    # For output formatting
    asset_idx: np.ndarray             # int32[S] - index into unique_assets
    straddle_idx: np.ndarray          # int32[S] - index into unique_straddles
    model_idx: np.ndarray             # int32[S] - index into unique_models

    # For Arrow dictionary encoding
    unique_assets: list[str]
    unique_straddles: list[str]
    unique_models: list[str]


def _prepare_backtest_arrays_sorted(
    straddle_list: list[tuple[str, str]],
    ticker_map: dict[str, dict[str, tuple[str, str]]],
    prices_numba: "prices_module.PricesNumba",
    stryms: list[str],
    ntrcs: list[str],
    amt_path: str,
    chain_csv: str | None = None,
    overrides_path: str | None = None,
) -> BacktestArraysSorted:
    """Prepare arrays for full_backtest_kernel_sorted.

    Similar to _prepare_backtest_arrays but builds ticker/field indices
    for sorted array lookup instead of price matrix row indices.

    Args:
        straddle_list: List of (asset, straddle) tuples
        ticker_map: Dict from _batch_resolve_tickers()
        prices_numba: PricesNumba object with sorted arrays and index mappings
        stryms: List of strym strings per straddle
        ntrcs: List of ntrc codes per straddle
        amt_path: Path to AMT YAML file
        chain_csv: Optional CSV for futures ticker normalization
        overrides_path: Path to overrides CSV (for OVERRIDE code)

    Returns:
        BacktestArraysSorted containing all pre-computed arrays
    """
    n_straddles = len(straddle_list)

    # === OPTIMIZATION: Pre-compute asset properties for unique assets ===
    # Instead of calling loader.get_asset() 231K times, call it once per unique asset (189 assets)
    unique_asset_names = set(asset for asset, _ in straddle_list)
    asset_to_data: dict[str, dict | None] = {}
    for asset_name in unique_asset_names:
        asset_to_data[asset_name] = loader.get_asset(amt_path, asset_name)

    # Pre-resolve chain CSV path once (avoids 462K Path.resolve() calls)
    chain_csv_resolved = str(Path(chain_csv).resolve()) if chain_csv else None

    # Pre-cache ticker normalizations to avoid repeated lookups
    ticker_norm_cache: dict[str, str | None] = {}

    def _normalize_ticker(ticker: str) -> str:
        """Normalize ticker with caching."""
        if ticker not in ticker_norm_cache:
            if chain_csv_resolved is not None:
                # Use the pre-resolved path and access the cache directly
                if chain_csv_resolved not in chain._ACTUAL_CACHE:
                    # Force cache population
                    chain.fut_act2norm(chain_csv_resolved, ticker)
                ticker_norm_cache[ticker] = chain._ACTUAL_CACHE[chain_csv_resolved].get(ticker)
            else:
                ticker_norm_cache[ticker] = None
        return ticker_norm_cache[ticker] if ticker_norm_cache[ticker] else ticker

    # Per-straddle ticker/field index arrays
    vol_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    vol_field_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge_field_idx = np.full(n_straddles, -1, dtype=np.int32)

    # Additional hedge ticker/field indices (for CDS and calc assets)
    hedge1_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge1_field_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge2_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge2_field_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge3_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge3_field_idx = np.full(n_straddles, -1, dtype=np.int32)
    n_hedges = np.ones(n_straddles, dtype=np.int8)  # Default to 1 (just primary hedge)

    # Entry/expiry anchor dates
    ntry_anchor_date32 = np.zeros(n_straddles, dtype=np.int32)
    xpry_anchor_date32 = np.zeros(n_straddles, dtype=np.int32)
    ntrv_offsets = np.zeros(n_straddles, dtype=np.int32)
    ntry_month_end = np.zeros(n_straddles, dtype=np.int32)
    xpry_month_end = np.zeros(n_straddles, dtype=np.int32)

    # For Arrow dictionary encoding (string -> index)
    unique_assets: dict[str, int] = {}
    unique_straddles: dict[str, int] = {}
    unique_models: dict[str, int] = {}
    asset_idx = np.empty(n_straddles, dtype=np.int32)
    straddle_idx = np.empty(n_straddles, dtype=np.int32)
    model_idx = np.empty(n_straddles, dtype=np.int32)

    for s, (asset, straddle) in enumerate(straddle_list):
        strym = stryms[s]
        ntrc = ntrcs[s]

        # Build unique asset/straddle lists for Arrow output
        if asset not in unique_assets:
            unique_assets[asset] = len(unique_assets)
        if straddle not in unique_straddles:
            unique_straddles[straddle] = len(unique_straddles)
        asset_idx[s] = unique_assets[asset]
        straddle_idx[s] = unique_straddles[straddle]

        # Get asset data from pre-computed cache (O(1) dict lookup)
        asset_data = asset_to_data[asset]
        if asset_data is None:
            if "" not in unique_models:
                unique_models[""] = len(unique_models)
            model_idx[s] = unique_models[""]
            continue

        # Get model name
        valuation = asset_data.get("Valuation", {})
        model_name = valuation.get("Model", "") if isinstance(valuation, dict) else ""
        if model_name not in unique_models:
            unique_models[model_name] = len(unique_models)
        model_idx[s] = unique_models[model_name]

        vol_cfg = asset_data.get("Vol")
        hedge_cfg = asset_data.get("Hedge")
        if vol_cfg is None or hedge_cfg is None:
            continue

        # Compute cache key
        cache_key = asset_straddle_tickers.asset_straddle_ticker_key(
            asset, strym, ntrc, vol_cfg, hedge_cfg
        )

        if cache_key in ticker_map:
            param_map = ticker_map[cache_key]

            # Get vol ticker/field indices
            # Use -2 for "required but ticker missing from prices" vs -1 for "not required"
            if "vol" in param_map:
                vol_ticker, vol_field = param_map["vol"]
                vol_ticker = _normalize_ticker(vol_ticker)
                # Look up indices in PricesNumba
                if vol_ticker in prices_numba.ticker_to_idx:
                    vol_ticker_idx[s] = prices_numba.ticker_to_idx[vol_ticker]
                else:
                    vol_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if vol_field in prices_numba.field_to_idx:
                    vol_field_idx[s] = prices_numba.field_to_idx[vol_field]

            # Get hedge ticker/field indices
            if "hedge" in param_map:
                hedge_ticker, hedge_field = param_map["hedge"]
                hedge_ticker = _normalize_ticker(hedge_ticker)
                # Look up indices in PricesNumba
                if hedge_ticker in prices_numba.ticker_to_idx:
                    hedge_ticker_idx[s] = prices_numba.ticker_to_idx[hedge_ticker]
                else:
                    hedge_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if hedge_field in prices_numba.field_to_idx:
                    hedge_field_idx[s] = prices_numba.field_to_idx[hedge_field]

            # Get hedge1 ticker/field indices (for CDS assets)
            if "hedge1" in param_map:
                h1_ticker, h1_field = param_map["hedge1"]
                h1_ticker = _normalize_ticker(h1_ticker)
                if h1_ticker in prices_numba.ticker_to_idx:
                    hedge1_ticker_idx[s] = prices_numba.ticker_to_idx[h1_ticker]
                else:
                    hedge1_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if h1_field in prices_numba.field_to_idx:
                    hedge1_field_idx[s] = prices_numba.field_to_idx[h1_field]
                n_hedges[s] = max(n_hedges[s], 2)

            # Get hedge2 ticker/field indices (for calc assets)
            if "hedge2" in param_map:
                h2_ticker, h2_field = param_map["hedge2"]
                h2_ticker = _normalize_ticker(h2_ticker)
                if h2_ticker in prices_numba.ticker_to_idx:
                    hedge2_ticker_idx[s] = prices_numba.ticker_to_idx[h2_ticker]
                else:
                    hedge2_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if h2_field in prices_numba.field_to_idx:
                    hedge2_field_idx[s] = prices_numba.field_to_idx[h2_field]
                n_hedges[s] = max(n_hedges[s], 3)

            # Get hedge3 ticker/field indices (for calc assets)
            if "hedge3" in param_map:
                h3_ticker, h3_field = param_map["hedge3"]
                h3_ticker = _normalize_ticker(h3_ticker)
                if h3_ticker in prices_numba.ticker_to_idx:
                    hedge3_ticker_idx[s] = prices_numba.ticker_to_idx[h3_ticker]
                else:
                    hedge3_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if h3_field in prices_numba.field_to_idx:
                    hedge3_field_idx[s] = prices_numba.field_to_idx[h3_field]
                n_hedges[s] = max(n_hedges[s], 4)

            # Handle hedge4 -> hedge3 mapping for calc assets
            if "hedge4" in param_map and hedge3_ticker_idx[s] == -1:
                h4_ticker, h4_field = param_map["hedge4"]
                h4_ticker = _normalize_ticker(h4_ticker)
                if h4_ticker in prices_numba.ticker_to_idx:
                    hedge3_ticker_idx[s] = prices_numba.ticker_to_idx[h4_ticker]
                else:
                    hedge3_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if h4_field in prices_numba.field_to_idx:
                    hedge3_field_idx[s] = prices_numba.field_to_idx[h4_field]
                n_hedges[s] = max(n_hedges[s], 4)

        # Parse straddle string for entry/expiry dates
        ntry_y = schedules.ntry(straddle)
        ntry_m = schedules.ntrm(straddle)
        xpry_y = schedules.xpry(straddle)
        xpry_m = schedules.xprm(straddle)

        # Get codes from straddle
        # Note: u8m format produces padded strings, so strip to get clean values
        xprc = schedules.xprc(straddle).strip()
        xprv = schedules.xprv(straddle).strip()
        ntrv_str = schedules.ntrv(straddle).strip()

        # Compute entry month end date
        _, ntry_num_days = calendar.monthrange(ntry_y, ntry_m)
        ntry_month_end[s] = valuation_numba.ymd_to_date32(ntry_y, ntry_m, ntry_num_days)

        # Compute expiry month end date
        _, xpry_num_days = calendar.monthrange(xpry_y, xpry_m)
        xpry_month_end[s] = valuation_numba.ymd_to_date32(xpry_y, xpry_m, xpry_num_days)

        # Parse ntrv as calendar day offset
        try:
            ntrv_offsets[s] = int(ntrv_str) if ntrv_str else 0
        except (ValueError, TypeError):
            ntrv_offsets[s] = 0

        # Compute entry anchor using _anchor_day
        entry_anchor = _anchor_day(xprc, xprv, ntry_y, ntry_m, asset, overrides_path)
        if entry_anchor is not None:
            y, m, d = map(int, entry_anchor.split("-"))
            ntry_anchor_date32[s] = valuation_numba.ymd_to_date32(y, m, d)
        else:
            ntry_anchor_date32[s] = np.iinfo(np.int32).max

        # Compute expiry anchor using _anchor_day
        expiry_anchor = _anchor_day(xprc, xprv, xpry_y, xpry_m, asset, overrides_path)
        if expiry_anchor is not None:
            y, m, d = map(int, expiry_anchor.split("-"))
            xpry_anchor_date32[s] = valuation_numba.ymd_to_date32(y, m, d)
        else:
            xpry_anchor_date32[s] = np.iinfo(np.int32).max

    return BacktestArraysSorted(
        vol_ticker_idx=vol_ticker_idx,
        vol_field_idx=vol_field_idx,
        hedge_ticker_idx=hedge_ticker_idx,
        hedge_field_idx=hedge_field_idx,
        hedge1_ticker_idx=hedge1_ticker_idx,
        hedge1_field_idx=hedge1_field_idx,
        hedge2_ticker_idx=hedge2_ticker_idx,
        hedge2_field_idx=hedge2_field_idx,
        hedge3_ticker_idx=hedge3_ticker_idx,
        hedge3_field_idx=hedge3_field_idx,
        n_hedges=n_hedges,
        ntry_anchor_date32=ntry_anchor_date32,
        xpry_anchor_date32=xpry_anchor_date32,
        ntrv_offsets=ntrv_offsets,
        ntry_month_end=ntry_month_end,
        xpry_month_end=xpry_month_end,
        asset_idx=asset_idx,
        straddle_idx=straddle_idx,
        model_idx=model_idx,
        unique_assets=list(unique_assets.keys()),
        unique_straddles=list(unique_straddles.keys()),
        unique_models=list(unique_models.keys()),
    )


def _compute_starts_lengths_from_parent_idx(parent_idx: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Compute straddle_starts and straddle_lengths from parent_idx array.

    Args:
        parent_idx: int32 array mapping each day to its source straddle index

    Returns:
        straddle_starts: int32 array of start positions for each straddle
        straddle_lengths: int32 array of lengths for each straddle
    """
    import numpy as np

    if len(parent_idx) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Find where parent_idx changes (boundaries between straddles)
    # Note: parent_idx is contiguous - all days for straddle 0 come first, then straddle 1, etc.
    n_straddles = int(parent_idx[-1]) + 1
    straddle_starts = np.zeros(n_straddles, dtype=np.int32)
    straddle_lengths = np.zeros(n_straddles, dtype=np.int32)

    # Count occurrences of each straddle index
    counts = np.bincount(parent_idx, minlength=n_straddles)
    straddle_lengths[:] = counts[:n_straddles].astype(np.int32)

    # Compute starts as cumsum of lengths (shifted by 1)
    straddle_starts[1:] = np.cumsum(straddle_lengths[:-1])

    return straddle_starts, straddle_lengths


def _prepare_backtest_arrays(
    straddle_list: list[tuple[str, str]],
    ticker_map: dict[str, dict[str, tuple[str, str]]],
    price_matrix: "prices_module.PriceMatrix",
    stryms: list[str],
    ntrcs: list[str],
    amt_path: str,
    chain_csv: str | None = None,
    overrides_path: str | None = None,
) -> BacktestArrays:
    """Prepare all numeric arrays for the full_backtest_kernel.

    This is a one-time preparation step that moves all string lookups
    out of the hot path. The result is pure numeric arrays that Numba
    can process efficiently.

    Args:
        straddle_list: List of (asset, straddle) tuples
        ticker_map: Dict from _batch_resolve_tickers()
        price_matrix: PriceMatrix object with price data
        stryms: List of strym strings per straddle
        ntrcs: List of ntrc codes per straddle
        amt_path: Path to AMT YAML file
        chain_csv: Optional CSV for futures ticker normalization
        overrides_path: Path to overrides CSV (for OVERRIDE code)

    Returns:
        BacktestArrays containing all pre-computed numeric arrays
    """
    n_straddles = len(straddle_list)

    # Per-straddle arrays
    vol_row_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge_row_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge1_row_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge2_row_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge3_row_idx = np.full(n_straddles, -1, dtype=np.int32)
    n_hedges = np.ones(n_straddles, dtype=np.int8)  # Default to 1 (just primary hedge)
    ntry_anchor_date32 = np.zeros(n_straddles, dtype=np.int32)
    xpry_anchor_date32 = np.zeros(n_straddles, dtype=np.int32)
    ntrv_offsets = np.zeros(n_straddles, dtype=np.int32)
    ntry_month_end = np.zeros(n_straddles, dtype=np.int32)
    xpry_month_end = np.zeros(n_straddles, dtype=np.int32)

    # For Arrow dictionary encoding (string -> index)
    unique_assets: dict[str, int] = {}
    unique_straddles: dict[str, int] = {}
    unique_models: dict[str, int] = {}
    asset_idx = np.empty(n_straddles, dtype=np.int32)
    straddle_idx = np.empty(n_straddles, dtype=np.int32)
    model_idx = np.empty(n_straddles, dtype=np.int32)

    # Cache to avoid repeated ticker resolution
    ticker_row_cache: dict[tuple[str, str], int] = {}

    for s, (asset, straddle) in enumerate(straddle_list):
        strym = stryms[s]
        ntrc = ntrcs[s]

        # Build unique asset/straddle lists for Arrow output
        if asset not in unique_assets:
            unique_assets[asset] = len(unique_assets)
        if straddle not in unique_straddles:
            unique_straddles[straddle] = len(unique_straddles)
        asset_idx[s] = unique_assets[asset]
        straddle_idx[s] = unique_straddles[straddle]

        # Get asset data for cache key and model
        asset_data = loader.get_asset(amt_path, asset)
        if asset_data is None:
            model_idx[s] = 0
            if "" not in unique_models:
                unique_models[""] = len(unique_models)
            model_idx[s] = unique_models[""]
            continue

        # Get model name
        valuation = asset_data.get("Valuation", {})
        model_name = valuation.get("Model", "") if isinstance(valuation, dict) else ""
        if model_name not in unique_models:
            unique_models[model_name] = len(unique_models)
        model_idx[s] = unique_models[model_name]

        vol_cfg = asset_data.get("Vol")
        hedge_cfg = asset_data.get("Hedge")
        if vol_cfg is None or hedge_cfg is None:
            continue

        # Compute cache key
        cache_key = asset_straddle_tickers.asset_straddle_ticker_key(
            asset, strym, ntrc, vol_cfg, hedge_cfg
        )

        if cache_key in ticker_map:
            param_map = ticker_map[cache_key]

            # Get vol ticker row index (with caching)
            if "vol" in param_map:
                vol_ticker, vol_field = param_map["vol"]
                if chain_csv is not None:
                    normalized = chain.fut_act2norm(chain_csv, vol_ticker)
                    if normalized is not None:
                        vol_ticker = normalized
                vol_key = (vol_ticker, vol_field)
                if vol_key not in ticker_row_cache:
                    ticker_row_cache[vol_key] = price_matrix.get_row_idx(vol_ticker, vol_field)
                vol_row_idx[s] = ticker_row_cache[vol_key]

            # Get hedge ticker row index (with caching)
            if "hedge" in param_map:
                hedge_ticker, hedge_field = param_map["hedge"]
                if chain_csv is not None:
                    normalized = chain.fut_act2norm(chain_csv, hedge_ticker)
                    if normalized is not None:
                        hedge_ticker = normalized
                hedge_key = (hedge_ticker, hedge_field)
                if hedge_key not in ticker_row_cache:
                    ticker_row_cache[hedge_key] = price_matrix.get_row_idx(hedge_ticker, hedge_field)
                hedge_row_idx[s] = ticker_row_cache[hedge_key]

            # Get hedge1 ticker row index (for CDS assets)
            if "hedge1" in param_map:
                h1_ticker, h1_field = param_map["hedge1"]
                if chain_csv is not None:
                    normalized = chain.fut_act2norm(chain_csv, h1_ticker)
                    if normalized is not None:
                        h1_ticker = normalized
                h1_key = (h1_ticker, h1_field)
                if h1_key not in ticker_row_cache:
                    ticker_row_cache[h1_key] = price_matrix.get_row_idx(h1_ticker, h1_field)
                hedge1_row_idx[s] = ticker_row_cache[h1_key]
                n_hedges[s] = max(n_hedges[s], 2)

            # Get hedge2 ticker row index (for calc assets)
            if "hedge2" in param_map:
                h2_ticker, h2_field = param_map["hedge2"]
                if chain_csv is not None:
                    normalized = chain.fut_act2norm(chain_csv, h2_ticker)
                    if normalized is not None:
                        h2_ticker = normalized
                h2_key = (h2_ticker, h2_field)
                if h2_key not in ticker_row_cache:
                    ticker_row_cache[h2_key] = price_matrix.get_row_idx(h2_ticker, h2_field)
                hedge2_row_idx[s] = ticker_row_cache[h2_key]
                n_hedges[s] = max(n_hedges[s], 3)

            # Get hedge3 ticker row index (for calc assets - maps from hedge4)
            # Note: asset_straddle_tickers uses hedge3 and hedge4, we map to hedge2/hedge3
            if "hedge3" in param_map:
                h3_ticker, h3_field = param_map["hedge3"]
                if chain_csv is not None:
                    normalized = chain.fut_act2norm(chain_csv, h3_ticker)
                    if normalized is not None:
                        h3_ticker = normalized
                h3_key = (h3_ticker, h3_field)
                if h3_key not in ticker_row_cache:
                    ticker_row_cache[h3_key] = price_matrix.get_row_idx(h3_ticker, h3_field)
                hedge3_row_idx[s] = ticker_row_cache[h3_key]
                n_hedges[s] = max(n_hedges[s], 4)

            # Handle hedge4 -> hedge3 mapping for calc assets
            if "hedge4" in param_map and hedge3_row_idx[s] == -1:
                h4_ticker, h4_field = param_map["hedge4"]
                if chain_csv is not None:
                    normalized = chain.fut_act2norm(chain_csv, h4_ticker)
                    if normalized is not None:
                        h4_ticker = normalized
                h4_key = (h4_ticker, h4_field)
                if h4_key not in ticker_row_cache:
                    ticker_row_cache[h4_key] = price_matrix.get_row_idx(h4_ticker, h4_field)
                hedge3_row_idx[s] = ticker_row_cache[h4_key]
                n_hedges[s] = max(n_hedges[s], 4)

        # Parse straddle string for entry/expiry dates
        ntry_y = schedules.ntry(straddle)
        ntry_m = schedules.ntrm(straddle)
        xpry_y = schedules.xpry(straddle)
        xpry_m = schedules.xprm(straddle)

        # Get codes from straddle
        # Note: u8m format produces padded strings, so strip to get clean values
        xprc = schedules.xprc(straddle).strip()
        xprv = schedules.xprv(straddle).strip()
        ntrv_str = schedules.ntrv(straddle).strip()

        # Compute entry month end date
        _, ntry_num_days = calendar.monthrange(ntry_y, ntry_m)
        ntry_month_end[s] = valuation_numba.ymd_to_date32(ntry_y, ntry_m, ntry_num_days)

        # Compute expiry month end date
        _, xpry_num_days = calendar.monthrange(xpry_y, xpry_m)
        xpry_month_end[s] = valuation_numba.ymd_to_date32(xpry_y, xpry_m, xpry_num_days)

        # Parse ntrv as calendar day offset
        try:
            ntrv_offsets[s] = int(ntrv_str) if ntrv_str else 0
        except (ValueError, TypeError):
            ntrv_offsets[s] = 0

        # Compute entry anchor using _anchor_day
        # If anchor lookup fails (None), mark straddle as invalid by setting
        # anchor to INT32_MAX - kernel will skip it
        entry_anchor = _anchor_day(xprc, xprv, ntry_y, ntry_m, asset, overrides_path)
        if entry_anchor is not None:
            # Parse ISO date string to date32
            y, m, d = map(int, entry_anchor.split("-"))
            ntry_anchor_date32[s] = valuation_numba.ymd_to_date32(y, m, d)
        else:
            # No anchor found - mark as invalid (INT32_MAX means skip)
            ntry_anchor_date32[s] = np.iinfo(np.int32).max

        # Compute expiry anchor using _anchor_day
        expiry_anchor = _anchor_day(xprc, xprv, xpry_y, xpry_m, asset, overrides_path)
        if expiry_anchor is not None:
            # Parse ISO date string to date32
            y, m, d = map(int, expiry_anchor.split("-"))
            xpry_anchor_date32[s] = valuation_numba.ymd_to_date32(y, m, d)
        else:
            # No anchor found - mark as invalid (INT32_MAX means skip)
            xpry_anchor_date32[s] = np.iinfo(np.int32).max

    return BacktestArrays(
        vol_row_idx=vol_row_idx,
        hedge_row_idx=hedge_row_idx,
        hedge1_row_idx=hedge1_row_idx,
        hedge2_row_idx=hedge2_row_idx,
        hedge3_row_idx=hedge3_row_idx,
        n_hedges=n_hedges,
        ntry_anchor_date32=ntry_anchor_date32,
        xpry_anchor_date32=xpry_anchor_date32,
        ntrv_offsets=ntrv_offsets,
        ntry_month_end=ntry_month_end,
        xpry_month_end=xpry_month_end,
        asset_idx=asset_idx,
        straddle_idx=straddle_idx,
        model_idx=model_idx,
        unique_assets=list(unique_assets.keys()),
        unique_straddles=list(unique_straddles.keys()),
        unique_models=list(unique_models.keys()),
    )


def _build_arrow_output(
    # Numeric arrays from kernel
    dates: np.ndarray,                 # int32[N] - date32 values
    vol: np.ndarray,                   # float64[N] - raw vol prices
    hedge: np.ndarray,                 # float64[N] - raw hedge prices
    hedge1: np.ndarray,                # float64[N] - raw hedge1 prices
    hedge2: np.ndarray,                # float64[N] - raw hedge2 prices
    hedge3: np.ndarray,                # float64[N] - raw hedge3 prices
    strike: np.ndarray,                # float64[N] - strike price
    strike1: np.ndarray,               # float64[N] - strike1 price
    strike2: np.ndarray,               # float64[N] - strike2 price
    strike3: np.ndarray,               # float64[N] - strike3 price
    mv: np.ndarray,                    # float64[N] - mark-to-market
    delta: np.ndarray,                 # float64[N] - delta
    opnl: np.ndarray,                  # float64[N] - option PnL
    hpnl: np.ndarray,                  # float64[N] - hedge PnL
    pnl: np.ndarray,                   # float64[N] - total PnL
    action: np.ndarray,                # int8[N] - action code

    # For string columns
    parent_idx: np.ndarray,            # int32[N] - which straddle each day belongs to
    backtest_arrays: BacktestArrays,   # Pre-computed array mappings

    # Straddle structure
    straddle_starts: np.ndarray,       # int32[S] - start index per straddle
    straddle_lengths: np.ndarray,      # int32[S] - length per straddle
    ntry_offsets: np.ndarray,          # int32[S] - entry offset per straddle
    xpry_offsets: np.ndarray,          # int32[S] - expiry offset per straddle

    # Options
    valid_only: bool = False,
) -> pa.Table:
    """Build Arrow table from numeric arrays (near zero-copy).

    This function converts the numeric output from full_backtest_kernel
    into a PyArrow Table. String columns use dictionary encoding for
    efficiency - the actual string values are stored once in a dictionary,
    and each row just stores an integer index.

    Args:
        dates: int32[N] - date32 values for each day
        vol: float64[N] - raw vol prices
        hedge: float64[N] - raw hedge prices
        hedge1: float64[N] - raw hedge1 prices (NaN if not used)
        hedge2: float64[N] - raw hedge2 prices (NaN if not used)
        hedge3: float64[N] - raw hedge3 prices (NaN if not used)
        strike: float64[N] - strike price (hedge at entry)
        strike1: float64[N] - strike1 price (hedge1 at entry)
        strike2: float64[N] - strike2 price (hedge2 at entry)
        strike3: float64[N] - strike3 price (hedge3 at entry)
        mv: float64[N] - mark-to-market value
        delta: float64[N] - option delta
        opnl: float64[N] - option PnL
        hpnl: float64[N] - hedge PnL
        pnl: float64[N] - total PnL
        action: int8[N] - action code (0=none, 1=ntry, 2=xpry)
        parent_idx: int32[N] - which straddle each day belongs to
        backtest_arrays: BacktestArrays with per-straddle index arrays
        straddle_starts: int32[S] - start index per straddle
        straddle_lengths: int32[S] - length per straddle
        ntry_offsets: int32[S] - entry offset per straddle (-1 if not found)
        xpry_offsets: int32[S] - expiry offset per straddle (-1 if not found)
        valid_only: If True, only include rows with valid mv values

    Returns:
        PyArrow Table with columns:
        asset, straddle, date, vol, hedge, hedge1, hedge2, hedge3, action, model,
        strike_vol, strike, strike1, strike2, strike3, expiry, mv, delta, opnl, hpnl, pnl
    """
    n_days = len(dates)
    n_straddles = len(straddle_starts)

    # Pre-compute strike_vol per straddle
    strike_vol = np.full(n_straddles, np.nan, dtype=np.float64)
    for s in range(n_straddles):
        ntry = ntry_offsets[s]
        if ntry >= 0:
            start = straddle_starts[s]
            strike_vol[s] = vol[start + ntry]

    # Pre-compute expiry date32 per straddle
    expiry_date32 = np.full(n_straddles, -1, dtype=np.int32)
    for s in range(n_straddles):
        xpry = xpry_offsets[s]
        if xpry >= 0:
            start = straddle_starts[s]
            expiry_date32[s] = dates[start + xpry]

    # Build valid_mask for filtering
    if valid_only:
        valid_mask = ~np.isnan(mv)
        n_valid = np.sum(valid_mask)
        indices = np.where(valid_mask)[0]

        # Filter arrays
        dates = dates[indices]
        vol = vol[indices]
        hedge = hedge[indices]
        hedge1 = hedge1[indices]
        hedge2 = hedge2[indices]
        hedge3 = hedge3[indices]
        strike = strike[indices]
        strike1 = strike1[indices]
        strike2 = strike2[indices]
        strike3 = strike3[indices]
        mv = mv[indices]
        delta = delta[indices]
        opnl = opnl[indices]
        hpnl = hpnl[indices]
        pnl = pnl[indices]
        action = action[indices]
        parent_idx = parent_idx[indices]
    else:
        n_valid = n_days

    # Build per-row arrays from per-straddle data
    # These use parent_idx to map each day back to its straddle
    asset_indices = backtest_arrays.asset_idx[parent_idx]
    straddle_indices = backtest_arrays.straddle_idx[parent_idx]
    model_indices = backtest_arrays.model_idx[parent_idx]
    strike_vol_values = strike_vol[parent_idx]
    expiry_values = expiry_date32[parent_idx]

    # Build dictionary-encoded string columns
    # Asset column
    asset_dict = pa.DictionaryArray.from_arrays(
        pa.array(asset_indices, type=pa.int32()),
        pa.array(backtest_arrays.unique_assets, type=pa.string())
    )

    # Straddle column
    straddle_dict = pa.DictionaryArray.from_arrays(
        pa.array(straddle_indices, type=pa.int32()),
        pa.array(backtest_arrays.unique_straddles, type=pa.string())
    )

    # Model column
    model_dict = pa.DictionaryArray.from_arrays(
        pa.array(model_indices, type=pa.int32()),
        pa.array(backtest_arrays.unique_models, type=pa.string())
    )

    # Action column (convert int8 to string)
    action_strs = ["", "ntry", "xpry"]  # 0, 1, 2
    action_dict = pa.DictionaryArray.from_arrays(
        pa.array(action, type=pa.int8()),
        pa.array(action_strs, type=pa.string())
    )

    # Date column (Arrow can handle date32 directly)
    date_array = pa.array(dates, type=pa.date32())

    # Expiry column (also date32)
    expiry_array = pa.array(expiry_values, type=pa.date32())

    # Build the table
    return pa.table({
        'asset': asset_dict,
        'straddle': straddle_dict,
        'date': date_array,
        'vol': pa.array(vol),
        'hedge': pa.array(hedge),
        'hedge1': pa.array(hedge1),
        'hedge2': pa.array(hedge2),
        'hedge3': pa.array(hedge3),
        'action': action_dict,
        'model': model_dict,
        'strike_vol': pa.array(strike_vol_values),
        'strike': pa.array(strike),
        'strike1': pa.array(strike1),
        'strike2': pa.array(strike2),
        'strike3': pa.array(strike3),
        'expiry': expiry_array,
        'mv': pa.array(mv),
        'delta': pa.array(delta),
        'opnl': pa.array(opnl),
        'hpnl': pa.array(hpnl),
        'pnl': pa.array(pnl),
    })


def _build_arrow_output_sorted(
    # Numeric arrays from kernel
    dates: np.ndarray,                 # int32[N] - date32 values
    vol: np.ndarray,                   # float64[N] - raw vol prices
    hedge: np.ndarray,                 # float64[N] - raw hedge prices
    hedge1: np.ndarray,                # float64[N] - raw hedge1 prices
    hedge2: np.ndarray,                # float64[N] - raw hedge2 prices
    hedge3: np.ndarray,                # float64[N] - raw hedge3 prices
    strike: np.ndarray,                # float64[N] - strike price
    strike1: np.ndarray,               # float64[N] - strike1 price
    strike2: np.ndarray,               # float64[N] - strike2 price
    strike3: np.ndarray,               # float64[N] - strike3 price
    mv: np.ndarray,                    # float64[N] - mark-to-market
    delta: np.ndarray,                 # float64[N] - delta
    opnl: np.ndarray,                  # float64[N] - option PnL
    hpnl: np.ndarray,                  # float64[N] - hedge PnL
    pnl: np.ndarray,                   # float64[N] - total PnL
    action: np.ndarray,                # int8[N] - action code

    # For string columns
    parent_idx: np.ndarray,            # int32[N] - which straddle each day belongs to
    backtest_arrays: BacktestArraysSorted,  # Pre-computed array mappings

    # Straddle structure
    straddle_starts: np.ndarray,       # int32[S] - start index per straddle
    straddle_lengths: np.ndarray,      # int32[S] - length per straddle
    ntry_offsets: np.ndarray,          # int32[S] - entry offset per straddle
    xpry_offsets: np.ndarray,          # int32[S] - expiry offset per straddle

    # Options
    valid_only: bool = False,
) -> pa.Table:
    """Build Arrow table from numeric arrays for sorted kernel output.

    Full version matching _build_arrow_output with hedge1/2/3 and strike1/2/3.
    """
    n_straddles = len(straddle_starts)

    # Pre-compute strike_vol per straddle
    strike_vol = np.full(n_straddles, np.nan, dtype=np.float64)
    for s in range(n_straddles):
        ntry = ntry_offsets[s]
        if ntry >= 0:
            start = straddle_starts[s]
            strike_vol[s] = vol[start + ntry]

    # Pre-compute expiry date32 per straddle
    expiry_date32 = np.full(n_straddles, -1, dtype=np.int32)
    for s in range(n_straddles):
        xpry = xpry_offsets[s]
        if xpry >= 0:
            start = straddle_starts[s]
            expiry_date32[s] = dates[start + xpry]

    # Build valid_mask for filtering (rows with valid mv)
    if valid_only:
        filter_mask = ~np.isnan(mv)
        indices = np.where(filter_mask)[0]

        # Filter arrays
        dates = dates[indices]
        vol = vol[indices]
        hedge = hedge[indices]
        hedge1 = hedge1[indices]
        hedge2 = hedge2[indices]
        hedge3 = hedge3[indices]
        strike = strike[indices]
        strike1 = strike1[indices]
        strike2 = strike2[indices]
        strike3 = strike3[indices]
        mv = mv[indices]
        delta = delta[indices]
        opnl = opnl[indices]
        hpnl = hpnl[indices]
        pnl = pnl[indices]
        action = action[indices]
        parent_idx = parent_idx[indices]

    # Build per-row arrays from per-straddle data
    asset_indices = backtest_arrays.asset_idx[parent_idx]
    straddle_indices = backtest_arrays.straddle_idx[parent_idx]
    model_indices = backtest_arrays.model_idx[parent_idx]
    strike_vol_values = strike_vol[parent_idx]
    expiry_values = expiry_date32[parent_idx]

    # Build dictionary-encoded string columns
    asset_dict = pa.DictionaryArray.from_arrays(
        pa.array(asset_indices, type=pa.int32()),
        pa.array(backtest_arrays.unique_assets, type=pa.string())
    )

    straddle_dict = pa.DictionaryArray.from_arrays(
        pa.array(straddle_indices, type=pa.int32()),
        pa.array(backtest_arrays.unique_straddles, type=pa.string())
    )

    model_dict = pa.DictionaryArray.from_arrays(
        pa.array(model_indices, type=pa.int32()),
        pa.array(backtest_arrays.unique_models, type=pa.string())
    )

    action_strs = ["", "ntry", "xpry"]
    action_dict = pa.DictionaryArray.from_arrays(
        pa.array(action, type=pa.int8()),
        pa.array(action_strs, type=pa.string())
    )

    date_array = pa.array(dates, type=pa.date32())
    expiry_array = pa.array(expiry_values, type=pa.date32())

    # Build table with all columns (matching _build_arrow_output)
    return pa.table({
        'asset': asset_dict,
        'straddle': straddle_dict,
        'date': date_array,
        'vol': pa.array(vol),
        'hedge': pa.array(hedge),
        'hedge1': pa.array(hedge1),
        'hedge2': pa.array(hedge2),
        'hedge3': pa.array(hedge3),
        'action': action_dict,
        'model': model_dict,
        'strike_vol': pa.array(strike_vol_values),
        'strike': pa.array(strike),
        'strike1': pa.array(strike1),
        'strike2': pa.array(strike2),
        'strike3': pa.array(strike3),
        'expiry': expiry_array,
        'mv': pa.array(mv),
        'delta': pa.array(delta),
        'opnl': pa.array(opnl),
        'hpnl': pa.array(hpnl),
        'pnl': pa.array(pnl),
    })


def _build_matrix_lookup_indices(
    straddle_list: list[tuple[str, str]],
    straddle_starts: np.ndarray,
    straddle_lengths: np.ndarray,
    dates: np.ndarray,
    ticker_map: dict[str, dict[str, tuple[str, str]]],
    stryms: list[str],
    ntrcs: list[str],
    amt_path: str,
    price_matrix: "prices_module.PriceMatrix",
    chain_csv: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build row/column indices for matrix-based price lookup.

    Pre-computes all the integer indices needed for the Numba batch_price_lookup
    kernel. This converts string ticker/field/date lookups into integer array
    indices that Numba can process efficiently.

    Args:
        straddle_list: List of (asset, straddle) tuples
        straddle_starts: int32[S] - start index for each straddle
        straddle_lengths: int32[S] - number of days per straddle
        dates: int32[N] - date32 values for all days
        ticker_map: Dict from _batch_resolve_tickers()
        stryms: List of strym strings per straddle
        ntrcs: List of ntrc codes per straddle
        amt_path: Path to AMT YAML file
        price_matrix: PriceMatrix object with price data
        chain_csv: Optional CSV for futures ticker normalization

    Returns:
        vol_row_idx: int32[N] - row index for vol lookup (-1 if invalid)
        hedge_row_idx: int32[N] - row index for hedge lookup (-1 if invalid)
        col_idx: int32[N] - column index for date lookup (-1 if invalid)
    """
    n_days = len(dates)
    n_straddles = len(straddle_list)

    # Build column indices using Numba kernel (vectorized)
    col_idx = valuation_numba.build_col_indices(
        dates, price_matrix.date32_to_col, price_matrix.min_date32
    )

    # Step 1: Build per-straddle row indices (one lookup per straddle, not per day)
    vol_row_per_straddle = np.full(n_straddles, -1, dtype=np.int32)
    hedge_row_per_straddle = np.full(n_straddles, -1, dtype=np.int32)

    # Cache to avoid repeated ticker resolution
    ticker_row_cache: dict[tuple[str, str], int] = {}

    for s in range(n_straddles):
        asset, straddle = straddle_list[s]
        strym = stryms[s]
        ntrc = ntrcs[s]

        # Get asset data for cache key
        asset_data = loader.get_asset(amt_path, asset)
        if asset_data is None:
            continue

        vol_cfg = asset_data.get("Vol")
        hedge_cfg = asset_data.get("Hedge")
        if vol_cfg is None or hedge_cfg is None:
            continue

        cache_key = asset_straddle_tickers.asset_straddle_ticker_key(
            asset, strym, ntrc, vol_cfg, hedge_cfg
        )

        if cache_key not in ticker_map:
            continue

        param_map = ticker_map[cache_key]

        # Get vol ticker row index (with caching)
        if "vol" in param_map:
            vol_ticker, vol_field = param_map["vol"]
            if chain_csv is not None:
                normalized = chain.fut_act2norm(chain_csv, vol_ticker)
                if normalized is not None:
                    vol_ticker = normalized
            vol_key = (vol_ticker, vol_field)
            if vol_key not in ticker_row_cache:
                ticker_row_cache[vol_key] = price_matrix.get_row_idx(vol_ticker, vol_field)
            vol_row_per_straddle[s] = ticker_row_cache[vol_key]

        # Get hedge ticker row index (with caching)
        if "hedge" in param_map:
            hedge_ticker, hedge_field = param_map["hedge"]
            if chain_csv is not None:
                normalized = chain.fut_act2norm(chain_csv, hedge_ticker)
                if normalized is not None:
                    hedge_ticker = normalized
            hedge_key = (hedge_ticker, hedge_field)
            if hedge_key not in ticker_row_cache:
                ticker_row_cache[hedge_key] = price_matrix.get_row_idx(hedge_ticker, hedge_field)
            hedge_row_per_straddle[s] = ticker_row_cache[hedge_key]

    # Step 2: Expand per-straddle indices to per-day indices using Numba
    vol_row_idx, hedge_row_idx = _expand_straddle_indices_to_days(
        vol_row_per_straddle, hedge_row_per_straddle,
        straddle_starts, straddle_lengths, n_days
    )

    return vol_row_idx, hedge_row_idx, col_idx


def _build_numba_sorted_lookup_indices(
    straddle_list: list[tuple[str, str]],
    straddle_starts: np.ndarray,
    straddle_lengths: np.ndarray,
    dates: np.ndarray,
    ticker_map: dict[str, dict[str, tuple[str, str]]],
    stryms: list[str],
    ntrcs: list[str],
    amt_path: str,
    prices_numba: "prices_module.PricesNumba",
    chain_csv: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build lookup indices for PricesNumba sorted array lookup.

    Pre-computes ticker/field indices and date offsets for the Numba
    batch_lookup_vol_hedge_sorted kernel.

    Args:
        straddle_list: List of (asset, straddle) tuples
        straddle_starts: int32[S] - start index for each straddle
        straddle_lengths: int32[S] - number of days per straddle
        dates: int32[N] - date32 values for all days
        ticker_map: Dict from _batch_resolve_tickers()
        stryms: List of strym strings per straddle
        ntrcs: List of ntrc codes per straddle
        amt_path: Path to AMT YAML file
        prices_numba: PricesNumba object with sorted arrays
        chain_csv: Optional CSV for futures ticker normalization

    Returns:
        vol_ticker_indices: int32[N] - ticker index for vol lookup (-1 if invalid)
        vol_field_indices: int32[N] - field index for vol lookup (-1 if invalid)
        hedge_ticker_indices: int32[N] - ticker index for hedge lookup (-1 if invalid)
        hedge_field_indices: int32[N] - field index for hedge lookup (-1 if invalid)
        date_offsets: int32[N] - date offset for lookup (date32 - min_date32)
    """
    n_days = len(dates)
    n_straddles = len(straddle_list)

    # Build date offsets (vectorized)
    date_offsets = (dates - prices_numba.min_date32).astype(np.int32)

    # Per-straddle ticker/field indices
    vol_ticker_per_straddle = np.full(n_straddles, -1, dtype=np.int32)
    vol_field_per_straddle = np.full(n_straddles, -1, dtype=np.int32)
    hedge_ticker_per_straddle = np.full(n_straddles, -1, dtype=np.int32)
    hedge_field_per_straddle = np.full(n_straddles, -1, dtype=np.int32)

    # Cache to avoid repeated ticker resolution
    ticker_idx_cache: dict[str, int] = {}
    field_idx_cache: dict[str, int] = {}

    for s in range(n_straddles):
        asset, straddle = straddle_list[s]
        strym = stryms[s]
        ntrc = ntrcs[s]

        # Get asset data for cache key
        asset_data = loader.get_asset(amt_path, asset)
        if asset_data is None:
            continue

        vol_cfg = asset_data.get("Vol")
        hedge_cfg = asset_data.get("Hedge")
        if vol_cfg is None or hedge_cfg is None:
            continue

        cache_key = asset_straddle_tickers.asset_straddle_ticker_key(
            asset, strym, ntrc, vol_cfg, hedge_cfg
        )

        if cache_key not in ticker_map:
            continue

        param_map = ticker_map[cache_key]

        # Get vol ticker/field indices
        if "vol" in param_map:
            vol_ticker, vol_field = param_map["vol"]
            if chain_csv is not None:
                normalized = chain.fut_act2norm(chain_csv, vol_ticker)
                if normalized is not None:
                    vol_ticker = normalized

            # Look up or cache ticker index
            if vol_ticker not in ticker_idx_cache:
                ticker_idx_cache[vol_ticker] = prices_numba.get_ticker_idx(vol_ticker)
            vol_ticker_per_straddle[s] = ticker_idx_cache[vol_ticker]

            # Look up or cache field index
            if vol_field not in field_idx_cache:
                field_idx_cache[vol_field] = prices_numba.get_field_idx(vol_field)
            vol_field_per_straddle[s] = field_idx_cache[vol_field]

        # Get hedge ticker/field indices
        if "hedge" in param_map:
            hedge_ticker, hedge_field = param_map["hedge"]
            if chain_csv is not None:
                normalized = chain.fut_act2norm(chain_csv, hedge_ticker)
                if normalized is not None:
                    hedge_ticker = normalized

            if hedge_ticker not in ticker_idx_cache:
                ticker_idx_cache[hedge_ticker] = prices_numba.get_ticker_idx(hedge_ticker)
            hedge_ticker_per_straddle[s] = ticker_idx_cache[hedge_ticker]

            if hedge_field not in field_idx_cache:
                field_idx_cache[hedge_field] = prices_numba.get_field_idx(hedge_field)
            hedge_field_per_straddle[s] = field_idx_cache[hedge_field]

    # Expand per-straddle indices to per-day arrays
    vol_ticker_indices, vol_field_indices, hedge_ticker_indices, hedge_field_indices = \
        _expand_straddle_tf_indices_to_days(
            vol_ticker_per_straddle, vol_field_per_straddle,
            hedge_ticker_per_straddle, hedge_field_per_straddle,
            straddle_starts, straddle_lengths, n_days
        )

    return vol_ticker_indices, vol_field_indices, hedge_ticker_indices, hedge_field_indices, date_offsets


@njit(cache=True, parallel=True)
def _expand_straddle_tf_indices_to_days(
    vol_ticker_per_straddle: np.ndarray,
    vol_field_per_straddle: np.ndarray,
    hedge_ticker_per_straddle: np.ndarray,
    hedge_field_per_straddle: np.ndarray,
    straddle_starts: np.ndarray,
    straddle_lengths: np.ndarray,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand per-straddle ticker/field indices to per-day arrays.

    Args:
        vol_ticker_per_straddle: int32[S] - vol ticker index for each straddle
        vol_field_per_straddle: int32[S] - vol field index for each straddle
        hedge_ticker_per_straddle: int32[S] - hedge ticker index for each straddle
        hedge_field_per_straddle: int32[S] - hedge field index for each straddle
        straddle_starts: int32[S] - start index for each straddle
        straddle_lengths: int32[S] - number of days per straddle
        n_days: Total number of days

    Returns:
        vol_ticker_indices: int32[N] - vol ticker index for each day
        vol_field_indices: int32[N] - vol field index for each day
        hedge_ticker_indices: int32[N] - hedge ticker index for each day
        hedge_field_indices: int32[N] - hedge field index for each day
    """
    vol_ticker_indices = np.full(n_days, -1, dtype=np.int32)
    vol_field_indices = np.full(n_days, -1, dtype=np.int32)
    hedge_ticker_indices = np.full(n_days, -1, dtype=np.int32)
    hedge_field_indices = np.full(n_days, -1, dtype=np.int32)
    n_straddles = len(straddle_starts)

    for s in prange(n_straddles):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        vt = vol_ticker_per_straddle[s]
        vf = vol_field_per_straddle[s]
        ht = hedge_ticker_per_straddle[s]
        hf = hedge_field_per_straddle[s]

        for i in range(length):
            idx = start + i
            vol_ticker_indices[idx] = vt
            vol_field_indices[idx] = vf
            hedge_ticker_indices[idx] = ht
            hedge_field_indices[idx] = hf

    return vol_ticker_indices, vol_field_indices, hedge_ticker_indices, hedge_field_indices


@njit(cache=True, parallel=True)
def _expand_straddle_indices_to_days(
    vol_row_per_straddle: np.ndarray,
    hedge_row_per_straddle: np.ndarray,
    straddle_starts: np.ndarray,
    straddle_lengths: np.ndarray,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand per-straddle indices to per-day arrays using Numba parallel.

    Args:
        vol_row_per_straddle: int32[S] - vol row index for each straddle
        hedge_row_per_straddle: int32[S] - hedge row index for each straddle
        straddle_starts: int32[S] - start index for each straddle
        straddle_lengths: int32[S] - number of days per straddle
        n_days: Total number of days

    Returns:
        vol_row_idx: int32[N] - vol row index for each day
        hedge_row_idx: int32[N] - hedge row index for each day
    """
    vol_row_idx = np.full(n_days, -1, dtype=np.int32)
    hedge_row_idx = np.full(n_days, -1, dtype=np.int32)
    n_straddles = len(straddle_starts)

    for s in prange(n_straddles):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        vol_row = vol_row_per_straddle[s]
        hedge_row = hedge_row_per_straddle[s]

        for i in range(length):
            idx = start + i
            vol_row_idx[idx] = vol_row
            hedge_row_idx[idx] = hedge_row

    return vol_row_idx, hedge_row_idx


def _duckdb_price_lookup(
    row_idx: np.ndarray,
    req_dates: list,
    tickers: list[str],
    fields: list[str],
    params: list[str],
    prices_parquet: str,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lookup prices using DuckDB join against parquet.

    Performs a SQL JOIN between the request table and the prices parquet
    file. DuckDB handles the join efficiently using columnar execution.

    Args:
        row_idx: int32[M] - index into output arrays for each request
        req_dates: list[M] - date objects for lookup
        tickers: list[M] - ticker strings
        fields: list[M] - field strings
        params: list[M] - param name ('vol' or 'hedge')
        prices_parquet: Path to prices parquet file
        n_days: Total number of days (size of output arrays)

    Returns:
        vol_array: float64[n_days] - vol values, NaN for missing
        hedge_array: float64[n_days] - hedge values, NaN for missing
    """
    import duckdb

    # Build request table as Arrow (DuckDB reads Arrow efficiently)
    request = pa.table({
        'row_idx': pa.array(row_idx, type=pa.int32()),
        'date': pa.array(req_dates),
        'ticker': pa.array(tickers),
        'field': pa.array(fields),
        'param': pa.array(params),
    })

    con = duckdb.connect()
    con.register('request', request)

    result = con.execute(f"""
        SELECT r.row_idx, r.param, p.value
        FROM request r
        LEFT JOIN read_parquet('{prices_parquet}') p
            ON r.ticker = p.ticker
            AND r.field = p.field
            AND r.date = p.date
    """).fetchall()
    con.close()

    # Pivot to arrays
    vol_array = np.full(n_days, np.nan, dtype=np.float64)
    hedge_array = np.full(n_days, np.nan, dtype=np.float64)

    for idx, param, value in result:
        if value is not None:
            try:
                float_val = float(value)
                if param == 'vol':
                    vol_array[idx] = float_val
                elif param == 'hedge':
                    hedge_array[idx] = float_val
            except (ValueError, TypeError):
                pass

    return vol_array, hedge_array


def get_straddle_backtests(
    pattern: str,
    start_year: int,
    end_year: int,
    amt_path: str,
    chain_csv: str | None = None,
    prices_parquet: str | None = None,
    valid_only: bool = False,
    price_lookup: str = "dict",
    output_format: str = "dict",
    overrides_path: str | None = None,
) -> dict[str, Any] | pa.Table:
    """Batch valuation for all straddles matching pattern.

    This is a vectorized implementation that processes all straddles at once.
    Uses Numba-accelerated model computation for performance.

    Args:
        pattern: Regex pattern to match assets (e.g., "^LA Comdty", ".")
        start_year: Start year for straddles
        end_year: End year for straddles
        amt_path: Path to AMT YAML file
        chain_csv: Optional CSV for futures ticker lookup
        prices_parquet: Path to prices parquet file (required for arrow/duckdb/matrix/numba lookup)
        valid_only: If True, only return rows with valid mv values (faster output)
        price_lookup: Strategy for price retrieval:
            - "dict": Use preloaded prices_dict (default, requires load_all_prices())
            - "arrow": Use PyArrow join (vectorized, cached Arrow table)
            - "duckdb": Use DuckDB join (vectorized, reads parquet directly)
            - "matrix": Use Numba-accelerated matrix lookup (requires load_prices_matrix())
            - "numba": Full Numba pipeline - single kernel for price lookup, actions,
              roll-forward, model, and PnL. Returns Arrow table. (FASTEST)
        output_format: Output format for the result:
            - "dict": Return dict with columns and rows (default, compatible with existing code)
            - "arrow": Return PyArrow Table (fastest, recommended for large results)
            Note: price_lookup="numba" ignores this and always returns Arrow table.
        overrides_path: Path to overrides CSV file for OVERRIDE code anchor dates.
            Defaults to "data/overrides.csv" if not specified.

    Returns:
        If output_format="dict": Table dict with columns matching get_straddle_valuation():
            asset, straddle, date, vol, hedge, ..., action, model,
            strike_vol, strike, expiry, mv, delta, opnl, hpnl, pnl
        If output_format="arrow" or price_lookup="numba": PyArrow Table

    Example:
        >>> from specparser.amt import get_straddle_backtests, load_all_prices, set_prices_dict
        >>> prices_dict = load_all_prices("data/prices.parquet", "2023-01-01", "2024-12-31")
        >>> set_prices_dict(prices_dict)
        >>> result = get_straddle_backtests("^LA Comdty", 2024, 2024, "data/amt.yml")
        >>> print(f"{len(result['rows']):,} rows")

        # Fastest: Full Numba pipeline returning Arrow table
        >>> result = get_straddle_backtests(".", 2001, 2026, "data/amt.yml",
        ...                                  prices_parquet="data/prices.parquet",
        ...                                  price_lookup="numba", valid_only=True)
        >>> print(f"{len(result):,} rows")  # result is Arrow table
    """
    # Phase 1+2: Use fast u8m-based date expansion (Numba kernel)
    # This replaces both find_straddle_yrs() and the Python loop with straddle_days()
    straddle_days_table = schedules.find_straddle_days_u8m(
        amt_path, start_year, end_year, pattern, live_only=True
    )

    # Extract u8m arrays and date32
    asset_u8m = straddle_days_table["rows"][0]  # (n_days, width) uint8
    straddle_u8m = straddle_days_table["rows"][1]  # (n_days, width) uint8
    dates = straddle_days_table["rows"][2]  # (n_days,) int32
    parent_idx = straddle_days_table["parent_idx"]  # (n_days,) int32

    n_days = len(dates)

    if n_days == 0:
        return {
            "orientation": "row",
            "columns": ["asset", "straddle", "date", "vol", "hedge", "hedge1", "hedge2", "hedge3",
                        "action", "model", "strike_vol", "strike", "strike1", "strike2", "strike3",
                        "expiry", "mv", "delta", "opnl", "hpnl", "pnl"],
            "rows": [],
        }

    # Compute straddle_starts and straddle_lengths from parent_idx
    straddle_starts, straddle_lengths = _compute_starts_lengths_from_parent_idx(parent_idx)
    n_straddles_actual = len(straddle_starts)

    # Build straddle_list from unique u8m rows (one per straddle, not per day)
    # Use parent_idx to get the first day of each straddle
    unique_indices = straddle_starts  # First day of each straddle
    unique_asset_u8m = asset_u8m[unique_indices]
    unique_straddle_u8m = straddle_u8m[unique_indices]

    # Convert u8m to strings for downstream compatibility
    # Note: u8m strings have trailing padding that must be stripped for asset lookups
    unique_assets = strings_module.u8m2s(unique_asset_u8m)
    unique_straddles = strings_module.u8m2s(unique_straddle_u8m)

    # Strip padding from assets (required for loader.get_asset() lookups)
    # Straddles don't need stripping - the parsing functions work with padding
    straddle_list = [(asset.strip(), straddle) for asset, straddle in zip(unique_assets.tolist(), unique_straddles.tolist())]
    n_straddles = len(straddle_list)

    # Phase 3: Resolve tickers (batched)
    # Extract unique (asset, strym, ntrc) combinations
    stryms = []
    ntrcs = []
    assets_for_tickers = []
    for asset, straddle in straddle_list:
        xpry = schedules.xpry(straddle)
        xprm = schedules.xprm(straddle)
        ntrc = schedules.ntrc(straddle)
        stryms.append(f"{xpry}-{xprm:02d}")
        ntrcs.append(ntrc)
        assets_for_tickers.append(asset)

    ticker_map = _batch_resolve_tickers(assets_for_tickers, stryms, ntrcs, amt_path)

    # Get model names per straddle (needed for Phase 6)
    model_names = []
    for s in range(n_straddles_actual):
        asset = straddle_list[s][0]  # (asset, straddle) tuple
        asset_data = loader.get_asset(amt_path, asset)
        if asset_data is not None:
            valuation = asset_data.get("Valuation", {})
            model = valuation.get("Model", "") if isinstance(valuation, dict) else ""
        else:
            model = ""
        model_names.append(model)

    # Phase 4: Price lookup - dispatch based on strategy
    vol_array = np.full(n_days, np.nan, dtype=np.float64)
    hedge_array = np.full(n_days, np.nan, dtype=np.float64)

    if price_lookup == "numba":
        # Full Numba pipeline - combines all computation into a single kernel
        if prices_parquet is None:
            raise ValueError("prices_parquet required for numba lookup")

        # Get or load the price matrix
        price_matrix = prices_module.get_prices_matrix()
        if price_matrix is None:
            price_matrix = prices_module.load_prices_matrix(prices_parquet)

        # Prepare all arrays (one-time string lookups)
        # Default to data/overrides.csv if not specified
        effective_overrides = overrides_path if overrides_path is not None else "data/overrides.csv"
        backtest_arrays = _prepare_backtest_arrays(
            straddle_list,
            ticker_map,
            price_matrix,
            stryms,
            ntrcs,
            amt_path,
            chain_csv,
            effective_overrides,
        )

        # Run the unified Numba kernel
        (vol_array, hedge_array, hedge1_array, hedge2_array, hedge3_array,
         ntry_offsets, xpry_offsets,
         strike_array, strike1_array, strike2_array, strike3_array,
         days_to_expiry, mv, delta, opnl, hpnl, pnl, action) = \
            valuation_numba.full_backtest_kernel(
                price_matrix.price_matrix,
                price_matrix.date32_to_col,
                price_matrix.min_date32,
                backtest_arrays.vol_row_idx,
                backtest_arrays.hedge_row_idx,
                backtest_arrays.hedge1_row_idx,
                backtest_arrays.hedge2_row_idx,
                backtest_arrays.hedge3_row_idx,
                backtest_arrays.n_hedges,
                straddle_starts,
                straddle_lengths,
                backtest_arrays.ntry_anchor_date32,
                backtest_arrays.xpry_anchor_date32,
                backtest_arrays.ntrv_offsets,
                backtest_arrays.ntry_month_end,
                backtest_arrays.xpry_month_end,
                dates,
            )

        # Build parent_idx mapping each day back to its straddle
        parent_idx = np.zeros(n_days, dtype=np.int32)
        for s in range(n_straddles_actual):
            start = straddle_starts[s]
            length = straddle_lengths[s]
            parent_idx[start:start+length] = s

        # Build Arrow output (near zero-copy)
        return _build_arrow_output(
            dates=dates,
            vol=vol_array,
            hedge=hedge_array,
            hedge1=hedge1_array,
            hedge2=hedge2_array,
            hedge3=hedge3_array,
            strike=strike_array,
            strike1=strike1_array,
            strike2=strike2_array,
            strike3=strike3_array,
            mv=mv,
            delta=delta,
            opnl=opnl,
            hpnl=hpnl,
            pnl=pnl,
            action=action,
            parent_idx=parent_idx,
            backtest_arrays=backtest_arrays,
            straddle_starts=straddle_starts,
            straddle_lengths=straddle_lengths,
            ntry_offsets=ntry_offsets,
            xpry_offsets=xpry_offsets,
            valid_only=valid_only,
        )

    elif price_lookup == "numba_sorted_kernel":
        # Full Numba pipeline with sorted array price lookup (fastest)
        if prices_parquet is None:
            raise ValueError("prices_parquet required for numba_sorted_kernel lookup")

        # Get or load the PricesNumba structure
        prices_numba = prices_module.get_prices_numba()
        if prices_numba is None:
            prices_numba = prices_module.load_prices_numba(prices_parquet)

        # Prepare all arrays (one-time string lookups)
        effective_overrides = overrides_path if overrides_path is not None else "data/overrides.csv"
        backtest_arrays = _prepare_backtest_arrays_sorted(
            straddle_list,
            ticker_map,
            prices_numba,
            stryms,
            ntrcs,
            amt_path,
            chain_csv,
            effective_overrides,
        )

        # Run the unified Numba kernel with sorted array lookup
        (vol_array, hedge_array, hedge1_array, hedge2_array, hedge3_array,
         ntry_offsets, xpry_offsets,
         strike_array, strike1_array, strike2_array, strike3_array,
         days_to_expiry, mv, delta, opnl, hpnl, pnl, action) = \
            valuation_numba.full_backtest_kernel_sorted(
                prices_numba.sorted_keys,
                prices_numba.sorted_values,
                prices_numba.n_fields,
                prices_numba.n_dates,
                prices_numba.min_date32,
                backtest_arrays.vol_ticker_idx,
                backtest_arrays.vol_field_idx,
                backtest_arrays.hedge_ticker_idx,
                backtest_arrays.hedge_field_idx,
                backtest_arrays.hedge1_ticker_idx,
                backtest_arrays.hedge1_field_idx,
                backtest_arrays.hedge2_ticker_idx,
                backtest_arrays.hedge2_field_idx,
                backtest_arrays.hedge3_ticker_idx,
                backtest_arrays.hedge3_field_idx,
                backtest_arrays.n_hedges,
                straddle_starts,
                straddle_lengths,
                backtest_arrays.ntry_anchor_date32,
                backtest_arrays.xpry_anchor_date32,
                backtest_arrays.ntrv_offsets,
                backtest_arrays.ntry_month_end,
                backtest_arrays.xpry_month_end,
                dates,
            )

        # Build parent_idx mapping each day back to its straddle
        parent_idx = np.zeros(n_days, dtype=np.int32)
        for s in range(n_straddles_actual):
            start = straddle_starts[s]
            length = straddle_lengths[s]
            parent_idx[start:start+length] = s

        # Build Arrow output
        return _build_arrow_output_sorted(
            dates=dates,
            vol=vol_array,
            hedge=hedge_array,
            hedge1=hedge1_array,
            hedge2=hedge2_array,
            hedge3=hedge3_array,
            strike=strike_array,
            strike1=strike1_array,
            strike2=strike2_array,
            strike3=strike3_array,
            mv=mv,
            delta=delta,
            opnl=opnl,
            hpnl=hpnl,
            pnl=pnl,
            action=action,
            parent_idx=parent_idx,
            backtest_arrays=backtest_arrays,
            straddle_starts=straddle_starts,
            straddle_lengths=straddle_lengths,
            ntry_offsets=ntry_offsets,
            xpry_offsets=xpry_offsets,
            valid_only=valid_only,
        )

    elif price_lookup == "matrix":
        # Numba-accelerated matrix lookup (fastest)
        if prices_parquet is None:
            raise ValueError("prices_parquet required for matrix lookup")

        # Get or load the price matrix
        price_matrix = prices_module.get_prices_matrix()
        if price_matrix is None:
            price_matrix = prices_module.load_prices_matrix(prices_parquet)

        # Build lookup indices
        vol_row_idx, hedge_row_idx, col_idx = _build_matrix_lookup_indices(
            straddle_list,
            straddle_starts,
            straddle_lengths,
            dates,
            ticker_map,
            stryms,
            ntrcs,
            amt_path,
            price_matrix,
            chain_csv,
        )

        # Run Numba kernel for batch lookup
        vol_array, hedge_array = valuation_numba.batch_price_lookup(
            price_matrix.price_matrix,
            vol_row_idx,
            hedge_row_idx,
            col_idx,
        )

    elif price_lookup in ("arrow", "duckdb"):
        # Vectorized lookup using Arrow or DuckDB join
        if prices_parquet is None:
            raise ValueError(f"prices_parquet required for {price_lookup} lookup")

        # Build request table
        row_idx, req_dates, tickers, fields, params = _build_price_request_table(
            straddle_list,
            straddle_starts,
            straddle_lengths,
            dates,
            ticker_map,
            stryms,
            ntrcs,
            amt_path,
            chain_csv,
        )

        if len(row_idx) > 0:
            if price_lookup == "arrow":
                vol_array, hedge_array = _arrow_price_lookup(
                    row_idx, req_dates, tickers, fields, params,
                    prices_parquet, n_days
                )
            else:  # duckdb
                vol_array, hedge_array = _duckdb_price_lookup(
                    row_idx, req_dates, tickers, fields, params,
                    prices_parquet, n_days
                )

    elif price_lookup == "numba_sorted":
        # Fast PyArrow-based loading with Numba binary search lookup
        if prices_parquet is None:
            raise ValueError("prices_parquet required for numba_sorted lookup")

        # Get or load the PricesNumba structure
        prices_numba = prices_module.get_prices_numba()
        if prices_numba is None:
            prices_numba = prices_module.load_prices_numba(prices_parquet)

        # Build lookup index arrays
        vol_ticker_indices, vol_field_indices, hedge_ticker_indices, hedge_field_indices, date_offsets = \
            _build_numba_sorted_lookup_indices(
                straddle_list,
                straddle_starts,
                straddle_lengths,
                dates,
                ticker_map,
                stryms,
                ntrcs,
                amt_path,
                prices_numba,
                chain_csv,
            )

        # Run Numba kernel for batch lookup
        vol_array, hedge_array = valuation_numba.batch_lookup_vol_hedge_sorted(
            prices_numba.sorted_keys,
            prices_numba.sorted_values,
            vol_ticker_indices,
            vol_field_indices,
            hedge_ticker_indices,
            hedge_field_indices,
            date_offsets,
            prices_numba.n_fields,
            prices_numba.n_dates,
        )

    else:
        # Default: dict-based lookup (original Python loop)
        if prices_module._PRICES_DICT is None:
            if prices_parquet is None:
                raise ValueError("No prices available: call load_all_prices() or provide prices_parquet")
            prices_module.load_all_prices(prices_parquet)

        prices_dict = prices_module._PRICES_DICT

        for s in range(n_straddles_actual):
            start = straddle_starts[s]
            length = straddle_lengths[s]
            asset, straddle = straddle_list[s]

            strym = stryms[s]
            ntrc = ntrcs[s]

            # Get asset data for cache key
            asset_data = loader.get_asset(amt_path, asset)
            if asset_data is None:
                continue

            vol_cfg = asset_data.get("Vol")
            hedge_cfg = asset_data.get("Hedge")
            if vol_cfg is None or hedge_cfg is None:
                continue

            cache_key = asset_straddle_tickers.asset_straddle_ticker_key(
                asset, strym, ntrc, vol_cfg, hedge_cfg
            )

            if cache_key not in ticker_map:
                continue

            param_map = ticker_map[cache_key]

            # Get vol ticker and look up prices directly
            if "vol" in param_map:
                vol_ticker, vol_field = param_map["vol"]
                # Normalize ticker if needed
                if chain_csv is not None:
                    normalized = chain.fut_act2norm(chain_csv, vol_ticker)
                    if normalized is not None:
                        vol_ticker = normalized
                # Look up prices for this straddle's dates
                for i in range(length):
                    date_str = prices_module._date32_to_isoformat(int(dates[start + i]))
                    key = f"{vol_ticker}|{vol_field}|{date_str}"
                    value_str = prices_dict.get(key)
                    if value_str is not None and value_str != "none":
                        try:
                            vol_array[start + i] = float(value_str)
                        except (ValueError, TypeError):
                            pass

            # Get hedge ticker and look up prices directly
            if "hedge" in param_map:
                hedge_ticker, hedge_field = param_map["hedge"]
                if chain_csv is not None:
                    normalized = chain.fut_act2norm(chain_csv, hedge_ticker)
                    if normalized is not None:
                        hedge_ticker = normalized
                # Look up prices for this straddle's dates
                for i in range(length):
                    date_str = prices_module._date32_to_isoformat(int(dates[start + i]))
                    key = f"{hedge_ticker}|{hedge_field}|{date_str}"
                    value_str = prices_dict.get(key)
                    if value_str is not None and value_str != "none":
                        try:
                            hedge_array[start + i] = float(value_str)
                        except (ValueError, TypeError):
                            pass

    # Phase 5: Compute actions (simplified)
    vol_valid = ~np.isnan(vol_array)
    hedge_valid = ~np.isnan(hedge_array)

    # Compute target entry/expiry offsets
    # Entry is at the beginning of entry month, expiry at beginning of expiry month
    # For simplicity, use offset 0 for both (first valid day)
    ntry_target_offsets = np.zeros(n_straddles_actual, dtype=np.int32)
    xpry_target_offsets = np.zeros(n_straddles_actual, dtype=np.int32)

    # Compute actual month offsets within each straddle
    for s in range(n_straddles_actual):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        _, straddle = straddle_list[s]

        ntry_y = schedules.ntry(straddle)
        ntry_m = schedules.ntrm(straddle)
        xpry_y = schedules.xpry(straddle)
        xpry_m = schedules.xprm(straddle)

        # Find offset where entry month starts
        ntry_month_start = valuation_numba.ymd_to_date32(ntry_y, ntry_m, 1)
        xpry_month_start = valuation_numba.ymd_to_date32(xpry_y, xpry_m, 1)

        # Find first day in entry month
        for i in range(length):
            if dates[start + i] >= ntry_month_start:
                ntry_target_offsets[s] = i
                break

        # Find first day in expiry month
        for i in range(length):
            if dates[start + i] >= xpry_month_start:
                xpry_target_offsets[s] = i
                break

    ntry_offsets, xpry_offsets = _compute_actions_batch_simple(
        dates, straddle_starts, straddle_lengths,
        vol_valid, hedge_valid,
        ntry_target_offsets, xpry_target_offsets
    )

    # Phase 6: Roll-forward and model computation
    vol_rolled = valuation_numba.roll_forward_by_straddle(
        vol_array, straddle_starts, straddle_lengths, ntry_offsets, xpry_offsets
    )
    hedge_rolled = valuation_numba.roll_forward_by_straddle(
        hedge_array, straddle_starts, straddle_lengths, ntry_offsets, xpry_offsets
    )

    # Build strike array (hedge at entry for each straddle)
    strike_array = np.full(n_days, np.nan, dtype=np.float64)
    for s in range(n_straddles_actual):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        ntry = ntry_offsets[s]
        xpry = xpry_offsets[s]

        if ntry >= 0:
            strike_val = hedge_rolled[start + ntry]
            for i in range(length):
                if ntry <= i <= xpry:
                    strike_array[start + i] = strike_val

    # Compute days to expiry
    days_to_expiry = valuation_numba.compute_days_to_expiry(
        dates, straddle_starts, straddle_lengths, xpry_offsets
    )

    # Build valid mask (only between ntry and xpry)
    valid_mask = np.zeros(n_days, dtype=np.bool_)
    for s in range(n_straddles_actual):
        start = straddle_starts[s]
        ntry = ntry_offsets[s]
        xpry = xpry_offsets[s]
        if ntry >= 0 and xpry >= 0:
            for i in range(ntry, xpry + 1):
                valid_mask[start + i] = True

    # Run vectorized model
    mv, delta = valuation_numba.model_ES_vectorized(
        hedge_rolled, strike_array, vol_rolled, days_to_expiry, valid_mask
    )

    # Phase 7: PnL computation
    opnl, hpnl, pnl = valuation_numba.compute_pnl_batch(
        mv, delta, hedge_rolled, strike_array,
        straddle_starts, straddle_lengths, ntry_offsets, xpry_offsets
    )

    # Phase 8: Assemble output table
    # Convert dates back to strings
    epoch = date(1970, 1, 1)
    date_strings = [(date.fromordinal(epoch.toordinal() + int(d))).isoformat() for d in dates]

    # Build action strings
    action_strings = ["-"] * n_days
    for s in range(n_straddles_actual):
        start = straddle_starts[s]
        ntry = ntry_offsets[s]
        xpry = xpry_offsets[s]
        if ntry >= 0:
            action_strings[start + ntry] = "ntry"
        if xpry >= 0:
            action_strings[start + xpry] = "xpry"

    # Build expiry strings
    expiry_strings = ["-"] * n_days
    for s in range(n_straddles_actual):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        ntry = ntry_offsets[s]
        xpry = xpry_offsets[s]
        if xpry >= 0:
            expiry_date = date_strings[start + xpry]
            for i in range(length):
                if ntry >= 0 and ntry <= i <= xpry:
                    expiry_strings[start + i] = expiry_date

    # Build model strings (per straddle)
    model_strings = ["-"] * n_days
    for s in range(n_straddles_actual):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        model = model_names[s]
        for i in range(length):
            model_strings[start + i] = model

    # Build output rows
    columns = ["asset", "straddle", "date", "vol", "hedge", "action",
               "model", "strike_vol", "strike", "expiry",
               "mv", "delta", "opnl", "hpnl", "pnl"]

    # Build parent_idx mapping each day back to its straddle
    parent_idx = np.zeros(n_days, dtype=np.int32)
    for s in range(n_straddles_actual):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        parent_idx[start:start+length] = s

    # Pre-compute strike_vol for each straddle
    strike_vol_by_straddle = []
    for s in range(n_straddles_actual):
        ntry = ntry_offsets[s]
        start = straddle_starts[s]
        if ntry >= 0 and not np.isnan(vol_rolled[start + ntry]):
            strike_vol_by_straddle.append(str(vol_rolled[start + ntry]))
        else:
            strike_vol_by_straddle.append("-")

    # Build rows
    rows = []
    for i in range(n_days):
        # Skip rows with invalid mv if valid_only is set
        if valid_only and np.isnan(mv[i]):
            continue

        s = parent_idx[i]

        # Format numeric values inline to avoid function call overhead
        vol_str = "-" if np.isnan(vol_array[i]) else str(vol_array[i])
        hedge_str = "-" if np.isnan(hedge_array[i]) else str(hedge_array[i])
        strike_str = "-" if np.isnan(strike_array[i]) else str(strike_array[i])
        mv_str = "-" if np.isnan(mv[i]) else str(mv[i])
        delta_str = "-" if np.isnan(delta[i]) else str(delta[i])
        opnl_str = "-" if np.isnan(opnl[i]) else str(opnl[i])
        hpnl_str = "-" if np.isnan(hpnl[i]) else str(hpnl[i])
        pnl_str = "-" if np.isnan(pnl[i]) else str(pnl[i])

        row = [
            expanded_assets[i],
            expanded_straddles[i],
            date_strings[i],
            vol_str,
            hedge_str,
            action_strings[i],
            model_strings[i],
            strike_vol_by_straddle[s] if valid_mask[i] else "-",
            strike_str,
            expiry_strings[i],
            mv_str,
            delta_str,
            opnl_str,
            hpnl_str,
            pnl_str,
        ]
        rows.append(row)

    return {"orientation": "row", "columns": columns, "rows": rows}
