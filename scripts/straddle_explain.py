#!/usr/bin/env python3
"""
Straddle explanation script.

Explains the selection of ntry and xpry dates for a straddle,
useful for debugging and validating straddle configurations.
"""
import argparse
import sys
from pathlib import Path

# Add src to path so we can import specparser
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from specparser.amt import loader, schedules
from specparser.amt.tickers import (
    get_straddle_actions,
    asset_straddle_tickers,
    _anchor_day,
    _add_calendar_days,
)


def count_good_days_in_month(rows: list[list], columns: list[str], year: int, month: int) -> int:
    """Count good days in a given month."""
    date_idx = columns.index("date") if "date" in columns else None
    vol_idx = columns.index("vol") if "vol" in columns else None

    if date_idx is None or vol_idx is None:
        return 0

    # Find hedge column indices
    hedge_indices = []
    for i, col in enumerate(columns):
        if col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
            hedge_indices.append(i)

    if not hedge_indices:
        return 0

    month_start = f"{year}-{month:02d}-01"
    import calendar
    _, num_days = calendar.monthrange(year, month)
    month_end = f"{year}-{month:02d}-{num_days:02d}"

    count = 0
    for row in rows:
        row_date = row[date_idx]
        if row_date < month_start:
            continue
        if row_date > month_end:
            break

        # Check if good day
        if row[vol_idx] == "none":
            continue
        if not all(row[idx] != "none" for idx in hedge_indices):
            continue
        count += 1

    return count


def count_good_days_in_range(rows: list[list], columns: list[str], ntry_idx: int, xpry_idx: int) -> int:
    """Count good days from ntry to xpry (inclusive)."""
    vol_idx = columns.index("vol") if "vol" in columns else None

    if vol_idx is None:
        return 0

    # Find hedge column indices
    hedge_indices = []
    for i, col in enumerate(columns):
        if col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
            hedge_indices.append(i)

    if not hedge_indices:
        return 0

    count = 0
    for i in range(ntry_idx, xpry_idx + 1):
        row = rows[i]
        if row[vol_idx] == "none":
            continue
        if not all(row[idx] != "none" for idx in hedge_indices):
            continue
        count += 1

    return count


def explain_straddle(
    amt_path: str,
    prices_path: str,
    chain_path: str,
    underlying: str,
    year: int,
    month: int,
    straddle_idx: int,
    just_check: bool = False
) -> bool:
    """Explain straddle entry/expiry selection. Returns True if both ntry and xpry found."""

    # Get straddle days table
    try:
        table = get_straddle_actions(underlying, year, month, straddle_idx, amt_path, chain_path, prices_path)
    except ValueError as e:
        if just_check:
            print("FALSE")
            return False
        print(f"Error: {e}")
        return False

    columns = table["columns"]
    rows = table["rows"]

    if not rows:
        if just_check:
            print("FALSE")
            return False
        print("Error: No data rows returned")
        return False

    # Get straddle string from first row
    straddle_idx_col = columns.index("straddle") if "straddle" in columns else None
    if straddle_idx_col is None:
        if just_check:
            print("FALSE")
            return False
        print("Error: No straddle column")
        return False

    straddle = rows[0][straddle_idx_col]

    # Parse straddle components
    ntry_year = schedules.ntry(straddle)
    ntry_month = schedules.ntrm(straddle)
    xpry_year = schedules.xpry(straddle)
    xpry_month = schedules.xprm(straddle)
    xprc = schedules.xprc(straddle)
    xprv = schedules.xprv(straddle)
    ntrv = schedules.ntrv(straddle)

    # Find ntry and xpry indices
    action_idx = columns.index("action") if "action" in columns else None
    date_idx = columns.index("date") if "date" in columns else None
    vol_idx = columns.index("vol") if "vol" in columns else None

    ntry_row_idx = None
    xpry_row_idx = None

    if action_idx is not None:
        for i, row in enumerate(rows):
            if row[action_idx] == "ntry":
                ntry_row_idx = i
            elif row[action_idx] == "xpry":
                xpry_row_idx = i

    # Just check mode
    if just_check:
        if ntry_row_idx is not None and xpry_row_idx is not None:
            print("TRUE")
            return True
        else:
            print("FALSE")
            return False

    # Full explanation
    print("=" * 60)
    print("STRADDLE EXPLANATION")
    print("=" * 60)
    print()

    print(f"Underlying: {underlying}")
    print(f"Straddle:   {straddle}")
    print()

    # Get ticker information
    try:
        ticker_table = asset_straddle_tickers(underlying, year, month, straddle_idx, amt_path, chain_path)
        ticker_cols = ticker_table["columns"]
        ticker_rows = ticker_table["rows"]

        param_idx_t = ticker_cols.index("param") if "param" in ticker_cols else None
        source_idx_t = ticker_cols.index("source") if "source" in ticker_cols else None
        ticker_idx_t = ticker_cols.index("ticker") if "ticker" in ticker_cols else None
        field_idx_t = ticker_cols.index("field") if "field" in ticker_cols else None

        if all(idx is not None for idx in [param_idx_t, source_idx_t, ticker_idx_t, field_idx_t]):
            print("TICKERS:")
            for trow in ticker_rows:
                param = trow[param_idx_t]
                source = trow[source_idx_t]
                ticker = trow[ticker_idx_t]
                field = trow[field_idx_t]
                # Format as source:ticker:field
                specstr = f"{source}:{ticker}:{field}"
                print(f"  {param}: {specstr}")
            print()
    except Exception:
        # If we can't get tickers, just skip this section
        pass

    print("STRADDLE PARAMETERS:")
    print(f"  Entry month:  {ntry_year}-{ntry_month:02d}")
    print(f"  Expiry month: {xpry_year}-{xpry_month:02d}")
    print(f"  Expiry code:  {xprc} (", end="")
    if xprc == "F":
        print("Friday", end="")
    elif xprc == "R":
        print("Thursday", end="")
    elif xprc == "W":
        print("Wednesday", end="")
    elif xprc == "BD":
        print("Business Day", end="")
    elif xprc == "OVERRIDE":
        print("Override lookup", end="")
    else:
        print("Unknown", end="")
    print(")")
    if xprc == "OVERRIDE":
        print(f"  Expiry value: {xprv} (ignored for OVERRIDE)")
    else:
        print(f"  Expiry value: {xprv} (Nth occurrence)")
    print(f"  Entry offset: {ntrv} (calendar days from anchor)")
    print()

    # Count good days
    good_days_entry = count_good_days_in_month(rows, columns, ntry_year, ntry_month)
    good_days_expiry = count_good_days_in_month(rows, columns, xpry_year, xpry_month)

    print("GOOD DAYS:")
    print(f"  In entry month ({ntry_year}-{ntry_month:02d}):  {good_days_entry}")
    print(f"  In expiry month ({xpry_year}-{xpry_month:02d}): {good_days_expiry}")
    print()

    # Explain anchor calculation
    print("ANCHOR CALCULATION:")
    entry_anchor = _anchor_day(xprc, xprv, ntry_year, ntry_month, underlying)
    expiry_anchor = _anchor_day(xprc, xprv, xpry_year, xpry_month, underlying)

    def ordinal(n):
        """Return ordinal string for number (1st, 2nd, 3rd, etc.)"""
        if 11 <= n % 100 <= 13:
            return f"{n}th"
        return f"{n}{['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]}"

    xprv_ord = ordinal(int(xprv)) if xprv and xprv.isdigit() else xprv

    if xprc == "OVERRIDE":
        print(f"  Entry anchor:  Override lookup for {underlying} {ntry_year}-{ntry_month:02d} = {entry_anchor or 'NOT FOUND'}")
        print(f"  Expiry anchor: Override lookup for {underlying} {xpry_year}-{xpry_month:02d} = {expiry_anchor or 'NOT FOUND'}")
    elif xprc == "BD":
        print(f"  Entry anchor:  {xprv_ord} business day of {ntry_year}-{ntry_month:02d} = {entry_anchor or 'NOT FOUND'}")
        print(f"  Expiry anchor: {xprv_ord} business day of {xpry_year}-{xpry_month:02d} = {expiry_anchor or 'NOT FOUND'}")
    else:
        weekday_name = {"F": "Friday", "R": "Thursday", "W": "Wednesday"}.get(xprc, xprc)
        print(f"  Entry anchor:  {xprv_ord} {weekday_name} of {ntry_year}-{ntry_month:02d} = {entry_anchor or 'NOT FOUND'}")
        print(f"  Expiry anchor: {xprv_ord} {weekday_name} of {xpry_year}-{xpry_month:02d} = {expiry_anchor or 'NOT FOUND'}")
    print()

    # Explain entry selection
    print("ENTRY DATE SELECTION:")
    if entry_anchor:
        ntrv_int = int(ntrv) if ntrv else 0
        target_date = _add_calendar_days(entry_anchor, ntrv_int)
        print(f"  Anchor date:     {entry_anchor}")
        print(f"  + {ntrv_int} calendar days = {target_date}")

        import calendar
        _, entry_num_days = calendar.monthrange(ntry_year, ntry_month)
        entry_month_end = f"{ntry_year}-{ntry_month:02d}-{entry_num_days:02d}"

        if target_date > entry_month_end:
            print(f"  Target {target_date} > month end {entry_month_end}")
            print(f"  -> Using last good day of entry month")
        else:
            print(f"  -> First good day at or after {target_date}")

        if ntry_row_idx is not None:
            ntry_date = rows[ntry_row_idx][date_idx]
            ntry_vol = rows[ntry_row_idx][vol_idx] if vol_idx else "-"
            print(f"  ENTRY DATE:      {ntry_date}")
            print(f"  Entry vol:       {ntry_vol}")

            # Find hedge values
            for i, col in enumerate(columns):
                if col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
                    print(f"  Entry {col}:     {rows[ntry_row_idx][i]}")
        else:
            print(f"  ENTRY DATE:      NOT FOUND (no good day at or after {target_date})")
    else:
        print(f"  Anchor not found - {xprc}/{xprv} doesn't exist in {ntry_year}-{ntry_month:02d}")
        print(f"  ENTRY DATE:      NOT FOUND")
    print()

    # Explain expiry selection
    print("EXPIRY DATE SELECTION:")
    if expiry_anchor:
        print(f"  Anchor date:     {expiry_anchor}")
        print(f"  -> First good day at or after anchor")

        if xpry_row_idx is not None:
            xpry_date = rows[xpry_row_idx][date_idx]
            print(f"  EXPIRY DATE:     {xpry_date}")
        else:
            print(f"  EXPIRY DATE:     NOT FOUND (no good day at or after anchor)")
    else:
        print(f"  Anchor not found - {xprc}/{xprv} doesn't exist in {xpry_year}-{xpry_month:02d}")
        print(f"  EXPIRY DATE:     NOT FOUND")
    print()

    # Summary
    print("SUMMARY:")
    if ntry_row_idx is not None and xpry_row_idx is not None:
        ntry_date = rows[ntry_row_idx][date_idx]
        xpry_date = rows[xpry_row_idx][date_idx]
        good_days_range = count_good_days_in_range(rows, columns, ntry_row_idx, xpry_row_idx)

        print(f"  Entry:  {ntry_date}")
        print(f"  Expiry: {xpry_date}")
        print(f"  Good days from entry to expiry: {good_days_range}")
        print(f"  Total calendar days: {xpry_row_idx - ntry_row_idx + 1}")
        print()
        print("  STATUS: OK")
        return True
    else:
        missing = []
        if ntry_row_idx is None:
            missing.append("entry")
        if xpry_row_idx is None:
            missing.append("expiry")
        print(f"  Missing: {', '.join(missing)}")
        print()
        print("  STATUS: INCOMPLETE")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Explain straddle entry/expiry date selection"
    )
    parser.add_argument("underlying", help="Asset underlying name")
    parser.add_argument("year", type=int, help="Expiry year")
    parser.add_argument("month", type=int, help="Expiry month")
    parser.add_argument("straddle_idx", type=int, help="Straddle index (0-based)")

    parser.add_argument("--amt", default="data/amt.yml",
                        help="Path to AMT YAML file (default: data/amt.yml)")
    parser.add_argument("--chain", default="data/futs.csv",
                        help="Path to futures chain CSV (default: data/futs.csv)")
    parser.add_argument("--prices", default="data/prices.parquet",
                        help="Path to prices parquet file (default: data/prices.parquet)")
    parser.add_argument("--just-check", action="store_true",
                        help="Only output TRUE or FALSE based on whether ntry and xpry were found")

    args = parser.parse_args()

    success = explain_straddle(
        args.amt,
        args.prices,
        args.chain,
        args.underlying,
        args.year,
        args.month,
        args.straddle_idx,
        args.just_check
    )

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
