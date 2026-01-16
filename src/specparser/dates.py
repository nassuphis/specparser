# -------------------------------------
# date/calendar utilities
# -------------------------------------
from datetime import date, timedelta
import re
import exchange_calendars as xcals

# Default calendar (NYSE)
_DEFAULT_CALENDAR = "XNYS"

# Weekday codes: M=Monday, T=Thursday, W=Wednesday, F=Friday
# Note: T is Thursday (market convention), Tuesday is rarely used for expiries
_WEEKDAY_MAP = {
    "M": 0,   # Monday
    "T": 3,   # Thursday (market convention)
    "W": 2,   # Wednesday
    "F": 4,   # Friday
}

# Regex patterns for expiry descriptors
_BD_PATTERN = re.compile(r"^BD(\d+)$")           # BD1, BD5, BD15
_LBD_PATTERN = re.compile(r"^LBD(\d*)$")         # LBD, LBD1, LBD2
_MFBD_PATTERN = re.compile(r"^MFBD(\d+)$")       # MFBD1 (modified following)
_MPBD_PATTERN = re.compile(r"^MPBD(\d+)$")       # MPBD1 (modified preceding)
_WEEKDAY_PATTERN = re.compile(r"^([MTWF])(\d+)$")  # F3, W1, T2
_MF_WEEKDAY_PATTERN = re.compile(r"^MF([MTWF])(\d+)$")  # MFF3 (modified following Friday)

def _parse_month(month_str: str) -> tuple[date, date]:
    """
    Parse a month string like '2010-01' into (first_day, last_day) of that month.
    """
    parts = month_str.strip().split("-")
    year = int(parts[0])
    month = int(parts[1])

    first_day = date(year, month, 1)

    # Calculate last day of month
    if month == 12:
        last_day = date(year + 1, 1, 1)
    else:
        last_day = date(year, month + 1, 1)

    # Go back one day to get actual last day of the month
    last_day = last_day - timedelta(days=1)

    return first_day, last_day


def _all_days_in_month(first_day: date, last_day: date) -> list[date]:
    """Return all calendar days in the range."""
    days = []
    current = first_day
    while current <= last_day:
        days.append(current)
        current += timedelta(days=1)
    return days


def _get_sessions_in_month(year: int, month: int, calendar: str = _DEFAULT_CALENDAR) -> list[date]:
    """Return list of business day date objects for a given month."""
    first_day, last_day = _parse_month(f"{year:04d}-{month:02d}")
    cal = xcals.get_calendar(calendar)
    sessions = cal.sessions_in_range(first_day, last_day)
    return [d.date() for d in sessions]


def _nth_weekday_in_month(year: int, month: int, weekday: int, n: int) -> date:
    """
    Return the Nth occurrence of a weekday in a month.

    Args:
        year: Year
        month: Month (1-12)
        weekday: Weekday (0=Monday, 4=Friday)
        n: Which occurrence (1=first, 2=second, etc.)

    Returns:
        The date of the Nth weekday

    Raises:
        ValueError: If there's no Nth occurrence of that weekday
    """
    first_day = date(year, month, 1)
    # Find the first occurrence of the weekday
    days_until = (weekday - first_day.weekday()) % 7
    first_occurrence = first_day + timedelta(days=days_until)

    # Add (n-1) weeks to get the Nth occurrence
    result = first_occurrence + timedelta(weeks=n - 1)

    # Check it's still in the same month
    if result.month != month:
        raise ValueError(f"No {n}th occurrence of weekday {weekday} in {year}-{month:02d}")

    return result


def _nth_last_weekday_in_month(year: int, month: int, weekday: int, n: int) -> date:
    """
    Return the Nth-to-last occurrence of a weekday in a month.

    Args:
        year: Year
        month: Month (1-12)
        weekday: Weekday (0=Monday, 4=Friday)
        n: Which occurrence from end (1=last, 2=second-to-last, etc.)
    """
    _, last_day = _parse_month(f"{year:04d}-{month:02d}")
    # Find the last occurrence of the weekday
    days_back = (last_day.weekday() - weekday) % 7
    last_occurrence = last_day - timedelta(days=days_back)

    # Subtract (n-1) weeks to get the Nth-to-last occurrence
    result = last_occurrence - timedelta(weeks=n - 1)

    # Check it's still in the same month
    if result.month != month:
        raise ValueError(f"No {n}th-to-last occurrence of weekday {weekday} in {year}-{month:02d}")

    return result


def _modified_following(d: date, sessions: set[date]) -> date:
    """
    Apply modified following business day convention.
    If date is not a business day, roll forward to next business day.
    If rolling forward crosses into next month, roll backward instead.
    """
    if d in sessions:
        return d

    original_month = d.month
    # Try rolling forward
    forward = d
    while forward not in sessions:
        forward += timedelta(days=1)
        if forward.month != original_month:
            # Crossed month boundary, roll backward instead
            backward = d
            while backward not in sessions:
                backward -= timedelta(days=1)
            return backward
    return forward


def _modified_preceding(d: date, sessions: set[date]) -> date:
    """
    Apply modified preceding business day convention.
    If date is not a business day, roll backward to previous business day.
    If rolling backward crosses into previous month, roll forward instead.
    """
    if d in sessions:
        return d

    original_month = d.month
    # Try rolling backward
    backward = d
    while backward not in sessions:
        backward -= timedelta(days=1)
        if backward.month != original_month:
            # Crossed month boundary, roll forward instead
            forward = d
            while forward not in sessions:
                forward += timedelta(days=1)
            return forward
    return backward


def expiry(year: int | str, month: int | str, descriptor: str, calendar: str = _DEFAULT_CALENDAR) -> str:
    """
    Return expiry date for given year, month, and descriptor.

    Args:
        year: Year (e.g., 2024 or "2024")
        month: Month (1-12 or "1"-"12")
        descriptor: Expiry descriptor string
        calendar: Exchange calendar code (default: 'XNYS' for NYSE)

    Descriptor patterns:
        BD{n}     - Nth business day (BD1, BD5, BD15)
        LBD       - Last business day
        LBD{n}    - Nth-to-last business day (LBD1=last, LBD2=2nd-to-last)
        MFBD{n}   - Modified following Nth business day
        MPBD{n}   - Modified preceding Nth business day
        {W}{n}    - Nth weekday (F3=3rd Friday, W1=1st Wednesday, T2=2nd Thursday)
        MF{W}{n}  - Modified following Nth weekday (MFF3=3rd Friday, adjusted)

    Weekday codes: M=Monday, T=Thursday, W=Wednesday, F=Friday

    Returns:
        Date string in 'YYYY-MM-DD' format

    Raises:
        ValueError: If descriptor is invalid or date doesn't exist
    """
    year = int(year)
    month = int(month)
    descriptor = descriptor.strip().upper()
    sessions = _get_sessions_in_month(year, month, calendar)
    sessions_set = set(sessions)

    # BD{n} - Nth business day
    m = _BD_PATTERN.match(descriptor)
    if m:
        n = int(m.group(1))
        if n < 1 or n > len(sessions):
            raise ValueError(f"BD{n} out of range for {year}-{month:02d} (has {len(sessions)} business days)")
        return sessions[n - 1].strftime("%Y-%m-%d")

    # LBD or LBD{n} - Last or Nth-to-last business day
    m = _LBD_PATTERN.match(descriptor)
    if m:
        n_str = m.group(1)
        n = int(n_str) if n_str else 1
        if n < 1 or n > len(sessions):
            raise ValueError(f"LBD{n} out of range for {year}-{month:02d}")
        return sessions[-n].strftime("%Y-%m-%d")

    # MFBD{n} - Modified following Nth business day
    m = _MFBD_PATTERN.match(descriptor)
    if m:
        n = int(m.group(1))
        # Get Nth calendar day, then apply modified following
        first_day, _ = _parse_month(f"{year:04d}-{month:02d}")
        target = first_day + timedelta(days=n - 1)
        result = _modified_following(target, sessions_set)
        return result.strftime("%Y-%m-%d")

    # MPBD{n} - Modified preceding Nth business day
    m = _MPBD_PATTERN.match(descriptor)
    if m:
        n = int(m.group(1))
        first_day, _ = _parse_month(f"{year:04d}-{month:02d}")
        target = first_day + timedelta(days=n - 1)
        result = _modified_preceding(target, sessions_set)
        return result.strftime("%Y-%m-%d")

    # MF{W}{n} - Modified following Nth weekday
    m = _MF_WEEKDAY_PATTERN.match(descriptor)
    if m:
        weekday_code = m.group(1)
        n = int(m.group(2))
        weekday = _WEEKDAY_MAP[weekday_code]
        target = _nth_weekday_in_month(year, month, weekday, n)
        result = _modified_following(target, sessions_set)
        return result.strftime("%Y-%m-%d")

    # {W}{n} - Nth weekday (F3, W1, T2, etc.)
    m = _WEEKDAY_PATTERN.match(descriptor)
    if m:
        weekday_code = m.group(1)
        n = int(m.group(2))
        weekday = _WEEKDAY_MAP[weekday_code]
        result = _nth_weekday_in_month(year, month, weekday, n)
        return result.strftime("%Y-%m-%d")

    raise ValueError(f"Invalid expiry descriptor: {descriptor}")


def entry(year: int | str, month: int | str, descriptor: str, near_far: str, calendar: str = _DEFAULT_CALENDAR) -> str:
    """
    Return the entry date for a Near or Far contract expiring in the given month.

    The entry date is the expiry date 1 month (Near) or 2 months (Far) before
    the specified expiry month.

    Args:
        year: Year of expiry (e.g., 2024 or "2024")
        month: Month of expiry (1-12 or "1"-"12")
        descriptor: Expiry descriptor string (BD1, F3, LBD, etc.)
        near_far: "N" for Near (1 month before) or "F" for Far (2 months before)
        calendar: Exchange calendar code (default: 'XNYS' for NYSE)

    Returns:
        Entry date string in 'YYYY-MM-DD' format

    Raises:
        ValueError: If near_far is not 'N' or 'F'
    """
    year = int(year)
    month = int(month)
    near_far = near_far.strip().upper()

    if near_far == "N":
        offset = 1
    elif near_far == "F":
        offset = 2
    else:
        raise ValueError(f"near_far must be 'N' (near) or 'F' (far), got: {near_far}")

    # Calculate entry month (go back offset months)
    total_months = year * 12 + (month - 1) - offset
    target_year = total_months // 12
    target_month = (total_months % 12) + 1

    return expiry(target_year, target_month, descriptor, calendar)


def good_days(month_str: str, calendar: str = _DEFAULT_CALENDAR) -> list[str]:
    """
    Return trading days in a month.

    Args:
        month_str: Month in format 'YYYY-MM' (e.g., '2010-01')
        calendar: Exchange calendar code (default: 'XNYS' for NYSE)

    Returns:
        List of trading day dates as strings in 'YYYY-MM-DD' format
    """
    first_day, last_day = _parse_month(month_str)
    cal = xcals.get_calendar(calendar)

    sessions = cal.sessions_in_range(first_day, last_day)
    return [d.strftime("%Y-%m-%d") for d in sessions]


def holidays(month_str: str, calendar: str = _DEFAULT_CALENDAR) -> list[str]:
    """
    Return non-trading days (holidays + weekends) in a month.

    Args:
        month_str: Month in format 'YYYY-MM' (e.g., '2010-01')
        calendar: Exchange calendar code (default: 'XNYS' for NYSE)

    Returns:
        List of non-trading day dates as strings in 'YYYY-MM-DD' format
    """
    first_day, last_day = _parse_month(month_str)
    cal = xcals.get_calendar(calendar)

    all_days = _all_days_in_month(first_day, last_day)
    # Convert Timestamps to date objects for comparison
    sessions = set(d.date() for d in cal.sessions_in_range(first_day, last_day))

    non_trading = [d for d in all_days if d not in sessions]
    return [d.strftime("%Y-%m-%d") for d in non_trading]


def weekday_holidays(month_str: str, calendar: str = _DEFAULT_CALENDAR) -> list[str]:
    """
    Return only weekday holidays (excluding weekends) in a month.
    These are the 'true' exchange holidays.

    Args:
        month_str: Month in format 'YYYY-MM' (e.g., '2010-01')
        calendar: Exchange calendar code (default: 'XNYS' for NYSE)

    Returns:
        List of weekday holiday dates as strings in 'YYYY-MM-DD' format
    """
    first_day, last_day = _parse_month(month_str)
    cal = xcals.get_calendar(calendar)

    all_days = _all_days_in_month(first_day, last_day)
    # Convert Timestamps to date objects for comparison
    sessions = set(d.date() for d in cal.sessions_in_range(first_day, last_day))

    # Weekdays are Monday (0) through Friday (4)
    weekday_non_trading = [
        d for d in all_days
        if d not in sessions and d.weekday() < 5
    ]
    return [d.strftime("%Y-%m-%d") for d in weekday_non_trading]


def calendars() -> list[str]:
    """Return list of available exchange calendar codes."""
    return xcals.get_calendar_names()


# ============================================================
# CLI
# ============================================================

def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Date/calendar utilities using exchange_calendars.",
        epilog="""
Expiry descriptors:
  BD{n}     Nth business day (BD1, BD5, BD15)
  LBD       Last business day
  LBD{n}    Nth-to-last business day (LBD2 = 2nd-to-last)
  MFBD{n}   Modified following Nth calendar day
  MPBD{n}   Modified preceding Nth calendar day
  {W}{n}    Nth weekday (F3=3rd Friday, W1=1st Wed, T2=2nd Thu, M1=1st Mon)
  MF{W}{n}  Modified following Nth weekday (MFF3=3rd Friday, adjusted)
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("month", nargs="?", help="Month in YYYY-MM format (e.g., 2024-01)")
    p.add_argument("--calendar", "-c", default=_DEFAULT_CALENDAR, help=f"Exchange calendar code (default: {_DEFAULT_CALENDAR})")
    p.add_argument("--good", "-g", action="store_true", help="Show trading days (good days)")
    p.add_argument("--holidays", "-o", action="store_true", help="Show all non-trading days (holidays + weekends)")
    p.add_argument("--weekday-holidays", "-w", action="store_true", help="Show only weekday holidays")
    p.add_argument("--expiry", "-e", metavar="DESC", help="Calculate expiry date for descriptor (e.g., BD1, F3, LBD)")
    p.add_argument("--list-calendars", "-l", action="store_true", help="List available exchange calendars")
    args = p.parse_args()

    if args.list_calendars:
        for name in calendars():
            print(name)
        return 0

    if args.month is None:
        p.error("month is required (e.g., 2024-01) unless --list-calendars is given")

    # Parse year and month
    parts = args.month.strip().split("-")
    year = int(parts[0])
    month = int(parts[1])

    # Handle expiry calculation
    if args.expiry:
        try:
            result = expiry(year, month, args.expiry, args.calendar)
            print(result)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        return 0

    # Default to --good if no option specified
    if not (args.good or args.holidays or args.weekday_holidays):
        args.good = True

    if args.good:
        print(f"Trading days in {args.month} ({args.calendar}):")
        for d in good_days(args.month, args.calendar):
            print(f"  {d}")

    if args.holidays:
        print(f"Non-trading days in {args.month} ({args.calendar}):")
        for d in holidays(args.month, args.calendar):
            print(f"  {d}")

    if args.weekday_holidays:
        print(f"Weekday holidays in {args.month} ({args.calendar}):")
        for d in weekday_holidays(args.month, args.calendar):
            print(f"  {d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
