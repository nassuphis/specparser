# -------------------------------------
# date/calendar utilities
# -------------------------------------
from datetime import date
import exchange_calendars as xcals

# Default calendar (NYSE)
_DEFAULT_CALENDAR = "XNYS"

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
    from datetime import timedelta
    last_day = last_day - timedelta(days=1)

    return first_day, last_day


def _all_days_in_month(first_day: date, last_day: date) -> list[date]:
    """Return all calendar days in the range."""
    from datetime import timedelta
    days = []
    current = first_day
    while current <= last_day:
        days.append(current)
        current += timedelta(days=1)
    return days


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
    sessions = set(cal.sessions_in_range(first_day, last_day))

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
    sessions = set(cal.sessions_in_range(first_day, last_day))

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

    p = argparse.ArgumentParser(description="Date/calendar utilities using exchange_calendars.")
    p.add_argument("month", nargs="?", help="Month in YYYY-MM format (e.g., 2010-01)")
    p.add_argument("--calendar", "-c", default=_DEFAULT_CALENDAR, help=f"Exchange calendar code (default: {_DEFAULT_CALENDAR})")
    p.add_argument("--good", "-g", action="store_true", help="Show trading days (good days)")
    p.add_argument("--holidays", "-o", action="store_true", help="Show all non-trading days (holidays + weekends)")
    p.add_argument("--weekday-holidays", "-w", action="store_true", help="Show only weekday holidays")
    p.add_argument("--list-calendars", "-l", action="store_true", help="List available exchange calendars")
    args = p.parse_args()

    if args.list_calendars:
        for name in calendars():
            print(name)
        return 0

    if args.month is None:
        p.error("month is required (e.g., 2010-01) unless --list-calendars is given")

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
