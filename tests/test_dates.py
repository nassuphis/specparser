"""Tests for specparser.dates module."""

import pytest

from specparser.dates import (
    good_days,
    holidays,
    weekday_holidays,
    calendars,
)


class TestGoodDays:
    """Tests for good_days (trading days) function."""

    def test_january_2024_nyse(self):
        """Test NYSE trading days in January 2024."""
        days = good_days("2024-01")
        # Should have ~21 trading days (excluding weekends and MLK Day)
        assert len(days) > 15
        assert len(days) < 25
        # First trading day should be Jan 2 (Jan 1 is holiday)
        assert days[0] == "2024-01-02"
        # All dates should be in correct format
        for d in days:
            assert d.startswith("2024-01-")

    def test_different_calendar(self):
        """Test with London Stock Exchange calendar."""
        nyse_days = good_days("2024-01", "XNYS")
        lse_days = good_days("2024-01", "XLON")
        # Different exchanges may have different trading days
        assert len(nyse_days) > 0
        assert len(lse_days) > 0

    def test_date_format(self):
        """Test that dates are in YYYY-MM-DD format."""
        days = good_days("2024-06")
        for d in days:
            parts = d.split("-")
            assert len(parts) == 3
            assert len(parts[0]) == 4  # Year
            assert len(parts[1]) == 2  # Month
            assert len(parts[2]) == 2  # Day


class TestHolidays:
    """Tests for holidays (non-trading days) function."""

    def test_includes_weekends(self):
        """Holidays should include all Saturdays and Sundays."""
        non_trading = holidays("2024-01")
        # January 2024 has 4 Saturdays and 4 Sundays = at least 8 non-trading days
        assert len(non_trading) >= 8

    def test_disjoint_from_trading_days(self):
        """Trading days and non-trading days should not overlap."""
        trading = set(good_days("2024-01"))
        non_trading = set(holidays("2024-01"))
        assert trading.isdisjoint(non_trading)

    def test_covers_all_days(self):
        """Trading + non-trading should cover all days in month."""
        trading = good_days("2024-01")
        non_trading = holidays("2024-01")
        # January has 31 days
        assert len(trading) + len(non_trading) == 31


class TestWeekdayHolidays:
    """Tests for weekday_holidays function."""

    def test_excludes_weekends(self):
        """Weekday holidays should not include Saturdays or Sundays."""
        from datetime import datetime
        wh = weekday_holidays("2024-01")
        for d in wh:
            dt = datetime.strptime(d, "%Y-%m-%d")
            # Monday=0, ..., Friday=4, Saturday=5, Sunday=6
            assert dt.weekday() < 5, f"{d} is a weekend day"

    def test_new_years_day_2024(self):
        """Jan 1, 2024 falls on Monday, should be a weekday holiday."""
        wh = weekday_holidays("2024-01")
        assert "2024-01-01" in wh

    def test_mlk_day_2024(self):
        """MLK Day 2024 (Jan 15) should be a weekday holiday."""
        wh = weekday_holidays("2024-01")
        assert "2024-01-15" in wh


class TestCalendars:
    """Tests for calendars function."""

    def test_returns_list(self):
        cals = calendars()
        assert isinstance(cals, list)

    def test_includes_common_exchanges(self):
        cals = calendars()
        # NYSE
        assert "XNYS" in cals
        # London
        assert "XLON" in cals
        # Tokyo
        assert "XTKS" in cals

    def test_non_empty(self):
        cals = calendars()
        assert len(cals) > 10  # Should have many exchanges
