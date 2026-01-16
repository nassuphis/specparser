"""Tests for specparser.dates module."""

import pytest

from specparser.dates import (
    good_days,
    holidays,
    weekday_holidays,
    calendars,
    expiry,
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


class TestExpiry:
    """Tests for expiry function."""

    # --- BD (Business Day) patterns ---

    def test_bd1_first_business_day(self):
        """BD1 should return first business day of month."""
        # Jan 1, 2024 is holiday, so BD1 is Jan 2
        assert expiry(2024, 1, "BD1") == "2024-01-02"

    def test_bd5_fifth_business_day(self):
        """BD5 should return 5th business day."""
        assert expiry(2024, 1, "BD5") == "2024-01-08"

    def test_bd15_fifteenth_business_day(self):
        """BD15 should return 15th business day."""
        assert expiry(2024, 1, "BD15") == "2024-01-23"

    def test_bd_case_insensitive(self):
        """Descriptor should be case insensitive."""
        assert expiry(2024, 1, "bd1") == expiry(2024, 1, "BD1")

    def test_bd_out_of_range(self):
        """BD with n > business days in month should raise."""
        with pytest.raises(ValueError, match="out of range"):
            expiry(2024, 1, "BD50")

    # --- LBD (Last Business Day) patterns ---

    def test_lbd_last_business_day(self):
        """LBD should return last business day of month."""
        assert expiry(2024, 1, "LBD") == "2024-01-31"

    def test_lbd1_same_as_lbd(self):
        """LBD1 should be same as LBD."""
        assert expiry(2024, 1, "LBD1") == expiry(2024, 1, "LBD")

    def test_lbd2_second_to_last(self):
        """LBD2 should return 2nd-to-last business day."""
        assert expiry(2024, 1, "LBD2") == "2024-01-30"

    def test_lbd3_third_to_last(self):
        """LBD3 should return 3rd-to-last business day."""
        assert expiry(2024, 1, "LBD3") == "2024-01-29"

    # --- Weekday patterns (F, W, T, M) ---

    def test_f3_third_friday(self):
        """F3 should return 3rd Friday of month."""
        # Jan 2024: Fridays are 5, 12, 19, 26
        assert expiry(2024, 1, "F3") == "2024-01-19"

    def test_f1_first_friday(self):
        """F1 should return 1st Friday of month."""
        assert expiry(2024, 1, "F1") == "2024-01-05"

    def test_w1_first_wednesday(self):
        """W1 should return 1st Wednesday of month."""
        # Jan 2024: First Wednesday is Jan 3
        assert expiry(2024, 1, "W1") == "2024-01-03"

    def test_t2_second_thursday(self):
        """T2 should return 2nd Thursday of month."""
        # Jan 2024: Thursdays are 4, 11, 18, 25
        assert expiry(2024, 1, "T2") == "2024-01-11"

    def test_m1_first_monday(self):
        """M1 should return 1st Monday of month."""
        # Jan 2024: First Monday is Jan 1 (but it's a holiday)
        assert expiry(2024, 1, "M1") == "2024-01-01"

    def test_weekday_out_of_range(self):
        """Weekday pattern with n too large should raise."""
        with pytest.raises(ValueError, match="No 6th occurrence"):
            expiry(2024, 1, "F6")  # No 6th Friday in any month

    # --- MF (Modified Following) weekday patterns ---

    def test_mff3_third_friday_modified(self):
        """MFF3 when 3rd Friday is trading day should return same."""
        # Jan 19, 2024 is a trading day
        assert expiry(2024, 1, "MFF3") == "2024-01-19"

    def test_mf_rolls_forward_on_holiday(self):
        """MF should roll forward when date is a holiday."""
        # Good Friday 2024 is March 29 (F5 in March, 5th Friday)
        # March 29, 2024 is Good Friday (NYSE closed)
        result = expiry(2024, 3, "MFF5")
        # Should roll to next business day (April 1, 2024)
        # But since April is next month, should roll backward to March 28
        assert result == "2024-03-28"

    # --- MFBD (Modified Following Business Day) patterns ---

    def test_mfbd1_first_calendar_day_adjusted(self):
        """MFBD1 on holiday should roll to next business day."""
        # Jan 1, 2024 is holiday, should roll to Jan 2
        assert expiry(2024, 1, "MFBD1") == "2024-01-02"

    def test_mfbd_on_weekend_rolls_forward(self):
        """MFBD on weekend should roll to Monday."""
        # Feb 3, 2024 is Saturday, should roll to Feb 5 (Monday)
        assert expiry(2024, 2, "MFBD3") == "2024-02-05"

    # --- MPBD (Modified Preceding Business Day) patterns ---

    def test_mpbd_on_weekend_rolls_backward(self):
        """MPBD on weekend should roll backward to Friday."""
        # Feb 3, 2024 is Saturday, should roll to Feb 2 (Friday)
        assert expiry(2024, 2, "MPBD3") == "2024-02-02"

    # --- Edge cases ---

    def test_invalid_descriptor_raises(self):
        """Invalid descriptor should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid expiry descriptor"):
            expiry(2024, 1, "INVALID")

    def test_different_calendar(self):
        """Expiry should work with different calendars."""
        # London calendar
        result = expiry(2024, 1, "BD1", calendar="XLON")
        assert result == "2024-01-02"  # Same as NYSE for this date

    def test_february_leap_year(self):
        """Test expiry in February of leap year."""
        # 2024 is a leap year, Feb has 29 days
        lbd = expiry(2024, 2, "LBD")
        assert lbd == "2024-02-29"

    def test_december_year_end(self):
        """Test expiry at year end."""
        lbd = expiry(2024, 12, "LBD")
        assert lbd == "2024-12-31"
