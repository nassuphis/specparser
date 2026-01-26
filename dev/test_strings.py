# Tests for dev/strings.py - Fast string/byte operations
#
# These test the experimental high-performance string building utilities.
# When the code moves to production (src/specparser/amt/), update imports.

import numpy as np
import pytest

from strings import (
    # Conversion utilities
    u8m2S,
    u8m2s,
    u82S,
    u82s,
    s2u8,
    strs2u8mat,
    sep,
    make_u8mat,
    # Date reading
    read_1digit,
    read_2digits,
    read_4digits,
    get_uint8_ym,
    get_uint8_ymd,
    # Date writing
    make_ym,
    add_months_ym,
    add_months_ym_inplace,
    add_months2specs_inplace,
    # Calendar functions
    is_leap_year,
    last_day_of_month,
    days_between,
    add_months,
    make_ym_matrix,
    make_ymd_matrix,
    make_calendar_from_ranges,
    make_calendar_from_ranges_par,
    make_calendar_from_specs_par,
    # Cartesian product
    cartesian_product,
    cartesian_product_np,
    cartesian_product_par,
    # Unfurl functions
    unfurl,
    unfurl_by_spec,
    unfurl_by_spec_sep,
    unfurl_concat,
    unfurl_concat_sep,
    # Constants
    ASCII_0,
    ASCII_DASH,
    ASCII_SPACE,
)


# -------------------------------------
# Conversion Utilities
# -------------------------------------

class TestS2U8:
    def test_ascii(self):
        result = s2u8("hello")
        expected = np.array([104, 101, 108, 108, 111], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_empty(self):
        result = s2u8("")
        assert len(result) == 0
        assert result.dtype == np.uint8

    def test_digits(self):
        result = s2u8("2024")
        expected = np.array([50, 48, 50, 52], dtype=np.uint8)  # ASCII codes
        np.testing.assert_array_equal(result, expected)


class TestStrs2U8Mat:
    def test_basic(self):
        result = strs2u8mat(["AA", "BB", "CC"])
        assert result.shape == (3, 2)
        assert result.dtype == np.uint8
        # Check first row is "AA"
        assert result[0, 0] == ord("A")
        assert result[0, 1] == ord("A")

    def test_padding(self):
        # Different length strings get padded with spaces
        result = strs2u8mat(["A", "BB", "CCC"])
        assert result.shape == (3, 3)
        # "A" should be "A  " (padded)
        assert result[0, 0] == ord("A")
        assert result[0, 1] == ord(" ")
        assert result[0, 2] == ord(" ")
        # "CCC" should be "CCC" (no padding)
        assert result[2, 0] == ord("C")
        assert result[2, 1] == ord("C")
        assert result[2, 2] == ord("C")

    def test_explicit_width(self):
        result = strs2u8mat(["AB", "CD"], width=5)
        assert result.shape == (2, 5)
        # Check padding
        assert result[0, 2] == ord(" ")

    def test_empty_list(self):
        result = strs2u8mat([])
        assert result.shape == (0, 0)

    def test_single_string(self):
        result = strs2u8mat(["hello"])
        assert result.shape == (1, 5)


class TestMakeU8Mat:
    def test_basic(self):
        result = make_u8mat(3, 5)
        assert result.shape == (3, 5)
        assert result.dtype == np.uint8
        assert np.all(result == ASCII_SPACE)

    def test_custom_fill(self):
        result = make_u8mat(2, 4, fill=ASCII_0)
        assert result.shape == (2, 4)
        assert np.all(result == ASCII_0)


class TestU8m2S:
    def test_roundtrip(self):
        original = ["hello", "world", "test!"]
        u8mat = strs2u8mat(original)
        result = u8m2S(u8mat)
        # Convert back to python strings for comparison
        result_strs = [s.decode('utf-8') for s in result]
        assert result_strs == original

    def test_shape(self):
        u8mat = np.array([[65, 66], [67, 68]], dtype=np.uint8)  # AB, CD
        result = u8m2S(u8mat)
        assert result.shape == (2,)
        assert result[0] == b"AB"
        assert result[1] == b"CD"


class TestU8m2s:
    def test_returns_string_dtype(self):
        u8mat = strs2u8mat(["hello", "world"])
        result = u8m2s(u8mat)
        assert result.dtype == np.dtypes.StringDType()
        assert result[0] == "hello"
        assert result[1] == "world"


class TestU82S:
    def test_vector_to_bytes(self):
        vec = s2u8("hello")
        result = u82S(vec)
        assert result == b"hello"


class TestU82s:
    def test_vector_to_string(self):
        vec = s2u8("hello")
        result = u82s(vec)
        assert result.dtype == np.dtypes.StringDType()
        # Returns 0-d array, item() extracts the scalar
        assert result.item() == "hello"


# -------------------------------------
# Date Reading
# -------------------------------------

class TestReadDigits:
    def test_read_1digit(self):
        row = s2u8("5")
        assert read_1digit(row, 0) == 5

    def test_read_2digits(self):
        row = s2u8("42")
        assert read_2digits(row, 0) == 42

    def test_read_2digits_with_offset(self):
        row = s2u8("XX12YY")
        assert read_2digits(row, 2) == 12

    def test_read_4digits(self):
        row = s2u8("2024")
        assert read_4digits(row, 0) == 2024

    def test_read_4digits_with_offset(self):
        row = s2u8("|2024-06|")
        assert read_4digits(row, 1) == 2024


class TestGetUint8Ym:
    def test_basic(self):
        row = s2u8("2024-06")
        y, m = get_uint8_ym(row)
        assert y == 2024
        assert m == 6

    def test_january(self):
        row = s2u8("2001-01")
        y, m = get_uint8_ym(row)
        assert y == 2001
        assert m == 1

    def test_december(self):
        row = s2u8("2025-12")
        y, m = get_uint8_ym(row)
        assert y == 2025
        assert m == 12

    def test_longer_string(self):
        # Extra characters after YYYY-MM are ignored
        row = s2u8("2024-06-15extra")
        y, m = get_uint8_ym(row)
        assert y == 2024
        assert m == 6


class TestGetUint8Ymd:
    def test_basic(self):
        row = s2u8("2024-06-15")
        y, m, d = get_uint8_ymd(row)
        assert y == 2024
        assert m == 6
        assert d == 15

    def test_first_of_month(self):
        row = s2u8("2024-01-01")
        y, m, d = get_uint8_ymd(row)
        assert y == 2024
        assert m == 1
        assert d == 1

    def test_end_of_month(self):
        row = s2u8("2024-02-29")
        y, m, d = get_uint8_ymd(row)
        assert y == 2024
        assert m == 2
        assert d == 29

    def test_with_calendar_output(self):
        # Verify round-trip with make_calendar_from_ranges
        src = np.array([[2024, 3, 2024, 3]], dtype=np.int64)
        _, cal = make_calendar_from_ranges(src)

        # First day of March 2024
        y, m, d = get_uint8_ymd(cal[0])
        assert (y, m, d) == (2024, 3, 1)

        # Last day of March 2024
        y, m, d = get_uint8_ymd(cal[-1])
        assert (y, m, d) == (2024, 3, 31)


# -------------------------------------
# Date Writing
# -------------------------------------

class TestMakeYm:
    def test_basic(self):
        result = make_ym(2024, 6)
        assert len(result) == 7
        assert u82S(result) == b"2024-06"

    def test_january(self):
        result = make_ym(2001, 1)
        assert u82S(result) == b"2001-01"

    def test_december(self):
        result = make_ym(2025, 12)
        assert u82S(result) == b"2025-12"


class TestAddMonthsYm:
    def test_add_zero(self):
        spec = s2u8("2024-06")
        result = add_months_ym(spec, 0)
        assert u82S(result) == b"2024-06"

    def test_add_one(self):
        spec = s2u8("2024-06")
        result = add_months_ym(spec, 1)
        assert u82S(result) == b"2024-07"

    def test_cross_year(self):
        spec = s2u8("2024-11")
        result = add_months_ym(spec, 3)
        assert u82S(result) == b"2025-02"

    def test_add_12(self):
        spec = s2u8("2024-06")
        result = add_months_ym(spec, 12)
        assert u82S(result) == b"2025-06"


class TestAddMonthsYmInplace:
    def test_basic(self):
        row = np.empty(7, dtype=np.uint8)
        spec = s2u8("2024-06")
        add_months_ym_inplace(row, spec, 3)
        assert u82S(row) == b"2024-09"

    def test_cross_year(self):
        row = np.empty(7, dtype=np.uint8)
        spec = s2u8("2024-11")
        add_months_ym_inplace(row, spec, 3)
        assert u82S(row) == b"2025-02"


class TestAddMonths2SpecsInplace:
    def test_batch(self):
        targets = np.empty((3, 7), dtype=np.uint8)
        sources = strs2u8mat(["2024-01", "2024-06", "2024-11"])
        months = np.array([1, 6, 3], dtype=np.int64)

        add_months2specs_inplace(targets, sources, months)

        results = [u82S(targets[i]) for i in range(3)]
        assert results == [b"2024-02", b"2024-12", b"2025-02"]


# -------------------------------------
# Calendar Functions
# -------------------------------------

class TestIsLeapYear:
    def test_regular_leap_years(self):
        assert is_leap_year(2000) == True
        assert is_leap_year(2004) == True
        assert is_leap_year(2020) == True
        assert is_leap_year(2024) == True

    def test_non_leap_years(self):
        assert is_leap_year(2001) == False
        assert is_leap_year(2019) == False
        assert is_leap_year(2023) == False

    def test_century_rules(self):
        # Divisible by 100 but not 400 -> not leap
        assert is_leap_year(1900) == False
        assert is_leap_year(2100) == False
        # Divisible by 400 -> leap
        assert is_leap_year(2000) == True
        assert is_leap_year(1600) == True


class TestLastDayOfMonth:
    def test_31_day_months(self):
        for month in [1, 3, 5, 7, 8, 10, 12]:
            assert last_day_of_month(2024, month) == 31

    def test_30_day_months(self):
        for month in [4, 6, 9, 11]:
            assert last_day_of_month(2024, month) == 30

    def test_february_leap_year(self):
        assert last_day_of_month(2024, 2) == 29
        assert last_day_of_month(2000, 2) == 29

    def test_february_non_leap_year(self):
        assert last_day_of_month(2023, 2) == 28
        assert last_day_of_month(1900, 2) == 28


class TestDaysBetween:
    def test_single_month(self):
        # January 2024
        assert days_between(2024, 1, 2024, 1) == 31

    def test_two_months(self):
        # Jan + Feb 2024 (leap year)
        assert days_between(2024, 1, 2024, 2) == 31 + 29

    def test_full_year(self):
        # 2024 is a leap year -> 366 days
        assert days_between(2024, 1, 2024, 12) == 366
        # 2023 is not a leap year -> 365 days
        assert days_between(2023, 1, 2023, 12) == 365

    def test_cross_year(self):
        # Dec 2023 + Jan 2024
        assert days_between(2023, 12, 2024, 1) == 31 + 31


class TestAddMonths:
    def test_same_year(self):
        y, m = add_months(2024, 1, 5)
        assert (y, m) == (2024, 6)

    def test_cross_year(self):
        y, m = add_months(2024, 11, 3)
        assert (y, m) == (2025, 2)

    def test_add_zero(self):
        y, m = add_months(2024, 6, 0)
        assert (y, m) == (2024, 6)


class TestMakeYmMatrix:
    def test_single_month(self):
        result = make_ym_matrix(np.array([2024, 1, 2024, 1]))
        assert result.shape == (1, 7)  # 1 month, 7 chars "YYYY-MM"
        assert u82S(result[0]) == b"2024-01"

    def test_three_months(self):
        result = make_ym_matrix(np.array([2024, 1, 2024, 3]))
        assert result.shape == (3, 7)
        assert u82S(result[0]) == b"2024-01"
        assert u82S(result[1]) == b"2024-02"
        assert u82S(result[2]) == b"2024-03"

    def test_cross_year(self):
        result = make_ym_matrix(np.array([2024, 11, 2025, 2]))
        assert result.shape == (4, 7)
        assert u82S(result[0]) == b"2024-11"
        assert u82S(result[3]) == b"2025-02"

    def test_empty_range(self):
        # End before start -> empty
        result = make_ym_matrix(np.array([2024, 3, 2024, 1]))
        assert result.shape[0] == 0


class TestMakeYmdMatrix:
    def test_single_month(self):
        result = make_ymd_matrix(np.array([2024, 1, 2024, 1]))
        assert result.shape == (31, 10)  # 31 days, 10 chars "YYYY-MM-DD"
        # Check first and last dates
        first = u8m2S(result[:1])[0].decode()
        last = u8m2S(result[-1:])[0].decode()
        assert first == "2024-01-01"
        assert last == "2024-01-31"

    def test_february_leap(self):
        result = make_ymd_matrix(np.array([2024, 2, 2024, 2]))
        assert result.shape == (29, 10)  # Leap year February
        last = u8m2S(result[-1:])[0].decode()
        assert last == "2024-02-29"

    def test_empty_range(self):
        # End before start -> empty
        result = make_ymd_matrix(np.array([2024, 3, 2024, 1]))
        assert result.shape[0] == 0


class TestMakeCalendarFromRanges:
    def test_single_range(self):
        src = np.array([[2024, 1, 2024, 1]], dtype=np.int64)
        src_idx, cal = make_calendar_from_ranges(src)

        assert len(src_idx) == 31
        assert cal.shape == (31, 10)
        assert np.all(src_idx == 0)  # All from row 0

    def test_multiple_ranges(self):
        src = np.array([
            [2024, 1, 2024, 1],  # 31 days
            [2024, 2, 2024, 2],  # 29 days (leap)
        ], dtype=np.int64)
        src_idx, cal = make_calendar_from_ranges(src)

        assert len(src_idx) == 31 + 29
        assert cal.shape == (60, 10)
        # First 31 from row 0, next 29 from row 1
        assert np.all(src_idx[:31] == 0)
        assert np.all(src_idx[31:] == 1)

    def test_src_idx_tracking(self):
        src = np.array([
            [2024, 1, 2024, 1],  # 31 days
            [2024, 4, 2024, 4],  # 30 days
            [2024, 2, 2024, 2],  # 29 days
        ], dtype=np.int64)
        src_idx, cal = make_calendar_from_ranges(src)

        # Check src_idx correctly tracks origin
        assert src_idx[0] == 0   # First day from row 0
        assert src_idx[30] == 0  # Last day of Jan from row 0
        assert src_idx[31] == 1  # First day of Apr from row 1
        assert src_idx[60] == 1  # Last day of Apr from row 1
        assert src_idx[61] == 2  # First day of Feb from row 2


class TestMakeCalendarFromRangesPar:
    """Test parallel version matches sequential."""

    def test_matches_sequential_single(self):
        src = np.array([[2024, 1, 2024, 1]], dtype=np.int64)
        idx_seq, cal_seq = make_calendar_from_ranges(src)
        idx_par, cal_par = make_calendar_from_ranges_par(src)
        np.testing.assert_array_equal(idx_seq, idx_par)
        np.testing.assert_array_equal(cal_seq, cal_par)

    def test_matches_sequential_multiple(self):
        src = np.array([
            [2024, 1, 2024, 1],
            [2024, 2, 2024, 2],
            [2024, 3, 2024, 3],
        ], dtype=np.int64)
        idx_seq, cal_seq = make_calendar_from_ranges(src)
        idx_par, cal_par = make_calendar_from_ranges_par(src)
        np.testing.assert_array_equal(idx_seq, idx_par)
        np.testing.assert_array_equal(cal_seq, cal_par)

    def test_empty_input(self):
        src = np.empty((0, 4), dtype=np.int64)
        idx_par, cal_par = make_calendar_from_ranges_par(src)
        assert len(idx_par) == 0
        assert cal_par.shape == (0, 10)

    def test_large_scale(self):
        """Test at realistic scale."""
        src = np.array(
            [[y, m+1, y, m+1] for y in range(2020, 2025) for m in range(12)],
            dtype=np.int64
        )
        idx_seq, cal_seq = make_calendar_from_ranges(src)
        idx_par, cal_par = make_calendar_from_ranges_par(src)
        np.testing.assert_array_equal(idx_seq, idx_par)
        np.testing.assert_array_equal(cal_seq, cal_par)

    def test_multi_month_ranges(self):
        """Test ranges spanning multiple months."""
        src = np.array([
            [2024, 1, 2024, 3],  # Jan-Mar
            [2024, 6, 2024, 8],  # Jun-Aug
        ], dtype=np.int64)
        idx_seq, cal_seq = make_calendar_from_ranges(src)
        idx_par, cal_par = make_calendar_from_ranges_par(src)
        np.testing.assert_array_equal(idx_seq, idx_par)
        np.testing.assert_array_equal(cal_seq, cal_par)


class TestMakeCalendarFromSpecsPar:
    """Test parsing straddle specs and expanding to calendars."""

    def test_single_month(self):
        specs = strs2u8mat(["|2024-01|2024-01|"])
        src_idx, cal = make_calendar_from_specs_par(specs)

        assert len(src_idx) == 31  # Jan has 31 days
        assert cal.shape == (31, 10)
        assert np.all(src_idx == 0)

    def test_multiple_specs(self):
        specs = strs2u8mat([
            "|2024-01|2024-01|extra",  # 31 days
            "|2024-02|2024-02|extra",  # 29 days (leap)
        ])
        src_idx, cal = make_calendar_from_specs_par(specs)

        assert len(src_idx) == 31 + 29
        assert cal.shape == (60, 10)
        assert np.all(src_idx[:31] == 0)
        assert np.all(src_idx[31:] == 1)

    def test_multi_month_range(self):
        specs = strs2u8mat(["|2024-01|2024-03|"])  # Jan-Mar
        src_idx, cal = make_calendar_from_specs_par(specs)

        # Jan=31, Feb=29 (leap), Mar=31 = 91 days
        assert len(src_idx) == 91
        assert cal.shape == (91, 10)

    def test_matches_ranges_version(self):
        """Verify output matches make_calendar_from_ranges_par."""
        specs = strs2u8mat(["|2024-01|2024-03|"])
        ranges = np.array([[2024, 1, 2024, 3]], dtype=np.int64)

        idx_specs, cal_specs = make_calendar_from_specs_par(specs)
        idx_ranges, cal_ranges = make_calendar_from_ranges_par(ranges)

        np.testing.assert_array_equal(idx_specs, idx_ranges)
        np.testing.assert_array_equal(cal_specs, cal_ranges)

    def test_empty_input(self):
        specs = np.empty((0, 20), dtype=np.uint8)
        src_idx, cal = make_calendar_from_specs_par(specs)
        assert len(src_idx) == 0
        assert cal.shape == (0, 10)

    def test_cross_year_range(self):
        specs = strs2u8mat(["|2024-11|2025-02|"])  # Nov-Feb
        src_idx, cal = make_calendar_from_specs_par(specs)

        # Nov=30, Dec=31, Jan=31, Feb=28 (2025 not leap) = 120 days
        assert len(src_idx) == 120


# -------------------------------------
# Cartesian Product Functions
# -------------------------------------

class TestCartesianProduct:
    def test_two_matrices(self):
        v1 = strs2u8mat(["A", "B"])
        v2 = strs2u8mat(["X", "Y", "Z"])
        result = cartesian_product((v1, v2))

        assert result.shape == (6, 2)  # 2*3 rows, 1+1 cols

        strs = [s.decode('utf-8') for s in u8m2S(result)]
        assert strs == ["AX", "AY", "AZ", "BX", "BY", "BZ"]

    def test_different_widths(self):
        v1 = strs2u8mat(["AA", "BB"])
        v2 = strs2u8mat(["XXX", "YYY"])
        result = cartesian_product((v1, v2))

        assert result.shape == (4, 5)  # 2*2 rows, 2+3 cols

        strs = [s.decode('utf-8') for s in u8m2S(result)]
        assert strs == ["AAXXX", "AAYYY", "BBXXX", "BBYYY"]

    def test_single_element(self):
        v1 = strs2u8mat(["A"])
        v2 = strs2u8mat(["X"])
        result = cartesian_product((v1, v2))

        assert result.shape == (1, 2)

    def test_order_preserved(self):
        # Verify order is: for each v1, iterate all v2
        v1 = strs2u8mat(["1", "2", "3"])
        v2 = strs2u8mat(["a", "b"])
        result = cartesian_product((v1, v2))

        strs = [s.decode('utf-8') for s in u8m2S(result)]
        assert strs == ["1a", "1b", "2a", "2b", "3a", "3b"]

    def test_with_separator(self):
        v1 = strs2u8mat(["AA", "BB"])
        v2 = strs2u8mat(["X", "Y"])
        pipe = sep(b"|")  # (1,1) matrix
        result = cartesian_product((v1, pipe, v2))

        assert result.shape == (4, 4)  # 2+1+1 cols

        strs = [s.decode('utf-8') for s in u8m2S(result)]
        assert strs == ["AA|X", "AA|Y", "BB|X", "BB|Y"]

    def test_three_matrices(self):
        v1 = strs2u8mat(["A", "B"])
        v2 = strs2u8mat(["X", "Y"])
        v3 = strs2u8mat(["1", "2"])
        result = cartesian_product((v1, v2, v3))

        assert result.shape == (8, 3)  # 2*2*2 rows

        strs = [s.decode('utf-8') for s in u8m2S(result)]
        expected = [
            "AX1", "AX2", "AY1", "AY2",
            "BX1", "BX2", "BY1", "BY2",
        ]
        assert strs == expected

    def test_three_with_separators(self):
        v1 = strs2u8mat(["A", "B"])
        v2 = strs2u8mat(["X"])
        v3 = strs2u8mat(["1"])
        pipe = sep(b"|")
        result = cartesian_product((v1, pipe, v2, pipe, v3))

        assert result.shape == (2, 5)  # 1+1+1+1+1 cols

        strs = [s.decode('utf-8') for s in u8m2S(result)]
        assert strs == ["A|X|1", "B|X|1"]

    def test_realistic_straddle_format(self):
        # Simulate: asset | entry_ym | expiry_ym
        assets = strs2u8mat(["CL", "GC"])
        entry_yms = strs2u8mat(["2024-01", "2024-02"])
        expiry_yms = strs2u8mat(["2024-03"])
        pipe = sep(b"|")

        result = cartesian_product((assets, pipe, entry_yms, pipe, expiry_yms))

        # 2 assets * 2 entries * 1 expiry = 4 rows
        assert result.shape == (4, 2 + 1 + 7 + 1 + 7)  # 18 cols

        strs = [s.decode('utf-8') for s in u8m2S(result)]
        assert strs == [
            "CL|2024-01|2024-03",
            "CL|2024-02|2024-03",
            "GC|2024-01|2024-03",
            "GC|2024-02|2024-03",
        ]


class TestCartesianProductNp:
    """Test NumPy repeat/tile implementation matches Numba version."""

    def test_two_matrices(self):
        v1 = strs2u8mat(["AA", "BB"])
        v2 = strs2u8mat(["X", "Y", "Z"])
        result = cartesian_product_np((v1, v2))
        expected = cartesian_product((v1, v2))
        np.testing.assert_array_equal(result, expected)

    def test_three_matrices(self):
        v1 = strs2u8mat(["A", "B"])
        v2 = strs2u8mat(["X", "Y"])
        v3 = strs2u8mat(["1", "2"])
        result = cartesian_product_np((v1, v2, v3))
        expected = cartesian_product((v1, v2, v3))
        np.testing.assert_array_equal(result, expected)

    def test_with_separator(self):
        v1 = strs2u8mat(["AA", "BB"])
        v2 = strs2u8mat(["X", "Y"])
        pipe = sep(b"|")
        result = cartesian_product_np((v1, pipe, v2))
        expected = cartesian_product((v1, pipe, v2))
        np.testing.assert_array_equal(result, expected)

    def test_single_row_inputs(self):
        v1 = strs2u8mat(["ONLY"])
        v2 = strs2u8mat(["X", "Y", "Z"])
        result = cartesian_product_np((v1, v2))
        expected = cartesian_product((v1, v2))
        np.testing.assert_array_equal(result, expected)


class TestCartesianProductPar:
    """Test parallel stride-based implementation matches sequential odometer."""

    def test_two_matrices(self):
        v1 = strs2u8mat(["AA", "BB"])
        v2 = strs2u8mat(["X", "Y", "Z"])
        result = cartesian_product_par((v1, v2))
        expected = cartesian_product((v1, v2))
        np.testing.assert_array_equal(result, expected)

    def test_three_matrices(self):
        v1 = strs2u8mat(["A", "B"])
        v2 = strs2u8mat(["X", "Y"])
        v3 = strs2u8mat(["1", "2"])
        result = cartesian_product_par((v1, v2, v3))
        expected = cartesian_product((v1, v2, v3))
        np.testing.assert_array_equal(result, expected)

    def test_with_separator(self):
        v1 = strs2u8mat(["AA", "BB"])
        v2 = strs2u8mat(["X", "Y"])
        pipe = sep(b"|")
        result = cartesian_product_par((v1, pipe, v2))
        expected = cartesian_product((v1, pipe, v2))
        np.testing.assert_array_equal(result, expected)

    def test_single_row_inputs(self):
        v1 = strs2u8mat(["ONLY"])
        v2 = strs2u8mat(["X", "Y", "Z"])
        result = cartesian_product_par((v1, v2))
        expected = cartesian_product((v1, v2))
        np.testing.assert_array_equal(result, expected)

    def test_large_scale(self):
        """Verify parallel version produces correct output at scale."""
        assets = strs2u8mat([f"A{i:03d}" for i in range(50)])
        months = strs2u8mat([f"{m:02d}" for m in range(1, 13)])
        result = cartesian_product_par((assets, months))
        expected = cartesian_product((assets, months))
        np.testing.assert_array_equal(result, expected)


# -------------------------------------
# Unfurl Functions
# -------------------------------------

class TestUnfurl:
    def test_basic(self):
        mat = strs2u8mat(["AA", "BB", "CC"])
        counts = np.array([2, 1, 3], dtype=np.uint8)

        out, src_idx = unfurl(mat, counts)

        assert out.shape == (6, 2)
        assert len(src_idx) == 6

        strs = [s.decode('utf-8') for s in u8m2S(out)]
        assert strs == ["AA", "AA", "BB", "CC", "CC", "CC"]

        np.testing.assert_array_equal(src_idx, [0, 0, 1, 2, 2, 2])

    def test_zero_counts(self):
        mat = strs2u8mat(["AA", "BB", "CC"])
        counts = np.array([2, 0, 1], dtype=np.uint8)

        out, src_idx = unfurl(mat, counts)

        assert out.shape == (3, 2)
        strs = [s.decode('utf-8') for s in u8m2S(out)]
        assert strs == ["AA", "AA", "CC"]
        np.testing.assert_array_equal(src_idx, [0, 0, 2])

    def test_single_row(self):
        mat = strs2u8mat(["TEST"])
        counts = np.array([5], dtype=np.uint8)

        out, src_idx = unfurl(mat, counts)

        assert out.shape == (5, 4)
        assert all(s.decode('utf-8') == "TEST" for s in u8m2S(out))
        np.testing.assert_array_equal(src_idx, [0, 0, 0, 0, 0])

    def test_large_total_no_overflow(self):
        """Verify sum > 255 works correctly (np.sum promotes to uint64)."""
        mat = strs2u8mat(["A", "B", "C"])
        # 3 rows with counts that sum to 754 (> 255)
        counts = np.array([255, 255, 244], dtype=np.uint8)

        out, src_idx = unfurl(mat, counts)

        assert out.shape == (754, 1)
        assert len(src_idx) == 754
        # Check distribution: 255 A's, 255 B's, 244 C's
        assert np.sum(src_idx == 0) == 255
        assert np.sum(src_idx == 1) == 255
        assert np.sum(src_idx == 2) == 244


class TestUnfurlBySpec:
    def test_single_byte_items(self):
        """Test with 1-byte data items (like NTRC codes N/F)."""
        mat = strs2u8mat(["STR1", "STR2"])
        # spec: [count, item0, item1, item2, item3]
        spec = np.array([
            [2, ord('N'), ord('F'), 0, 0],           # 2 expansions: N, F
            [4, ord('N'), ord('N'), ord('F'), ord('F')],  # 4 expansions
        ], dtype=np.uint8)

        out, src_idx = unfurl_by_spec(mat, spec)

        assert out.shape == (6, 5)  # 4 (mat width) + 1 (item width)

        strs = [s.decode('utf-8') for s in u8m2S(out)]
        assert strs == ["STR1N", "STR1F", "STR2N", "STR2N", "STR2F", "STR2F"]

        np.testing.assert_array_equal(src_idx, [0, 0, 1, 1, 1, 1])

    def test_two_byte_items(self):
        """Test with 2-byte data items."""
        mat = strs2u8mat(["AA", "BB"])
        # spec: [count, b0, b1, b0, b1, b0, b1] for max 3 items of 2 bytes each
        spec = np.array([
            [2, ord('N'), ord('N'), ord('F'), ord('F'), 0, 0],
            [3, ord('X'), ord('X'), ord('Y'), ord('Y'), ord('Z'), ord('Z')],
        ], dtype=np.uint8)

        out, src_idx = unfurl_by_spec(mat, spec)

        assert out.shape == (5, 4)  # 2 (mat width) + 2 (item width)

        strs = [s.decode('utf-8') for s in u8m2S(out)]
        assert strs == ["AANN", "AAFF", "BBXX", "BBYY", "BBZZ"]

    def test_varying_counts(self):
        """Test with different counts per row."""
        mat = strs2u8mat(["A", "B", "C"])
        spec = np.array([
            [1, ord('X'), 0, 0],
            [3, ord('1'), ord('2'), ord('3')],
            [2, ord('P'), ord('Q'), 0],
        ], dtype=np.uint8)

        out, src_idx = unfurl_by_spec(mat, spec)

        assert out.shape == (6, 2)  # 1 + 1
        strs = [s.decode('utf-8') for s in u8m2S(out)]
        assert strs == ["AX", "B1", "B2", "B3", "CP", "CQ"]

    def test_empty_spec(self):
        """Test with all zero counts."""
        mat = strs2u8mat(["AA", "BB"])
        spec = np.array([
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.uint8)

        out, src_idx = unfurl_by_spec(mat, spec)

        assert out.shape[0] == 0


class TestUnfurlBySpecSep:
    def test_with_separator(self):
        mat = strs2u8mat(["STR1", "STR2"])
        spec = np.array([
            [2, ord('N'), ord('F')],
            [1, ord('X'), 0],
        ], dtype=np.uint8)
        sep_byte = np.uint8(ord('|'))

        out, src_idx = unfurl_by_spec_sep(mat, spec, sep_byte)

        assert out.shape == (3, 6)  # 4 + 1 + 1

        strs = [s.decode('utf-8') for s in u8m2S(out)]
        assert strs == ["STR1|N", "STR1|F", "STR2|X"]


class TestUnfurlConcat:
    def test_basic(self):
        mat = strs2u8mat(["AA", "BB"])
        values = strs2u8mat(["X", "Y", "Z"])
        src_idx = np.array([0, 0, 1], dtype=np.int64)

        out = unfurl_concat(mat, values, src_idx)

        assert out.shape == (3, 3)  # 2 + 1
        strs = [s.decode('utf-8') for s in u8m2S(out)]
        assert strs == ["AAX", "AAY", "BBZ"]

    def test_with_calendar(self):
        """Test using calendar output."""
        assets = strs2u8mat(["CL", "GC"])
        ranges = np.array([
            [2024, 1, 2024, 1],  # Jan 2024: 31 days
            [2024, 2, 2024, 2],  # Feb 2024: 29 days (leap)
        ], dtype=np.int64)

        cal_idx, cal_dates = make_calendar_from_ranges(ranges)
        result = unfurl_concat(assets, cal_dates, cal_idx)

        assert result.shape == (60, 12)  # 2 + 10

        # Check first and last
        strs = u8m2S(result).astype("U12")
        assert strs[0] == "CL2024-01-01"
        assert strs[30] == "CL2024-01-31"
        assert strs[31] == "GC2024-02-01"
        assert strs[-1] == "GC2024-02-29"


class TestUnfurlConcatSep:
    def test_with_separator(self):
        mat = strs2u8mat(["AA", "BB"])
        values = strs2u8mat(["XX", "YY", "ZZ"])
        src_idx = np.array([0, 1, 1], dtype=np.int64)
        sep_byte = np.uint8(ord('|'))

        out = unfurl_concat_sep(mat, values, src_idx, sep_byte)

        assert out.shape == (3, 5)  # 2 + 1 + 2
        strs = [s.decode('utf-8') for s in u8m2S(out)]
        assert strs == ["AA|XX", "BB|YY", "BB|ZZ"]


# -------------------------------------
# Integration Tests
# -------------------------------------

class TestIntegration:
    def test_calendar_to_strings(self):
        """Verify calendar output can be converted to date strings."""
        src = np.array([[2024, 6, 2024, 6]], dtype=np.int64)
        src_idx, cal = make_calendar_from_ranges(src)

        # Convert to strings
        date_strs = u8m2S(cal).astype("U10")

        assert len(date_strs) == 30  # June has 30 days
        assert date_strs[0] == "2024-06-01"
        assert date_strs[-1] == "2024-06-30"

    def test_cartesian_with_calendar(self):
        """Test combining cartesian product with calendar expansion."""
        # Create asset-yearmonth combinations
        assets = strs2u8mat(["CL", "GC"])
        yms = strs2u8mat(["2024-01", "2024-02"])
        pipe = sep(b"|")

        combinations = cartesian_product((assets, pipe, yms))
        combo_strs = [s.decode('utf-8') for s in u8m2S(combinations)]

        assert len(combo_strs) == 4
        assert "CL|2024-01" in combo_strs
        assert "GC|2024-02" in combo_strs

    def test_large_scale(self):
        """Test with realistic data sizes."""
        # 50 assets, 120 yearmonths (10 years)
        assets = strs2u8mat([f"Asset{i:03d}" for i in range(50)])
        yms = strs2u8mat([f"{y}-{m:02d}" for y in range(2020, 2030) for m in range(1, 13)])

        result = cartesian_product((assets, yms))

        assert result.shape[0] == 50 * 120  # 6000 rows
        assert result.shape[1] == 8 + 7  # "Asset000" + "2020-01"

    def test_unfurl_pipeline(self):
        """Test full unfurl pipeline: assets -> calendar -> concatenate."""
        # Setup
        assets = strs2u8mat(["ASSET1", "ASSET2", "ASSET3"])
        ranges = np.array([
            [2024, 1, 2024, 1],  # 31 days
            [2024, 6, 2024, 6],  # 30 days
            [2024, 2, 2024, 2],  # 29 days (leap)
        ], dtype=np.int64)

        # Generate calendar
        cal_idx, cal_dates = make_calendar_from_ranges(ranges)

        # Concatenate with separator
        result = unfurl_concat_sep(assets, cal_dates, cal_idx, np.uint8(ord('|')))

        # Verify
        assert result.shape[0] == 31 + 30 + 29  # 90 rows
        assert result.shape[1] == 6 + 1 + 10  # 17 cols

        strs = u8m2S(result).astype("U17")
        assert strs[0] == "ASSET1|2024-01-01"
        assert strs[30] == "ASSET1|2024-01-31"
        assert strs[31] == "ASSET2|2024-06-01"
        assert strs[-1] == "ASSET3|2024-02-29"
