"""Tests for specparser.files module."""

import pytest
import tempfile
import os

from specparser.files import (
    all_lines,
    get_line,
    get_lines,
    get_random_line,
    get_lines_paired,
    set_seed,
    clear_cache,
)


@pytest.fixture
def temp_file():
    """Create a temporary file with test content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("line0\n")
        f.write("line1\n")
        f.write("line2\n")
        f.write("line3\n")
        f.write("line4\n")
        path = f.name
    yield path
    os.unlink(path)
    clear_cache()


@pytest.fixture
def empty_file():
    """Create an empty temporary file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)
    clear_cache()


class TestAllLines:
    """Tests for all_lines function."""

    def test_read_all_lines(self, temp_file):
        lines = all_lines(temp_file)
        assert lines == ["line0", "line1", "line2", "line3", "line4"]

    def test_caching(self, temp_file):
        lines1 = all_lines(temp_file)
        lines2 = all_lines(temp_file)
        assert lines1 is lines2  # Same object due to caching


class TestGetLine:
    """Tests for get_line function."""

    def test_get_first_line(self, temp_file):
        assert get_line(temp_file, 0) == "line0"

    def test_get_middle_line(self, temp_file):
        assert get_line(temp_file, 2) == "line2"

    def test_get_last_line(self, temp_file):
        assert get_line(temp_file, 4) == "line4"

    def test_out_of_range(self, temp_file):
        with pytest.raises(IndexError, match="out of range"):
            get_line(temp_file, 10)


class TestGetLines:
    """Tests for get_lines function."""

    def test_get_multiple_lines(self, temp_file):
        set_seed(42)
        lines = get_lines(temp_file, 3)
        assert len(lines) == 3
        assert all(line in ["line0", "line1", "line2", "line3", "line4"] for line in lines)

    def test_reproducible_with_seed(self, temp_file):
        set_seed(12345)
        lines1 = get_lines(temp_file, 3)
        set_seed(12345)
        lines2 = get_lines(temp_file, 3)
        assert lines1 == lines2

    def test_empty_file_raises(self, empty_file):
        with pytest.raises(ValueError, match="empty"):
            get_lines(empty_file, 1)


class TestGetRandomLine:
    """Tests for get_random_line function."""

    def test_returns_valid_line(self, temp_file):
        set_seed(42)
        line = get_random_line(temp_file)
        assert line in ["line0", "line1", "line2", "line3", "line4"]

    def test_reproducible_with_seed(self, temp_file):
        set_seed(12345)
        line1 = get_random_line(temp_file)
        set_seed(12345)
        line2 = get_random_line(temp_file)
        assert line1 == line2

    def test_empty_file_raises(self, empty_file):
        with pytest.raises(ValueError, match="empty"):
            get_random_line(empty_file)


class TestGetLinesPaired:
    """Tests for get_lines_paired function."""

    def test_default_delimiter(self, temp_file):
        set_seed(42)
        pairs = get_lines_paired(temp_file, 2)
        assert len(pairs) == 2
        for pair in pairs:
            assert ":" in pair

    def test_custom_delimiter(self, temp_file):
        set_seed(42)
        pairs = get_lines_paired(temp_file, 2, delim="|")
        assert len(pairs) == 2
        for pair in pairs:
            assert "|" in pair

    def test_reproducible_with_seed(self, temp_file):
        set_seed(12345)
        pairs1 = get_lines_paired(temp_file, 2)
        set_seed(12345)
        pairs2 = get_lines_paired(temp_file, 2)
        assert pairs1 == pairs2


class TestSetSeed:
    """Tests for set_seed function."""

    def test_returns_seed(self):
        seed = set_seed(42)
        assert seed == 42

    def test_none_returns_random_seed(self):
        seed = set_seed(None)
        assert isinstance(seed, int)


class TestClearCache:
    """Tests for clear_cache function."""

    def test_clears_cache(self, temp_file):
        # Read file to populate cache
        all_lines(temp_file)
        clear_cache()
        # No way to directly check cache, but should not raise
        all_lines(temp_file)
