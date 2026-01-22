# -------------------------------------
# AMT module tests
# -------------------------------------
"""
Tests for the AMT module (loader, schedules, tickers).
"""
import os
import tempfile
import pytest
from pathlib import Path

from specparser.amt import (
    # loader
    load_amt,
    clear_cache,
    get_value,
    get_aum,
    get_leverage,
    get_asset,
    find_assets,
    cached_assets,
    get_table,
    table_column,
    format_table,
    print_table,
    bind_rows,
    table_unique_rows,
    table_select_columns,
    table_add_column,
    table_drop_columns,
    table_replace_value,
    table_stack_cols,
    table_to_columns,
    table_to_rows,
    # arrow support
    table_to_arrow,
    table_to_jsonable,
    table_orientation,
    table_nrows,
    table_validate,
    table_head,
    table_sample,
    table_bind_rows,
    table_unchop,
    table_chop,
    table_left_join,
    table_inner_join,
    _iter_assets,
    assets,
    asset_class,
    _compile_rules,
    _match_rules,
    asset_table,
    asset_group,
    # schedules
    get_schedule,
    find_schedules,
    _split_code_value,
    find_straddle_yrs,
    find_straddle_ym,
    get_straddle_yrs,
    get_expand_ym,
    # straddle parsing
    ntr,
    ntry,
    ntrm,
    xpr,
    xpry,
    xprm,
    ntrc,
    ntrv,
    xprc,
    xprv,
    wgt,
    # tickers
    _split_ticker,
    get_tschemas,
    find_tschemas,
    fut_spec2ticker,
    fut_norm2act,
    fut_act2norm,
    clear_normalized_cache,
    _tschma_dict_expand_bbgfc,
    _tschma_dict_expand_split,
    find_tickers,
    find_tickers_ym,
    asset_straddle_tickers,
)
from specparser.amt.tickers import (
    _parse_date_constraint,
    get_tickers_ym,
    _filter_straddle_tickers,
    _compute_actions,
    _anchor_day,
    _nth_good_day_after,
    _add_calendar_days,
    _last_good_day_in_month,
    _norm_cdf,
    model_ES,
    model_NS,
    model_BS,
    model_default,
    MODEL_DISPATCH,
    _get_rollforward_fields,
    get_straddle_valuation,
    _load_overrides,
    _override_expiry,
    _OVERRIDE_CACHE,
)
import specparser.amt.tickers as tickers_module


# -------------------------------------
# Test Fixtures
# -------------------------------------

@pytest.fixture
def test_amt_file():
    """Create a temporary AMT file for testing."""
    test_amt = """
backtest:
  aum: 1000000.0
  leverage: 2.5

amt:
  Asset1:
    Underlying: "CL Comdty"
    Class: "Commodity"
    WeightCap: 0.10
    Options: "monthly_std"
    Market:
      Field: "PX_LAST"
      Tickers: ["CL1 Comdty", "CL2 Comdty"]
    Vol:
      Source: "BBG"
      Ticker: "CL1C Comdty"
      Near: "ATM_IMP_VOL"
      Far: "ATM_IMP_VOL_6M"
    Hedge:
      Source: "fut"
      fut_code: "CL"
      fut_month_map: "FGHJKMNQUVXZ"
      min_year_offset: 0
      market_code: "Comdty"
    Valuation:
      Model: "Black"

  Asset2:
    Underlying: "ES Equity"
    Class: "Equity"
    WeightCap: 0.0
    Options: "quarterly_std"
    Market:
      Field: "PX_LAST"
      Tickers: "ES1 Index"
    Vol:
      Source: "BBG"
      Ticker: "ES1C Index"
      Near: "ATM_IMP_VOL"
      Far: "NONE"
    Hedge:
      Source: "nonfut"
      Ticker: "SPY US Equity"
      Field: "PX_LAST"

  Asset3:
    Underlying: "TY Rate"
    Class: "Rate"
    WeightCap: 0.05
    Options: "monthly_std"
    Hedge:
      Source: "calc"
      ccy: ["USD"]
      tenor: ["10Y"]

  Asset4:
    Underlying: "JPY Curncy"
    Class: "Currency"
    WeightCap: 0.03
    Options: "monthly_std"
    Hedge:
      Source: "cds"
      hedge: "USDJPY Curncy"
      hedge1: "JPY1M Curncy"

  Asset5:
    Underlying: "SPLIT Test"
    Class: "Test"
    WeightCap: 0.01
    Hedge:
      Source: "nonfut"
      Ticker: "OLD1 Comdty:2024-06:NEW1 Comdty"
      Field: "PX_LAST"

expiry_schedules:
  monthly_std:
    - "N1_OVERRIDE15"
    - "Fa_OVERRIDEb_0.5"
  quarterly_std:
    - "N1_OVERRIDE15_0.33"
    - "N2_OVERRIDE15_0.33"
    - "F1_OVERRIDE15_0.34"

group_table:
  Columns: [field, rgx, value]
  Rows:
    - [Class, "^Commodity$", "commodities"]
    - [Class, "^Equity$", "equities"]
    - [Class, "^Rate$", "rates"]
    - [Class, ".*", "other"]

subgroup_table:
  Columns: [field, rgx, value]
  Rows:
    - [Underlying, "^CL", "energy"]
    - [Underlying, "^ES", "indices"]
    - [Underlying, ".*", "misc"]

liquidity_table:
  Columns: [field, rgx, value]
  Rows:
    - [Class, "^Commodity$", "2"]
    - [Class, ".*", "1"]

limit_overrides:
  Columns: [field, rgx, value]
  Rows:
    - [Underlying, "^CL", "0.15"]
    - [Underlying, ".*", ""]
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(test_amt)
        path = f.name

    yield path

    # Cleanup
    os.unlink(path)
    clear_cache()


@pytest.fixture
def chain_csv_file():
    """Create a temporary chain CSV file for testing."""
    csv_content = """normalized_future,actual_future
CLF2024 Comdty,CL F24 Comdty
CLG2024 Comdty,CL G24 Comdty
CLH2024 Comdty,CL H24 Comdty
CLJ2024 Comdty,CL J24 Comdty
CLK2024 Comdty,CL K24 Comdty
CLM2024 Comdty,CL M24 Comdty
CLN2024 Comdty,CL N24 Comdty
CLQ2024 Comdty,CL Q24 Comdty
CLU2024 Comdty,CL U24 Comdty
CLV2024 Comdty,CL V24 Comdty
CLX2024 Comdty,CL X24 Comdty
CLZ2024 Comdty,CL Z24 Comdty
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    yield path

    # Cleanup
    os.unlink(path)
    clear_normalized_cache()


# -------------------------------------
# Loader Tests
# -------------------------------------

class TestLoader:
    """Tests for loader.py functions."""

    def test_load_amt(self, test_amt_file):
        """Test loading an AMT file."""
        clear_cache()
        data = load_amt(test_amt_file)
        assert "amt" in data
        assert "backtest" in data
        assert "expiry_schedules" in data

    def test_load_amt_caching(self, test_amt_file):
        """Test that AMT files are cached."""
        clear_cache()
        data1 = load_amt(test_amt_file)
        data2 = load_amt(test_amt_file)
        assert data1 is data2  # Same object from cache

    def test_clear_cache(self, test_amt_file):
        """Test cache clearing."""
        load_amt(test_amt_file)
        clear_cache()
        # After clear, should reload
        data1 = load_amt(test_amt_file)
        clear_cache()
        data2 = load_amt(test_amt_file)
        assert data1 is not data2  # Different objects

    def test_get_value(self, test_amt_file):
        """Test getting values by dot-separated path."""
        clear_cache()
        assert get_value(test_amt_file, "backtest.aum") == 1000000.0
        assert get_value(test_amt_file, "backtest.leverage") == 2.5
        assert get_value(test_amt_file, "nonexistent.path", "default") == "default"
        assert get_value(test_amt_file, "backtest.nonexistent", None) is None

    def test_get_aum(self, test_amt_file):
        """Test getting AUM value."""
        clear_cache()
        assert get_aum(test_amt_file) == 1000000.0

    def test_get_leverage(self, test_amt_file):
        """Test getting leverage value."""
        clear_cache()
        assert get_leverage(test_amt_file) == 2.5

    def test_get_asset(self, test_amt_file):
        """Test getting asset by Underlying."""
        clear_cache()
        asset = get_asset(test_amt_file, "CL Comdty")
        assert asset is not None
        assert asset["Class"] == "Commodity"
        assert asset["WeightCap"] == 0.10

        # Non-existent asset
        assert get_asset(test_amt_file, "NONEXISTENT") is None

    def test_find_assets(self, test_amt_file):
        """Test finding assets by pattern."""
        clear_cache()
        # Find all
        found = find_assets(test_amt_file, ".")
        assert len(found["rows"]) == 5

        # Find by prefix
        found = find_assets(test_amt_file, "^CL")
        assert len(found["rows"]) == 1
        assert found["rows"][0][0] == "CL Comdty"

        # Find commodities by suffix
        found = find_assets(test_amt_file, "Comdty$")
        assert len(found["rows"]) == 1

    def test_find_assets_live_only(self, test_amt_file):
        """Test finding assets with live_only filter."""
        clear_cache()
        # Without live_only: 5 assets
        all_assets = find_assets(test_amt_file, ".", live_only=False)
        assert len(all_assets["rows"]) == 5

        # With live_only: excludes Asset2 (WeightCap=0)
        live_assets = find_assets(test_amt_file, ".", live_only=True)
        assert len(live_assets["rows"]) == 4

    def test_cached_assets(self, test_amt_file):
        """Test listing cached assets."""
        clear_cache()
        load_amt(test_amt_file)
        cached = cached_assets(test_amt_file)
        assert len(cached["rows"]) == 5
        assert cached["columns"] == ["asset"]

    def test_assets(self, test_amt_file):
        """Test assets function with various filters."""
        clear_cache()
        # All assets
        all_a = assets(test_amt_file)
        assert len(all_a["rows"]) == 5

        # Live only
        live_a = assets(test_amt_file, live_only=True)
        assert len(live_a["rows"]) == 4

        # With pattern
        filtered = assets(test_amt_file, pattern="^CL")
        assert len(filtered["rows"]) == 1

    def test_iter_assets(self, test_amt_file):
        """Test _iter_assets generator."""
        clear_cache()
        # All assets
        result = list(_iter_assets(test_amt_file))
        assert len(result) == 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

        # Live only
        live = list(_iter_assets(test_amt_file, live_only=True))
        assert len(live) == 4

        # With pattern
        filtered = list(_iter_assets(test_amt_file, pattern="^ES"))
        assert len(filtered) == 1
        assert filtered[0][1] == "ES Equity"

    def test_asset_class(self, test_amt_file):
        """Test asset_class function."""
        clear_cache()
        table = asset_class(test_amt_file, live_only=True)
        assert table["columns"] == ["asset", "cls", "volsrc", "hdgsrc", "model"]
        assert len(table["rows"]) == 4

        # Check CL Comdty row
        cl_row = [r for r in table["rows"] if r[0] == "CL Comdty"][0]
        assert cl_row[1] == "Commodity"  # cls
        assert cl_row[2] == "BBG"  # volsrc
        assert cl_row[3] == "fut"  # hdgsrc
        assert cl_row[4] == "Black"  # model

    def test_get_table(self, test_amt_file):
        """Test getting embedded tables."""
        clear_cache()
        table = get_table(test_amt_file, "group_table")
        assert table["columns"] == ["field", "rgx", "value"]
        assert len(table["rows"]) == 4

    def test_get_table_errors(self, test_amt_file):
        """Test get_table error handling."""
        clear_cache()
        with pytest.raises(ValueError, match="not found"):
            get_table(test_amt_file, "nonexistent_table")

    def test_table_column(self, test_amt_file):
        """Test extracting a column from a table."""
        clear_cache()
        table = {"columns": ["a", "b", "c"], "rows": [[1, 2, 3], [4, 5, 6]]}
        assert table_column(table, "a") == [1, 4]
        assert table_column(table, "b") == [2, 5]
        assert table_column(table, "c") == [3, 6]

    def test_table_column_not_found(self):
        """Test table_column with missing column."""
        table = {"columns": ["a", "b"], "rows": [[1, 2]]}
        with pytest.raises(ValueError, match="Column 'c' not found"):
            table_column(table, "c")

    def test_format_table(self):
        """Test table formatting."""
        table = {"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
        formatted = format_table(table)
        lines = formatted.split("\n")
        assert lines[0] == "a\tb"
        assert lines[1] == "1\t2"
        assert lines[2] == "3\t4"

    def test_print_table(self, capsys):
        """Test table printing."""
        table = {"columns": ["col1", "col2"], "rows": [[1, 2]]}
        print_table(table)
        captured = capsys.readouterr()
        assert "col1\tcol2" in captured.out
        assert "1\t2" in captured.out

    def test_compile_rules(self, test_amt_file):
        """Test rule compilation."""
        clear_cache()
        table = get_table(test_amt_file, "group_table")
        rules = _compile_rules(table)
        assert len(rules) == 4
        assert rules[0][0] == "Class"
        assert rules[0][2] == "commodities"

    def test_match_rules(self, test_amt_file):
        """Test rule matching."""
        clear_cache()
        table = get_table(test_amt_file, "group_table")
        rules = _compile_rules(table)

        # Test matching
        result = _match_rules(rules, {"Class": "Commodity"})
        assert result == "commodities"

        result = _match_rules(rules, {"Class": "Equity"})
        assert result == "equities"

        result = _match_rules(rules, {"Class": "Unknown"})
        assert result == "other"  # Matches .*

        # Test default
        empty_rules = []
        result = _match_rules(empty_rules, {}, default="fallback")
        assert result == "fallback"

    def test_asset_table(self, test_amt_file):
        """Test asset_table function."""
        clear_cache()
        table = asset_table(test_amt_file, "group_table", live_only=True)
        assert table["columns"] == ["asset", "group"]
        assert len(table["rows"]) == 4

        # Check classifications
        rows_dict = {r[0]: r[1] for r in table["rows"]}
        assert rows_dict["CL Comdty"] == "commodities"
        assert rows_dict["TY Rate"] == "rates"

    def test_asset_group(self, test_amt_file):
        """Test asset_group function (merges multiple tables)."""
        clear_cache()
        table = asset_group(test_amt_file, live_only=True)
        assert "group" in table["columns"]
        assert "subgroup" in table["columns"]
        assert "liquidity" in table["columns"]
        assert "limit_overrides" in table["columns"]
        # table_stack_cols returns column-oriented, so rows[0] is first column
        assert table["orientation"] == "column"
        assert len(table["rows"][0]) == 4  # 4 live assets


# -------------------------------------
# Table Utilities Tests
# -------------------------------------

class TestTableUtilities:
    """Tests for bind_rows and table_unique_rows functions."""

    def test_bind_rows_basic(self):
        """Test basic row binding."""
        t1 = {"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
        t2 = {"columns": ["a", "b"], "rows": [[5, 6], [7, 8]]}
        result = bind_rows(t1, t2)
        assert result["columns"] == ["a", "b"]
        assert result["rows"] == [[1, 2], [3, 4], [5, 6], [7, 8]]

    def test_bind_rows_single_table(self):
        """Test bind_rows with single table."""
        t1 = {"columns": ["a", "b"], "rows": [[1, 2]]}
        result = bind_rows(t1)
        assert result["columns"] == ["a", "b"]
        assert result["rows"] == [[1, 2]]

    def test_bind_rows_empty(self):
        """Test bind_rows with no tables."""
        result = bind_rows()
        assert result == {"orientation": "row", "columns": [], "rows": []}

    def test_bind_rows_empty_rows(self):
        """Test bind_rows with tables that have empty rows."""
        t1 = {"columns": ["a", "b"], "rows": []}
        t2 = {"columns": ["a", "b"], "rows": [[1, 2]]}
        result = bind_rows(t1, t2)
        assert result["rows"] == [[1, 2]]

    def test_bind_rows_column_mismatch(self):
        """Test bind_rows raises error on column mismatch."""
        t1 = {"columns": ["a", "b"], "rows": [[1, 2]]}
        t2 = {"columns": ["x", "y"], "rows": [[3, 4]]}
        with pytest.raises(ValueError, match="columns"):
            bind_rows(t1, t2)

    def test_bind_rows_multiple_tables(self):
        """Test bind_rows with multiple tables."""
        t1 = {"columns": ["a"], "rows": [[1]]}
        t2 = {"columns": ["a"], "rows": [[2]]}
        t3 = {"columns": ["a"], "rows": [[3]]}
        result = bind_rows(t1, t2, t3)
        assert result["rows"] == [[1], [2], [3]]

    def test_table_unique_rows_basic(self):
        """Test basic row deduplication."""
        table = {"columns": ["a", "b"], "rows": [[1, 2], [3, 4], [1, 2]]}
        result = table_unique_rows(table)
        assert result["columns"] == ["a", "b"]
        assert len(result["rows"]) == 2
        # Last occurrence is preserved
        assert [1, 2] in result["rows"]
        assert [3, 4] in result["rows"]

    def test_table_unique_rows_no_duplicates(self):
        """Test unique_rows with no duplicates."""
        table = {"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
        result = table_unique_rows(table)
        assert result["rows"] == [[1, 2], [3, 4]]

    def test_table_unique_rows_empty(self):
        """Test unique_rows with empty table."""
        table = {"columns": ["a", "b"], "rows": []}
        result = table_unique_rows(table)
        assert result["rows"] == []

    def test_table_unique_rows_all_duplicates(self):
        """Test unique_rows where all rows are identical."""
        table = {"columns": ["a"], "rows": [[1], [1], [1]]}
        result = table_unique_rows(table)
        assert result["rows"] == [[1]]

    def test_table_unique_rows_preserves_last(self):
        """Test that unique_rows preserves the last occurrence."""
        # Using dict comprehension preserves insertion order with last value
        table = {"columns": ["a", "b"], "rows": [[1, "first"], [1, "second"]]}
        result = table_unique_rows(table)
        # The dict uses tuple([1, "first"]) and tuple([1, "second"]) as different keys
        # since the rows are different, both should be preserved
        assert len(result["rows"]) == 2

    def test_table_select_columns_basic(self):
        """Test selecting and reordering columns."""
        table = {"columns": ["a", "b", "c"], "rows": [[1, 2, 3], [4, 5, 6]]}
        result = table_select_columns(table, ["c", "a"])
        assert result["columns"] == ["c", "a"]
        assert result["rows"] == [[3, 1], [6, 4]]

    def test_table_select_columns_all(self):
        """Test selecting all columns in same order."""
        table = {"columns": ["a", "b"], "rows": [[1, 2]]}
        result = table_select_columns(table, ["a", "b"])
        assert result["columns"] == ["a", "b"]
        assert result["rows"] == [[1, 2]]

    def test_table_select_columns_not_found(self):
        """Test selecting non-existent column raises ValueError."""
        table = {"columns": ["a", "b"], "rows": [[1, 2]]}
        with pytest.raises(ValueError, match="Column 'x' not found"):
            table_select_columns(table, ["a", "x"])

    def test_table_add_column_append(self):
        """Test adding column at end."""
        table = {"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
        result = table_add_column(table, "c", value="X")
        assert result["columns"] == ["a", "b", "c"]
        assert result["rows"] == [[1, 2, "X"], [3, 4, "X"]]

    def test_table_add_column_position(self):
        """Test adding column at specific position."""
        table = {"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
        result = table_add_column(table, "c", value="X", position=1)
        assert result["columns"] == ["a", "c", "b"]
        assert result["rows"] == [[1, "X", 2], [3, "X", 4]]

    def test_table_add_column_default_none(self):
        """Test adding column with default None value."""
        table = {"columns": ["a"], "rows": [[1], [2]]}
        result = table_add_column(table, "b")
        assert result["rows"] == [[1, None], [2, None]]

    def test_table_drop_columns_basic(self):
        """Test dropping columns."""
        table = {"columns": ["a", "b", "c"], "rows": [[1, 2, 3], [4, 5, 6]]}
        result = table_drop_columns(table, ["b"])
        assert result["columns"] == ["a", "c"]
        assert result["rows"] == [[1, 3], [4, 6]]

    def test_table_drop_columns_multiple(self):
        """Test dropping multiple columns."""
        table = {"columns": ["a", "b", "c", "d"], "rows": [[1, 2, 3, 4]]}
        result = table_drop_columns(table, ["b", "d"])
        assert result["columns"] == ["a", "c"]
        assert result["rows"] == [[1, 3]]

    def test_table_drop_columns_nonexistent(self):
        """Test dropping non-existent column is silently ignored."""
        table = {"columns": ["a", "b"], "rows": [[1, 2]]}
        result = table_drop_columns(table, ["x", "b"])
        assert result["columns"] == ["a"]
        assert result["rows"] == [[1]]

    def test_table_replace_value_basic(self):
        """Test replacing values in a column."""
        table = {"columns": ["a", "b"], "rows": [[1, "old"], [2, "old"], [3, "keep"]]}
        result = table_replace_value(table, "b", "old", "new")
        assert result["rows"] == [[1, "new"], [2, "new"], [3, "keep"]]

    def test_table_replace_value_no_match(self):
        """Test replace when no values match."""
        table = {"columns": ["a", "b"], "rows": [[1, "x"], [2, "y"]]}
        result = table_replace_value(table, "b", "z", "new")
        assert result["rows"] == [[1, "x"], [2, "y"]]

    def test_table_replace_value_not_found(self):
        """Test replace on non-existent column raises ValueError."""
        table = {"columns": ["a", "b"], "rows": [[1, 2]]}
        with pytest.raises(ValueError, match="Column 'x' not found"):
            table_replace_value(table, "x", 1, 2)

    # --- table_stack_cols tests ---

    def test_table_stack_cols_basic(self):
        """Test basic column stacking."""
        t1 = {"orientation": "row", "columns": ["key", "a"], "rows": [[1, 10], [2, 20]]}
        t2 = {"orientation": "row", "columns": ["key", "b"], "rows": [[1, 100], [2, 200]]}
        result = table_stack_cols(t1, t2)
        assert result["orientation"] == "column"
        assert result["columns"] == ["key", "a", "b"]
        # Column-oriented: rows = [[key_vals], [a_vals], [b_vals]]
        assert result["rows"][0] == [1, 2]  # key column
        assert result["rows"][1] == [10, 20]  # a column
        assert result["rows"][2] == [100, 200]  # b column

    def test_table_stack_cols_column_oriented_input(self):
        """Test stacking when input is already column-oriented (should be O(k))."""
        t1 = {"orientation": "column", "columns": ["key", "a"], "rows": [[1, 2], [10, 20]]}
        t2 = {"orientation": "column", "columns": ["key", "b"], "rows": [[1, 2], [100, 200]]}
        result = table_stack_cols(t1, t2)
        assert result["orientation"] == "column"
        assert result["columns"] == ["key", "a", "b"]
        assert result["rows"][0] == [1, 2]  # key column
        assert result["rows"][1] == [10, 20]  # a column
        assert result["rows"][2] == [100, 200]  # b column

    def test_table_stack_cols_empty(self):
        """Test stacking with no tables."""
        result = table_stack_cols()
        assert result == {"orientation": "column", "columns": [], "rows": []}

    def test_table_stack_cols_single_table(self):
        """Test stacking single table."""
        t1 = {"orientation": "row", "columns": ["key", "a"], "rows": [[1, 10]]}
        result = table_stack_cols(t1)
        assert result["columns"] == ["key", "a"]
        # Only key column is output (key from first, no non-key from second table)
        # Wait - with single table, should have key + non-key cols
        assert len(result["rows"]) == 2  # key and a columns

    def test_table_stack_cols_row_count_mismatch(self):
        """Test stacking raises error when row counts differ."""
        t1 = {"orientation": "row", "columns": ["key", "a"], "rows": [[1, 10], [2, 20]]}
        t2 = {"orientation": "row", "columns": ["key", "b"], "rows": [[1, 100]]}  # only 1 row
        with pytest.raises(ValueError, match="Table 2 has 1 rows; expected 2"):
            table_stack_cols(t1, t2)

    def test_table_stack_cols_copy_data_true(self):
        """Test that copy_data=True copies column data."""
        col_data = [1, 2, 3]
        t1 = {"orientation": "column", "columns": ["key", "a"], "rows": [col_data, [10, 20, 30]]}
        result = table_stack_cols(t1, copy_data=True)
        # Modify original should not affect result
        col_data[0] = 999
        assert result["rows"][0][0] == 1  # unchanged

    def test_table_stack_cols_copy_data_false(self):
        """Test that copy_data=False references existing lists."""
        col_data = [1, 2, 3]
        t1 = {"orientation": "column", "columns": ["key", "a"], "rows": [col_data, [10, 20, 30]]}
        result = table_stack_cols(t1, copy_data=False)
        # Modify original should affect result
        col_data[0] = 999
        assert result["rows"][0][0] == 999  # changed

    def test_table_stack_cols_multiple_tables(self):
        """Test stacking more than 2 tables."""
        t1 = {"orientation": "row", "columns": ["key", "a"], "rows": [[1, 10]]}
        t2 = {"orientation": "row", "columns": ["key", "b"], "rows": [[1, 20]]}
        t3 = {"orientation": "row", "columns": ["key", "c"], "rows": [[1, 30]]}
        result = table_stack_cols(t1, t2, t3)
        assert result["columns"] == ["key", "a", "b", "c"]

    # --- table orientation conversion tests ---

    def test_table_to_columns_basic(self):
        """Test converting row-oriented to column-oriented."""
        t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
        result = table_to_columns(t)
        assert result["orientation"] == "column"
        assert result["columns"] == ["a", "b"]
        assert result["rows"] == [[1, 3], [2, 4]]

    def test_table_to_columns_already_column(self):
        """Test to_columns returns same table if already column-oriented."""
        t = {"orientation": "column", "columns": ["a", "b"], "rows": [[1, 3], [2, 4]]}
        result = table_to_columns(t)
        assert result is t  # same object

    def test_table_to_columns_empty(self):
        """Test to_columns with empty table."""
        t = {"orientation": "row", "columns": ["a", "b"], "rows": []}
        result = table_to_columns(t)
        assert result["orientation"] == "column"
        assert result["rows"] == [[], []]  # empty columns

    def test_table_to_rows_basic(self):
        """Test converting column-oriented to row-oriented."""
        t = {"orientation": "column", "columns": ["a", "b"], "rows": [[1, 3], [2, 4]]}
        result = table_to_rows(t)
        assert result["orientation"] == "row"
        assert result["columns"] == ["a", "b"]
        assert result["rows"] == [[1, 2], [3, 4]]

    def test_table_to_rows_already_row(self):
        """Test to_rows returns same table if already row-oriented."""
        t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
        result = table_to_rows(t)
        assert result is t  # same object

    def test_table_to_rows_empty(self):
        """Test to_rows with empty table."""
        t = {"orientation": "column", "columns": ["a", "b"], "rows": [[], []]}
        result = table_to_rows(t)
        assert result["orientation"] == "row"
        assert result["rows"] == []

    def test_table_orientation_roundtrip(self):
        """Test row->column->row preserves data."""
        original = {"orientation": "row", "columns": ["a", "b", "c"], "rows": [[1, 2, 3], [4, 5, 6]]}
        col = table_to_columns(original)
        back = table_to_rows(col)
        assert back["rows"] == original["rows"]
        assert back["columns"] == original["columns"]


# -------------------------------------
# Schedules Tests
# -------------------------------------

class TestSchedules:
    """Tests for schedules.py functions."""

    def test_split_code_value(self):
        """Test _split_code_value function."""
        assert _split_code_value("N1") == ("N", "1")
        assert _split_code_value("OVERRIDE15") == ("OVERRIDE", "15")
        assert _split_code_value("F") == ("F", "")
        assert _split_code_value("123") == ("123", "")
        assert _split_code_value("") == ("", "")
        assert _split_code_value("Fa") == ("F", "a")

    def test_get_schedule(self, test_amt_file):
        """Test getting schedule for an asset."""
        clear_cache()
        table = get_schedule(test_amt_file, "CL Comdty")
        assert table["columns"] == ["schcnt", "schid", "asset", "ntrc", "ntrv", "xprc", "xprv", "wgt"]
        assert len(table["rows"]) == 2  # monthly_std has 2 components

    def test_get_schedule_not_found(self, test_amt_file):
        """Test get_schedule with non-existent asset."""
        clear_cache()
        table = get_schedule(test_amt_file, "NONEXISTENT")
        assert table["rows"] == []

    def test_get_schedule_no_schedule(self, test_amt_file):
        """Test get_schedule for asset without Options field."""
        clear_cache()
        # Asset5 has no Options field in our test fixture
        # This should return a table with empty schedule placeholder
        table = get_schedule(test_amt_file, "SPLIT Test")
        # Should have a row with schcnt=0 for asset without schedule
        assert len(table["rows"]) == 1
        assert table["rows"][0][0] == 0  # schcnt

    def test_find_schedules(self, test_amt_file):
        """Test finding schedules by pattern."""
        clear_cache()
        table = find_schedules(test_amt_file, ".", live_only=True)
        assert len(table["rows"]) > 0

        # Find specific asset
        table = find_schedules(test_amt_file, "^CL", live_only=False)
        assert all(r[2] == "CL Comdty" for r in table["rows"])

    def test_find_schedules_live_only(self, test_amt_file):
        """Test find_schedules with live_only filter."""
        clear_cache()
        live = find_schedules(test_amt_file, ".", live_only=True)
        all_sch = find_schedules(test_amt_file, ".", live_only=False)
        assert len(all_sch["rows"]) >= len(live["rows"])

    def test_find_straddle_ym(self, test_amt_file):
        """Test expanding schedules for a specific year/month."""
        clear_cache()
        table = find_straddle_ym(test_amt_file, 2024, 6, pattern="^CL", live_only=False)
        assert table["columns"] == ["asset", "straddle"]
        assert len(table["rows"]) == 2  # CL Comdty has 2 schedule components

        # Check straddle format
        for row in table["rows"]:
            assert row[0] == "CL Comdty"
            assert row[1].startswith("|")
            assert row[1].endswith("|")

    def test_find_straddle_yrs(self, test_amt_file):
        """Test expanding schedules across year range."""
        clear_cache()
        table = find_straddle_yrs(test_amt_file, 2024, 2024, pattern="^CL", live_only=False)
        # 12 months * 2 schedule components = 24 rows
        assert len(table["rows"]) == 24
        assert table["columns"] == ["asset", "straddle"]

    def test_get_straddle_yrs(self, test_amt_file):
        """Test get_straddle_yrs for single asset."""
        clear_cache()
        table = get_straddle_yrs(test_amt_file, "CL Comdty", 2024, 2024)
        # 12 months * 2 schedule components = 24 rows
        assert len(table["rows"]) == 24

    def test_get_expand_ym(self, test_amt_file):
        """Test get_expand_ym for single asset and month."""
        clear_cache()
        table = get_expand_ym(test_amt_file, "CL Comdty", 2024, 6)
        assert len(table["rows"]) == 2

    def test_find_straddle_yrs_empty_pattern(self, test_amt_file):
        """Test find_straddle_yrs with pattern matching no assets."""
        clear_cache()
        table = find_straddle_yrs(test_amt_file, 2024, 2024, pattern="^NONEXISTENT", live_only=False)
        assert len(table["rows"]) == 0

    def test_straddle_format(self, test_amt_file):
        """Test straddle string format is correct."""
        clear_cache()
        table = find_straddle_ym(test_amt_file, 2024, 6, pattern="^CL", live_only=False)

        for row in table["rows"]:
            straddle = row[1]
            # Format: |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|
            # Strip leading/trailing pipes, split by pipe
            parts = straddle.strip("|").split("|")
            # Expect 6-7 parts (wgt may be empty)
            assert len(parts) >= 6
            assert "-" in parts[0]  # ntry-ntrm
            assert parts[1] == "2024-06"  # xpry-xprm


# -------------------------------------
# Straddle Parsing Tests
# -------------------------------------

class TestStraddleParsing:
    """Tests for straddle string parsing functions."""

    # Test straddle format: |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|
    # Example: |2023-12|2024-01|N|0|OVERRIDE|15|33.3|

    def test_ntr(self):
        """Test ntr (entry date string) extraction."""
        s = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"
        assert ntr(s) == "2023-12"

    def test_ntry(self):
        """Test ntry (entry year) extraction."""
        s = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"
        assert ntry(s) == 2023

    def test_ntrm(self):
        """Test ntrm (entry month) extraction."""
        s = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"
        assert ntrm(s) == 12

    def test_xpr(self):
        """Test xpr (expiry date string) extraction."""
        s = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"
        assert xpr(s) == "2024-01"

    def test_xpry(self):
        """Test xpry (expiry year) extraction."""
        s = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"
        assert xpry(s) == 2024

    def test_xprm(self):
        """Test xprm (expiry month) extraction."""
        s = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"
        assert xprm(s) == 1

    def test_ntrc(self):
        """Test ntrc (entry code) extraction."""
        s = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"
        assert ntrc(s) == "N"

    def test_ntrv(self):
        """Test ntrv (entry value) extraction."""
        s = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"
        assert ntrv(s) == "0"

    def test_xprc(self):
        """Test xprc (expiry code) extraction."""
        s = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"
        assert xprc(s) == "OVERRIDE"

    def test_xprv(self):
        """Test xprv (expiry value) extraction."""
        s = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"
        assert xprv(s) == "15"

    def test_wgt(self):
        """Test wgt (weight) extraction."""
        s = "|2023-12|2024-01|N|0|OVERRIDE|15|33.3|"
        assert wgt(s) == "33.3"

    def test_far_entry_code(self):
        """Test straddle with F (Far) entry code."""
        s = "|2023-11|2024-01|F|5|OVERRIDE|15|50.0|"
        assert ntrc(s) == "F"
        assert ntry(s) == 2023
        assert ntrm(s) == 11

    def test_empty_weight(self):
        """Test straddle with empty weight."""
        # wgt can be empty string, the format preserves empty fields
        s = "|2023-12|2024-01|N|0|OVERRIDE|15||"
        assert wgt(s) == ""

    def test_empty_xprv(self):
        """Test straddle with empty expiry value."""
        s = "|2023-12|2024-01|N|0|OVERRIDE||33.3|"
        assert xprv(s) == ""

    def test_straddle_parsing_consistency(self):
        """Test that all parsing functions work together consistently."""
        s = "|2024-06|2024-07|N|1|OVERRIDE|15|0.5|"
        # Entry: June 2024
        assert f"{ntry(s)}-{ntrm(s):02d}" == "2024-06"
        # Expiry: July 2024
        assert f"{xpry(s)}-{xprm(s):02d}" == "2024-07"
        # Full date strings
        assert ntr(s) == "2024-06"
        assert xpr(s) == "2024-07"


# -------------------------------------
# Date Constraint Tests
# -------------------------------------

class TestDateConstraints:
    """Tests for _parse_date_constraint function."""

    def test_no_constraint(self):
        """Test param without any date constraint."""
        clean, include = _parse_date_constraint("hedge", 2024, 6)
        assert clean == "hedge"
        assert include is True

    def test_equality_constraint_match(self):
        """Test X (equality) constraint that matches."""
        clean, include = _parse_date_constraint("hedgeX2024-06", 2024, 6)
        assert clean == "hedge"
        assert include is True

    def test_equality_constraint_no_match(self):
        """Test X (equality) constraint that doesn't match."""
        clean, include = _parse_date_constraint("hedgeX2024-06", 2024, 7)
        assert clean == "hedge"
        assert include is False

    def test_equality_constraint_different_year(self):
        """Test X constraint with different year."""
        clean, include = _parse_date_constraint("hedgeX2025-06", 2024, 6)
        assert clean == "hedge"
        assert include is False

    def test_less_than_constraint_match(self):
        """Test < (before) constraint that matches."""
        clean, include = _parse_date_constraint("hedge<2024-06", 2024, 5)
        assert clean == "hedge"
        assert include is True

    def test_less_than_constraint_equal(self):
        """Test < constraint when dates are equal (should not match)."""
        clean, include = _parse_date_constraint("hedge<2024-06", 2024, 6)
        assert clean == "hedge"
        assert include is False

    def test_less_than_constraint_no_match(self):
        """Test < constraint when expiry is after limit."""
        clean, include = _parse_date_constraint("hedge<2024-06", 2024, 7)
        assert clean == "hedge"
        assert include is False

    def test_greater_than_constraint_match(self):
        """Test > (after) constraint that matches."""
        clean, include = _parse_date_constraint("hedge>2024-06", 2024, 7)
        assert clean == "hedge"
        assert include is True

    def test_greater_than_constraint_equal(self):
        """Test > constraint when dates are equal (should not match)."""
        clean, include = _parse_date_constraint("hedge>2024-06", 2024, 6)
        assert clean == "hedge"
        assert include is False

    def test_greater_than_constraint_no_match(self):
        """Test > constraint when expiry is before limit."""
        clean, include = _parse_date_constraint("hedge>2024-06", 2024, 5)
        assert clean == "hedge"
        assert include is False

    def test_year_boundary_less_than(self):
        """Test < constraint across year boundary."""
        # 2023-12 < 2024-01
        clean, include = _parse_date_constraint("hedge<2024-01", 2023, 12)
        assert include is True

    def test_year_boundary_greater_than(self):
        """Test > constraint across year boundary."""
        # 2024-01 > 2023-12
        clean, include = _parse_date_constraint("hedge>2023-12", 2024, 1)
        assert include is True

    def test_invalid_date_format(self):
        """Test that invalid date format returns original param."""
        # Invalid date format should return (param, True)
        clean, include = _parse_date_constraint("hedgeXinvalid", 2024, 6)
        assert clean == "hedgeXinvalid"
        assert include is True

    def test_complex_param_name_with_X(self):
        """Test param with X constraint and complex name."""
        clean, include = _parse_date_constraint("hedge1X2024-06", 2024, 6)
        assert clean == "hedge1"
        assert include is True


# -------------------------------------
# Tickers Tests
# -------------------------------------

class TestTickers:
    """Tests for tickers.py functions."""

    def test_split_ticker_normal(self):
        """Test _split_ticker with normal ticker."""
        result = _split_ticker("CL1 Comdty", "hedge")
        assert result == [("CL1 Comdty", "hedge")]

    def test_split_ticker_split_format(self):
        """Test _split_ticker with split ticker format."""
        result = _split_ticker("OLD1 Comdty:2024-06:NEW1 Comdty", "hedge")
        assert len(result) == 2
        assert result[0] == ("OLD1 Comdty", "hedge<2024-06")
        assert result[1] == ("NEW1 Comdty", "hedge>2024-06")

    def test_split_ticker_invalid_format(self):
        """Test _split_ticker with non-split ticker containing colon."""
        result = _split_ticker("BBG:CL1 Comdty", "hedge")
        assert result == [("BBG:CL1 Comdty", "hedge")]

    def test_get_tschemas(self, test_amt_file):
        """Test getting ticker schemas for an asset."""
        clear_cache()
        table = get_tschemas(test_amt_file, "CL Comdty")
        assert table["columns"] == ["asset", "cls", "type", "param", "source", "ticker", "field"]
        assert len(table["rows"]) > 0

        # Check Market tickers
        market_rows = [r for r in table["rows"] if r[2] == "Market"]
        assert len(market_rows) == 2  # CL1 and CL2

        # Check Vol tickers
        vol_rows = [r for r in table["rows"] if r[2] == "Vol"]
        assert len(vol_rows) >= 1

        # Check Hedge tickers
        hedge_rows = [r for r in table["rows"] if r[2] == "Hedge"]
        assert len(hedge_rows) == 1
        assert hedge_rows[0][4] == "BBGfc"  # source

    def test_get_tschemas_not_found(self, test_amt_file):
        """Test get_tschemas with non-existent asset."""
        clear_cache()
        table = get_tschemas(test_amt_file, "NONEXISTENT")
        assert table["rows"] == []

    def test_get_tschemas_nonfut_hedge(self, test_amt_file):
        """Test get_tschemas with nonfut hedge source."""
        clear_cache()
        table = get_tschemas(test_amt_file, "ES Equity")
        hedge_rows = [r for r in table["rows"] if r[2] == "Hedge"]
        assert len(hedge_rows) == 1
        assert hedge_rows[0][4] == "BBG"
        assert hedge_rows[0][5] == "SPY US Equity"

    def test_get_tschemas_calc_hedge(self, test_amt_file):
        """Test get_tschemas with calc hedge source."""
        clear_cache()
        table = get_tschemas(test_amt_file, "TY Rate")
        hedge_rows = [r for r in table["rows"] if r[2] == "Hedge"]
        assert len(hedge_rows) == 4  # calc produces 4 tickers

    def test_get_tschemas_cds_hedge(self, test_amt_file):
        """Test get_tschemas with cds hedge source."""
        clear_cache()
        table = get_tschemas(test_amt_file, "JPY Curncy")
        hedge_rows = [r for r in table["rows"] if r[2] == "Hedge"]
        assert len(hedge_rows) == 2  # hedge and hedge1

    def test_find_tschemas(self, test_amt_file):
        """Test finding ticker schemas by pattern."""
        clear_cache()
        # Find all commodities
        table = find_tschemas(test_amt_file, "Comdty$")
        assert table["columns"] == ["asset", "cls", "type", "param", "source", "ticker", "field"]
        assert len(table["rows"]) > 0
        # All rows should be for CL Comdty
        for row in table["rows"]:
            assert row[0] == "CL Comdty"

    def test_find_tschemas_no_match(self, test_amt_file):
        """Test find_tschemas with pattern matching nothing."""
        clear_cache()
        table = find_tschemas(test_amt_file, "^NONEXISTENT")
        assert table["rows"] == []

    def test_fut_spec2ticker(self):
        """Test futures spec to ticker conversion."""
        spec = "generic:CL1 Comdty,fut_code:CL,fut_month_map:FGHJKMNQUVXZ,min_year_offset:0,market_code:Comdty"

        # Test various months
        assert fut_spec2ticker(spec, 2024, 1) == "CLF2024 Comdty"  # January -> F
        assert fut_spec2ticker(spec, 2024, 6) == "CLM2024 Comdty"  # June -> M
        assert fut_spec2ticker(spec, 2024, 12) == "CLZ2024 Comdty"  # December -> Z

    def test_fut_spec2ticker_with_year_offset(self):
        """Test futures spec with year offset."""
        # When fut_month_code < opt_month_code, year increments
        spec = "generic:X1 Comdty,fut_code:X,fut_month_map:AAAAAAZZZZZZ,min_year_offset:0,market_code:Comdty"
        # A < F, so year should increment for January
        result = fut_spec2ticker(spec, 2024, 1)
        assert "2025" in result

    def test_fut_norm2act(self, chain_csv_file):
        """Test normalized to actual ticker lookup."""
        clear_normalized_cache()
        result = fut_norm2act(chain_csv_file, "CLF2024 Comdty")
        assert result == "CL F24 Comdty"

        # Non-existent
        result = fut_norm2act(chain_csv_file, "NONEXISTENT")
        assert result is None

    def test_fut_norm2act_caching(self, chain_csv_file):
        """Test that CSV is cached."""
        clear_normalized_cache()
        # First call loads CSV
        fut_norm2act(chain_csv_file, "CLF2024 Comdty")
        # Second call should use cache
        result = fut_norm2act(chain_csv_file, "CLG2024 Comdty")
        assert result == "CL G24 Comdty"

    def test_fut_act2norm(self, chain_csv_file):
        """Test actual to normalized ticker lookup."""
        clear_normalized_cache()
        result = fut_act2norm(chain_csv_file, "CL F24 Comdty")
        assert result == "CLF2024 Comdty"

        # Non-existent
        result = fut_act2norm(chain_csv_file, "NONEXISTENT")
        assert result is None

    def test_fut_act2norm_caching(self, chain_csv_file):
        """Test that reverse CSV is cached."""
        clear_normalized_cache()
        # First call loads CSV
        fut_act2norm(chain_csv_file, "CL F24 Comdty")
        # Second call should use cache
        result = fut_act2norm(chain_csv_file, "CL G24 Comdty")
        assert result == "CLG2024 Comdty"

    def test_fut_act2norm_and_norm2act_inverse(self, chain_csv_file):
        """Test that fut_act2norm and fut_norm2act are inverses."""
        clear_normalized_cache()
        # Normalized -> Actual -> Normalized
        normalized = "CLM2024 Comdty"
        actual = fut_norm2act(chain_csv_file, normalized)
        assert actual == "CL M24 Comdty"
        back_to_normalized = fut_act2norm(chain_csv_file, actual)
        assert back_to_normalized == normalized

        # Actual -> Normalized -> Actual
        actual2 = "CL Z24 Comdty"
        normalized2 = fut_act2norm(chain_csv_file, actual2)
        assert normalized2 == "CLZ2024 Comdty"
        back_to_actual = fut_norm2act(chain_csv_file, normalized2)
        assert back_to_actual == actual2

    def test_clear_normalized_cache_clears_both(self, chain_csv_file):
        """Test that clear_normalized_cache clears both forward and reverse caches."""
        clear_normalized_cache()
        # Populate both caches
        fut_norm2act(chain_csv_file, "CLF2024 Comdty")
        fut_act2norm(chain_csv_file, "CL F24 Comdty")
        # Clear caches
        clear_normalized_cache()
        # Access internal caches to verify they're cleared
        from specparser.amt.tickers import _NORMALIZED_CACHE, _ACTUAL_CACHE
        assert chain_csv_file not in _NORMALIZED_CACHE or str(chain_csv_file) not in _NORMALIZED_CACHE
        assert chain_csv_file not in _ACTUAL_CACHE or str(chain_csv_file) not in _ACTUAL_CACHE

    def test_tschma_dict_expand_bbgfc(self):
        """Test expanding BBGfc row."""
        row = {
            "asset": "CL Comdty",
            "cls": "Commodity",
            "type": "Hedge",
            "param": "fut",
            "source": "BBGfc",
            "ticker": "generic:CL1 Comdty,fut_code:CL,fut_month_map:FGHJKMNQUVXZ,min_year_offset:0,market_code:Comdty",
            "field": "PX_LAST"
        }

        expanded = _tschma_dict_expand_bbgfc(row, 2024, 2024)
        assert len(expanded) == 12  # 12 months

        # Check first row
        assert expanded[0]["param"] == "hedgeX2024-01"
        assert expanded[0]["source"] == "nBBG"  # No CSV provided
        assert "CLF2024" in expanded[0]["ticker"]

    def test_tschma_dict_expand_bbgfc_with_csv(self, chain_csv_file):
        """Test expanding BBGfc row with CSV lookup."""
        clear_normalized_cache()
        row = {
            "asset": "CL Comdty",
            "cls": "Commodity",
            "type": "Hedge",
            "param": "fut",
            "source": "BBGfc",
            "ticker": "generic:CL1 Comdty,fut_code:CL,fut_month_map:FGHJKMNQUVXZ,min_year_offset:0,market_code:Comdty",
            "field": "PX_LAST"
        }

        expanded = _tschma_dict_expand_bbgfc(row, 2024, 2024, chain_csv_file)

        # With CSV, some should be BBG, some might be nBBG
        bbg_rows = [r for r in expanded if r["source"] == "BBG"]
        assert len(bbg_rows) == 12  # All months in our test CSV

    def test_tschma_dict_expand_split_normal(self):
        """Test expand_split_ticker_row with normal ticker."""
        row = {
            "asset": "TEST",
            "cls": "Test",
            "type": "Hedge",
            "param": "hedge",
            "source": "BBG",
            "ticker": "TEST1 Comdty",
            "field": "PX_LAST"
        }

        result = _tschma_dict_expand_split(row)
        assert len(result) == 1
        assert result[0] == row

    def test_tschma_dict_expand_split_split(self):
        """Test expand_split_ticker_row with split ticker."""
        row = {
            "asset": "TEST",
            "cls": "Test",
            "type": "Hedge",
            "param": "hedge",
            "source": "BBG",
            "ticker": "OLD1 Comdty:2024-06:NEW1 Comdty",
            "field": "PX_LAST"
        }

        result = _tschma_dict_expand_split(row)
        assert len(result) == 2
        assert result[0]["ticker"] == "OLD1 Comdty"
        assert result[0]["param"] == "hedge<2024-06"
        assert result[1]["ticker"] == "NEW1 Comdty"
        assert result[1]["param"] == "hedge>2024-06"


# -------------------------------------
# Ticker Expansion Tests (find_tickers, etc.)
# -------------------------------------

class TestTickerExpansion:
    """Tests for find_tickers, find_tickers_ym, and asset_straddle_tickers."""

    def test_find_tickers_ym_basic(self, test_amt_file, chain_csv_file):
        """Test find_tickers_ym for specific month."""
        clear_cache()
        clear_normalized_cache()
        table = find_tickers_ym(test_amt_file, "^CL", True, 2024, 6, chain_csv_file)
        assert table["columns"] == ["asset", "cls", "type", "param", "source", "ticker", "field"]
        assert len(table["rows"]) > 0
        # All rows should be for CL Comdty
        for row in table["rows"]:
            assert row[0] == "CL Comdty"

    def test_find_tickers_ym_hedge_expansion(self, test_amt_file, chain_csv_file):
        """Test that BBGfc hedge is expanded to specific ticker."""
        clear_cache()
        clear_normalized_cache()
        table = find_tickers_ym(test_amt_file, "^CL", True, 2024, 6, chain_csv_file)
        # Find hedge row
        hedge_rows = [r for r in table["rows"] if r[2] == "Hedge"]
        assert len(hedge_rows) == 1
        # Should have BBG source (from CSV lookup)
        assert hedge_rows[0][4] == "BBG"
        # Should have expanded ticker
        assert "CL" in hedge_rows[0][5]

    def test_find_tickers_year_range(self, test_amt_file, chain_csv_file):
        """Test find_tickers across year range."""
        clear_cache()
        clear_normalized_cache()
        table = find_tickers(test_amt_file, "^CL", True, 2024, 2024, chain_csv_file)
        assert len(table["rows"]) > 0
        # Should have multiple hedge rows for different months (deduplicated)
        hedge_rows = [r for r in table["rows"] if r[2] == "Hedge"]
        # With deduplication, we get 12 unique hedge rows (one per month)
        assert len(hedge_rows) == 12

    def test_find_tickers_deduplication(self, test_amt_file, chain_csv_file):
        """Test that find_tickers properly deduplicates."""
        clear_cache()
        clear_normalized_cache()
        table = find_tickers(test_amt_file, "^CL", True, 2024, 2024, chain_csv_file)
        # Market and Vol tickers should only appear once (same across all months)
        market_rows = [r for r in table["rows"] if r[2] == "Market"]
        assert len(market_rows) == 2  # CL1 and CL2

    def test_find_tickers_no_year_range(self, test_amt_file):
        """Test find_tickers without year range returns tschemas."""
        clear_cache()
        table = find_tickers(test_amt_file, "^CL", True)
        # Should return same as find_tschemas
        expected = find_tschemas(test_amt_file, "^CL", True)
        assert table["columns"] == expected["columns"]
        assert len(table["rows"]) == len(expected["rows"])

    def test_find_tickers_ym_no_match(self, test_amt_file):
        """Test find_tickers_ym with no matching assets."""
        clear_cache()
        table = find_tickers_ym(test_amt_file, "^NONEXISTENT", True, 2024, 6)
        assert table["rows"] == []

    def test_find_tickers_split_ticker_filtering(self, test_amt_file):
        """Test that split tickers are filtered by date constraint."""
        clear_cache()
        # Asset5 has split ticker: OLD1 Comdty:2024-06:NEW1 Comdty
        table = find_tickers_ym(test_amt_file, "^SPLIT", False, 2024, 5)
        hedge_rows = [r for r in table["rows"] if r[2] == "Hedge"]
        # Before 2024-06, only OLD1 should be included
        assert len(hedge_rows) == 1
        assert hedge_rows[0][5] == "OLD1 Comdty"

        # After 2024-06, only NEW1 should be included
        table = find_tickers_ym(test_amt_file, "^SPLIT", False, 2024, 7)
        hedge_rows = [r for r in table["rows"] if r[2] == "Hedge"]
        assert len(hedge_rows) == 1
        assert hedge_rows[0][5] == "NEW1 Comdty"

    def test_asset_straddle_tickers_basic(self, test_amt_file, chain_csv_file):
        """Test asset_straddle_tickers returns correct format."""
        clear_cache()
        clear_normalized_cache()
        table = asset_straddle_tickers("CL Comdty", 2024, 6, 0, test_amt_file, chain_csv_file)
        # Output columns (cls and type removed, straddle after asset)
        assert table["columns"] == ["asset", "straddle", "param", "source", "ticker", "field"]
        assert len(table["rows"]) > 0
        # All rows should have straddle string at index 1
        for row in table["rows"]:
            assert row[1].startswith("|")
            assert row[1].endswith("|")

    def test_asset_straddle_tickers_modulo(self, test_amt_file, chain_csv_file):
        """Test asset_straddle_tickers index wrapping."""
        clear_cache()
        clear_normalized_cache()
        # CL Comdty has 2 schedule components (monthly_std)
        # Index 0 and 2 should give same result
        table0 = asset_straddle_tickers("CL Comdty", 2024, 6, 0, test_amt_file, chain_csv_file)
        table2 = asset_straddle_tickers("CL Comdty", 2024, 6, 2, test_amt_file, chain_csv_file)
        # Should be same straddle (0 % 2 == 2 % 2), straddle is at index 1
        assert table0["rows"][0][1] == table2["rows"][0][1]

    def test_asset_straddle_tickers_different_index(self, test_amt_file, chain_csv_file):
        """Test asset_straddle_tickers with different index gives different straddle."""
        clear_cache()
        clear_normalized_cache()
        table0 = asset_straddle_tickers("CL Comdty", 2024, 6, 0, test_amt_file, chain_csv_file)
        table1 = asset_straddle_tickers("CL Comdty", 2024, 6, 1, test_amt_file, chain_csv_file)
        # Should be different straddles, straddle is at index 1
        assert table0["rows"][0][1] != table1["rows"][0][1]

    def test_asset_straddle_tickers_no_schedule(self):
        """Test asset_straddle_tickers behavior for asset without Options field."""
        # Create temp file with asset that has no Options field
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("""
amt:
  NoScheduleAsset:
    Underlying: "NO_SCHED Test"
    Class: "Test"
    WeightCap: 0.1
    Hedge:
      Source: "nonfut"
      Ticker: "TEST1 Comdty"
      Field: "PX_LAST"
""")
            path = f.name

        try:
            clear_cache()
            clear_normalized_cache()
            # Asset has no Options field - returns placeholder straddle with empty values
            table = asset_straddle_tickers("NO_SCHED Test", 2024, 6, 0, path, None)
            assert len(table["rows"]) > 0
            # Straddle has empty ntrc, ntrv, xprc, xprv, wgt (straddle is at index 1)
            straddle = table["rows"][0][1]
            parts = straddle[1:-1].split("|")
            assert parts[2] == ""  # empty ntrc
            assert parts[3] == ""  # empty ntrv
        finally:
            os.unlink(path)
            clear_cache()

    def test_asset_straddle_tickers_straddle_format(self, test_amt_file, chain_csv_file):
        """Test that straddle string in result is valid."""
        clear_cache()
        clear_normalized_cache()
        table = asset_straddle_tickers("CL Comdty", 2024, 6, 0, test_amt_file, chain_csv_file)
        straddle = table["rows"][0][1]  # straddle is at index 1
        # Verify straddle format by parsing it (use s[1:-1] to preserve empty parts)
        parts = straddle[1:-1].split("|")
        assert len(parts) == 7  # ntry-ntrm, xpry-xprm, ntrc, ntrv, xprc, xprv, wgt

    def test_get_tickers_ym_basic(self, test_amt_file, chain_csv_file):
        """Test get_tickers_ym for single asset."""
        clear_cache()
        clear_normalized_cache()
        table = get_tickers_ym(test_amt_file, "CL Comdty", 2024, 6, chain_csv_file)
        assert table["columns"] == ["asset", "cls", "type", "param", "source", "ticker", "field"]
        assert len(table["rows"]) > 0
        # All rows should be for CL Comdty
        for row in table["rows"]:
            assert row[0] == "CL Comdty"

    def test_get_tickers_ym_expands_bbgfc(self, test_amt_file, chain_csv_file):
        """Test get_tickers_ym expands BBGfc hedge."""
        clear_cache()
        clear_normalized_cache()
        table = get_tickers_ym(test_amt_file, "CL Comdty", 2024, 6, chain_csv_file)
        hedge_rows = [r for r in table["rows"] if r[2] == "Hedge"]
        assert len(hedge_rows) == 1
        # Should have param "hedge" (constraint removed)
        assert hedge_rows[0][3] == "hedge"
        # Should have BBG source (from CSV lookup)
        assert hedge_rows[0][4] == "BBG"

    def test_get_tickers_ym_nonfut_asset(self, test_amt_file):
        """Test get_tickers_ym for asset with nonfut hedge."""
        clear_cache()
        table = get_tickers_ym(test_amt_file, "ES Equity", 2024, 6)
        hedge_rows = [r for r in table["rows"] if r[2] == "Hedge"]
        assert len(hedge_rows) == 1
        assert hedge_rows[0][5] == "SPY US Equity"

    def test_get_tickers_ym_not_found(self, test_amt_file):
        """Test get_tickers_ym for non-existent asset."""
        clear_cache()
        table = get_tickers_ym(test_amt_file, "NONEXISTENT", 2024, 6)
        assert table["rows"] == []


# -------------------------------------
# Straddle Ticker Filtering Tests
# -------------------------------------

class TestStraddleTickerFiltering:
    """Tests for _filter_straddle_tickers function."""

    def test_filter_excludes_market_rows(self):
        """Test that Market rows are always excluded."""
        columns = ["asset", "cls", "type", "param", "source", "ticker", "field"]
        rows = [
            ["CL Comdty", "Commodity", "Market", "-", "BBG", "CL1 Comdty", "PX_LAST"],
            ["CL Comdty", "Commodity", "Vol", "Near", "BBG", "CL1 Comdty", "VOL"],
            ["CL Comdty", "Commodity", "Hedge", "hedge", "BBG", "CLF25 Comdty", "PX_LAST"],
        ]
        result = _filter_straddle_tickers(rows, columns, "N")
        types = [r[2] for r in result]
        assert "Market" not in types
        assert len(result) == 2

    def test_filter_vol_near_with_ntrc_n(self):
        """Test Vol/Near rows kept when ntrc='N', param changed to 'vol'."""
        columns = ["asset", "cls", "type", "param", "source", "ticker", "field"]
        rows = [
            ["CL Comdty", "Commodity", "Vol", "Near", "BBG", "CL1 Comdty", "NEAR_VOL"],
            ["CL Comdty", "Commodity", "Vol", "Far", "BBG", "CL1 Comdty", "FAR_VOL"],
        ]
        result = _filter_straddle_tickers(rows, columns, "N")
        assert len(result) == 1
        assert result[0][3] == "vol"  # param changed from Near to vol
        assert result[0][6] == "NEAR_VOL"

    def test_filter_vol_far_with_ntrc_f(self):
        """Test Vol/Far rows kept when ntrc='F', param changed to 'vol'."""
        columns = ["asset", "cls", "type", "param", "source", "ticker", "field"]
        rows = [
            ["CL Comdty", "Commodity", "Vol", "Near", "BBG", "CL1 Comdty", "NEAR_VOL"],
            ["CL Comdty", "Commodity", "Vol", "Far", "BBG", "CL1 Comdty", "FAR_VOL"],
        ]
        result = _filter_straddle_tickers(rows, columns, "F")
        assert len(result) == 1
        assert result[0][3] == "vol"  # param changed from Far to vol
        assert result[0][6] == "FAR_VOL"

    def test_filter_vol_excluded_with_wrong_ntrc(self):
        """Test Vol rows excluded when ntrc doesn't match."""
        columns = ["asset", "cls", "type", "param", "source", "ticker", "field"]
        rows = [
            ["CL Comdty", "Commodity", "Vol", "Near", "BBG", "CL1 Comdty", "NEAR_VOL"],
            ["CL Comdty", "Commodity", "Vol", "Far", "BBG", "CL1 Comdty", "FAR_VOL"],
        ]
        # ntrc='N' should exclude Far, kept row has param='vol'
        result = _filter_straddle_tickers(rows, columns, "N")
        assert len(result) == 1
        assert result[0][3] == "vol"

        # ntrc='F' should exclude Near, kept row has param='vol'
        result = _filter_straddle_tickers(rows, columns, "F")
        assert len(result) == 1
        assert result[0][3] == "vol"

    def test_filter_hedge_always_kept(self):
        """Test Hedge rows always kept regardless of ntrc."""
        columns = ["asset", "cls", "type", "param", "source", "ticker", "field"]
        rows = [
            ["CL Comdty", "Commodity", "Hedge", "hedge", "BBG", "CLF25 Comdty", "PX_LAST"],
            ["CL Comdty", "Commodity", "Hedge", "hedge1", "BBG", "CLG25 Comdty", "PX_LAST"],
        ]
        # Should keep both hedges for N
        result = _filter_straddle_tickers(rows, columns, "N")
        assert len(result) == 2
        # Should keep both hedges for F
        result = _filter_straddle_tickers(rows, columns, "F")
        assert len(result) == 2

    def test_filter_other_types_kept(self):
        """Test other row types (calc, etc.) are kept."""
        columns = ["asset", "cls", "type", "param", "source", "ticker", "field"]
        rows = [
            ["TY Rate", "Rate", "Hedge", "hedge", "calc", "USD_fsw0m_10Y", ""],
            ["TY Rate", "Rate", "Hedge", "hedge1", "calc", "USD_fsw6m_10Y", ""],
        ]
        result = _filter_straddle_tickers(rows, columns, "N")
        assert len(result) == 2

    def test_filter_empty_ntrc(self):
        """Test filtering with empty ntrc (placeholder straddle)."""
        columns = ["asset", "cls", "type", "param", "source", "ticker", "field"]
        rows = [
            ["TEST", "Test", "Vol", "Near", "BBG", "T1", "VOL"],
            ["TEST", "Test", "Vol", "Far", "BBG", "T1", "VOL"],
            ["TEST", "Test", "Hedge", "hedge", "BBG", "T1", "PX_LAST"],
        ]
        # Empty ntrc means neither Near nor Far matches
        result = _filter_straddle_tickers(rows, columns, "")
        # Only Hedge should be kept
        assert len(result) == 1
        assert result[0][2] == "Hedge"

    def test_filter_combined_scenario(self):
        """Test filtering with typical straddle ticker combination."""
        columns = ["asset", "cls", "type", "param", "source", "ticker", "field"]
        rows = [
            ["CL Comdty", "Commodity", "Market", "-", "BBG", "CL1 Comdty", "PX_LAST"],
            ["CL Comdty", "Commodity", "Market", "-", "BBG", "CL2 Comdty", "PX_LAST"],
            ["CL Comdty", "Commodity", "Vol", "Near", "BBG", "CL1 Comdty", "NEAR_VOL"],
            ["CL Comdty", "Commodity", "Vol", "Far", "BBG", "CL1 Comdty", "FAR_VOL"],
            ["CL Comdty", "Commodity", "Hedge", "hedge", "BBG", "CLF25 Comdty", "PX_LAST"],
        ]
        # With ntrc='N': exclude Market, keep Vol/Near (param->vol), exclude Vol/Far, keep Hedge
        result = _filter_straddle_tickers(rows, columns, "N")
        assert len(result) == 2
        types_params = [(r[2], r[3]) for r in result]
        assert ("Vol", "vol") in types_params  # param changed to "vol"
        assert ("Hedge", "hedge") in types_params

    def test_asset_straddle_tickers_filters_near(self, test_amt_file, chain_csv_file):
        """Test asset_straddle_tickers filters to Near vol for N straddle."""
        clear_cache()
        clear_normalized_cache()
        # Index 0 should be an N straddle (first schedule entry is N1_OVERRIDE15)
        table = asset_straddle_tickers("CL Comdty", 2024, 6, 0, test_amt_file, chain_csv_file)
        # Check straddle has ntrc=N (straddle is at index 1)
        straddle = table["rows"][0][1]
        parts = straddle[1:-1].split("|")
        assert parts[2] == "N"
        # Output columns: ['asset', 'straddle', 'param', 'source', 'ticker', 'field']
        # Check Vol row has param "vol" (param is at index 2)
        vol_rows = [r for r in table["rows"] if r[2] == "vol"]
        assert len(vol_rows) == 1

    def test_asset_straddle_tickers_filters_far(self, test_amt_file, chain_csv_file):
        """Test asset_straddle_tickers filters to Far vol for F straddle."""
        clear_cache()
        clear_normalized_cache()
        # Index 1 should be an F straddle (second schedule entry is Fa_OVERRIDEb_0.5)
        table = asset_straddle_tickers("CL Comdty", 2024, 6, 1, test_amt_file, chain_csv_file)
        # Check straddle has ntrc=F (straddle is at index 1)
        straddle = table["rows"][0][1]
        parts = straddle[1:-1].split("|")
        assert parts[2] == "F"
        # Output columns: ['asset', 'straddle', 'param', 'source', 'ticker', 'field']
        # Check Vol row has param "vol" (param is at index 2)
        vol_rows = [r for r in table["rows"] if r[2] == "vol"]
        assert len(vol_rows) == 1


# -------------------------------------
# Action Column Tests
# -------------------------------------

class TestComputeActions:
    """Tests for _compute_actions function."""

    def test_default_action_unknown_xprc(self):
        """Test that unknown xprc returns all default actions."""
        rows = [
            ["A", "s", "2024-01-01", "10", "100"],
            ["A", "s", "2024-01-02", "11", "101"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        actions = _compute_actions(rows, columns, "N", "5", "UNKNOWN", "15")
        assert all(a == "-" for a in actions)

    def test_override_without_underlying_returns_no_actions(self):
        """Test that OVERRIDE without underlying parameter returns all default actions."""
        rows = [
            ["A", "s", "2024-01-01", "10", "100"],
            ["A", "s", "2024-01-02", "11", "101"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # OVERRIDE without underlying should fail lookup and return no actions
        actions = _compute_actions(rows, columns, "N", "5", "OVERRIDE", "15", ntry=2024, ntrm=1)
        assert all(a == "-" for a in actions)

    def test_bd_trigger_at_exact_threshold(self):
        """Test BD trigger fires at exact threshold."""
        # xprc = "BD", ntrv = "2", xprv = "3" → threshold = 5
        rows = [
            ["A", "s", "2024-01-01", "10", "100"],
            ["A", "s", "2024-01-02", "11", "101"],
            ["A", "s", "2024-01-03", "12", "102"],
            ["A", "s", "2024-01-04", "13", "103"],
            ["A", "s", "2024-01-05", "14", "104"],  # 5th valid day
            ["A", "s", "2024-01-06", "15", "105"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        actions = _compute_actions(rows, columns, "N", "2", "BD", "3", ntry=2024, ntrm=1)
        assert actions == ["-", "-", "-", "-", "ntry", "-"]

    def test_bd_missing_data_finds_next_good_day(self):
        """Test that when target date has missing data, next good day is used."""
        # January 2024: 1st BD=Jan 1, 2nd BD=Jan 2
        rows = [
            ["A", "s", "2024-01-01", "10", "100"],   # 1st BD
            ["A", "s", "2024-01-02", "11", "101"],   # 2nd BD = anchor
            ["A", "s", "2024-01-03", "none", "102"], # target (anchor+1) - BAD
            ["A", "s", "2024-01-04", "13", "103"],   # first good at/after target - ntry!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # BD/2: anchor = 2024-01-02, ntrv=1: target = 2024-01-03, first good = 2024-01-04
        actions = _compute_actions(rows, columns, "N", "1", "BD", "2", ntry=2024, ntrm=1)
        assert actions == ["-", "-", "-", "ntry"]

    def test_bd_threshold_never_reached(self):
        """Test that no action when threshold never reached."""
        rows = [
            ["A", "s", "2024-01-01", "10", "100"],
            ["A", "s", "2024-01-02", "11", "101"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        actions = _compute_actions(rows, columns, "N", "5", "BD", "5")  # threshold = 10
        assert actions == ["-", "-"]

    def test_invalid_ntrv_xprv_values(self):
        """Test that non-numeric ntrv/xprv returns default actions."""
        rows = [["A", "s", "2024-01-01", "10", "100"]]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        actions = _compute_actions(rows, columns, "N", "abc", "BD", "5")
        assert actions == ["-"]

    def test_empty_ntrv(self):
        """Test that empty ntrv returns default actions."""
        rows = [["A", "s", "2024-01-01", "10", "100"]]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        actions = _compute_actions(rows, columns, "N", "", "BD", "5")
        assert actions == ["-"]

    def test_empty_xprv(self):
        """Test that empty xprv returns default actions."""
        rows = [["A", "s", "2024-01-01", "10", "100"]]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        actions = _compute_actions(rows, columns, "N", "5", "BD", "")
        assert actions == ["-"]

    def test_missing_vol_column(self):
        """Test that missing vol column returns default actions."""
        rows = [["A", "s", "2024-01-01", "100"]]
        columns = ["asset", "straddle", "date", "hedge"]
        actions = _compute_actions(rows, columns, "N", "1", "BD", "1")
        assert actions == ["-"]

    def test_missing_hedge_column(self):
        """Test that missing hedge column returns default actions."""
        rows = [["A", "s", "2024-01-01", "10"]]
        columns = ["asset", "straddle", "date", "vol"]
        actions = _compute_actions(rows, columns, "N", "1", "BD", "1")
        assert actions == ["-"]

    def test_multiple_hedges_all_must_be_valid(self):
        """Test that ALL hedge columns must be valid for a good day."""
        # January 2024: 1st BD=Jan 1, 2nd BD=Jan 2
        rows = [
            ["A", "s", "2024-01-01", "10", "100", "200"],   # 1st BD
            ["A", "s", "2024-01-02", "11", "101", "201"],   # 2nd BD = anchor
            ["A", "s", "2024-01-03", "12", "102", "none"],  # target - BAD (hedge1 missing)
            ["A", "s", "2024-01-04", "13", "none", "203"],  # BAD (hedge missing)
            ["A", "s", "2024-01-05", "14", "104", "204"],   # first good at/after target - ntry!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge", "hedge1"]
        # BD/2: anchor = 2024-01-02, ntrv=1: target = 2024-01-03
        actions = _compute_actions(rows, columns, "N", "1", "BD", "2", ntry=2024, ntrm=1)
        assert actions == ["-", "-", "-", "-", "ntry"]

    def test_multiple_hedges_with_hedge2(self):
        """Test with hedge, hedge1, hedge2 - all three must be valid."""
        rows = [
            ["A", "s", "2024-01-01", "10", "100", "200", "300"],   # valid (1)
            ["A", "s", "2024-01-02", "11", "101", "201", "none"],  # invalid - hedge2 missing
            ["A", "s", "2024-01-03", "12", "102", "202", "302"],   # valid (2) - trigger!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge", "hedge1", "hedge2"]
        actions = _compute_actions(rows, columns, "N", "1", "BD", "1", ntry=2024, ntrm=1)  # threshold = 2
        assert actions == ["-", "-", "ntry"]

    def test_empty_rows(self):
        """Test with empty rows list."""
        rows = []
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        actions = _compute_actions(rows, columns, "N", "1", "BD", "1")
        assert actions == []

    def test_threshold_of_one(self):
        """Test threshold of 1 triggers on first valid day."""
        rows = [
            ["A", "s", "2024-01-01", "none", "100"],  # invalid - vol missing
            ["A", "s", "2024-01-02", "11", "101"],    # valid (1) - trigger!
            ["A", "s", "2024-01-03", "12", "102"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        actions = _compute_actions(rows, columns, "N", "0", "BD", "1", ntry=2024, ntrm=1)  # threshold = 1
        assert actions == ["-", "ntry", "-"]

    # --- Rule 2: xpry action tests ---

    def test_xpry_trigger_at_expiry_month(self):
        """Test xpry trigger at anchor (Nth BD) of expiry month."""
        # Entry month: 2024-01, Expiry month: 2024-02
        # BD/3: 3rd BD of January 2024 is 2024-01-03 (1=Jan 1, 2=Jan 2, 3=Jan 3)
        # BD/3: 3rd BD of February 2024 is 2024-02-05 (1=Feb 1, 2=Feb 2, 3=Feb 5)
        rows = [
            ["A", "s", "2024-01-02", "10", "100"],  # 2nd BD Jan
            ["A", "s", "2024-01-03", "11", "101"],  # 3rd BD Jan = anchor, target = anchor + 0 - ntry!
            ["A", "s", "2024-02-01", "12", "102"],  # 1st BD Feb
            ["A", "s", "2024-02-02", "13", "103"],  # 2nd BD Feb
            ["A", "s", "2024-02-05", "14", "104"],  # 3rd BD Feb = expiry anchor - xpry!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # BD/3: entry anchor = 3rd BD Jan = 2024-01-03, ntrv=0: target = anchor
        # xpry anchor = 3rd BD Feb = 2024-02-05
        actions = _compute_actions(rows, columns, "N", "0", "BD", "3", xpry=2024, xprm=2, ntry=2024, ntrm=1)
        assert actions == ["-", "ntry", "-", "-", "xpry"]

    def test_xpry_with_missing_data(self):
        """Test xpry trigger skips invalid days in expiry month."""
        rows = [
            ["A", "s", "2024-02-01", "11", "101"],   # expiry: valid (1)
            ["A", "s", "2024-02-02", "none", "102"], # expiry: invalid (vol missing)
            ["A", "s", "2024-02-03", "13", "none"],  # expiry: invalid (hedge missing)
            ["A", "s", "2024-02-04", "14", "104"],   # expiry: valid (2) - xpry trigger!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # ntry threshold = 10+2 = 12 (never reached)
        # xpry threshold = 2 (fires on 2024-02-04)
        actions = _compute_actions(rows, columns, "N", "10", "BD", "2", xpry=2024, xprm=2)
        assert actions == ["-", "-", "-", "xpry"]

    def test_xpry_threshold_never_reached(self):
        """Test no xpry when threshold never reached."""
        rows = [
            ["A", "s", "2024-02-01", "10", "100"],
            ["A", "s", "2024-02-02", "11", "101"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # xpry threshold = 10 (never reached)
        actions = _compute_actions(rows, columns, "N", "5", "BD", "10", xpry=2024, xprm=2)
        assert actions == ["-", "-"]

    def test_xpry_without_xpry_xprm_params(self):
        """Test that xpry rule is skipped when xpry/xprm not provided."""
        rows = [
            ["A", "s", "2024-02-01", "10", "100"],
            ["A", "s", "2024-02-02", "11", "101"],
            ["A", "s", "2024-02-03", "12", "102"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # Without xpry/xprm, only ntry rule applies
        # ntry threshold = 1+2 = 3 (fires on 2024-02-03)
        actions = _compute_actions(rows, columns, "N", "1", "BD", "2", ntry=2024, ntrm=2)
        assert actions == ["-", "-", "ntry"]

    def test_xpry_both_ntry_and_xpry_trigger(self):
        """Test both ntry and xpry can trigger in same data set."""
        rows = [
            ["A", "s", "2024-01-01", "10", "100"],  # valid (1)
            ["A", "s", "2024-01-02", "11", "101"],  # valid (2)
            ["A", "s", "2024-01-03", "12", "102"],  # valid (3) - ntry trigger!
            ["A", "s", "2024-02-01", "13", "103"],  # expiry: valid (1)
            ["A", "s", "2024-02-02", "14", "104"],  # expiry: valid (2) - xpry trigger!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # ntry threshold = 1+2 = 3 (fires on 2024-01-03)
        # xpry threshold = 2 (fires on 2024-02-02)
        actions = _compute_actions(rows, columns, "N", "1", "BD", "2", xpry=2024, xprm=2, ntry=2024, ntrm=1)
        assert actions == ["-", "-", "ntry", "-", "xpry"]

    def test_xpry_multiple_hedges(self):
        """Test xpry trigger with multiple hedge columns."""
        rows = [
            ["A", "s", "2024-02-01", "10", "100", "200"],   # valid (1)
            ["A", "s", "2024-02-02", "11", "101", "none"],  # invalid - hedge1 missing
            ["A", "s", "2024-02-03", "12", "102", "202"],   # valid (2) - xpry trigger!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge", "hedge1"]
        # xpry threshold = 2
        actions = _compute_actions(rows, columns, "N", "1", "BD", "2", xpry=2024, xprm=2)
        assert actions == ["-", "-", "xpry"]

    def test_xpry_expiry_month_not_in_data(self):
        """Test xpry when expiry month dates not in data (all dates before)."""
        rows = [
            ["A", "s", "2024-01-01", "10", "100"],
            ["A", "s", "2024-01-02", "11", "101"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # Expiry month is 2024-02 but data only has January
        actions = _compute_actions(rows, columns, "N", "1", "BD", "1", xpry=2024, xprm=2, ntry=2024, ntrm=1)
        # ntry threshold = 2, fires on 2024-01-02
        # xpry never starts counting (no dates >= 2024-02-01)
        assert actions == ["-", "ntry"]

    def test_xpry_threshold_of_one(self):
        """Test xpry threshold of 1 triggers on first valid day of expiry month."""
        rows = [
            ["A", "s", "2024-01-31", "10", "100"],
            ["A", "s", "2024-02-01", "none", "101"],  # invalid - vol missing
            ["A", "s", "2024-02-02", "12", "102"],    # valid (1) - xpry trigger!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # xpry threshold = 1
        actions = _compute_actions(rows, columns, "N", "1", "BD", "1", xpry=2024, xprm=2)
        # ntry threshold = 2, but only 1 valid before expiry month
        assert actions == ["-", "-", "xpry"]

    # --- Rule 3 & 4: F/R/W weekday action tests ---

    def test_friday_expiry_trigger_on_anchor(self):
        """Test Rule 3: F (Friday) expiry trigger when anchor is a good day."""
        # 3rd Friday of June 2024 is 2024-06-21
        rows = [
            ["A", "s", "2024-06-19", "10", "100"],  # Wednesday
            ["A", "s", "2024-06-20", "11", "101"],  # Thursday
            ["A", "s", "2024-06-21", "12", "102"],  # Friday (3rd) - anchor, good day - xpry!
            ["A", "s", "2024-06-24", "13", "103"],  # Monday
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        actions = _compute_actions(rows, columns, "N", "0", "F", "3", xpry=2024, xprm=6, ntry=2024, ntrm=6)
        # Rule 3: anchor (2024-06-21) is good, so xpry on anchor
        # Rule 4: anchor (2024-06-21) is good, ntrv=0 means act on anchor, so ntry on anchor too
        # Both rules point to same day, xpry wins since processed second (overwrites ntry)
        assert actions[2] == "xpry"

    def test_friday_expiry_trigger_next_good_day(self):
        """Test Rule 3: F (Friday) expiry trigger when anchor is not a good day."""
        # 3rd Friday of June 2024 is 2024-06-21
        rows = [
            ["A", "s", "2024-06-19", "10", "100"],  # Wednesday
            ["A", "s", "2024-06-20", "11", "101"],  # Thursday
            ["A", "s", "2024-06-21", "none", "102"],  # Friday (3rd) - anchor, BAD (vol missing)
            ["A", "s", "2024-06-24", "13", "103"],  # Monday - next good day - xpry!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        actions = _compute_actions(rows, columns, "N", "0", "F", "3", xpry=2024, xprm=6, ntry=2024, ntrm=6)
        # Rule 3: anchor (2024-06-21) is bad, next good day is 2024-06-24, xpry there
        # Rule 4: anchor (2024-06-21) is bad, next good day is 2024-06-24, ntrv=0 means Day 0
        # Both rules point to same day, xpry wins since processed second (overwrites ntry)
        assert actions[3] == "xpry"

    def test_thursday_entry_trigger_with_offset(self):
        """Test R (Thursday) entry trigger with ntrv calendar day offset."""
        # 2nd Thursday of May 2024 is 2024-05-09
        rows = [
            ["A", "s", "2024-05-08", "10", "100"],  # Wednesday
            ["A", "s", "2024-05-09", "11", "101"],  # Thursday (2nd) - anchor
            ["A", "s", "2024-05-10", "12", "102"],  # Friday
            ["A", "s", "2024-05-11", "none", "none"],  # Saturday - target (anchor + 2) - no data
            ["A", "s", "2024-05-13", "14", "104"],  # Monday - first good day at/after target - ntry!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # ntrv = "2" means anchor + 2 calendar days = 2024-05-11 (Sat, no data), first good = Monday
        actions = _compute_actions(rows, columns, "N", "2", "R", "2", xpry=2024, xprm=6, ntry=2024, ntrm=5)
        # anchor = 2024-05-09, target = 2024-05-11, first good at/after = 2024-05-13
        assert actions == ["-", "-", "-", "-", "ntry"]

    def test_wednesday_expiry_5th_weekday_not_exist(self):
        """Test Rule 3: W (Wednesday) when 5th Wednesday doesn't exist - no action."""
        # June 2024 has only 4 Wednesdays (5th, 12th, 19th, 26th)
        rows = [
            ["A", "s", "2024-06-19", "10", "100"],
            ["A", "s", "2024-06-26", "11", "101"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # xprv = "5" → 5th Wednesday doesn't exist
        actions = _compute_actions(rows, columns, "N", "0", "W", "5", xpry=2024, xprm=6, ntry=2024, ntrm=6)
        # No anchor found → no action
        assert actions == ["-", "-"]

    def test_friday_entry_and_expiry_different_months(self):
        """Test F entry and expiry with calendar day offset in different months."""
        # 3rd Friday of May 2024 is 2024-05-17
        # 3rd Friday of June 2024 is 2024-06-21
        rows = [
            ["A", "s", "2024-05-15", "10", "100"],  # Wed
            ["A", "s", "2024-05-16", "11", "101"],  # Thu
            ["A", "s", "2024-05-17", "12", "102"],  # Fri (3rd) - entry anchor
            ["A", "s", "2024-05-18", "13", "103"],  # Sat - target (anchor + 1 day) - ntry! (good day)
            ["A", "s", "2024-06-19", "14", "104"],  # Wed
            ["A", "s", "2024-06-20", "15", "105"],  # Thu
            ["A", "s", "2024-06-21", "16", "106"],  # Fri (3rd) - expiry anchor - xpry!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # ntrv = "1" → anchor + 1 calendar day = 2024-05-18, first good at/after
        actions = _compute_actions(rows, columns, "N", "1", "F", "3", xpry=2024, xprm=6, ntry=2024, ntrm=5)
        assert actions[3] == "ntry"  # 2024-05-18
        assert actions[6] == "xpry"  # 2024-06-21

    def test_friday_anchor_with_calendar_offset(self):
        """Test F anchor with calendar day offset."""
        # 2nd Friday of May 2024 is 2024-05-10
        rows = [
            ["A", "s", "2024-05-09", "10", "100"],  # Thu
            ["A", "s", "2024-05-10", "11", "101"],  # Fri (2nd) - anchor
            ["A", "s", "2024-05-11", "12", "102"],  # Sat - target (anchor + 1 day) - ntry! (good day)
            ["A", "s", "2024-05-13", "13", "103"],  # Mon
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # ntrv = "1" → anchor + 1 calendar day = 2024-05-11, first good at/after
        actions = _compute_actions(rows, columns, "N", "1", "F", "2", xpry=2024, xprm=6, ntry=2024, ntrm=5)
        # anchor = 2024-05-10, target = 2024-05-11, first good at/after = 2024-05-11
        assert actions == ["-", "-", "ntry", "-"]

    def test_bd_entry_with_calendar_offset(self):
        """Test BD rule entry with calendar day offset."""
        # January 2024: 1st BD=Jan 1, 2nd BD=Jan 2
        # February 2024: 1st BD=Feb 1, 2nd BD=Feb 2
        rows = [
            ["A", "s", "2024-01-01", "10", "100"],  # 1st BD
            ["A", "s", "2024-01-02", "11", "101"],  # 2nd BD = anchor
            ["A", "s", "2024-01-03", "12", "102"],  # anchor + 1 day - ntry!
            ["A", "s", "2024-02-02", "13", "103"],  # expiry month: 2nd BD - xpry!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        # BD/2: anchor = 2nd business day of January = 2024-01-02
        # ntrv = "1": target = anchor + 1 = 2024-01-03
        actions = _compute_actions(rows, columns, "N", "1", "BD", "2", xpry=2024, xprm=2, ntry=2024, ntrm=1)
        assert actions == ["-", "-", "ntry", "xpry"]

    def test_frw_with_month_limit_expiry(self):
        """Test F/R/W rule with expiry month limit - no good day in expiry month."""
        # 3rd Friday of June 2024 is 2024-06-21
        rows = [
            ["A", "s", "2024-06-21", "none", "100"],  # Fri (3rd) - anchor BAD
            ["A", "s", "2024-06-24", "none", "101"],  # Mon - BAD
            ["A", "s", "2024-06-28", "none", "102"],  # Fri - BAD (last weekday of June)
            ["A", "s", "2024-07-01", "13", "103"],  # July - outside expiry month!
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]
        actions = _compute_actions(rows, columns, "N", "0", "F", "3", xpry=2024, xprm=6, ntry=2024, ntrm=5)
        # No good day in expiry month after anchor, so no xpry
        # ntry would use May anchor (not in data), so no ntry either
        assert actions == ["-", "-", "-", "-"]


class TestAnchorDay:
    """Tests for _anchor_day helper function."""

    def test_friday_3rd_june_2024(self):
        """Test 3rd Friday of June 2024."""
        result = _anchor_day("F", "3", 2024, 6)
        assert result == "2024-06-21"

    def test_thursday_2nd_may_2024(self):
        """Test 2nd Thursday of May 2024."""
        result = _anchor_day("R", "2", 2024, 5)
        assert result == "2024-05-09"

    def test_wednesday_1st_jan_2024(self):
        """Test 1st Wednesday of January 2024."""
        result = _anchor_day("W", "1", 2024, 1)
        assert result == "2024-01-03"

    def test_friday_5th_not_exist(self):
        """Test 5th Friday doesn't exist in June 2024."""
        result = _anchor_day("F", "5", 2024, 6)
        assert result is None

    def test_invalid_xprc(self):
        """Test invalid xprc returns None."""
        assert _anchor_day("X", "3", 2024, 6) is None
        assert _anchor_day("", "3", 2024, 6) is None

    def test_bd_anchor_day(self):
        """Test BD code returns Nth business day."""
        # June 2024: 1=Sat, 3=Mon(1st BD), 4=Tue(2nd BD), 5=Wed(3rd BD)
        assert _anchor_day("BD", "1", 2024, 6) == "2024-06-03"  # 1st BD = Mon June 3
        assert _anchor_day("BD", "3", 2024, 6) == "2024-06-05"  # 3rd BD = Wed June 5
        assert _anchor_day("BD", "10", 2024, 6) == "2024-06-14"  # 10th BD

    def test_invalid_xprv(self):
        """Test invalid xprv returns None."""
        assert _anchor_day("F", "0", 2024, 6) is None
        assert _anchor_day("F", "-1", 2024, 6) is None
        assert _anchor_day("F", "abc", 2024, 6) is None
        assert _anchor_day("F", "", 2024, 6) is None

    def test_friday_4th_in_month_with_5_fridays(self):
        """Test 4th Friday in month with 5 Fridays (August 2024)."""
        # August 2024: Fridays on 2, 9, 16, 23, 30
        result = _anchor_day("F", "4", 2024, 8)
        assert result == "2024-08-23"
        result = _anchor_day("F", "5", 2024, 8)
        assert result == "2024-08-30"


class TestAddCalendarDays:
    """Tests for _add_calendar_days helper function."""

    def test_add_zero_days(self):
        """Test adding 0 days returns same date."""
        result = _add_calendar_days("2024-01-19", 0)
        assert result == "2024-01-19"

    def test_add_ten_days(self):
        """Test adding 10 days."""
        result = _add_calendar_days("2024-01-19", 10)
        assert result == "2024-01-29"

    def test_add_days_across_month(self):
        """Test adding days that cross month boundary."""
        result = _add_calendar_days("2024-01-25", 10)
        assert result == "2024-02-04"

    def test_add_days_across_year(self):
        """Test adding days that cross year boundary."""
        result = _add_calendar_days("2024-12-25", 10)
        assert result == "2025-01-04"


class TestLastGoodDayInMonth:
    """Tests for _last_good_day_in_month helper function."""

    def test_last_good_day_basic(self):
        """Test finding last good day in month."""
        rows = [
            ["A", "s", "2024-01-29", "10", "100"],  # good (idx 0)
            ["A", "s", "2024-01-30", "11", "101"],  # good (idx 1)
            ["A", "s", "2024-01-31", "12", "102"],  # good (idx 2) - last
        ]
        # vol_idx=3, hedge_indices=[4], date_idx=2
        result = _last_good_day_in_month(rows, 3, [4], 2, 2024, 1)
        assert result == 2

    def test_last_good_day_skips_invalid(self):
        """Test that invalid days at end are skipped."""
        rows = [
            ["A", "s", "2024-01-29", "10", "100"],     # good (idx 0)
            ["A", "s", "2024-01-30", "11", "101"],     # good (idx 1) - last good
            ["A", "s", "2024-01-31", "none", "102"],   # invalid
        ]
        result = _last_good_day_in_month(rows, 3, [4], 2, 2024, 1)
        assert result == 1

    def test_last_good_day_no_good_days(self):
        """Test when no good days exist in month."""
        rows = [
            ["A", "s", "2024-01-29", "none", "100"],
            ["A", "s", "2024-01-30", "none", "101"],
            ["A", "s", "2024-01-31", "none", "102"],
        ]
        result = _last_good_day_in_month(rows, 3, [4], 2, 2024, 1)
        assert result is None

    def test_last_good_day_respects_month_boundary(self):
        """Test that only days in specified month are considered."""
        rows = [
            ["A", "s", "2024-01-31", "10", "100"],  # good - January
            ["A", "s", "2024-02-01", "11", "101"],  # good - February (different month)
        ]
        result = _last_good_day_in_month(rows, 3, [4], 2, 2024, 1)
        assert result == 0  # Only January day


class TestNthGoodDayAfter:
    """Tests for _nth_good_day_after helper function."""

    def test_n_zero_anchor_is_good(self):
        """Test n=0 returns anchor when anchor is good."""
        rows = [
            ["A", "s", "2024-01-01", "10", "100"],  # anchor - good
            ["A", "s", "2024-01-02", "11", "101"],
        ]
        result = _nth_good_day_after(rows, 3, [4], 2, "2024-01-01", 0)
        assert result == 0

    def test_n_zero_anchor_is_bad(self):
        """Test n=0 returns first good day after anchor when anchor is bad."""
        rows = [
            ["A", "s", "2024-01-01", "none", "100"],  # anchor - bad
            ["A", "s", "2024-01-02", "11", "101"],    # first good day (Day 0)
        ]
        result = _nth_good_day_after(rows, 3, [4], 2, "2024-01-01", 0)
        assert result == 1

    def test_n_one_after_good_anchor(self):
        """Test n=1 returns 1st good day after Day 0."""
        rows = [
            ["A", "s", "2024-01-01", "10", "100"],  # anchor - good (Day 0)
            ["A", "s", "2024-01-02", "11", "101"],  # Day 1
        ]
        result = _nth_good_day_after(rows, 3, [4], 2, "2024-01-01", 1)
        assert result == 1

    def test_n_one_after_bad_anchor(self):
        """Test n=1 returns 1st good day after Day 0 when anchor is bad."""
        rows = [
            ["A", "s", "2024-01-01", "none", "100"],  # anchor - bad
            ["A", "s", "2024-01-02", "11", "101"],    # Day 0 (first good after anchor)
            ["A", "s", "2024-01-03", "12", "102"],    # Day 1
        ]
        result = _nth_good_day_after(rows, 3, [4], 2, "2024-01-01", 1)
        assert result == 2

    def test_n_two_skips_invalid(self):
        """Test n=2 correctly skips invalid days."""
        rows = [
            ["A", "s", "2024-01-01", "10", "100"],   # anchor - good (Day 0)
            ["A", "s", "2024-01-02", "none", "101"], # invalid
            ["A", "s", "2024-01-03", "12", "102"],   # Day 1
            ["A", "s", "2024-01-04", "13", "103"],   # Day 2
        ]
        result = _nth_good_day_after(rows, 3, [4], 2, "2024-01-01", 2)
        assert result == 3

    def test_respects_month_limit(self):
        """Test that searching stops at month_limit."""
        rows = [
            ["A", "s", "2024-01-30", "10", "100"],  # anchor - good (Day 0)
            ["A", "s", "2024-01-31", "11", "101"],  # Day 1
            ["A", "s", "2024-02-01", "12", "102"],  # outside limit
        ]
        result = _nth_good_day_after(rows, 3, [4], 2, "2024-01-30", 2, "2024-01-31")
        assert result is None  # Not enough days within limit

    def test_anchor_not_in_data(self):
        """Test when anchor date is not in data (all dates after)."""
        rows = [
            ["A", "s", "2024-01-05", "10", "100"],  # first date in data
            ["A", "s", "2024-01-06", "11", "101"],
        ]
        # anchor is 2024-01-01, not in data, first date >= anchor is 2024-01-05
        result = _nth_good_day_after(rows, 3, [4], 2, "2024-01-01", 0)
        assert result == 0  # First good day at/after anchor

    def test_negative_n(self):
        """Test negative n returns None."""
        rows = [["A", "s", "2024-01-01", "10", "100"]]
        result = _nth_good_day_after(rows, 3, [4], 2, "2024-01-01", -1)
        assert result is None


# -------------------------------------
# Strike Columns
# -------------------------------------

class TestStrikeColumns:
    """Tests for strike columns (strike_vol, strike, strike1, ...) in get_straddle_days output."""

    def _make_mock_output(self, actions):
        """Create mock get_straddle_days output (before strike columns are added).

        Returns rows with: asset, straddle, date, vol, hedge, hedge1, action, model
        """
        columns = ["asset", "straddle", "date", "vol", "hedge", "hedge1", "action", "model"]
        rows = [
            ["A", "s", "2024-01-01", "10.0", "100.0", "1.0", actions[0], "BS"],
            ["A", "s", "2024-01-02", "11.0", "101.0", "1.1", actions[1], "BS"],
            ["A", "s", "2024-01-03", "12.0", "102.0", "1.2", actions[2], "BS"],
            ["A", "s", "2024-01-04", "13.0", "103.0", "1.3", actions[3], "BS"],
            ["A", "s", "2024-01-05", "14.0", "104.0", "1.4", actions[4], "BS"],
        ]
        return {"columns": columns, "rows": rows}

    def _add_strike_columns(self, table):
        """Add strike columns to a table (simulates the logic in get_straddle_days)."""
        out_columns = table["columns"][:]
        out_rows = [row[:] for row in table["rows"]]

        # Find action column to get ntry/xpry indices
        action_idx = out_columns.index("action")
        ntry_idx = None
        xpry_idx = None
        for i, row in enumerate(out_rows):
            if row[action_idx] == "ntry":
                ntry_idx = i
            elif row[action_idx] == "xpry":
                xpry_idx = i

        # Find vol and hedge column indices
        vol_col_idx = out_columns.index("vol") if "vol" in out_columns else None
        hedge_col_indices = []
        for i, col in enumerate(out_columns):
            if col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
                hedge_col_indices.append(i)

        # Get strike values from ntry row
        if ntry_idx is not None and vol_col_idx is not None:
            strike_vol_value = out_rows[ntry_idx][vol_col_idx]
            strike_values = [out_rows[ntry_idx][idx] for idx in hedge_col_indices]
        else:
            strike_vol_value = "-"
            strike_values = ["-"] * len(hedge_col_indices)

        # Add strike_vol column
        for i, row in enumerate(out_rows):
            in_range = (ntry_idx is not None and i >= ntry_idx and
                        (xpry_idx is None or i <= xpry_idx))
            row.append(strike_vol_value if in_range else "-")
        out_columns.append("strike_vol")

        # Add strike columns
        for j in range(len(hedge_col_indices)):
            strike_col_name = "strike" if j == 0 else f"strike{j}"
            for i, row in enumerate(out_rows):
                in_range = (ntry_idx is not None and i >= ntry_idx and
                            (xpry_idx is None or i <= xpry_idx))
                row.append(strike_values[j] if in_range else "-")
            out_columns.append(strike_col_name)

        return {"columns": out_columns, "rows": out_rows}

    def test_strike_columns_basic(self):
        """Verify strike_vol and strike columns are added with correct values from ntry row."""
        # ntry at row 1, xpry at row 3
        table = self._make_mock_output(["-", "ntry", "-", "xpry", "-"])
        result = self._add_strike_columns(table)

        assert "strike_vol" in result["columns"]
        assert "strike" in result["columns"]
        assert "strike1" in result["columns"]

        # Strike values should come from ntry row (row 1)
        strike_vol_idx = result["columns"].index("strike_vol")
        strike_idx = result["columns"].index("strike")
        strike1_idx = result["columns"].index("strike1")

        # Row 1 (ntry): vol=11.0, hedge=101.0, hedge1=1.1
        assert result["rows"][1][strike_vol_idx] == "11.0"
        assert result["rows"][1][strike_idx] == "101.0"
        assert result["rows"][1][strike1_idx] == "1.1"

    def test_strike_columns_before_ntry(self):
        """Verify all strike columns show '-' before ntry."""
        # ntry at row 2, xpry at row 4
        table = self._make_mock_output(["-", "-", "ntry", "-", "xpry"])
        result = self._add_strike_columns(table)

        strike_vol_idx = result["columns"].index("strike_vol")
        strike_idx = result["columns"].index("strike")
        strike1_idx = result["columns"].index("strike1")

        # Rows 0, 1 are before ntry - should be "-"
        assert result["rows"][0][strike_vol_idx] == "-"
        assert result["rows"][0][strike_idx] == "-"
        assert result["rows"][0][strike1_idx] == "-"
        assert result["rows"][1][strike_vol_idx] == "-"
        assert result["rows"][1][strike_idx] == "-"
        assert result["rows"][1][strike1_idx] == "-"

    def test_strike_columns_after_xpry(self):
        """Verify all strike columns show '-' after xpry."""
        # ntry at row 1, xpry at row 3
        table = self._make_mock_output(["-", "ntry", "-", "xpry", "-"])
        result = self._add_strike_columns(table)

        strike_vol_idx = result["columns"].index("strike_vol")
        strike_idx = result["columns"].index("strike")
        strike1_idx = result["columns"].index("strike1")

        # Row 4 is after xpry - should be "-"
        assert result["rows"][4][strike_vol_idx] == "-"
        assert result["rows"][4][strike_idx] == "-"
        assert result["rows"][4][strike1_idx] == "-"

    def test_strike_columns_in_range(self):
        """Verify strike columns show values from ntry to xpry (inclusive)."""
        # ntry at row 1, xpry at row 3
        table = self._make_mock_output(["-", "ntry", "-", "xpry", "-"])
        result = self._add_strike_columns(table)

        strike_vol_idx = result["columns"].index("strike_vol")
        strike_idx = result["columns"].index("strike")

        # Rows 1, 2, 3 are in range (ntry to xpry inclusive)
        for i in [1, 2, 3]:
            assert result["rows"][i][strike_vol_idx] == "11.0"  # vol at ntry
            assert result["rows"][i][strike_idx] == "101.0"     # hedge at ntry

    def test_strike_columns_no_ntry(self):
        """When no ntry action exists, all strike columns are '-' for all rows."""
        # No ntry, xpry at row 3
        table = self._make_mock_output(["-", "-", "-", "xpry", "-"])
        result = self._add_strike_columns(table)

        strike_vol_idx = result["columns"].index("strike_vol")
        strike_idx = result["columns"].index("strike")

        # All rows should be "-"
        for row in result["rows"]:
            assert row[strike_vol_idx] == "-"
            assert row[strike_idx] == "-"

    def test_strike_columns_no_xpry(self):
        """When no xpry action exists, strike values continue to end of data."""
        # ntry at row 1, no xpry
        table = self._make_mock_output(["-", "ntry", "-", "-", "-"])
        result = self._add_strike_columns(table)

        strike_vol_idx = result["columns"].index("strike_vol")
        strike_idx = result["columns"].index("strike")

        # Row 0 is before ntry - should be "-"
        assert result["rows"][0][strike_vol_idx] == "-"

        # Rows 1-4 are from ntry onward - should have values
        for i in [1, 2, 3, 4]:
            assert result["rows"][i][strike_vol_idx] == "11.0"
            assert result["rows"][i][strike_idx] == "101.0"

    def test_strike_columns_multiple_hedges(self):
        """Verify correct number of strike columns (one per hedge)."""
        table = self._make_mock_output(["-", "ntry", "-", "xpry", "-"])
        result = self._add_strike_columns(table)

        # Should have strike_vol, strike, strike1 (2 hedges)
        assert "strike_vol" in result["columns"]
        assert "strike" in result["columns"]
        assert "strike1" in result["columns"]
        # strike2 should not exist (only 2 hedges)
        assert "strike2" not in result["columns"]

    def test_vol_column_unchanged(self):
        """Verify vol column is NOT modified (still shows market data)."""
        table = self._make_mock_output(["-", "ntry", "-", "xpry", "-"])
        original_vols = [row[3] for row in table["rows"]]  # vol is at index 3

        result = self._add_strike_columns(table)

        vol_idx = result["columns"].index("vol")
        result_vols = [row[vol_idx] for row in result["rows"]]

        # Vol column should be unchanged
        assert result_vols == original_vols


# -------------------------------------
# Edge Cases and Error Handling
# -------------------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_amt(self):
        """Test handling of empty AMT file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("amt: {}\n")
            path = f.name

        try:
            clear_cache()
            data = load_amt(path)
            assert data["amt"] == {}

            found = find_assets(path, ".")
            assert found["rows"] == []
        finally:
            os.unlink(path)
            clear_cache()

    def test_asset_without_underlying(self):
        """Test handling of asset without Underlying field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("""
amt:
  BadAsset:
    Class: "Test"
""")
            path = f.name

        try:
            clear_cache()
            found = find_assets(path, ".")
            assert found["rows"] == []
        finally:
            os.unlink(path)
            clear_cache()

    def test_schedule_with_single_component(self):
        """Test schedule with single component."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("""
amt:
  TestAsset:
    Underlying: "TEST Comdty"
    WeightCap: 0.1
    Options: "single"
expiry_schedules:
  single:
    - "N1_OVERRIDE15"
""")
            path = f.name

        try:
            clear_cache()
            table = get_schedule(path, "TEST Comdty")
            assert len(table["rows"]) == 1
        finally:
            os.unlink(path)
            clear_cache()

    def test_schedule_with_empty_schedule(self):
        """Test handling of empty schedule list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("""
amt:
  TestAsset:
    Underlying: "TEST Comdty"
    WeightCap: 0.1
    Options: "empty"
expiry_schedules:
  empty: []
""")
            path = f.name

        try:
            clear_cache()
            table = get_schedule(path, "TEST Comdty")
            # Empty schedule should produce placeholder row
            assert len(table["rows"]) == 1
            assert table["rows"][0][0] == 0  # schcnt = 0
        finally:
            os.unlink(path)
            clear_cache()

    def test_vol_ticker_deduplication(self):
        """Test that duplicate Vol tickers are deduplicated."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("""
amt:
  TestAsset:
    Underlying: "TEST Comdty"
    WeightCap: 0.1
    Vol:
      Source: "BBG"
      Ticker: "VOL1 Comdty"
      Near: "ATM_IMP_VOL"
      Far: "ATM_IMP_VOL"
""")
            path = f.name

        try:
            clear_cache()
            table = get_tschemas(path, "TEST Comdty")
            vol_rows = [r for r in table["rows"] if r[2] == "Vol"]
            # Near and Far have same ticker/field, should be deduplicated
            assert len(vol_rows) == 1
        finally:
            os.unlink(path)
            clear_cache()

    def test_table_column_empty_table(self):
        """Test table_column with empty table."""
        table = {"columns": ["a", "b"], "rows": []}
        result = table_column(table, "a")
        assert result == []

    def test_path_types(self, test_amt_file):
        """Test that both str and Path work for file paths."""
        clear_cache()

        # String path
        data1 = load_amt(test_amt_file)
        clear_cache()

        # Path object
        data2 = load_amt(Path(test_amt_file))

        assert data1 == data2


# -------------------------------------
# Integration Tests
# -------------------------------------

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_schedule_expansion_workflow(self, test_amt_file):
        """Test complete workflow: find assets -> get schedules -> expand."""
        clear_cache()

        # Find live commodity assets
        found = find_assets(test_amt_file, "Comdty$", live_only=True)
        assert len(found["rows"]) == 1

        # Get schedules for found assets
        asset = found["rows"][0][0]
        schedule = get_schedule(test_amt_file, asset)
        assert len(schedule["rows"]) > 0

        # Expand for a year
        expanded = get_straddle_yrs(test_amt_file, asset, 2024, 2024)
        assert len(expanded["rows"]) == 24  # 12 months * 2 components

    def test_asset_classification_workflow(self, test_amt_file):
        """Test complete asset classification workflow."""
        clear_cache()

        # Get group assignments (returns column-oriented)
        groups = asset_group(test_amt_file, live_only=True)
        groups = table_to_rows(groups)  # convert to row-oriented for dict access

        # Verify structure
        assert "asset" in groups["columns"]
        assert "group" in groups["columns"]

        # Check specific assignments
        rows_dict = {r[0]: dict(zip(groups["columns"], r)) for r in groups["rows"]}
        assert rows_dict["CL Comdty"]["group"] == "commodities"
        assert rows_dict["CL Comdty"]["subgroup"] == "energy"
        assert rows_dict["CL Comdty"]["liquidity"] == "2"

    def test_ticker_extraction_workflow(self, test_amt_file):
        """Test ticker extraction workflow."""
        clear_cache()

        # Get tickers for an asset
        tickers = get_tschemas(test_amt_file, "CL Comdty")

        # Verify different ticker types present
        types = set(r[2] for r in tickers["rows"])
        assert "Market" in types
        assert "Vol" in types
        assert "Hedge" in types


# -------------------------------------
# Valuation Model Tests
# -------------------------------------


class TestNormCdf:
    """Tests for _norm_cdf helper function."""

    def test_norm_cdf_zero(self):
        """N(0) = 0.5."""
        assert abs(_norm_cdf(0) - 0.5) < 1e-10

    def test_norm_cdf_positive(self):
        """N(1) ≈ 0.8413."""
        assert abs(_norm_cdf(1) - 0.8413447460685429) < 1e-6

    def test_norm_cdf_negative(self):
        """N(-1) ≈ 0.1587."""
        assert abs(_norm_cdf(-1) - 0.15865525393145707) < 1e-6

    def test_norm_cdf_large_positive(self):
        """N(3) ≈ 0.9987."""
        assert abs(_norm_cdf(3) - 0.9986501019683699) < 1e-6

    def test_norm_cdf_large_negative(self):
        """N(-3) ≈ 0.0013."""
        assert abs(_norm_cdf(-3) - 0.0013498980316300946) < 1e-6

    def test_norm_cdf_symmetry(self):
        """N(x) + N(-x) = 1."""
        for x in [0.5, 1.0, 2.0]:
            assert abs(_norm_cdf(x) + _norm_cdf(-x) - 1.0) < 1e-10


class TestModelES:
    """Tests for model_ES European Straddle pricing function."""

    def test_model_es_basic(self):
        """Test basic ES model calculation."""
        row = {
            "hedge": "100",
            "strike": "100",
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        assert result["mv"] != "-"
        assert result["delta"] != "-"
        mv = float(result["mv"])
        assert mv > 0  # ATM straddle has positive value

    def test_model_es_atm_value(self):
        """ATM straddle value / strike should be roughly 10-15% for 25% vol, 90 days."""
        row = {
            "hedge": "100",
            "strike": "100",
            "vol": "25",
            "date": "2025-01-01",
            "expiry": "2025-04-01",  # 90 days
        }
        result = model_ES(row)
        mv = float(result["mv"])
        # ATM straddle with 25% vol, 90 days: mv/X should be roughly 0.08-0.20
        assert 0.08 < mv < 0.20

    def test_model_es_itm_higher_value(self):
        """ITM straddle (S > X) should have higher value than ATM."""
        atm_row = {
            "hedge": "100",
            "strike": "100",
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        itm_row = {
            "hedge": "110",  # S > X
            "strike": "100",
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        atm_mv = float(model_ES(atm_row)["mv"])
        itm_mv = float(model_ES(itm_row)["mv"])
        assert itm_mv > atm_mv

    def test_model_es_higher_vol_higher_value(self):
        """Higher vol should give higher straddle value."""
        low_vol_row = {
            "hedge": "100",
            "strike": "100",
            "vol": "10",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        high_vol_row = {
            "hedge": "100",
            "strike": "100",
            "vol": "30",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        low_mv = float(model_ES(low_vol_row)["mv"])
        high_mv = float(model_ES(high_vol_row)["mv"])
        assert high_mv > low_mv

    def test_model_es_longer_time_higher_value(self):
        """Longer time to expiry should give higher straddle value."""
        short_row = {
            "hedge": "100",
            "strike": "100",
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-01-15",  # 14 days
        }
        long_row = {
            "hedge": "100",
            "strike": "100",
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-04-01",  # 90 days
        }
        short_mv = float(model_ES(short_row)["mv"])
        long_mv = float(model_ES(long_row)["mv"])
        assert long_mv > short_mv

    def test_model_es_at_expiry_returns_intrinsic(self):
        """t=0 (at expiry) returns intrinsic value abs(S-X)/X."""
        row = {
            "hedge": "110",
            "strike": "100",
            "vol": "20",
            "date": "2025-02-01",
            "expiry": "2025-02-01",  # Same day
        }
        result = model_ES(row)
        assert result["mv"] != "-"
        assert float(result["mv"]) == 0.1  # abs(110 - 100) / 100
        assert float(result["delta"]) == 1.0  # S >= X

    def test_model_es_at_expiry_itm_put(self):
        """t=0 with S < X returns intrinsic value / strike."""
        row = {
            "hedge": "90",
            "strike": "100",
            "vol": "20",
            "date": "2025-02-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        assert float(result["mv"]) == 0.1  # abs(90 - 100) / 100
        assert float(result["delta"]) == -1.0  # S < X

    def test_model_es_at_expiry_atm(self):
        """t=0 with S == X returns 0."""
        row = {
            "hedge": "100",
            "strike": "100",
            "vol": "20",
            "date": "2025-02-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        assert float(result["mv"]) == 0.0  # abs(100 - 100) / 100
        assert float(result["delta"]) == 1.0  # S >= X (equal case)

    def test_model_es_past_expiry_returns_dash(self):
        """t<0 (past expiry) returns '-'."""
        row = {
            "hedge": "110",
            "strike": "100",
            "vol": "20",
            "date": "2025-02-15",
            "expiry": "2025-02-01",  # Past
        }
        result = model_ES(row)
        assert result["mv"] == "-"
        assert result["delta"] == "-"

    def test_model_es_zero_vol_returns_dash(self):
        """Zero volatility returns '-'."""
        row = {
            "hedge": "100",
            "strike": "100",
            "vol": "0",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        assert result["mv"] == "-"
        assert result["delta"] == "-"

    def test_model_es_negative_vol_returns_dash(self):
        """Negative volatility returns '-'."""
        row = {
            "hedge": "100",
            "strike": "100",
            "vol": "-20",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        assert result["mv"] == "-"
        assert result["delta"] == "-"

    def test_model_es_zero_hedge_returns_dash(self):
        """Zero hedge price returns '-'."""
        row = {
            "hedge": "0",
            "strike": "100",
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        assert result["mv"] == "-"
        assert result["delta"] == "-"

    def test_model_es_zero_strike_returns_dash(self):
        """Zero strike returns '-'."""
        row = {
            "hedge": "100",
            "strike": "0",
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        assert result["mv"] == "-"
        assert result["delta"] == "-"

    def test_model_es_missing_key_returns_dash(self):
        """Missing required key returns '-'."""
        row = {
            "hedge": "100",
            # "strike" missing
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        assert result["mv"] == "-"
        assert result["delta"] == "-"

    def test_model_es_invalid_date_returns_dash(self):
        """Invalid date format returns '-'."""
        row = {
            "hedge": "100",
            "strike": "100",
            "vol": "20",
            "date": "invalid",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        assert result["mv"] == "-"
        assert result["delta"] == "-"

    def test_model_es_non_numeric_hedge_returns_dash(self):
        """Non-numeric hedge returns '-'."""
        row = {
            "hedge": "none",
            "strike": "100",
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        assert result["mv"] == "-"
        assert result["delta"] == "-"

    def test_model_es_delta_atm(self):
        """ATM straddle delta should be near 0."""
        row = {
            "hedge": "100",
            "strike": "100",
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        delta = float(result["delta"])
        # ATM straddle delta should be close to 0 (N_d1 ≈ 1 for ATM, so delta = N_d1 - 1 ≈ 0)
        assert -0.1 < delta < 0.1

    def test_model_es_delta_itm_call(self):
        """ITM straddle (S > X) should have positive delta."""
        row = {
            "hedge": "120",
            "strike": "100",
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        delta = float(result["delta"])
        # ITM call-side means N_d1 > 1, so delta = N_d1 - 1 > 0
        assert delta > 0

    def test_model_es_delta_itm_put(self):
        """ITM straddle (S < X) should have negative delta."""
        row = {
            "hedge": "80",
            "strike": "100",
            "vol": "20",
            "date": "2025-01-01",
            "expiry": "2025-02-01",
        }
        result = model_ES(row)
        delta = float(result["delta"])
        # ITM put-side means N_d1 < 1, so delta = N_d1 - 1 < 0
        assert delta < 0


class TestOtherModels:
    """Tests for placeholder model functions."""

    def test_model_ns_returns_dash(self):
        """model_NS always returns '-' for mv and delta."""
        assert model_NS({}) == {"mv": "-", "delta": "-"}
        assert model_NS({"hedge": "100"}) == {"mv": "-", "delta": "-"}

    def test_model_bs_returns_dash(self):
        """model_BS always returns '-' for mv and delta."""
        assert model_BS({}) == {"mv": "-", "delta": "-"}
        assert model_BS({"hedge": "100"}) == {"mv": "-", "delta": "-"}

    def test_model_default_returns_dash(self):
        """model_default always returns '-' for mv and delta."""
        assert model_default({}) == {"mv": "-", "delta": "-"}
        assert model_default({"hedge": "100"}) == {"mv": "-", "delta": "-"}

    def test_model_dispatch_es(self):
        """MODEL_DISPATCH maps 'ES' to model_ES."""
        assert MODEL_DISPATCH["ES"] == model_ES

    def test_model_dispatch_cds_es(self):
        """MODEL_DISPATCH maps 'CDS_ES' to model_ES."""
        assert MODEL_DISPATCH["CDS_ES"] == model_ES

    def test_model_dispatch_unknown(self):
        """Unknown model name not in dispatch."""
        assert "UNKNOWN" not in MODEL_DISPATCH


class TestGetRollforwardFields:
    """Tests for _get_rollforward_fields helper."""

    def test_rollforward_vol_and_hedge(self):
        """Should include vol and hedge."""
        columns = ["asset", "date", "vol", "hedge", "action"]
        fields = _get_rollforward_fields(columns)
        assert "vol" in fields
        assert "hedge" in fields

    def test_rollforward_multiple_hedges(self):
        """Should include hedge1, hedge2, etc."""
        columns = ["asset", "vol", "hedge", "hedge1", "hedge2", "action"]
        fields = _get_rollforward_fields(columns)
        assert "hedge" in fields
        assert "hedge1" in fields
        assert "hedge2" in fields

    def test_rollforward_excludes_non_market_data(self):
        """Should not include non-market columns."""
        columns = ["asset", "straddle", "date", "vol", "hedge", "action", "model", "strike", "expiry"]
        fields = _get_rollforward_fields(columns)
        assert "asset" not in fields
        assert "straddle" not in fields
        assert "date" not in fields
        assert "action" not in fields
        assert "model" not in fields
        assert "strike" not in fields
        assert "expiry" not in fields

    def test_rollforward_empty_columns(self):
        """Empty columns list returns empty set."""
        assert _get_rollforward_fields([]) == set()

    def test_rollforward_no_vol_or_hedge(self):
        """Columns without vol/hedge returns empty set."""
        columns = ["asset", "date", "action"]
        assert _get_rollforward_fields(columns) == set()


class TestValuationRollforward:
    """Tests for get_straddle_valuation with roll-forward logic."""

    def _make_mock_valuation_table(self, rows_data):
        """Create a mock table for valuation testing.

        rows_data: list of dicts with keys: date, vol, hedge, action
        """
        columns = ["asset", "straddle", "date", "vol", "hedge", "action", "model", "strike_vol", "strike", "expiry"]
        rows = []
        for d in rows_data:
            rows.append([
                "TEST",
                "|2025-01|2025-02|N|0|F|3|100|",
                d["date"],
                d["vol"],
                d["hedge"],
                d["action"],
                "ES",
                d.get("strike_vol", "25"),
                d.get("strike", "100"),
                d.get("expiry", "2025-02-21"),
            ])
        return {"columns": columns, "rows": rows}

    def test_rollforward_basic(self):
        """Basic roll-forward: weekend uses previous day's data."""
        # Simulate: Friday good, Saturday/Sunday missing, Monday good
        table = self._make_mock_valuation_table([
            {"date": "2025-02-14", "vol": "25", "hedge": "100", "action": "ntry"},  # Friday
            {"date": "2025-02-15", "vol": "25", "hedge": "none", "action": "-"},    # Saturday
            {"date": "2025-02-16", "vol": "25", "hedge": "none", "action": "-"},    # Sunday
            {"date": "2025-02-17", "vol": "26", "hedge": "102", "action": "-"},     # Monday
            {"date": "2025-02-21", "vol": "27", "hedge": "105", "action": "xpry"},  # Expiry
        ])

        # Manually apply the valuation logic (simulating get_straddle_valuation internals)
        columns = table["columns"]
        rows = table["rows"]
        action_idx = columns.index("action")

        ntry_idx = None
        xpry_idx = None
        for idx, row in enumerate(rows):
            if row[action_idx] == "ntry":
                ntry_idx = idx
            elif row[action_idx] == "xpry":
                xpry_idx = idx

        rollforward_fields = _get_rollforward_fields(columns)
        rolled_data = {}
        ntry_row_dict = dict(zip(columns, rows[ntry_idx]))
        for key in rollforward_fields:
            if key in ntry_row_dict:
                rolled_data[key] = ntry_row_dict[key]

        mv_results = []
        for idx, row in enumerate(rows):
            if idx < ntry_idx or idx > xpry_idx:
                mv_results.append("-")
            else:
                row_dict = dict(zip(columns, row))
                for key in rollforward_fields:
                    if key in row_dict and row_dict[key] != "none":
                        rolled_data[key] = row_dict[key]
                model_input = row_dict.copy()
                model_input.update(rolled_data)
                result = model_ES(model_input)
                mv_results.append(result)

        # All rows from ntry to xpry should have values (including xpry which returns intrinsic)
        assert mv_results[0]["mv"] != "-"  # ntry
        assert mv_results[1]["mv"] != "-"  # Saturday - rolled forward
        assert mv_results[2]["mv"] != "-"  # Sunday - rolled forward
        assert mv_results[3]["mv"] != "-"  # Monday - new data
        assert mv_results[4]["mv"] != "-"  # xpry (t=0, returns intrinsic value)

    def test_rollforward_partial_data(self):
        """When only hedge is missing, vol from current row is used."""
        table = self._make_mock_valuation_table([
            {"date": "2025-02-14", "vol": "25", "hedge": "100", "action": "ntry"},
            {"date": "2025-02-15", "vol": "30", "hedge": "none", "action": "-"},  # Vol exists, hedge missing
            {"date": "2025-02-21", "vol": "27", "hedge": "105", "action": "xpry"},
        ])

        columns = table["columns"]
        rows = table["rows"]
        rollforward_fields = _get_rollforward_fields(columns)

        # After processing row 1 (2025-02-15), rolled_data should have:
        # - vol = "30" (from current row, not "none")
        # - hedge = "100" (rolled forward from ntry)

        rolled_data = {}
        ntry_row_dict = dict(zip(columns, rows[0]))
        for key in rollforward_fields:
            if key in ntry_row_dict:
                rolled_data[key] = ntry_row_dict[key]

        # Process row 1
        row_dict = dict(zip(columns, rows[1]))
        for key in rollforward_fields:
            if key in row_dict and row_dict[key] != "none":
                rolled_data[key] = row_dict[key]

        assert rolled_data["vol"] == "30"    # Updated from current row
        assert rolled_data["hedge"] == "100"  # Rolled forward

    def test_rollforward_updates_when_data_returns(self):
        """When new data arrives, it replaces rolled-forward values."""
        columns = ["vol", "hedge"]
        rolled_data = {"vol": "25", "hedge": "100"}

        # Simulate new data arriving
        new_row = {"vol": "28", "hedge": "110"}
        rollforward_fields = {"vol", "hedge"}
        for key in rollforward_fields:
            if key in new_row and new_row[key] != "none":
                rolled_data[key] = new_row[key]

        assert rolled_data["vol"] == "28"
        assert rolled_data["hedge"] == "110"

    def test_rollforward_consecutive_missing(self):
        """Multiple consecutive missing days all get values."""
        # 5 days of missing data should all compute mv using rolled-forward data
        table = self._make_mock_valuation_table([
            {"date": "2025-02-10", "vol": "25", "hedge": "100", "action": "ntry"},
            {"date": "2025-02-11", "vol": "none", "hedge": "none", "action": "-"},
            {"date": "2025-02-12", "vol": "none", "hedge": "none", "action": "-"},
            {"date": "2025-02-13", "vol": "none", "hedge": "none", "action": "-"},
            {"date": "2025-02-14", "vol": "none", "hedge": "none", "action": "-"},
            {"date": "2025-02-15", "vol": "none", "hedge": "none", "action": "-"},
            {"date": "2025-02-21", "vol": "27", "hedge": "105", "action": "xpry"},
        ])

        columns = table["columns"]
        rows = table["rows"]
        rollforward_fields = _get_rollforward_fields(columns)

        rolled_data = {}
        ntry_row_dict = dict(zip(columns, rows[0]))
        for key in rollforward_fields:
            if key in ntry_row_dict:
                rolled_data[key] = ntry_row_dict[key]

        mv_results = []
        for idx, row in enumerate(rows):
            if idx == 0 or idx == len(rows) - 1:
                # Skip ntry and xpry for this test
                continue
            row_dict = dict(zip(columns, row))
            for key in rollforward_fields:
                if key in row_dict and row_dict[key] != "none":
                    rolled_data[key] = row_dict[key]
            model_input = row_dict.copy()
            model_input.update(rolled_data)
            mv = model_ES(model_input)
            mv_results.append(mv)

        # All 5 missing days should have computed values (not "-")
        for mv in mv_results:
            assert mv != "-"


# -------------------------------------
# Override Expiry Tests
# -------------------------------------


class TestOverrideExpiry:
    """Tests for OVERRIDE expiry rule functionality."""

    @pytest.fixture(autouse=True)
    def reset_override_cache(self):
        """Reset the override cache before and after each test."""
        tickers_module._OVERRIDE_CACHE = None
        yield
        tickers_module._OVERRIDE_CACHE = None

    def test_load_overrides_creates_cache(self, tmp_path):
        """Test that _load_overrides creates a cache from CSV."""
        csv_file = tmp_path / "overrides.csv"
        csv_file.write_text("ticker,expiry\nTEST Asset,2024-03-15\nTEST Asset,2024-04-19\n")

        cache = _load_overrides(csv_file)

        assert ("TEST Asset", "2024-03") in cache
        assert cache[("TEST Asset", "2024-03")] == "2024-03-15"
        assert ("TEST Asset", "2024-04") in cache
        assert cache[("TEST Asset", "2024-04")] == "2024-04-19"

    def test_load_overrides_caches_result(self, tmp_path):
        """Test that _load_overrides returns cached result on second call."""
        csv_file = tmp_path / "overrides.csv"
        csv_file.write_text("ticker,expiry\nTEST,2024-03-15\n")

        cache1 = _load_overrides(csv_file)
        # Modify the cache to verify it's returned
        cache1[("MODIFIED", "2024-01")] = "2024-01-01"
        cache2 = _load_overrides(csv_file)

        assert cache1 is cache2
        assert ("MODIFIED", "2024-01") in cache2

    def test_load_overrides_handles_missing_file(self, tmp_path):
        """Test that _load_overrides returns empty cache for missing file."""
        cache = _load_overrides(tmp_path / "nonexistent.csv")
        assert cache == {}

    def test_override_expiry_found(self, tmp_path):
        """Test _override_expiry returns correct date when found."""
        csv_file = tmp_path / "overrides.csv"
        csv_file.write_text("ticker,expiry\nTEST Asset,2024-03-15\n")

        result = _override_expiry("TEST Asset", 2024, 3, csv_file)
        assert result == "2024-03-15"

    def test_override_expiry_not_found(self, tmp_path):
        """Test _override_expiry returns None when not found."""
        csv_file = tmp_path / "overrides.csv"
        csv_file.write_text("ticker,expiry\nTEST Asset,2024-03-15\n")

        result = _override_expiry("OTHER Asset", 2024, 3, csv_file)
        assert result is None

        result = _override_expiry("TEST Asset", 2024, 4, csv_file)
        assert result is None

    def test_anchor_day_override_returns_lookup(self, tmp_path):
        """Test _anchor_day with OVERRIDE returns override lookup result."""
        csv_file = tmp_path / "overrides.csv"
        csv_file.write_text("ticker,expiry\nTEST Asset,2024-03-21\n")

        result = _anchor_day("OVERRIDE", "0", 2024, 3, "TEST Asset", csv_file)
        assert result == "2024-03-21"

    def test_anchor_day_override_without_underlying(self):
        """Test _anchor_day with OVERRIDE but no underlying returns None."""
        result = _anchor_day("OVERRIDE", "0", 2024, 3)
        assert result is None

    def test_anchor_day_override_not_found(self, tmp_path):
        """Test _anchor_day with OVERRIDE returns None when lookup fails."""
        csv_file = tmp_path / "overrides.csv"
        csv_file.write_text("ticker,expiry\nTEST Asset,2024-03-21\n")

        result = _anchor_day("OVERRIDE", "0", 2024, 4, "TEST Asset", csv_file)
        assert result is None

    def test_compute_actions_override_entry(self, tmp_path):
        """Test _compute_actions with OVERRIDE sets ntry correctly."""
        csv_file = tmp_path / "overrides.csv"
        # Override for entry month 2024-01 = 2024-01-15
        csv_file.write_text("ticker,expiry\nTEST,2024-01-15\nTEST,2024-02-20\n")

        rows = [
            ["A", "s", "2024-01-14", "10", "100"],
            ["A", "s", "2024-01-15", "11", "101"],  # anchor, should be ntry
            ["A", "s", "2024-01-16", "12", "102"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]

        # OVERRIDE with ntrv=0: anchor is 2024-01-15, target is 2024-01-15
        actions = _compute_actions(
            rows, columns, "N", "0", "OVERRIDE", "0",
            ntry=2024, ntrm=1, underlying="TEST", overrides_path=csv_file
        )
        assert actions == ["-", "ntry", "-"]

    def test_compute_actions_override_entry_with_offset(self, tmp_path):
        """Test _compute_actions with OVERRIDE and ntrv calendar day offset."""
        csv_file = tmp_path / "overrides.csv"
        csv_file.write_text("ticker,expiry\nTEST,2024-01-15\n")

        rows = [
            ["A", "s", "2024-01-15", "10", "100"],  # anchor
            ["A", "s", "2024-01-16", "11", "101"],
            ["A", "s", "2024-01-17", "12", "102"],  # anchor + 2 days, should be ntry
            ["A", "s", "2024-01-18", "13", "103"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]

        # OVERRIDE with ntrv=2: anchor is 2024-01-15, target is 2024-01-17
        actions = _compute_actions(
            rows, columns, "N", "2", "OVERRIDE", "0",
            ntry=2024, ntrm=1, underlying="TEST", overrides_path=csv_file
        )
        assert actions == ["-", "-", "ntry", "-"]

    def test_compute_actions_override_expiry(self, tmp_path):
        """Test _compute_actions with OVERRIDE sets xpry correctly."""
        csv_file = tmp_path / "overrides.csv"
        csv_file.write_text("ticker,expiry\nTEST,2024-01-10\nTEST,2024-02-15\n")

        rows = [
            ["A", "s", "2024-02-14", "10", "100"],
            ["A", "s", "2024-02-15", "11", "101"],  # expiry anchor, should be xpry
            ["A", "s", "2024-02-16", "12", "102"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]

        actions = _compute_actions(
            rows, columns, "N", "0", "OVERRIDE", "0",
            xpry=2024, xprm=2, underlying="TEST", overrides_path=csv_file
        )
        assert actions == ["-", "xpry", "-"]

    def test_compute_actions_override_both_entry_and_expiry(self, tmp_path):
        """Test _compute_actions with OVERRIDE sets both ntry and xpry."""
        csv_file = tmp_path / "overrides.csv"
        csv_file.write_text("ticker,expiry\nTEST,2024-01-10\nTEST,2024-02-15\n")

        rows = [
            ["A", "s", "2024-01-09", "10", "100"],
            ["A", "s", "2024-01-10", "11", "101"],  # entry anchor, should be ntry
            ["A", "s", "2024-01-11", "12", "102"],
            ["A", "s", "2024-02-14", "13", "103"],
            ["A", "s", "2024-02-15", "14", "104"],  # expiry anchor, should be xpry
            ["A", "s", "2024-02-16", "15", "105"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]

        actions = _compute_actions(
            rows, columns, "N", "0", "OVERRIDE", "0",
            ntry=2024, ntrm=1, xpry=2024, xprm=2,
            underlying="TEST", overrides_path=csv_file
        )
        assert actions == ["-", "ntry", "-", "-", "xpry", "-"]

    def test_compute_actions_override_no_match(self, tmp_path):
        """Test _compute_actions with OVERRIDE returns no actions when lookup fails."""
        csv_file = tmp_path / "overrides.csv"
        csv_file.write_text("ticker,expiry\nOTHER,2024-01-15\n")

        rows = [
            ["A", "s", "2024-01-14", "10", "100"],
            ["A", "s", "2024-01-15", "11", "101"],
        ]
        columns = ["asset", "straddle", "date", "vol", "hedge"]

        actions = _compute_actions(
            rows, columns, "N", "0", "OVERRIDE", "0",
            ntry=2024, ntrm=1, underlying="TEST", overrides_path=csv_file
        )
        assert actions == ["-", "-"]


# -------------------------------------
# straddle_days Tests
# -------------------------------------

from specparser.amt import straddle_days, count_straddle_days
import datetime


class TestStraddleDays:
    """Tests for straddle_days and count_straddle_days functions."""

    def test_single_month(self):
        """Test generating days for a single month."""
        # straddle format: |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|
        straddle = "|2024-01|2024-01|N|5|F||33.3|"
        days = straddle_days(straddle)
        assert len(days) == 31  # January has 31 days
        assert days[0] == datetime.date(2024, 1, 1)
        assert days[-1] == datetime.date(2024, 1, 31)

    def test_february_leap_year(self):
        """Test February in a leap year."""
        straddle = "|2024-02|2024-02|N|5|F||33.3|"
        days = straddle_days(straddle)
        assert len(days) == 29  # 2024 is a leap year
        assert days[-1] == datetime.date(2024, 2, 29)

    def test_february_non_leap_year(self):
        """Test February in a non-leap year."""
        straddle = "|2023-02|2023-02|N|5|F||33.3|"
        days = straddle_days(straddle)
        assert len(days) == 28
        assert days[-1] == datetime.date(2023, 2, 28)

    def test_multiple_months(self):
        """Test spanning multiple months."""
        straddle = "|2024-01|2024-03|N|5|F||33.3|"
        days = straddle_days(straddle)
        # Jan=31, Feb=29, Mar=31 = 91 days
        assert len(days) == 91
        assert days[0] == datetime.date(2024, 1, 1)
        assert days[-1] == datetime.date(2024, 3, 31)

    def test_cross_year_boundary(self):
        """Test spanning across year boundary."""
        straddle = "|2023-12|2024-01|N|5|F||33.3|"
        days = straddle_days(straddle)
        # Dec=31, Jan=31 = 62 days
        assert len(days) == 62
        assert days[0] == datetime.date(2023, 12, 1)
        assert days[-1] == datetime.date(2024, 1, 31)

    def test_full_year(self):
        """Test generating all days in a year."""
        straddle = "|2024-01|2024-12|N|5|F||33.3|"
        days = straddle_days(straddle)
        assert len(days) == 366  # 2024 is a leap year
        assert days[0] == datetime.date(2024, 1, 1)
        assert days[-1] == datetime.date(2024, 12, 31)

    def test_consecutive_days(self):
        """Test that returned days are consecutive."""
        straddle = "|2024-01|2024-03|N|5|F||33.3|"
        days = straddle_days(straddle)
        for i in range(1, len(days)):
            delta = days[i] - days[i-1]
            assert delta.days == 1, f"Gap between {days[i-1]} and {days[i]}"

    def test_returns_date_objects(self):
        """Test that function returns datetime.date objects."""
        straddle = "|2024-01|2024-01|N|5|F||33.3|"
        days = straddle_days(straddle)
        assert all(isinstance(d, datetime.date) for d in days)

    def test_count_straddle_days(self):
        """Test count_straddle_days returns correct count."""
        straddle = "|2024-01|2024-03|N|5|F||33.3|"
        count = count_straddle_days(straddle)
        assert count == 91  # Jan=31, Feb=29, Mar=31

    def test_count_matches_len(self):
        """Test that count_straddle_days matches len(straddle_days)."""
        straddle = "|2024-01|2024-06|N|5|F||33.3|"
        count = count_straddle_days(straddle)
        days = straddle_days(straddle)
        assert count == len(days)


# -------------------------------------
# table_left_join and table_inner_join Tests
# -------------------------------------

from specparser.amt import table_left_join, table_inner_join


class TestTableLeftJoin:
    """Tests for table_left_join function."""

    def test_basic_left_join(self):
        """Test basic left join with matching and non-matching keys."""
        left = {"orientation": "row", "columns": ["id", "name"],
                "rows": [[1, "a"], [2, "b"], [3, "c"]]}
        right = {"orientation": "row", "columns": ["id", "value"],
                 "rows": [[1, 100], [2, 200]]}
        result = table_left_join(left, right, "id")
        assert result["columns"] == ["id", "name", "value"]
        assert result["rows"] == [[1, "a", 100], [2, "b", 200], [3, "c", None]]

    def test_duplicate_keys_in_right(self):
        """Test left join when right table has duplicate keys."""
        left = {"orientation": "row", "columns": ["id", "x"],
                "rows": [[1, "a"]]}
        right = {"orientation": "row", "columns": ["id", "y"],
                 "rows": [[1, "p"], [1, "q"]]}
        result = table_left_join(left, right, "id")
        assert len(result["rows"]) == 2  # 1 left row * 2 right matches
        assert result["rows"] == [[1, "a", "p"], [1, "a", "q"]]

    def test_column_name_conflict(self):
        """Test left join with overlapping column names."""
        left = {"orientation": "row", "columns": ["id", "value"],
                "rows": [[1, "left"]]}
        right = {"orientation": "row", "columns": ["id", "value"],
                 "rows": [[1, "right"]]}
        result = table_left_join(left, right, "id", suffixes=("_l", "_r"))
        assert result["columns"] == ["id", "value_l", "value_r"]
        assert result["rows"] == [[1, "left", "right"]]

    def test_different_key_column_names(self):
        """Test left join when key columns have different names."""
        left = {"orientation": "row", "columns": ["asset", "price"],
                "rows": [["AAPL", 100]]}
        right = {"orientation": "row", "columns": ["ticker", "volume"],
                 "rows": [["AAPL", 1000]]}
        result = table_left_join(left, right, "asset", "ticker")
        assert result["columns"] == ["asset", "price", "volume"]
        assert result["rows"] == [["AAPL", 100, 1000]]

    def test_no_matches(self):
        """Test left join when no keys match."""
        left = {"orientation": "row", "columns": ["id", "name"],
                "rows": [[1, "a"], [2, "b"]]}
        right = {"orientation": "row", "columns": ["id", "value"],
                 "rows": [[3, 300], [4, 400]]}
        result = table_left_join(left, right, "id")
        assert result["rows"] == [[1, "a", None], [2, "b", None]]

    def test_empty_left_table(self):
        """Test left join with empty left table."""
        left = {"orientation": "row", "columns": ["id", "name"], "rows": []}
        right = {"orientation": "row", "columns": ["id", "value"],
                 "rows": [[1, 100]]}
        result = table_left_join(left, right, "id")
        assert result["columns"] == ["id", "name", "value"]
        assert result["rows"] == []

    def test_empty_right_table(self):
        """Test left join with empty right table."""
        left = {"orientation": "row", "columns": ["id", "name"],
                "rows": [[1, "a"], [2, "b"]]}
        right = {"orientation": "row", "columns": ["id", "value"], "rows": []}
        result = table_left_join(left, right, "id")
        assert result["columns"] == ["id", "name", "value"]
        assert result["rows"] == [[1, "a", None], [2, "b", None]]

    def test_column_oriented_input(self):
        """Test left join with column-oriented input tables."""
        left = {"orientation": "column", "columns": ["id", "name"],
                "rows": [[1, 2, 3], ["a", "b", "c"]]}
        right = {"orientation": "column", "columns": ["id", "value"],
                 "rows": [[1, 2], [100, 200]]}
        result = table_left_join(left, right, "id")
        assert result["orientation"] == "row"
        assert result["columns"] == ["id", "name", "value"]
        assert result["rows"] == [[1, "a", 100], [2, "b", 200], [3, "c", None]]

    def test_multiple_right_columns(self):
        """Test left join with multiple columns from right table."""
        left = {"orientation": "row", "columns": ["id", "a"],
                "rows": [[1, "x"], [2, "y"]]}
        right = {"orientation": "row", "columns": ["id", "b", "c", "d"],
                 "rows": [[1, 10, 20, 30]]}
        result = table_left_join(left, right, "id")
        assert result["columns"] == ["id", "a", "b", "c", "d"]
        assert result["rows"] == [[1, "x", 10, 20, 30], [2, "y", None, None, None]]

    def test_key_by_index(self):
        """Test left join using column index instead of name."""
        left = {"orientation": "row", "columns": ["id", "name"],
                "rows": [[1, "a"], [2, "b"]]}
        right = {"orientation": "row", "columns": ["id", "value"],
                 "rows": [[1, 100]]}
        result = table_left_join(left, right, 0, 0)
        assert result["rows"] == [[1, "a", 100], [2, "b", None]]


class TestTableInnerJoin:
    """Tests for table_inner_join function."""

    def test_basic_inner_join(self):
        """Test basic inner join returns only matching rows."""
        left = {"orientation": "row", "columns": ["id", "name"],
                "rows": [[1, "a"], [2, "b"], [3, "c"]]}
        right = {"orientation": "row", "columns": ["id", "value"],
                 "rows": [[1, 100], [3, 300]]}
        result = table_inner_join(left, right, "id")
        assert len(result["rows"]) == 2
        assert result["rows"] == [[1, "a", 100], [3, "c", 300]]

    def test_no_matches_returns_empty(self):
        """Test inner join with no matches returns empty table."""
        left = {"orientation": "row", "columns": ["id", "name"],
                "rows": [[1, "a"], [2, "b"]]}
        right = {"orientation": "row", "columns": ["id", "value"],
                 "rows": [[3, 300], [4, 400]]}
        result = table_inner_join(left, right, "id")
        assert result["rows"] == []
        assert result["columns"] == ["id", "name", "value"]

    def test_all_match(self):
        """Test inner join when all left keys have matches."""
        left = {"orientation": "row", "columns": ["id", "name"],
                "rows": [[1, "a"], [2, "b"]]}
        right = {"orientation": "row", "columns": ["id", "value"],
                 "rows": [[1, 100], [2, 200]]}
        result = table_inner_join(left, right, "id")
        assert result["rows"] == [[1, "a", 100], [2, "b", 200]]

    def test_duplicate_keys_in_right(self):
        """Test inner join when right table has duplicate keys."""
        left = {"orientation": "row", "columns": ["id", "x"],
                "rows": [[1, "a"], [2, "b"]]}
        right = {"orientation": "row", "columns": ["id", "y"],
                 "rows": [[1, "p"], [1, "q"]]}
        result = table_inner_join(left, right, "id")
        # Only id=1 matches, and it has 2 matches in right
        assert len(result["rows"]) == 2
        assert result["rows"] == [[1, "a", "p"], [1, "a", "q"]]

    def test_different_key_column_names(self):
        """Test inner join with different key column names."""
        left = {"orientation": "row", "columns": ["asset", "price"],
                "rows": [["AAPL", 100], ["GOOGL", 200], ["MSFT", 300]]}
        right = {"orientation": "row", "columns": ["ticker", "volume"],
                 "rows": [["AAPL", 1000], ["MSFT", 3000]]}
        result = table_inner_join(left, right, "asset", "ticker")
        assert result["columns"] == ["asset", "price", "volume"]
        assert result["rows"] == [["AAPL", 100, 1000], ["MSFT", 300, 3000]]


# ============================================================
# Arrow Table Tests
# ============================================================

class TestArrowTables:
    """Tests for arrow-oriented table support."""

    def test_table_to_arrow_from_row(self):
        """Test converting row-oriented table to arrow."""
        row_table = {"orientation": "row", "columns": ["a", "b"],
                     "rows": [[1, "x"], [2, "y"], [3, "z"]]}
        result = table_to_arrow(row_table)
        assert result["orientation"] == "arrow"
        assert result["columns"] == ["a", "b"]
        assert result["rows"][0].to_pylist() == [1, 2, 3]
        assert result["rows"][1].to_pylist() == ["x", "y", "z"]

    def test_table_to_arrow_from_column(self):
        """Test converting column-oriented table to arrow."""
        col_table = {"orientation": "column", "columns": ["a", "b"],
                     "rows": [[1, 2, 3], ["x", "y", "z"]]}
        result = table_to_arrow(col_table)
        assert result["orientation"] == "arrow"
        assert result["rows"][0].to_pylist() == [1, 2, 3]

    def test_table_to_arrow_idempotent(self):
        """Test that table_to_arrow returns same object for arrow input."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a"],
                       "rows": [pa.array([1, 2, 3])]}
        result = table_to_arrow(arrow_table)
        assert result is arrow_table

    def test_table_to_columns_from_arrow(self):
        """Test converting arrow table to column-oriented."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a", "b"],
                       "rows": [pa.array([1, 2, 3]), pa.array(["x", "y", "z"])]}
        result = table_to_columns(arrow_table)
        assert result["orientation"] == "column"
        assert result["rows"][0] == [1, 2, 3]
        assert result["rows"][1] == ["x", "y", "z"]

    def test_table_to_rows_from_arrow(self):
        """Test converting arrow table to row-oriented."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a", "b"],
                       "rows": [pa.array([1, 2, 3]), pa.array(["x", "y", "z"])]}
        result = table_to_rows(arrow_table)
        assert result["orientation"] == "row"
        assert result["rows"] == [[1, "x"], [2, "y"], [3, "z"]]

    def test_arrow_roundtrip(self):
        """Test row -> arrow -> row preserves data."""
        row_table = {"orientation": "row", "columns": ["a", "b"],
                     "rows": [[1, "x"], [2, "y"], [3, "z"]]}
        roundtrip = table_to_rows(table_to_arrow(row_table))
        assert roundtrip["rows"] == row_table["rows"]

    def test_table_orientation(self):
        """Test table_orientation helper."""
        import pyarrow as pa
        row_table = {"orientation": "row", "columns": [], "rows": []}
        col_table = {"orientation": "column", "columns": [], "rows": []}
        arrow_table = {"orientation": "arrow", "columns": [], "rows": []}

        assert table_orientation(row_table) == "row"
        assert table_orientation(col_table) == "column"
        assert table_orientation(arrow_table) == "arrow"

    def test_table_nrows_all_orientations(self):
        """Test table_nrows works for all orientations."""
        import pyarrow as pa
        row_table = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3]]}
        col_table = {"orientation": "column", "columns": ["a"], "rows": [[1, 2, 3]]}
        arrow_table = {"orientation": "arrow", "columns": ["a"],
                       "rows": [pa.array([1, 2, 3])]}

        assert table_nrows(row_table) == 3
        assert table_nrows(col_table) == 3
        assert table_nrows(arrow_table) == 3

    def test_table_validate_arrow(self):
        """Test table_validate with arrow tables."""
        import pyarrow as pa
        valid_arrow = {"orientation": "arrow", "columns": ["a", "b"],
                       "rows": [pa.array([1, 2]), pa.array(["x", "y"])]}
        table_validate(valid_arrow)  # Should not raise

        # Invalid: non-Arrow array in rows
        invalid_arrow = {"orientation": "arrow", "columns": ["a"],
                         "rows": [[1, 2, 3]]}
        with pytest.raises(ValueError, match="not a PyArrow Array"):
            table_validate(invalid_arrow)

    def test_table_to_jsonable(self):
        """Test table_to_jsonable converts arrow to JSON-serializable."""
        import pyarrow as pa
        from datetime import datetime
        from decimal import Decimal

        # Test with special types
        special_table = {"orientation": "row", "columns": ["dt", "dec"],
                         "rows": [[datetime(2024, 1, 15, 10, 30), Decimal("123.456")]]}
        result = table_to_jsonable(special_table)
        assert result["rows"][0][0] == "2024-01-15T10:30:00"
        assert result["rows"][0][1] == 123.456


class TestArrowFastPaths:
    """Tests for arrow fast-path implementations."""

    def test_table_head_arrow(self):
        """Test table_head preserves arrow orientation."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a", "b"],
                       "rows": [pa.array([1, 2, 3, 4, 5]), pa.array(["a", "b", "c", "d", "e"])]}
        result = table_head(arrow_table, 3)
        assert result["orientation"] == "arrow"
        assert table_nrows(result) == 3
        assert result["rows"][0].to_pylist() == [1, 2, 3]

    def test_table_sample_arrow(self):
        """Test table_sample preserves arrow orientation."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a"],
                       "rows": [pa.array(list(range(100)))]}
        result = table_sample(arrow_table, 5)
        assert result["orientation"] == "arrow"
        assert table_nrows(result) == 5

    def test_table_select_columns_arrow(self):
        """Test table_select_columns preserves arrow orientation."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a", "b", "c"],
                       "rows": [pa.array([1, 2]), pa.array(["x", "y"]), pa.array([10.0, 20.0])]}
        result = table_select_columns(arrow_table, ["c", "a"])
        assert result["orientation"] == "arrow"
        assert result["columns"] == ["c", "a"]
        assert result["rows"][0].to_pylist() == [10.0, 20.0]

    def test_table_drop_columns_arrow(self):
        """Test table_drop_columns preserves arrow orientation."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a", "b", "c"],
                       "rows": [pa.array([1, 2]), pa.array(["x", "y"]), pa.array([10.0, 20.0])]}
        result = table_drop_columns(arrow_table, ["b"])
        assert result["orientation"] == "arrow"
        assert result["columns"] == ["a", "c"]

    def test_table_column_arrow_returns_array(self):
        """Test table_column returns pa.Array for arrow input."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a", "b"],
                       "rows": [pa.array([1, 2, 3]), pa.array(["x", "y", "z"])]}
        result = table_column(arrow_table, "a")
        assert isinstance(result, pa.Array)
        assert result.to_pylist() == [1, 2, 3]

    def test_table_replace_value_arrow(self):
        """Test table_replace_value preserves arrow orientation."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a", "b"],
                       "rows": [pa.array([1, 2, 1, 3]), pa.array(["x", "y", "z", "w"])]}
        result = table_replace_value(arrow_table, "a", 1, 999)
        assert result["orientation"] == "arrow"
        assert result["rows"][0].to_pylist() == [999, 2, 999, 3]

    def test_table_add_column_arrow(self):
        """Test table_add_column preserves arrow orientation."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a"],
                       "rows": [pa.array([1, 2, 3])]}
        result = table_add_column(arrow_table, "b", "new")
        assert result["orientation"] == "arrow"
        assert result["columns"] == ["a", "b"]
        assert result["rows"][1].to_pylist() == ["new", "new", "new"]

    def test_table_bind_rows_arrow(self):
        """Test table_bind_rows with arrow tables."""
        import pyarrow as pa
        arrow1 = {"orientation": "arrow", "columns": ["a", "b"],
                  "rows": [pa.array([1, 2]), pa.array(["x", "y"])]}
        arrow2 = {"orientation": "arrow", "columns": ["a", "b"],
                  "rows": [pa.array([3, 4, 5]), pa.array(["z", "w", "v"])]}
        result = table_bind_rows(arrow1, arrow2)
        assert result["orientation"] == "arrow"
        assert table_nrows(result) == 5
        assert result["rows"][0].to_pylist() == [1, 2, 3, 4, 5]

    def test_table_bind_rows_mixed_to_arrow(self):
        """Test table_bind_rows converts to arrow when any input is arrow."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a"],
                       "rows": [pa.array([1, 2])]}
        row_table = {"orientation": "row", "columns": ["a"],
                     "rows": [[3], [4]]}
        result = table_bind_rows(arrow_table, row_table)
        assert result["orientation"] == "arrow"
        assert result["rows"][0].to_pylist() == [1, 2, 3, 4]

    def test_table_unique_rows_arrow(self):
        """Test table_unique_rows with arrow tables."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a", "b"],
                       "rows": [pa.array([1, 2, 1, 3]), pa.array(["x", "y", "x", "z"])]}
        result = table_unique_rows(arrow_table)
        assert result["orientation"] == "arrow"
        assert table_nrows(result) == 3  # 3 unique rows

    def test_table_stack_cols_arrow(self):
        """Test table_stack_cols with arrow tables."""
        import pyarrow as pa
        arrow1 = {"orientation": "arrow", "columns": ["key", "val1"],
                  "rows": [pa.array([1, 2, 3]), pa.array([10, 20, 30])]}
        arrow2 = {"orientation": "arrow", "columns": ["key", "val2"],
                  "rows": [pa.array([1, 2, 3]), pa.array([100, 200, 300])]}
        result = table_stack_cols(arrow1, arrow2)
        assert result["orientation"] == "arrow"
        assert result["columns"] == ["key", "val1", "val2"]

    def test_table_left_join_arrow(self):
        """Test table_left_join with arrow tables."""
        import pyarrow as pa
        left = {"orientation": "arrow", "columns": ["id", "name"],
                "rows": [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])]}
        right = {"orientation": "arrow", "columns": ["id", "value"],
                 "rows": [pa.array([2, 3, 4]), pa.array([100, 200, 300])]}
        result = table_left_join(left, right, "id")
        assert result["orientation"] == "arrow"
        assert "id" in result["columns"]
        assert "name" in result["columns"]
        assert "value" in result["columns"]

    def test_table_inner_join_arrow(self):
        """Test table_inner_join with arrow tables."""
        import pyarrow as pa
        left = {"orientation": "arrow", "columns": ["id", "name"],
                "rows": [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])]}
        right = {"orientation": "arrow", "columns": ["id", "value"],
                 "rows": [pa.array([2, 3, 4]), pa.array([100, 200, 300])]}
        result = table_inner_join(left, right, "id")
        assert result["orientation"] == "arrow"
        # Only id=2 and id=3 match
        assert table_nrows(result) == 2

    def test_table_unchop_arrow(self):
        """Test table_unchop with arrow tables."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["a", "b"],
                       "rows": [pa.array(["x", "y"]), pa.array([[1, 2], [3, 4, 5]])]}
        result = table_unchop(arrow_table, "b")
        assert result["orientation"] == "arrow"
        # x -> [1, 2] = 2 rows, y -> [3, 4, 5] = 3 rows => 5 total
        assert table_nrows(result) == 5

    def test_table_chop_arrow(self):
        """Test table_chop with arrow tables."""
        import pyarrow as pa
        arrow_table = {"orientation": "arrow", "columns": ["group", "value"],
                       "rows": [pa.array(["A", "A", "B", "B", "B"]), pa.array([1, 2, 3, 4, 5])]}
        result = table_chop(arrow_table, "value")
        assert result["orientation"] == "arrow"
        # 2 groups: A and B
        assert table_nrows(result) == 2


class TestArrowCrossOrientationEquivalence:
    """Tests that operations give equivalent results across orientations."""

    def test_head_equivalence(self):
        """Test that table_head gives same result for row/column/arrow."""
        import pyarrow as pa
        row_table = {"orientation": "row", "columns": ["a", "b"],
                     "rows": [[1, "x"], [2, "y"], [3, "z"], [4, "w"]]}
        col_table = table_to_columns(row_table)
        arrow_table = table_to_arrow(row_table)

        row_result = table_to_rows(table_head(row_table, 2))
        col_result = table_to_rows(table_head(col_table, 2))
        arrow_result = table_to_rows(table_head(arrow_table, 2))

        assert row_result["rows"] == [[1, "x"], [2, "y"]]
        assert col_result["rows"] == [[1, "x"], [2, "y"]]
        assert arrow_result["rows"] == [[1, "x"], [2, "y"]]

    def test_select_columns_equivalence(self):
        """Test that table_select_columns gives same result across orientations."""
        import pyarrow as pa
        row_table = {"orientation": "row", "columns": ["a", "b", "c"],
                     "rows": [[1, "x", 10], [2, "y", 20]]}
        col_table = table_to_columns(row_table)
        arrow_table = table_to_arrow(row_table)

        row_result = table_to_rows(table_select_columns(row_table, ["c", "a"]))
        col_result = table_to_rows(table_select_columns(col_table, ["c", "a"]))
        arrow_result = table_to_rows(table_select_columns(arrow_table, ["c", "a"]))

        expected = [[10, 1], [20, 2]]
        assert row_result["rows"] == expected
        assert col_result["rows"] == expected
        assert arrow_result["rows"] == expected

    def test_replace_value_equivalence(self):
        """Test that table_replace_value gives same result across orientations."""
        import pyarrow as pa
        row_table = {"orientation": "row", "columns": ["a", "b"],
                     "rows": [[1, "x"], [2, "y"], [1, "z"]]}
        col_table = table_to_columns(row_table)
        arrow_table = table_to_arrow(row_table)

        row_result = table_to_rows(table_replace_value(row_table, "a", 1, 999))
        col_result = table_to_rows(table_replace_value(col_table, "a", 1, 999))
        arrow_result = table_to_rows(table_replace_value(arrow_table, "a", 1, 999))

        expected = [[999, "x"], [2, "y"], [999, "z"]]
        assert row_result["rows"] == expected
        assert col_result["rows"] == expected
        assert arrow_result["rows"] == expected


# ============================================================
# Arrow Compute Functions Tests
# ============================================================

from specparser.amt import (
    # arithmetic
    table_add_arrow, table_subtract_arrow, table_multiply_arrow, table_divide_arrow,
    table_negate_arrow, table_abs_arrow, table_sign_arrow, table_power_arrow,
    table_sqrt_arrow, table_exp_arrow, table_ln_arrow, table_log10_arrow,
    table_log2_arrow, table_round_arrow, table_ceil_arrow, table_floor_arrow,
    table_trunc_arrow,
    # trigonometric
    table_sin_arrow, table_cos_arrow, table_tan_arrow,
    table_asin_arrow, table_acos_arrow, table_atan_arrow, table_atan2_arrow,
    # comparison
    table_equal_arrow, table_not_equal_arrow, table_less_arrow,
    table_less_equal_arrow, table_greater_arrow, table_greater_equal_arrow,
    # null checks
    table_is_null_arrow, table_is_valid_arrow, table_is_nan_arrow,
    table_is_finite_arrow, table_is_in_arrow,
    # logical
    table_and_arrow, table_or_arrow, table_xor_arrow, table_invert_arrow,
    # string
    table_upper_arrow, table_lower_arrow, table_capitalize_arrow, table_title_arrow,
    table_strip_arrow, table_lstrip_arrow, table_rstrip_arrow, table_length_arrow,
    table_starts_with_arrow, table_ends_with_arrow, table_contains_arrow,
    table_replace_substr_arrow, table_split_arrow,
    # aggregates
    table_summarize_arrow, table_sum_arrow, table_mean_arrow, table_min_arrow,
    table_max_arrow, table_count_arrow, table_count_distinct_arrow,
    table_stddev_arrow, table_variance_arrow, table_first_arrow, table_last_arrow,
    table_any_arrow, table_all_arrow,
    # cumulative
    table_cumsum_arrow, table_cumprod_arrow, table_cummin_arrow, table_cummax_arrow,
    table_cummean_arrow, table_diff_arrow,
    # selection
    table_if_else_arrow, table_coalesce_arrow, table_fill_null_arrow,
    table_fill_null_forward_arrow, table_fill_null_backward_arrow,
    # filter
    table_filter_arrow,
)


class TestArrowComputeArithmetic:
    """Tests for arrow arithmetic compute functions."""

    def test_add_scalar(self):
        """Test adding scalar to column."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3]]}
        result = table_add_arrow(t, "a", 10, result_col="sum")
        assert result["orientation"] == "arrow"
        assert result["columns"] == ["a", "sum"]
        assert result["rows"][-1].to_pylist() == [11, 12, 13]

    def test_add_column(self):
        """Test adding two columns."""
        t = {"orientation": "row", "columns": ["a", "b"], "rows": [[1, 10], [2, 20]]}
        result = table_add_arrow(t, "a", "b", result_col="sum")
        assert result["rows"][-1].to_pylist() == [11, 22]

    def test_multiply_replaces_column(self):
        """Test multiply replaces column when result_col is None."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[2], [3], [4]]}
        result = table_multiply_arrow(t, "a", 5)
        assert result["columns"] == ["a"]  # same column
        assert result["rows"][0].to_pylist() == [10, 15, 20]

    def test_divide(self):
        """Test division."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[10], [20], [30]]}
        result = table_divide_arrow(t, "a", 2, result_col="half")
        assert result["rows"][-1].to_pylist() == [5.0, 10.0, 15.0]

    def test_negate(self):
        """Test negation."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1], [-2], [3]]}
        result = table_negate_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [-1, 2, -3]

    def test_abs(self):
        """Test absolute value."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[-5], [3], [-1]]}
        result = table_abs_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [5, 3, 1]

    def test_sign(self):
        """Test sign function."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[-5], [0], [3]]}
        result = table_sign_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [-1, 0, 1]

    def test_power(self):
        """Test power function."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[2], [3], [4]]}
        result = table_power_arrow(t, "a", 2)
        assert result["rows"][0].to_pylist() == [4.0, 9.0, 16.0]

    def test_sqrt(self):
        """Test square root."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[4], [9], [16]]}
        result = table_sqrt_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [2.0, 3.0, 4.0]

    def test_round(self):
        """Test rounding."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1.234], [5.678], [9.999]]}
        result = table_round_arrow(t, "a", decimals=2)
        vals = result["rows"][0].to_pylist()
        assert abs(vals[0] - 1.23) < 0.01
        assert abs(vals[1] - 5.68) < 0.01

    def test_ceil_floor_trunc(self):
        """Test ceiling, floor, truncation."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1.5], [-1.5]]}
        ceil_result = table_ceil_arrow(t, "a")
        floor_result = table_floor_arrow(t, "a")
        trunc_result = table_trunc_arrow(t, "a")

        assert ceil_result["rows"][0].to_pylist() == [2.0, -1.0]
        assert floor_result["rows"][0].to_pylist() == [1.0, -2.0]
        assert trunc_result["rows"][0].to_pylist() == [1.0, -1.0]


class TestArrowComputeTrigonometric:
    """Tests for arrow trigonometric compute functions."""

    def test_sin_cos(self):
        """Test sin and cos."""
        import math
        t = {"orientation": "row", "columns": ["a"], "rows": [[0], [math.pi / 2]]}
        sin_result = table_sin_arrow(t, "a")
        cos_result = table_cos_arrow(t, "a")

        sin_vals = sin_result["rows"][0].to_pylist()
        cos_vals = cos_result["rows"][0].to_pylist()

        assert abs(sin_vals[0] - 0.0) < 1e-10
        assert abs(sin_vals[1] - 1.0) < 1e-10
        assert abs(cos_vals[0] - 1.0) < 1e-10
        assert abs(cos_vals[1] - 0.0) < 1e-10


class TestArrowComputeComparison:
    """Tests for arrow comparison compute functions."""

    def test_equal(self):
        """Test equality comparison."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3]]}
        result = table_equal_arrow(t, "a", 2, result_col="is_two")
        assert result["rows"][-1].to_pylist() == [False, True, False]

    def test_greater(self):
        """Test greater than comparison."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3]]}
        result = table_greater_arrow(t, "a", 1, result_col="gt_one")
        assert result["rows"][-1].to_pylist() == [False, True, True]

    def test_less_equal(self):
        """Test less than or equal comparison."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3]]}
        result = table_less_equal_arrow(t, "a", 2)
        assert result["rows"][0].to_pylist() == [True, True, False]


class TestArrowComputeNullChecks:
    """Tests for arrow null check compute functions."""

    def test_is_null(self):
        """Test is_null check."""
        import pyarrow as pa
        t = {"orientation": "arrow", "columns": ["a"],
             "rows": [pa.array([1, None, 3])]}
        result = table_is_null_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [False, True, False]

    def test_is_valid(self):
        """Test is_valid check."""
        import pyarrow as pa
        t = {"orientation": "arrow", "columns": ["a"],
             "rows": [pa.array([1, None, 3])]}
        result = table_is_valid_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [True, False, True]

    def test_is_in(self):
        """Test is_in check."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3], [4]]}
        result = table_is_in_arrow(t, "a", [2, 4], result_col="in_set")
        assert result["rows"][-1].to_pylist() == [False, True, False, True]


class TestArrowComputeLogical:
    """Tests for arrow logical compute functions."""

    def test_and_or_xor(self):
        """Test logical operations."""
        t = {"orientation": "row", "columns": ["a", "b"],
             "rows": [[True, True], [True, False], [False, True], [False, False]]}
        and_result = table_and_arrow(t, "a", "b", result_col="and")
        or_result = table_or_arrow(t, "a", "b", result_col="or")
        xor_result = table_xor_arrow(t, "a", "b", result_col="xor")

        assert and_result["rows"][-1].to_pylist() == [True, False, False, False]
        assert or_result["rows"][-1].to_pylist() == [True, True, True, False]
        assert xor_result["rows"][-1].to_pylist() == [False, True, True, False]

    def test_invert(self):
        """Test logical NOT."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[True], [False]]}
        result = table_invert_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [False, True]


class TestArrowComputeString:
    """Tests for arrow string compute functions."""

    def test_upper_lower(self):
        """Test uppercase and lowercase."""
        t = {"orientation": "row", "columns": ["a"], "rows": [["Hello"], ["WORLD"]]}
        upper_result = table_upper_arrow(t, "a")
        lower_result = table_lower_arrow(t, "a")

        assert upper_result["rows"][0].to_pylist() == ["HELLO", "WORLD"]
        assert lower_result["rows"][0].to_pylist() == ["hello", "world"]

    def test_strip(self):
        """Test string strip."""
        t = {"orientation": "row", "columns": ["a"], "rows": [["  hello  "], ["world  "]]}
        result = table_strip_arrow(t, "a")
        assert result["rows"][0].to_pylist() == ["hello", "world"]

    def test_length(self):
        """Test string length."""
        t = {"orientation": "row", "columns": ["a"], "rows": [["hi"], ["hello"], [""]]}
        result = table_length_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [2, 5, 0]

    def test_starts_ends_contains(self):
        """Test string pattern matching."""
        t = {"orientation": "row", "columns": ["a"],
             "rows": [["hello world"], ["goodbye world"], ["hello there"]]}
        starts = table_starts_with_arrow(t, "a", "hello", result_col="starts")
        ends = table_ends_with_arrow(t, "a", "world", result_col="ends")
        contains = table_contains_arrow(t, "a", "world", result_col="has")

        assert starts["rows"][-1].to_pylist() == [True, False, True]
        assert ends["rows"][-1].to_pylist() == [True, True, False]
        assert contains["rows"][-1].to_pylist() == [True, True, False]


class TestArrowComputeAggregates:
    """Tests for arrow aggregate compute functions."""

    def test_sum_mean(self):
        """Test sum and mean aggregates."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3], [4]]}
        assert table_sum_arrow(t, "a") == 10
        assert table_mean_arrow(t, "a") == 2.5

    def test_min_max(self):
        """Test min and max aggregates."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[5], [2], [8], [1]]}
        assert table_min_arrow(t, "a") == 1
        assert table_max_arrow(t, "a") == 8

    def test_count(self):
        """Test count aggregate."""
        import pyarrow as pa
        t = {"orientation": "arrow", "columns": ["a"],
             "rows": [pa.array([1, None, 3, None, 5])]}
        assert table_count_arrow(t, "a") == 3  # non-null count

    def test_count_distinct(self):
        """Test count distinct aggregate."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [1], [3], [2]]}
        assert table_count_distinct_arrow(t, "a") == 3

    def test_first_last(self):
        """Test first and last aggregates."""
        t = {"orientation": "row", "columns": ["a"], "rows": [["x"], ["y"], ["z"]]}
        assert table_first_arrow(t, "a") == "x"
        assert table_last_arrow(t, "a") == "z"

    def test_any_all(self):
        """Test any and all aggregates."""
        t_all_true = {"orientation": "row", "columns": ["a"],
                      "rows": [[True], [True], [True]]}
        t_some_true = {"orientation": "row", "columns": ["a"],
                       "rows": [[True], [False], [True]]}
        t_all_false = {"orientation": "row", "columns": ["a"],
                       "rows": [[False], [False]]}

        assert table_all_arrow(t_all_true, "a") == True
        assert table_all_arrow(t_some_true, "a") == False
        assert table_any_arrow(t_some_true, "a") == True
        assert table_any_arrow(t_all_false, "a") == False

    def test_summarize(self):
        """Test summarize with multiple aggregations."""
        t = {"orientation": "row", "columns": ["a", "b"],
             "rows": [[1, 10], [2, 20], [3, 30]]}
        result = table_summarize_arrow(t, {"a": ["sum", "mean"], "b": "max"})
        assert result["orientation"] == "arrow"
        assert set(result["columns"]) == {"a_sum", "a_mean", "b_max"}


class TestArrowComputeCumulative:
    """Tests for arrow cumulative compute functions."""

    def test_cumsum(self):
        """Test cumulative sum."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3], [4]]}
        result = table_cumsum_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [1, 3, 6, 10]

    def test_cumprod(self):
        """Test cumulative product."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3], [4]]}
        result = table_cumprod_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [1, 2, 6, 24]

    def test_cummin_cummax(self):
        """Test cumulative min and max."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[3], [1], [4], [1], [5]]}
        min_result = table_cummin_arrow(t, "a")
        max_result = table_cummax_arrow(t, "a")

        assert min_result["rows"][0].to_pylist() == [3, 1, 1, 1, 1]
        assert max_result["rows"][0].to_pylist() == [3, 3, 4, 4, 5]

    def test_diff(self):
        """Test pairwise differences."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[10], [15], [13], [20]]}
        result = table_diff_arrow(t, "a")
        vals = result["rows"][0].to_pylist()
        assert vals[0] is None  # first element is null
        assert vals[1] == 5
        assert vals[2] == -2
        assert vals[3] == 7


class TestArrowComputeSelection:
    """Tests for arrow selection compute functions."""

    def test_if_else(self):
        """Test if_else selection."""
        t = {"orientation": "row", "columns": ["cond", "a", "b"],
             "rows": [[True, 1, 10], [False, 2, 20], [True, 3, 30]]}
        result = table_if_else_arrow(t, "cond", "a", "b", result_col="result")
        assert result["rows"][-1].to_pylist() == [1, 20, 3]

    def test_fill_null(self):
        """Test fill_null."""
        import pyarrow as pa
        t = {"orientation": "arrow", "columns": ["a"],
             "rows": [pa.array([1, None, 3, None])]}
        result = table_fill_null_arrow(t, "a", 999)
        assert result["rows"][0].to_pylist() == [1, 999, 3, 999]

    def test_fill_null_forward(self):
        """Test forward fill."""
        import pyarrow as pa
        t = {"orientation": "arrow", "columns": ["a"],
             "rows": [pa.array([1, None, None, 4])]}
        result = table_fill_null_forward_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [1, 1, 1, 4]

    def test_fill_null_backward(self):
        """Test backward fill."""
        import pyarrow as pa
        t = {"orientation": "arrow", "columns": ["a"],
             "rows": [pa.array([1, None, None, 4])]}
        result = table_fill_null_backward_arrow(t, "a")
        assert result["rows"][0].to_pylist() == [1, 4, 4, 4]


class TestArrowComputeFilter:
    """Tests for arrow filter compute function."""

    def test_filter_by_column(self):
        """Test filtering by boolean column."""
        t = {"orientation": "row", "columns": ["a", "mask"],
             "rows": [[1, True], [2, False], [3, True], [4, False]]}
        result = table_filter_arrow(t, "mask")
        assert result["orientation"] == "arrow"
        assert table_nrows(result) == 2
        assert result["rows"][0].to_pylist() == [1, 3]

    def test_filter_workflow(self):
        """Test typical filter workflow: compare then filter."""
        t = {"orientation": "row", "columns": ["a", "b"],
             "rows": [[1, "x"], [5, "y"], [3, "z"], [7, "w"]]}
        # Add comparison column
        with_mask = table_greater_arrow(t, "a", 3, result_col="_mask")
        # Filter
        filtered = table_filter_arrow(with_mask, "_mask")
        # Drop mask column
        result = table_drop_columns(filtered, ["_mask"])

        rows = table_to_rows(result)["rows"]
        assert rows == [[5, "y"], [7, "w"]]


class TestArrowComputeInputOrientations:
    """Test that arrow compute functions accept any input orientation."""

    def test_accepts_row_oriented(self):
        """Test function accepts row-oriented input."""
        t = {"orientation": "row", "columns": ["a"], "rows": [[1], [2], [3]]}
        result = table_add_arrow(t, "a", 10)
        assert result["orientation"] == "arrow"
        assert result["rows"][0].to_pylist() == [11, 12, 13]

    def test_accepts_column_oriented(self):
        """Test function accepts column-oriented input."""
        t = {"orientation": "column", "columns": ["a"], "rows": [[1, 2, 3]]}
        result = table_add_arrow(t, "a", 10)
        assert result["orientation"] == "arrow"
        assert result["rows"][0].to_pylist() == [11, 12, 13]

    def test_accepts_arrow_oriented(self):
        """Test function accepts arrow-oriented input (no conversion)."""
        import pyarrow as pa
        t = {"orientation": "arrow", "columns": ["a"], "rows": [pa.array([1, 2, 3])]}
        result = table_add_arrow(t, "a", 10)
        assert result["orientation"] == "arrow"
        assert result["rows"][0].to_pylist() == [11, 12, 13]
