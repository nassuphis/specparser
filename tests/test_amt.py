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
    expand,
    expand_ym,
    get_expand,
    get_expand_ym,
    # tickers
    _split_ticker,
    get_tschemas,
    find_tschemas,
    fut_spec2ticker,
    fut_norm2act,
    clear_normalized_cache,
    _tschma_dict_expand_bbgfc,
    _tschma_dict_expand_split,
)


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
      Source: "BBG"
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
        assert len(table["rows"]) == 4


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

    def test_expand_ym(self, test_amt_file):
        """Test expanding schedules for a specific year/month."""
        clear_cache()
        table = expand_ym(test_amt_file, 2024, 6, pattern="^CL", live_only=False)
        assert table["columns"] == ["asset", "straddle"]
        assert len(table["rows"]) == 2  # CL Comdty has 2 schedule components

        # Check straddle format
        for row in table["rows"]:
            assert row[0] == "CL Comdty"
            assert row[1].startswith("|")
            assert row[1].endswith("|")

    def test_expand(self, test_amt_file):
        """Test expanding schedules across year range."""
        clear_cache()
        table = expand(test_amt_file, 2024, 2024, pattern="^CL", live_only=False)
        # 12 months * 2 schedule components = 24 rows
        assert len(table["rows"]) == 24
        assert table["columns"] == ["asset", "straddle"]

    def test_get_expand(self, test_amt_file):
        """Test get_expand for single asset."""
        clear_cache()
        table = get_expand(test_amt_file, "CL Comdty", 2024, 2024)
        # 12 months * 2 schedule components = 24 rows
        assert len(table["rows"]) == 24

    def test_get_expand_ym(self, test_amt_file):
        """Test get_expand_ym for single asset and month."""
        clear_cache()
        table = get_expand_ym(test_amt_file, "CL Comdty", 2024, 6)
        assert len(table["rows"]) == 2

    def test_expand_empty_pattern(self, test_amt_file):
        """Test expand with pattern matching no assets."""
        clear_cache()
        table = expand(test_amt_file, 2024, 2024, pattern="^NONEXISTENT", live_only=False)
        assert len(table["rows"]) == 0

    def test_straddle_format(self, test_amt_file):
        """Test straddle string format is correct."""
        clear_cache()
        table = expand_ym(test_amt_file, 2024, 6, pattern="^CL", live_only=False)

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
        expanded = get_expand(test_amt_file, asset, 2024, 2024)
        assert len(expanded["rows"]) == 24  # 12 months * 2 components

    def test_asset_classification_workflow(self, test_amt_file):
        """Test complete asset classification workflow."""
        clear_cache()

        # Get group assignments
        groups = asset_group(test_amt_file, live_only=True)

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
