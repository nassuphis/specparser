# -------------------------------------
# AMT loader - Core file access
# -------------------------------------
"""
Core AMT file loading and querying utilities.

Handles YAML loading, caching, asset queries, and rule matching.
"""
import re
from pathlib import Path
from typing import Any

import yaml


# Module-level cache for loaded AMT data
_AMT_CACHE: dict[str, dict[str, Any]] = {}

# Cache for underlying -> asset_data lookup (built during load_amt)
_ASSET_BY_UNDERLYING: dict[str, dict[str, dict[str, Any]]] = {}


def load_amt(path: str | Path) -> dict[str, Any]:
    """
    Load an AMT YAML file and return the full parsed data.

    Args:
        path: Path to the AMT YAML file

    Returns:
        Parsed YAML data as a dict

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file is not valid YAML
    """
    path = Path(path)
    path_str = str(path.resolve())

    if path_str in _AMT_CACHE:
        return _AMT_CACHE[path_str]

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    _AMT_CACHE[path_str] = data

    # Build underlying -> asset_data lookup
    amt = data.get("amt", {})
    underlying_map: dict[str, dict[str, Any]] = {}
    for asset_data in amt.values():
        if isinstance(asset_data, dict):
            underlying = asset_data.get("Underlying")
            if underlying:
                underlying_map[underlying] = asset_data
    _ASSET_BY_UNDERLYING[path_str] = underlying_map

    return data


def clear_cache():
    """Clear the AMT file cache and underlying lookup cache."""
    _AMT_CACHE.clear()
    _ASSET_BY_UNDERLYING.clear()

def _iter_assets(path: str | Path, live_only: bool = False, pattern: str = "."):
    """
    Iterate over assets in an AMT file.

    Args:
        path: Path to the AMT YAML file
        live_only: If True, only yield assets with WeightCap > 0
        pattern: Regex pattern to filter Underlying values (default "." matches all)

    Yields:
        Tuples of (asset_data, underlying)
    """
    data = load_amt(path)
    amt = data.get("amt", {})
    regex = re.compile(pattern)

    for asset_data in amt.values():
        if isinstance(asset_data, dict):
            underlying = asset_data.get("Underlying")
            if underlying is not None:
                if not regex.search(underlying):
                    continue
                if live_only:
                    wcap = asset_data.get("WeightCap")
                    if wcap is None or wcap <= 0:
                        continue
                yield asset_data, underlying

#
# values in the amt
#
def get_value(path: str | Path, key_path: str, default: Any = None) -> Any:
    """Get a value from an AMT file by its dot-separated key path."""
    data = load_amt(path)

    current = data
    for key in key_path.split("."):
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]

    return current


def get_aum(path: str | Path) -> float | None:
    """Get the AUM (Assets Under Management) value from an AMT file."""
    return get_value(path, "backtest.aum")


def get_leverage(path: str | Path) -> float | None:
    """Get the leverage value from an AMT file."""
    return get_value(path, "backtest.leverage")

#
# assets
#
def get_asset(path: str | Path, underlying: str) -> dict[str, Any] | None:
    """Get asset data by its Underlying value."""
    path = Path(path)
    path_str = str(path.resolve())
    if path_str not in _ASSET_BY_UNDERLYING:
        load_amt(path)
    return _ASSET_BY_UNDERLYING.get(path_str, {}).get(underlying)


def find_assets(path: str | Path, pattern: str, live_only: bool = False) -> dict[str, Any]:
    """Find all Underlying values matching a regex pattern."""
    rows = [[underlying] for _, underlying in _iter_assets(path, live_only=live_only, pattern=pattern)]
    return {"columns": ["asset"], "rows": rows}


def cached_assets(path: str | Path) -> dict[str, Any]:
    """List all asset Underlying values from the cache."""
    path = Path(path)
    path_str = str(path.resolve())
    if path_str not in _ASSET_BY_UNDERLYING: load_amt(path)
    assets = _ASSET_BY_UNDERLYING.get(path_str, {})
    rows = [[u] for u in assets.keys()]
    return { "columns": ["asset"], "rows": rows }

def assets(path: str | Path, live_only: bool = False, pattern: str = ".") -> dict[str, Any]:
    """Get assets with their Underlying values."""
    rows = [[underlying] for _, underlying in _iter_assets(path, live_only=live_only, pattern=pattern)]
    return {"columns": ["asset"], "rows": rows}


def asset_class(path: str | Path, live_only: bool = False, pattern: str = ".") -> dict[str, Any]:
    """Get all live assets (WeightCap > 0) with their class and source information."""
    rows = []
    for asset_data, underlying in _iter_assets(path, live_only=live_only, pattern=pattern):
        cls = asset_data.get("Class", "")
        vol = asset_data.get("Vol", {})
        volsrc = vol.get("Source", "") if isinstance(vol, dict) else ""
        hedge = asset_data.get("Hedge", {})
        hdgsrc = hedge.get("Source", "") if isinstance(hedge, dict) else ""
        valuation = asset_data.get("Valuation", {})
        model = valuation.get("Model", "") if isinstance(valuation, dict) else ""
        rows.append([underlying, cls, volsrc, hdgsrc, model])

    return {
        "columns": ["asset", "cls", "volsrc", "hdgsrc", "model"],
        "rows": rows,
    }

#
# embedded tables
#
def get_table(path: str | Path, key_path: str) -> dict[str, Any]:
    """Get an embedded table from an AMT file by its key path."""
    _missing = object()
    current = get_value(path, key_path, default=_missing)

    if current is _missing:
        raise ValueError(f"Key path '{key_path}' not found")

    if not isinstance(current, dict):
        raise ValueError(f"Path '{key_path}' does not lead to a dict")

    if "Columns" not in current:
        raise ValueError(f"Table at '{key_path}' is missing 'Columns' key")

    if "Rows" not in current:
        raise ValueError(f"Table at '{key_path}' is missing 'Rows' key")

    columns = current["Columns"]
    types = current.get("Types")
    rows = current["Rows"]

    if not isinstance(columns, list):
        raise ValueError(f"'Columns' at '{key_path}' is not a list")

    if not isinstance(rows, list):
        raise ValueError(f"'Rows' at '{key_path}' is not a list")

    return { "columns": columns, "types": types, "rows": rows, }


def table_column(table: dict[str, Any], colname: str) -> list[Any]:
    """Extract a single column from a table as a list.

    Args:
        table: Dict with 'columns' and 'rows'
        colname: Name of the column to extract

    Returns:
        List of values from that column

    Raises:
        ValueError: If column name not found
    """
    try:
        idx = table["columns"].index(colname)
    except ValueError:
        raise ValueError(f"Column '{colname}' not found in table columns: {table['columns']}")
    return [row[idx] for row in table["rows"]]


def format_table(table: dict[str, Any]) -> str:
    """Format a table dict as a tab-separated string with header."""
    lines = []

    # Header
    lines.append("\t".join(str(c) for c in table["columns"]))

    # Rows
    for row in table["rows"]:
        lines.append("\t".join(str(v) for v in row))

    return "\n".join(lines)


def print_table(table: dict[str, Any]) -> None:
    """Print a table with header and rows to stdout."""
    # Header
    print("\t".join(str(c) for c in table["columns"]))

    # Rows
    for row in table["rows"]:
        print("\t".join(str(v) for v in row))


def _merge_tables(*tables: dict[str, Any], key_col: int = 0) -> dict[str, Any]:
    """
    Merge multiple tables by combining their columns.

    Takes the key column from the first table, then appends all non-key columns
    from each table. All tables must have the same number of rows.

    Args:
        *tables: Tables to merge (each has 'columns' and 'rows')
        key_col: Index of the key column (default 0, typically 'asset')

    Returns:
        Merged table with combined columns and rows
    """
    if not tables:
        return {"columns": [], "rows": []}

    first = tables[0]
    n_rows = len(first["rows"])

    # Start with key column from first table
    columns = [first["columns"][key_col]]

    # Add non-key columns from each table
    for tbl in tables:
        for i, col in enumerate(tbl["columns"]):
            if i != key_col:
                columns.append(col)

    # Build rows: key value + non-key values from each table
    rows = []
    for row_idx in range(n_rows):
        row = [first["rows"][row_idx][key_col]]
        for tbl in tables:
            for i, val in enumerate(tbl["rows"][row_idx]):
                if i != key_col:
                    row.append(val)
        rows.append(row)

    return {"columns": columns, "rows": rows}


def _compile_rules(table: dict[str, Any]) -> list[tuple[str, re.Pattern, str]]:
    """
    Compile rules from a table with columns [field, rgx, value].

    Args:
        table: Dict with 'columns' and 'rows' from get_table()

    Returns:
        List of (field, compiled_pattern, value) tuples
    """
    cols = table["columns"]
    field_idx = cols.index("field")
    rgx_idx = cols.index("rgx")
    value_idx = cols.index("value")

    rules = []
    for rule_row in table["rows"]:
        field = rule_row[field_idx]
        pattern = re.compile(rule_row[rgx_idx])
        value = rule_row[value_idx]
        rules.append((field, pattern, value))
    return rules


def _match_rules(
    rules: list[tuple[str, re.Pattern, str]],
    field_values: dict[str, str],
    default: str = "error",
) -> str:
    """
    Find the first matching rule and return its value.

    Args:
        rules: List of (field, compiled_pattern, value) tuples
        field_values: Dict mapping field names to values
        default: Default value if no rule matches

    Returns:
        The value from the first matching rule, or default
    """
    for field, pattern, value in rules:
        field_val = field_values.get(field, "")
        if pattern.match(field_val):
            return value
    return default


def asset_table(
        path: str | Path, 
        table_name: str, 
        default: str = "",
        live_only: bool = False, 
        pattern: str = "."
    ) -> dict[str, Any]:
    """
    Evaluate a classification rule table against live assets 

    Matches each live asset against the rules in the specified table from the AMT file.
    Rules are evaluated in order, and the first matching rule determines the classification.

    Each rule specifies:
    - field: Which asset field to check ('Underlying' or 'Class')
    - rgx: Regex pattern to match against the field value
    - value: Value to assign if the pattern matches

    Args:
        path: Path to the AMT YAML file
        table_name: Name of the rule table (e.g., "group_table", "limit_overrides")
        default: Default value if no rule matches

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['asset', '<column_name>'] where column_name is table_name
        with '_table' suffix removed (if present)

    Example:
        >>> table = live_table("data/amt.yml", "group_table")
        >>> table['columns']
        ['asset', 'group']
        >>> table = live_table("data/amt.yml", "limit_overrides")
        >>> table['columns']
        ['asset', 'limit_overrides']
    """
    # Compute column name: remove _table suffix if present
    if table_name.endswith("_table"):
        col_name = table_name[:-6]  # Remove "_table"
    else:
        col_name = table_name

    rules = _compile_rules(get_table(path, table_name))

    rows = []
    for asset_data, underlying in _iter_assets(path, live_only=live_only,pattern=pattern):
        cls = asset_data.get("Class", "")
        field_values = {"Underlying": underlying, "Class": cls}
        classification = _match_rules(rules, field_values, default=default)
        rows.append([underlying, classification])

    return {
        "columns": ["asset", col_name],
        "rows": rows,
    }


#
# group table
#
def asset_group(path: str | Path, live_only: bool = False, pattern: str = ".") -> dict[str, Any]:
    """live with group, subgroup, liquidity, and limit override."""
    return _merge_tables(
        asset_table(path, "group_table", default="error",live_only=live_only,pattern=pattern),
        asset_table(path, "subgroup_table", default="",live_only=live_only,pattern=pattern),
        asset_table(path, "liquidity_table", default="1",live_only=live_only,pattern=pattern),
        asset_table(path, "limit_overrides", default="",live_only=live_only,pattern=pattern),
    )


# -------------------------------------
# CLI
# -------------------------------------
def _main() -> int:
    import argparse
    import yaml

    p = argparse.ArgumentParser(
        description="AMT loader - asset queries and table utilities.",
    )
    p.add_argument("path", nargs="?", help="Path to AMT YAML file")
    p.add_argument("--selftest", action="store_true", help="Run self-tests")
    p.add_argument("--get", "-g", metavar="UNDERLYING", help="Get asset by Underlying value")
    p.add_argument("--find", "-f", metavar="PATTERN", help="Find assets by regex pattern on Underlying")
    p.add_argument("--table", "-t", metavar="KEY_PATH", help="Get embedded table by key path")
    p.add_argument("--list", "-l", action="store_true", help="List all asset names from cache")
    p.add_argument("--all", "-a", action="store_true", help="List all assets")
    p.add_argument("--live", action="store_true", help="List all live assets (WeightCap > 0)")
    p.add_argument("--class", dest="live_class", action="store_true", help="List live assets with class info")
    p.add_argument("--group", dest="live_group", action="store_true", help="List live assets with group assignment")
    p.add_argument("--live-table", metavar="TABLE_NAME", help="Evaluate rule table against live assets")
    p.add_argument("--merge", "-m", metavar="TABLES", help="Merge live_table() results from comma-separated table names")
    p.add_argument("--value", "-v", metavar="KEY_PATH", help="Get value by dot-separated key path")
    p.add_argument("--aum", action="store_true", help="Get AUM value")
    p.add_argument("--leverage", action="store_true", help="Get leverage value")
    args = p.parse_args()

    if args.selftest:
        return _selftest()

    if not args.path:
        p.print_help()
        return 1

    if args.get:
        asset = get_asset(args.path, args.get)
        if asset:
            print(yaml.dump(asset, default_flow_style=False))
        else:
            print(f"Asset not found: {args.get}")
            return 1
    elif args.find:
        table = find_assets(args.path, args.find)
        if not table["rows"]:
            print(f"No assets found matching: {args.find}")
            return 1
        print_table(table)
    elif args.table:
        try:
            table = get_table(args.path, args.table)
            print(format_table(table))
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    elif args.list:
        print_table(cached_assets(args.path))
    elif args.all:
        print_table(assets(args.path))
    elif args.live:
        print_table(assets(args.path, live_only=True))
    elif args.live_class:
        print_table(asset_class(args.path, live_only=True))
    elif args.live_group:
        print_table(asset_group(args.path, live_only=True))
    elif args.live_table:
        try:
            print_table(asset_table(args.path, args.live_table, live_only=True))
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    elif args.merge:
        try:
            table_names = [t.strip() for t in args.merge.split(",")]
            tables = [asset_table(args.path, name, live_only=True) for name in table_names]
            print_table(_merge_tables(*tables))
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    elif args.value:
        val = get_value(args.path, args.value)
        if val is not None:
            print(val)
        else:
            print(f"Value not found: {args.value}")
            return 1
    elif args.aum:
        val = get_aum(args.path)
        if val is not None:
            print(val)
        else:
            print("AUM not found")
            return 1
    elif args.leverage:
        val = get_leverage(args.path)
        if val is not None:
            print(val)
        else:
            print("Leverage not found")
            return 1
    else:
        p.print_help()

    return 0


def _selftest() -> int:
    """Run self-tests for the loader module."""
    import tempfile
    import os

    print("Running loader self-tests...")

    # Create a temporary AMT file for testing
    test_amt = """
backtest:
  aum: 1000.0
  leverage: 10.0

amt:
  Asset1:
    Underlying: "TEST1 Comdty"
    Class: "Commodity"
    WeightCap: 0.05
  Asset2:
    Underlying: "TEST2 Equity"
    Class: "Equity"
    WeightCap: 0.0
  Asset3:
    Underlying: "TEST3 Rate"
    Class: "Rate"
    WeightCap: 0.10

group_table:
  Columns: [field, rgx, value]
  Rows:
    - [Class, "^Commodity$", "commodities"]
    - [Class, "^Equity$", "equities"]
    - [Class, ".*", "other"]
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(test_amt)
        test_path = f.name

    try:
        clear_cache()

        # Test load_amt
        data = load_amt(test_path)
        assert "amt" in data, "load_amt: missing 'amt' key"
        assert "backtest" in data, "load_amt: missing 'backtest' key"
        print("  load_amt: OK")

        # Test get_value
        assert get_value(test_path, "backtest.aum") == 1000.0, "get_value: wrong aum"
        assert get_value(test_path, "backtest.leverage") == 10.0, "get_value: wrong leverage"
        assert get_value(test_path, "nonexistent", "default") == "default", "get_value: default not returned"
        print("  get_value: OK")

        # Test get_aum / get_leverage
        assert get_aum(test_path) == 1000.0, "get_aum: wrong value"
        assert get_leverage(test_path) == 10.0, "get_leverage: wrong value"
        print("  get_aum/get_leverage: OK")

        # Test get_asset
        asset = get_asset(test_path, "TEST1 Comdty")
        assert asset is not None, "get_asset: not found"
        assert asset["Class"] == "Commodity", "get_asset: wrong class"
        assert get_asset(test_path, "NONEXISTENT") is None, "get_asset: should return None"
        print("  get_asset: OK")

        # Test find_assets
        found = find_assets(test_path, "^TEST")
        assert len(found["rows"]) == 3, f"find_assets: expected 3, got {len(found['rows'])}"
        found = find_assets(test_path, "Comdty$")
        assert len(found["rows"]) == 1, f"find_assets: expected 1, got {len(found['rows'])}"
        print("  find_assets: OK")

        # Test cached_assets
        cached = cached_assets(test_path)
        assert len(cached["rows"]) == 3, f"cached_assets: expected 3, got {len(cached['rows'])}"
        print("  cached_assets: OK")

        # Test assets
        all_a = assets(test_path)
        assert len(all_a["rows"]) == 3, f"assets: expected 3, got {len(all_a['rows'])}"
        print("  assets: OK")

        # Test assets with live_only=True (WeightCap > 0)
        live = assets(test_path, live_only=True)
        assert len(live["rows"]) == 2, f"assets(live_only=True): expected 2, got {len(live['rows'])}"
        print("  assets(live_only=True): OK")

        # Test get_table
        table = get_table(test_path, "group_table")
        assert table["columns"] == ["field", "rgx", "value"], "get_table: wrong columns"
        assert len(table["rows"]) == 3, f"get_table: expected 3 rows, got {len(table['rows'])}"
        print("  get_table: OK")

        # Test asset_table
        lt = asset_table(test_path, "group_table", live_only=True)
        assert lt["columns"] == ["asset", "group"], f"asset_table: wrong columns {lt['columns']}"
        assert len(lt["rows"]) == 2, f"asset_table: expected 2, got {len(lt['rows'])}"
        # TEST1 Comdty -> commodities, TEST3 Rate -> other
        print("  asset_table: OK")

        # Test _merge_tables
        t1 = {"columns": ["key", "a"], "rows": [["k1", 1], ["k2", 2]]}
        t2 = {"columns": ["key", "b"], "rows": [["k1", 10], ["k2", 20]]}
        merged = _merge_tables(t1, t2)
        assert merged["columns"] == ["key", "a", "b"], f"_merge_tables: wrong columns {merged['columns']}"
        assert merged["rows"] == [["k1", 1, 10], ["k2", 2, 20]], f"_merge_tables: wrong rows {merged['rows']}"
        print("  _merge_tables: OK")

        print("All loader self-tests passed!")
        return 0

    finally:
        os.unlink(test_path)
        clear_cache()


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(_main())