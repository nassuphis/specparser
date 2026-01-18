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


def get_value(path: str | Path, key_path: str, default: Any = None) -> Any:
    """
    Get a value from an AMT file by its dot-separated key path.

    Args:
        path: Path to the AMT YAML file
        key_path: Dot-separated path to the value (e.g., "backtest.aum")
        default: Default value if the key is not found

    Returns:
        The value at the key path, or default if not found

    Example:
        >>> get_value("data/amt.yml", "backtest.aum")
        800.0
        >>> get_value("data/amt.yml", "backtest.leverage")
        20.0
    """
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
    """
    Get the AUM (Assets Under Management) value from an AMT file.

    Args:
        path: Path to the AMT YAML file

    Returns:
        The AUM value, or None if not found

    Example:
        >>> get_aum("data/amt.yml")
        800.0
    """
    return get_value(path, "backtest.aum")


def get_leverage(path: str | Path) -> float | None:
    """
    Get the leverage value from an AMT file.

    Args:
        path: Path to the AMT YAML file

    Returns:
        The leverage value, or None if not found

    Example:
        >>> get_leverage("data/amt.yml")
        20.0
    """
    return get_value(path, "backtest.leverage")


def get_asset(path: str | Path, underlying: str) -> dict[str, Any] | None:
    """
    Get asset data by its Underlying value.

    Uses a cached lookup for O(1) access.

    Args:
        path: Path to the AMT YAML file
        underlying: The Underlying value to search for

    Returns:
        The asset dict if found, None otherwise

    Example:
        >>> asset = get_asset("data/amt.yml", "LA Comdty")
        >>> asset["Description"]
        'LME PRI ALUM FUTR'
    """
    path = Path(path)
    path_str = str(path.resolve())

    # Ensure cache is built
    if path_str not in _ASSET_BY_UNDERLYING:
        load_amt(path)

    return _ASSET_BY_UNDERLYING.get(path_str, {}).get(underlying)


def find_underlyings(path: str | Path, pattern: str) -> list[str]:
    """
    Find all Underlying values matching a regex pattern.

    Args:
        path: Path to the AMT YAML file
        pattern: Regex pattern to match against Underlying values

    Returns:
        List of matching Underlying values

    Example:
        >>> find_underlyings("data/amt.yml", "^LA.*")
        ['LA Comdty', 'LA Comdty OLD']
        >>> find_underlyings("data/amt.yml", ".*Equity$")
        ['AAPL US Equity', 'MSFT US Equity', ...]
    """
    path = Path(path)
    path_str = str(path.resolve())

    # Ensure cache is built
    if path_str not in _ASSET_BY_UNDERLYING:
        load_amt(path)

    regex = re.compile(pattern)
    return [u for u in _ASSET_BY_UNDERLYING.get(path_str, {}) if regex.search(u)]


def list_assets(path: str | Path) -> list[str]:
    """
    List all asset names (YAML keys) in an AMT file.

    Args:
        path: Path to the AMT YAML file

    Returns:
        List of asset names (YAML keys under 'amt')
    """
    data = load_amt(path)
    amt = data.get("amt", {})

    names = []
    for name, asset_data in amt.items():
        if isinstance(asset_data, dict):
            names.append(name)

    return names


def get_table(path: str | Path, key_path: str) -> dict[str, Any]:
    """
    Get an embedded table from an AMT file by its key path.

    Tables have the structure:
        Columns: [col1, col2, ...]     # Required
        Types:   [type1, type2, ...]   # Optional
        Rows:                          # Required
        - [val1, val2, ...]
        - [val1, val2, ...]

    Args:
        path: Path to the AMT YAML file
        key_path: Dot-separated path to the table (e.g., "group_risk_multiplier_table")

    Returns:
        Dict with keys: 'columns' (list), 'types' (list or None), 'rows' (list of lists)

    Raises:
        ValueError: If the path doesn't lead to a valid table

    Example:
        >>> table = get_table("data/amt.yml", "group_risk_multiplier_table")
        >>> table['columns']
        ['group', 'multiplier']
        >>> table['rows'][0]
        ['rates', 1.0]
    """
    data = load_amt(path)

    # Navigate to the key path
    current = data
    for key in key_path.split("."):
        if not isinstance(current, dict):
            raise ValueError(f"Cannot navigate to '{key}' - parent is not a dict")
        if key not in current:
            raise ValueError(f"Key '{key}' not found in path '{key_path}'")
        current = current[key]

    # Validate table structure
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

    return {
        "columns": columns,
        "types": types,
        "rows": rows,
    }


def format_table(table: dict[str, Any]) -> str:
    """
    Format a table dict as a tab-separated string with header.

    Args:
        table: Dict with 'columns' and 'rows' keys (from get_table)

    Returns:
        Tab-separated string with header row

    Example:
        >>> table = {'columns': ['a', 'b'], 'rows': [[1, 2], [3, 4]]}
        >>> print(format_table(table))
        a\tb
        1\t2
        3\t4
    """
    lines = []

    # Header
    lines.append("\t".join(str(c) for c in table["columns"]))

    # Rows
    for row in table["rows"]:
        lines.append("\t".join(str(v) for v in row))

    return "\n".join(lines)


def print_table(table: dict[str, Any]) -> None:
    """
    Print a table with header and rows to stdout.

    Args:
        table: Dict with 'columns' and 'rows' keys (from get_table)
    """
    # Header
    print("\t".join(str(c) for c in table["columns"]))

    # Rows
    for row in table["rows"]:
        print("\t".join(str(v) for v in row))


def _iter_assets(path: str | Path, live_only: bool = False):
    """
    Iterate over assets in an AMT file.

    Args:
        path: Path to the AMT YAML file
        live_only: If True, only yield assets with WeightCap > 0

    Yields:
        Tuples of (asset_id, name, asset_data, underlying, wcap)
    """
    data = load_amt(path)
    amt = data.get("amt", {})

    for asset_id, (name, asset_data) in enumerate(amt.items()):
        if isinstance(asset_data, dict):
            underlying = asset_data.get("Underlying")
            wcap = asset_data.get("WeightCap")
            if underlying is not None:
                if live_only and (wcap is None or wcap <= 0):
                    continue
                yield asset_id, name, asset_data, underlying, wcap


def assets(path: str | Path) -> dict[str, Any]:
    """
    Get all assets with their Underlying and WeightCap values.

    Args:
        path: Path to the AMT YAML file

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['asset', 'wcap']

    Example:
        >>> table = assets("data/amt.yml")
        >>> table['columns']
        ['asset', 'wcap']
    """
    rows = []
    for _, _, _, underlying, wcap in _iter_assets(path, live_only=False):
        rows.append([underlying, wcap])

    return {
        "columns": ["asset", "wcap"],
        "rows": rows,
    }


def live_assets(path: str | Path) -> dict[str, Any]:
    """
    Get all live assets (WeightCap > 0) with their Underlying and WeightCap values.

    Args:
        path: Path to the AMT YAML file

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['asset', 'wcap']

    Example:
        >>> table = live_assets("data/amt.yml")
        >>> table['columns']
        ['asset', 'wcap']
    """
    rows = []
    for _, _, _, underlying, wcap in _iter_assets(path, live_only=True):
        rows.append([underlying, wcap])

    return {
        "columns": ["asset", "wcap"],
        "rows": rows,
    }


def live_class(path: str | Path) -> dict[str, Any]:
    """
    Get all live assets (WeightCap > 0) with their class and source information.

    Args:
        path: Path to the AMT YAML file

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['asset', 'cls', 'volsrc', 'hdgsrc', 'model']

    Example:
        >>> table = live_class("data/amt.yml")
        >>> table['columns']
        ['asset', 'cls', 'volsrc', 'hdgsrc', 'model']
    """
    rows = []
    for _, _, asset_data, underlying, _ in _iter_assets(path, live_only=True):
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


def live_table(path: str | Path, table_name: str, default: str = "") -> dict[str, Any]:
    """
    Get all live assets with values from a rule table.

    Matches each live asset against the rules in the specified table from the AMT file.
    Rules are evaluated in order, and the first matching rule determines the value.

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
    for _, _, asset_data, underlying, _ in _iter_assets(path, live_only=True):
        cls = asset_data.get("Class", "")
        field_values = {"Underlying": underlying, "Class": cls}
        value = _match_rules(rules, field_values, default=default)
        rows.append([underlying, value])

    return {
        "columns": ["asset", col_name],
        "rows": rows,
    }


def live_group(path: str | Path) -> dict[str, Any]:
    """
    Get all live assets (WeightCap > 0) with their group, subgroup, liquidity, and limit override.

    Group, subgroup, liquidity, and limit override are determined by matching each asset
    against the rules in the 'group_table', 'subgroup_table', 'liquidity_table', and
    'limit_overrides' from the AMT file. Rules are evaluated in order, and the first
    matching rule determines the value.

    Each rule specifies:
    - field: Which asset field to check ('Underlying' or 'Class')
    - rgx: Regex pattern to match against the field value
    - value: Value to assign if the pattern matches

    Args:
        path: Path to the AMT YAML file

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['asset', 'grp', 'sgrp', 'lqdty', 'lmtovr']

    Example:
        >>> table = live_group("data/amt.yml")
        >>> table['columns']
        ['asset', 'grp', 'sgrp', 'lqdty', 'lmtovr']
    """
    # Compile rules from all tables
    group_rules = _compile_rules(get_table(path, "group_table"))
    subgroup_rules = _compile_rules(get_table(path, "subgroup_table"))
    liquidity_rules = _compile_rules(get_table(path, "liquidity_table"))
    limit_rules = _compile_rules(get_table(path, "limit_overrides"))

    rows = []
    for _, _, asset_data, underlying, _ in _iter_assets(path, live_only=True):
        cls = asset_data.get("Class", "")

        # Build field lookup dict
        field_values = {
            "Underlying": underlying,
            "Class": cls,
        }

        # Find first matching rule for group, subgroup, liquidity, and limit override
        grp = _match_rules(group_rules, field_values, default="error")
        sgrp = _match_rules(subgroup_rules, field_values, default="")
        lqdty = _match_rules(liquidity_rules, field_values, default="1")
        lmtovr = _match_rules(limit_rules, field_values, default="")

        rows.append([underlying, grp, sgrp, lqdty, lmtovr])

    return {
        "columns": ["asset", "grp", "sgrp", "lqdty", "lmtovr"],
        "rows": rows,
    }
