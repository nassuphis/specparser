# -------------------------------------
# AMT (Asset Management Table) utilities
# -------------------------------------
"""
Module for processing AMT YAML files.

AMT files are YAML files with a specific structure containing asset definitions
under the 'amt' key, where each asset has an 'Underlying' identifier.
"""
import re
from pathlib import Path
from typing import Any

import yaml


# Module-level cache for loaded AMT data
_AMT_CACHE: dict[str, dict[str, Any]] = {}


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
    return data


def clear_cache():
    """Clear the AMT file cache."""
    _AMT_CACHE.clear()


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


def get_asset(path: str | Path, name: str) -> dict[str, Any] | None:
    """
    Get asset data by its YAML key (asset name).

    Args:
        path: Path to the AMT YAML file
        name: The asset name (YAML key under 'amt')

    Returns:
        The asset dict if found, None otherwise

    Example:
        >>> asset = get_asset("data/amt.yml", "LA Comdty OLD")
        >>> asset["Underlying"]
        'LA Comdty'
    """
    data = load_amt(path)
    amt = data.get("amt", {})

    asset = amt.get(name)
    if isinstance(asset, dict):
        return asset

    return None


def find_by_underlying(path: str | Path, underlying: str) -> list[tuple[str, dict[str, Any]]]:
    """
    Find all assets with a given Underlying value.

    Args:
        path: Path to the AMT YAML file
        underlying: The Underlying value to search for

    Returns:
        List of (name, asset_dict) tuples for matching assets

    Example:
        >>> matches = find_by_underlying("data/amt.yml", "LA Comdty")
        >>> [(name, a["Description"]) for name, a in matches]
        [('LA Comdty OLD', 'LME PRI ALUM FUTR')]
    """
    data = load_amt(path)
    amt = data.get("amt", {})

    matches = []
    for name, asset_data in amt.items():
        if isinstance(asset_data, dict) and asset_data.get("Underlying") == underlying:
            matches.append((name, asset_data))

    return matches


def get_schedule(path: str | Path, name: str) -> list[str] | None:
    """
    Get the expiry schedule for an asset by its YAML key.

    Looks up the asset's 'Options' field to get the schedule name,
    then returns the schedule from 'expiry_schedules'.

    Args:
        path: Path to the AMT YAML file
        name: The asset name (YAML key under 'amt')

    Returns:
        List of schedule entries, or None if asset or schedule not found

    Example:
        >>> get_schedule("data/amt.yml", "LA Comdty OLD")
        ['N0_OVERRIDE_33.3', 'N5_OVERRIDE_33.3', 'F10_OVERRIDE_12.5', 'F15_OVERRIDE_12.5']
    """
    data = load_amt(path)
    amt = data.get("amt", {})

    asset = amt.get(name)
    if not isinstance(asset, dict):
        return None

    schedule_name = asset.get("Options")
    if not schedule_name:
        return None

    schedules = data.get("expiry_schedules", {})
    schedule = schedules.get(schedule_name)

    if isinstance(schedule, list):
        return schedule

    return None


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


def _split_ticker(ticker: str, param: str) -> list[tuple[str, str]]:
    """
    Split a ticker if it's a "split ticker" format: ticker1:YYYY-MM:ticker2

    Returns a list of (ticker, param) tuples.
    For normal tickers, returns [(ticker, param)].
    For split tickers, returns [(ticker1, param<YYYY-MM), (ticker2, param>YYYY-MM)].

    The < suffix indicates the ticker is valid before the date.
    The > suffix indicates the ticker is valid after the date.
    """
    # Match pattern: something:YYYY-MM:something
    # The date part is 4 digits, dash, 2 digits
    match = re.match(r'^(.+):(\d{4}-\d{2}):(.+)$', ticker)
    if match:
        ticker1 = match.group(1)
        date = match.group(2)
        ticker2 = match.group(3)
        return [(ticker1, f"{param}<{date}"), (ticker2, f"{param}>{date}")]
    return [(ticker, param)]


def asset_tickers(path: str | Path, underlying: str) -> dict[str, Any]:
    """
    Get all tickers for an asset by its Underlying value.

    Returns a table with columns: ['asset', 'cls', 'type', 'param', 'source', 'ticker', 'field']

    Market tickers have source "BBG" and use the Market.Field value.
    Vol tickers use Vol.Source and have fields "Near" and "Far".

    Args:
        path: Path to the AMT YAML file
        underlying: The Underlying value to search for

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['asset', 'cls', 'type', 'param', 'source', 'ticker', 'field']

    Example:
        >>> table = asset_tickers("data/amt.yml", "LA Comdty")
        >>> table['columns']
        ['asset', 'cls', 'type', 'param', 'source', 'ticker', 'field']
    """
    matches = find_by_underlying(path, underlying)
    if not matches:
        return {"columns": ["asset", "cls", "type", "param", "source", "ticker", "field"], "rows": []}

    rows = []
    for name, asset_data in matches:
        asset_underlying = asset_data.get("Underlying", "")
        asset_class = asset_data.get("Class", "")

        # Market tickers - source is always "BBG"
        market = asset_data.get("Market", {})
        if isinstance(market, dict):
            field = market.get("Field", "")
            tickers = market.get("Tickers", "")
            # Handle both string and list of tickers
            if isinstance(tickers, str):
                ticker_list = [tickers] if tickers else []
            elif isinstance(tickers, list):
                ticker_list = tickers
            else:
                ticker_list = []
            for ticker in ticker_list:
                rows.append([asset_underlying, asset_class, "Market", "-", "BBG", ticker, field])

        # Vol tickers
        vol = asset_data.get("Vol", {})
        if isinstance(vol, dict):
            source = vol.get("Source", "")
            ticker = vol.get("Ticker", "")
            near = vol.get("Near", "")
            far = vol.get("Far", "")
            # Track (ticker, field) pairs to avoid duplicates
            vol_seen = set()
            # Skip "NONE" values
            if near and near != "NONE":
                key = (ticker, near)
                if key not in vol_seen:
                    vol_seen.add(key)
                    rows.append([asset_underlying, asset_class, "Vol", "Near", source, ticker, near])
            if far and far != "NONE":
                key = (ticker, far)
                if key not in vol_seen:
                    vol_seen.add(key)
                    rows.append([asset_underlying, asset_class, "Vol", "Far", source, ticker, far])

        # Hedge tickers
        hedge = asset_data.get("Hedge", {})
        if isinstance(hedge, dict):
            source = hedge.get("Source", "")
            if source == "nonfut":
                # nonfut has Ticker and Field, source is always "BBG"
                ticker = hedge.get("Ticker", "")
                field = hedge.get("Field", "")
                rows.append([asset_underlying, asset_class, "Hedge", "nonfut", "BBG", ticker, field])
            elif source == "cds":
                # cds: two rows for hedge and hedge1, each with their ticker value
                hedge_ticker = hedge.get("hedge", "")
                hedge1_ticker = hedge.get("hedge1", "")
                if hedge_ticker:
                    rows.append([asset_underlying, asset_class, "Hedge", "hedge", "BBG", hedge_ticker, "PX_LAST"])
                if hedge1_ticker:
                    rows.append([asset_underlying, asset_class, "Hedge", "hedge1", "BBG", hedge1_ticker, "PX_LAST"])
            elif source == "fut":
                # fut: source is BBGfc, spec string goes to ticker, field is PX_LAST
                spec_parts = []
                for key, val in hedge.items():
                    if key != "Source":
                        spec_parts.append(f"{key}:{val}")
                spec_str = ",".join(spec_parts)
                rows.append([asset_underlying, asset_class, "Hedge", "fut", "BBGfc", spec_str, "PX_LAST"])
            elif source == "calc":
                # calc: spec string goes to ticker column, field is empty
                spec_parts = []
                for key, val in hedge.items():
                    if key != "Source":
                        spec_parts.append(f"{key}:{val}")
                spec_str = ",".join(spec_parts)
                rows.append([asset_underlying, asset_class, "Hedge", "calc", "calc", spec_str, ""])
            else:
                # other sources - condense all non-Source fields into a spec string in field column
                spec_parts = []
                for key, val in hedge.items():
                    if key != "Source":
                        spec_parts.append(f"{key}:{val}")
                spec_str = ",".join(spec_parts)
                rows.append([asset_underlying, asset_class, "Hedge", source, source, "", spec_str])

    return {
        "columns": ["asset", "cls", "type", "param", "source", "ticker", "field"],
        "rows": rows,
    }


def fut_ticker(spec: str, year: int, month: int) -> str:
    """
    Compute the actual futures ticker from a spec string and year/month.

    The spec string is in the format:
        generic:LA1 Comdty,fut_code:LA,fut_month_map:FGHJKMNQUVXZ,min_year_offset:0,market_code:Comdty

    The calculation:
    1. Get fut_month_code from fut_month_map at position (month - 1)
    2. Get opt_month_code from standard month map "FGHJKMNQUVXZ" at position (month - 1)
    3. If fut_month_code < opt_month_code alphabetically, the futures contract is
       in the next year, so add 1 to year. Also respect min_year_offset.
    4. Build ticker as "{fut_code}{fut_month_code}{year} {market_code}"

    Args:
        spec: Spec string with fut_code, fut_month_map, min_year_offset, market_code
        year: Option expiry year (e.g., 2024)
        month: Option expiry month (1-12)

    Returns:
        Futures ticker string (e.g., "LAG2024 Comdty")

    Example:
        >>> fut_ticker("generic:LA1 Comdty,fut_code:LA,fut_month_map:FGHJKMNQUVXZ,min_year_offset:0,market_code:Comdty", 2024, 7)
        'LAN2024 Comdty'
    """
    # Parse the spec string into a dict
    spec_dict = {}
    for part in spec.split(","):
        if ":" in part:
            key, val = part.split(":", 1)
            spec_dict[key] = val

    fut_code = spec_dict.get("fut_code", "")
    fut_month_map = spec_dict.get("fut_month_map", "FGHJKMNQUVXZ")
    min_year_offset = int(spec_dict.get("min_year_offset", "0"))
    market_code = spec_dict.get("market_code", "")

    # Standard month map for comparison
    month_map = "FGHJKMNQUVXZ"

    # Get month codes (month is 1-based, so use month-1 for 0-based index)
    fut_month_code = fut_month_map[month - 1]
    opt_month_code = month_map[month - 1]

    # Calculate year offset: if fut_month_code < opt_month_code, add 1 year
    # (this means the futures contract rolls to next year before the option expires)
    year_offset = max(1 if fut_month_code < opt_month_code else 0, min_year_offset)
    fut_year = year + year_offset

    # Build ticker
    return f"{fut_code}{fut_month_code}{fut_year} {market_code}"


# Cache for normalized to actual futures mapping
_NORMALIZED_CACHE: dict[str, dict[str, str]] = {}


def normalized2actual(csv_path: str | Path, ticker: str) -> str | None:
    """
    Convert a normalized BBG futures ticker to the actual BBG ticker.

    Uses the CSV lookup table to map normalized tickers (e.g., "LAF2025 Comdty")
    to actual BBG tickers (e.g., "LA F25 Comdty").

    The CSV is loaded once and cached for subsequent lookups.

    Args:
        csv_path: Path to the CSV file with normalized_future,actual_future columns
        ticker: The normalized futures ticker to look up

    Returns:
        The actual BBG ticker if found, or None if not found

    Example:
        >>> actual = normalized2actual("data/current_bbg_chain_data.csv", "LAF2025 Comdty")
        >>> actual  # Returns actual ticker or None
    """
    csv_path = str(Path(csv_path).resolve())

    # Load and cache the CSV if not already cached
    if csv_path not in _NORMALIZED_CACHE:
        import csv
        mapping = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = row.get("normalized_future", "")
                actual = row.get("actual_future", "")
                if normalized and actual:
                    mapping[normalized] = actual
        _NORMALIZED_CACHE[csv_path] = mapping

    return _NORMALIZED_CACHE[csv_path].get(ticker)


def clear_normalized_cache() -> None:
    """Clear the normalized to actual futures cache."""
    _NORMALIZED_CACHE.clear()


def _expand_bbgfc_row(
    row: dict,
    start_year: int,
    end_year: int,
    chain_csv: str | Path | None = None
) -> list[dict]:
    """
    Expand a BBGfc row into monthly ticker rows.

    Takes a row with source "BBGfc" and expands it into individual monthly ticker rows
    for each month from January of start_year through December of end_year.

    If chain_csv is provided, normalized tickers are looked up and converted to actual
    BBG tickers. If lookup succeeds, source becomes "BBG". If lookup fails, source is "nBBG".

    Args:
        row: A dict with keys [asset, cls, type, param, source, ticker, field] where source is "BBGfc"
             and ticker is the futures spec string (e.g., "LA{M}{YY} Comdty")
        start_year: Start year for expansion (inclusive)
        end_year: End year for expansion (inclusive)
        chain_csv: Optional path to CSV with normalized_future,actual_future columns

    Returns:
        List of expanded row dicts, one per month in the date range
    """
    expanded_rows = []
    spec = row["ticker"]

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            ticker = fut_ticker(spec, year, month)
            new_row = row.copy()
            new_row["param"] = f"hedgeX{year}-{month:02d}"

            # Try to look up actual ticker if chain_csv provided
            if chain_csv is not None:
                actual = normalized2actual(chain_csv, ticker)
                if actual is not None:
                    new_row["source"] = "BBG"
                    new_row["ticker"] = actual
                else:
                    new_row["source"] = "nBBG"
                    new_row["ticker"] = ticker
            else:
                new_row["source"] = "nBBG"
                new_row["ticker"] = ticker

            expanded_rows.append(new_row)

    return expanded_rows


def _expand_split_ticker_row(row: dict) -> list[dict]:
    """
    Expand a BBG row if it contains a split ticker.

    Split ticker format: ticker1:YYYY-MM:ticker2
    - ticker1 is valid before the date (param gets <YYYY-MM suffix)
    - ticker2 is valid after the date (param gets >YYYY-MM suffix)

    If the ticker is not a split ticker, returns the row unchanged (as single-item list).

    Args:
        row: A dict with keys [asset, cls, type, param, source, ticker, field] where source is "BBG"

    Returns:
        List of row dicts - either one (unchanged) or two (split into before/after)
    """
    ticker = row["ticker"]
    param = row["param"]

    split_result = _split_ticker(ticker, param)
    if len(split_result) == 1:
        # Not a split ticker, return unchanged
        return [row]

    # Split ticker - create two rows
    expanded_rows = []
    for new_ticker, new_param in split_result:
        new_row = row.copy()
        new_row["param"] = new_param
        new_row["ticker"] = new_ticker
        expanded_rows.append(new_row)

    return expanded_rows


def asset_straddle(
    path: str | Path,
    underlying: str,
    straddle: str,
    chain_csv: str | Path | None = None
) -> dict[str, Any]:
    """
    Build a straddle info table with asset metadata and relevant tickers.

    Takes an underlying and a packed straddle string (e.g., |2023-12|2024-01|N|0|OVERRIDE||33.3|)
    and returns a table with name/value pairs containing:
    - asset: the underlying
    - straddle: the packed straddle string
    - Vol and Hedge tickers formatted as source:ticker:field

    The straddle format is: |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|

    Tickers are selected based on the expiry year/month from the straddle:
    - Vol tickers (Near/Far) are included
    - Hedge tickers matching the expiry month (hedgeX{year}-{month:02d}) are included

    Args:
        path: Path to the AMT YAML file
        underlying: The Underlying value to search for
        straddle: Packed straddle string (e.g., |2023-12|2024-01|N|0|OVERRIDE||33.3|)
        chain_csv: Optional path to CSV for normalized to actual ticker lookup

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['name', 'value']

    Example:
        >>> table = asset_straddle("data/amt.yml", "C Comdty", "|2023-12|2024-01|N|0|OVERRIDE||33.3|")
        >>> table['columns']
        ['name', 'value']
    """
    # Parse straddle: |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|
    parts = straddle.strip("|").split("|")
    if len(parts) < 7:
        raise ValueError(f"Invalid straddle format: {straddle}")

    expiry_part = parts[1]  # "2024-01"

    # Parse expiry year and month
    try:
        xpry, xprm = expiry_part.split("-")
        xpry = int(xpry)
        xprm = int(xprm)
    except ValueError as e:
        raise ValueError(f"Invalid expiry format in straddle: {expiry_part}") from e

    # Get tickers for the asset
    ticker_table = asset_tickers(path, underlying)
    if not ticker_table["rows"]:
        raise ValueError(f"No asset found with Underlying: {underlying}")

    ticker_columns = ticker_table["columns"]
    rows = []

    # Add asset and straddle info
    rows.append(["asset", underlying])
    rows.append(["straddle", straddle])

    # Process tickers: expand BBGfc and split tickers for the expiry month
    for list_row in ticker_table["rows"]:
        row = dict(zip(ticker_columns, list_row))
        ticker_type = row["type"]
        source = row["source"]
        param = row["param"]

        if ticker_type == "Vol":
            # Vol tickers - include as-is with param as name
            # Format: source:ticker:field
            value = f"{row['source']}:{row['ticker']}:{row['field']}"
            rows.append([f"Vol.{param}", value])

        elif ticker_type == "Hedge":
            if source == "BBGfc":
                # Expand for the specific expiry month
                expanded = _expand_bbgfc_row(row, xpry, xpry, chain_csv)
                # Filter to the specific month
                target_param = f"hedgeX{xpry}-{xprm:02d}"
                for exp_row in expanded:
                    if exp_row["param"] == target_param:
                        value = f"{exp_row['source']}:{exp_row['ticker']}:{exp_row['field']}"
                        # Strip date suffix from param (hedgeXYYYY-MM -> hedge)
                        clean_param = "hedge"
                        rows.append([f"Hedge.{clean_param}", value])
            elif source == "BBG":
                # Check for split tickers and apply date filter
                expanded = _expand_split_ticker_row(row)
                for exp_row in expanded:
                    exp_param = exp_row["param"]
                    # For split tickers with date constraints, check if valid for expiry
                    include = True
                    clean_param = exp_param  # Default to unchanged
                    if "<" in exp_param:
                        # param<YYYY-MM means valid before that date
                        clean_param, date_str = exp_param.split("<", 1)
                        try:
                            limit_year, limit_month = map(int, date_str.split("-"))
                            if (xpry, xprm) >= (limit_year, limit_month):
                                include = False
                        except ValueError:
                            pass  # Keep the ticker if date parsing fails
                    elif ">" in exp_param:
                        # param>YYYY-MM means valid after that date
                        clean_param, date_str = exp_param.split(">", 1)
                        try:
                            limit_year, limit_month = map(int, date_str.split("-"))
                            if (xpry, xprm) <= (limit_year, limit_month):
                                include = False
                        except ValueError:
                            pass  # Keep the ticker if date parsing fails

                    if include:
                        value = f"{exp_row['source']}:{exp_row['ticker']}:{exp_row['field']}"
                        rows.append([f"Hedge.{clean_param}", value])
            else:
                # Other hedge sources - include as-is
                value = f"{row['source']}:{row['ticker']}:{row['field']}"
                rows.append([f"Hedge.{param}", value])

    return {
        "columns": ["name", "value"],
        "rows": rows,
    }


def live_tickers(path: str | Path, start_year: int | None = None, end_year: int | None = None, chain_csv: str | Path | None = None) -> dict[str, Any]:
    """
    Get all tickers for all live assets.

    Returns a table with columns: ['asset', 'cls', 'type', 'param', 'source', 'ticker', 'field']

    Market tickers have source "BBG" and use the Market.Field value.
    Vol tickers use Vol.Source and have fields "Near" and "Far".

    If start_year and end_year are provided, BBGfc rows are expanded into monthly
    tickers using the fut_ticker function for each month from Jan start_year to Dec end_year.

    If chain_csv is provided, normalized tickers (nBBG) are looked up and converted
    to actual BBG tickers. If the lookup succeeds, source becomes "BBG" and the ticker
    is replaced with the actual ticker. If the lookup fails, source stays "nBBG".

    Args:
        path: Path to the AMT YAML file
        start_year: Optional start year for BBGfc expansion
        end_year: Optional end year for BBGfc expansion
        chain_csv: Optional path to CSV with normalized_future,actual_future columns

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['asset', 'cls', 'type', 'param', 'source', 'ticker', 'field']

    Example:
        >>> table = live_tickers("data/amt.yml")
        >>> table['columns']
        ['asset', 'cls', 'type', 'param', 'source', 'ticker', 'field']
        >>> table = live_tickers("data/amt.yml", 2024, 2025)  # Expands BBGfc to monthly tickers
        >>> table = live_tickers("data/amt.yml", 2024, 2025, "data/current_bbg_chain_data.csv")  # With lookup
    """
    columns = ["asset", "cls", "type", "param", "source", "ticker", "field"]
    rows = []

    for _, _, _, underlying, _ in _iter_assets(path, live_only=True):
        asset_table = asset_tickers(path, underlying)
        for list_row in asset_table["rows"]:
            # Convert list row to dict for easier manipulation
            row = dict(zip(columns, list_row))
            source = row["source"]

            if source == "BBGfc" and start_year is not None and end_year is not None:
                rows.extend(_expand_bbgfc_row(row, start_year, end_year, chain_csv))
            elif source == "BBG":
                rows.extend(_expand_split_ticker_row(row))
            else:
                rows.append(row)

    # Convert dict rows back to list rows
    list_rows = [[row[col] for col in columns] for row in rows]

    return {
        "columns": columns,
        "rows": list_rows,
    }


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


def _split_code_value(s: str) -> tuple[str, str]:
    """
    Split a string into uppercase alphabetic code prefix and trailing value.

    The code is the leading uppercase letters. The value is everything after
    (digits or lowercase letters like a,b,c,d for fix_expiry).

    Examples:
        'N5' -> ('N', '5')
        'BD15' -> ('BD', '15')
        'LBD' -> ('LBD', '')
        'F3' -> ('F', '3')
        'BDa' -> ('BD', 'a')
        'BDb' -> ('BD', 'b')
    """
    m = re.match(r"^([A-Z]+)(.*)$", s)
    if m:
        return m.group(1), m.group(2)
    return s, ""


def live_schedules(path: str | Path) -> dict[str, Any]:
    """
    Get all live assets (WeightCap > 0) with their schedule components.

    Each schedule component is expanded to its own row, with the component split
    on '_' into entry code/value and expiry code/value columns.

    Args:
        path: Path to the AMT YAML file

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['assid', 'schcnt', 'schid', 'asset', 'wcap', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']

    Example:
        >>> table = live_schedules("data/amt.yml")
        >>> table['columns']
        ['assid', 'schcnt', 'schid', 'asset', 'wcap', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']
    """
    data = load_amt(path)
    schedules = data.get("expiry_schedules", {})

    rows = []
    for assid, _, asset_data, underlying, wcap in _iter_assets(path, live_only=True):
        schedule_name = asset_data.get("Options")
        schedule = schedules.get(schedule_name) if schedule_name else None
        if isinstance(schedule, list) and schedule:
            schcnt = len(schedule)
            for schid, component in enumerate(schedule, start=1):
                # Split component on '_' into entry, expiry, weight
                parts = component.split("_")
                entry = parts[0] if len(parts) > 0 else ""
                expiry = parts[1] if len(parts) > 1 else ""
                wgt = parts[2] if len(parts) > 2 else ""
                # Split entry into code and value
                ntrc, ntrv = _split_code_value(entry)
                # Split expiry into code and value
                xprc, xprv = _split_code_value(expiry)
                rows.append([assid, schcnt, schid, underlying, wcap, ntrc, ntrv, xprc, xprv, wgt])
        else:
            # No schedule - still include the row with empty columns
            rows.append([assid, 0, 0, underlying, wcap, "", "", "", "", ""])

    return {
        "columns": ["assid", "schcnt", "schid", "asset", "wcap", "ntrc", "ntrv", "xprc", "xprv", "wgt"],
        "rows": rows,
    }


def fix_expiry(table: dict[str, Any]) -> dict[str, Any]:
    """
    Transform expiry values with lowercase letters [a,b,c,d] to computed values.

    Replaces ntrv/xprv values like 'a', 'b', 'c', 'd' with a computed number:
        (schedule_id - 1) * day_stride + day_offset
    where:
        day_offset = asset_id % 5 + 1
        day_stride = 20 / (schedule_count + 1)

    Args:
        table: Dict with 'columns' and 'rows' from live_schedules()

    Returns:
        New table dict with transformed entv/xprv values

    Example:
        >>> table = live_schedules("data/amt.yml")
        >>> fixed = fix_expiry(table)
    """
    import re

    # Pattern to match lowercase a, b, c, d (possibly with digits after)
    pattern = re.compile(r"^([abcd])(\d*)$")

    # Get column indices
    cols = table["columns"]
    asset_id_idx = cols.index("assid")
    schedule_count_idx = cols.index("schcnt")
    schedule_id_idx = cols.index("schid")
    ntrv_idx = cols.index("ntrv")
    xprv_idx = cols.index("xprv")

    new_rows = []
    for row in table["rows"]:
        new_row = list(row)
        asset_id = row[asset_id_idx]
        schedule_count = row[schedule_count_idx]
        schedule_id = row[schedule_id_idx]

        # Compute the replacement value
        if schedule_count > 0:
            day_offset = int(asset_id % 5 + 1)
            day_stride = int(20 / (schedule_count + 1))
            value = int(schedule_id - 1) * day_stride + day_offset
        else:
            value = int(asset_id % 5 + 1)

        # Transform ntrv if it matches pattern
        ntrv = str(row[ntrv_idx])
        m = pattern.match(ntrv)
        if m:
            new_row[ntrv_idx] = str(value)

        # Transform xprv if it matches pattern
        xprv = str(row[xprv_idx])
        m = pattern.match(xprv)
        if m:
            new_row[xprv_idx] = str(value)

        new_rows.append(new_row)

    return {
        "columns": table["columns"],
        "rows": new_rows,
    }


def _expand_schedules(table: dict[str, Any], start_year: int, end_year: int) -> dict[str, Any]:
    """
    Expand a schedules table across a year/month range.

    Computes the cartesian product of:
    - years: start_year to end_year (inclusive)
    - months: 1 to 12
    - rows from the input table

    Also computes entry year (ntry) and entry month (ntrm) based on ntrc:
    - N (Near): 1 month before expiry
    - F (Far): 2 months before expiry

    Args:
        table: Dict with 'columns' and 'rows' from live_schedules() or fix_expiry()
        start_year: Start year (inclusive)
        end_year: End year (inclusive)

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
    """
    asset_rows = table["rows"]
    cols = table["columns"]
    ntrc_idx = cols.index("ntrc")

    rows = []
    for xpry in range(start_year, end_year + 1):
        for xprm in range(1, 13):
            for asset_row in asset_rows:
                ntrc = asset_row[ntrc_idx]
                # Compute entry year/month based on Near (N) or Far (F)
                if ntrc == "N":
                    offset = 1
                elif ntrc == "F":
                    offset = 2
                else:
                    offset = 0
                total_months = xpry * 12 + (xprm - 1) - offset
                ntry = total_months // 12
                ntrm = (total_months % 12) + 1
                rows.append([xpry, xprm, ntry, ntrm] + asset_row)

    return {
        "columns": ["xpry", "xprm", "ntry", "ntrm"] + table["columns"],
        "rows": rows,
    }


def expand_schedules(path: str | Path, start_year: int, end_year: int) -> dict[str, Any]:
    """
    Expand all live schedules across a year/month range.

    Computes the cartesian product of:
    - years: start_year to end_year (inclusive)
    - months: 1 to 12
    - rows from live_schedules() (already expanded by schedule component)

    Args:
        path: Path to the AMT YAML file
        start_year: Start year (inclusive)
        end_year: End year (inclusive)

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['xpry', 'xprm', 'ntry', 'ntrm', 'assid', 'schcnt', 'schid', 'asset', 'wcap', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']

    Example:
        >>> table = expand_schedules("data/amt.yml", 2024, 2025)
        >>> table['columns']
        ['xpry', 'xprm', 'ntry', 'ntrm', 'assid', 'schcnt', 'schid', 'asset', 'wcap', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']
    """
    live = live_schedules(path)
    return _expand_schedules(live, start_year, end_year)


def expand_schedules_fixed(path: str | Path, start_year: int, end_year: int) -> dict[str, Any]:
    """
    Expand all live schedules across a year/month range, with expiry codes fixed.

    Same as expand_schedules but applies fix_expiry() first to transform
    expiry values like 'a', 'b' to computed numbers.

    Args:
        path: Path to the AMT YAML file
        start_year: Start year (inclusive)
        end_year: End year (inclusive)

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['xpry', 'xprm', 'ntry', 'ntrm', 'assid', 'schcnt', 'schid', 'asset', 'wcap', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']

    Example:
        >>> table = expand_schedules_fixed("data/amt.yml", 2024, 2025)
        >>> table['columns']
        ['xpry', 'xprm', 'ntry', 'ntrm', 'assid', 'schcnt', 'schid', 'asset', 'wcap', 'ntrc', 'ntrv', 'xprc', 'xprv', 'wgt']
    """
    live = live_schedules(path)
    fixed = fix_expiry(live)
    return _expand_schedules(fixed, start_year, end_year)


def pack_straddle(table: dict[str, Any]) -> dict[str, Any]:
    """
    Pack expanded schedule rows into straddle strings grouped by asset.

    Takes an expanded table (from expand_schedules_fixed) and packs
    each row into a pipe-delimited string:
        |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|

    Returns a table with columns ['asset', 'straddle'].

    Args:
        table: Dict with 'columns' and 'rows' from expand_schedules_fixed()

    Returns:
        Dict with keys: 'columns' (list), 'rows' (list of lists)
        Columns are ['asset', 'straddle']

    Example:
        >>> expanded = expand_schedules_fixed("data/amt.yml", 2024, 2024)
        >>> packed = pack_straddle(expanded)
        >>> packed['columns']
        ['asset', 'straddle']
    """
    cols = table["columns"]
    xpry_idx = cols.index("xpry")
    xprm_idx = cols.index("xprm")
    ntry_idx = cols.index("ntry")
    ntrm_idx = cols.index("ntrm")
    asset_idx = cols.index("asset")
    ntrc_idx = cols.index("ntrc")
    ntrv_idx = cols.index("ntrv")
    xprc_idx = cols.index("xprc")
    xprv_idx = cols.index("xprv")
    wgt_idx = cols.index("wgt")

    rows = []
    for row in table["rows"]:
        ntry = row[ntry_idx]
        ntrm = row[ntrm_idx]
        xpry = row[xpry_idx]
        xprm = row[xprm_idx]
        asset = row[asset_idx]
        ntrc = row[ntrc_idx]
        ntrv = row[ntrv_idx]
        xprc = row[xprc_idx]
        xprv = row[xprv_idx]
        wgt = row[wgt_idx]

        # Format: |ntry-ntrm|xpry-xprm|ntrc|ntrv|xprc|xprv|wgt|
        straddle = f"|{ntry}-{ntrm:02d}|{xpry}-{xprm:02d}|{ntrc}|{ntrv}|{xprc}|{xprv}|{wgt}|"
        rows.append([asset, straddle])

    return {
        "columns": ["asset", "straddle"],
        "rows": rows,
    }


# ============================================================
# CLI
# ============================================================

def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="AMT (Asset Management Table) utilities.",
    )
    p.add_argument("path", help="Path to AMT YAML file")
    p.add_argument("--get", "-g", metavar="NAME", help="Get asset by name (YAML key)")
    p.add_argument("--find", "-f", metavar="UNDERLYING", help="Find assets by Underlying value")
    p.add_argument("--schedule", "-s", metavar="UNDERLYING", help="Get expiry schedule for asset by Underlying value")
    p.add_argument("--table", "-t", metavar="KEY_PATH", help="Get embedded table by key path (e.g., group_risk_multiplier_table)")
    p.add_argument("--list", "-l", action="store_true", help="List all asset names")
    p.add_argument("--all", "-a", action="store_true", help="List all assets with their weight caps")
    p.add_argument("--live", action="store_true", help="List all live assets (weight_cap > 0)")
    p.add_argument("--class", dest="live_class", action="store_true", help="List live assets with class and source info")
    p.add_argument("--group", dest="live_group", action="store_true", help="List live assets with group assignment")
    p.add_argument("--live-table", metavar="TABLE_NAME", help="List live assets with values from a rule table")
    p.add_argument("--asset-tickers", metavar="UNDERLYING", help="Get all tickers for an asset by Underlying value")
    p.add_argument("--live-tickers", nargs="*", type=int, metavar=("START_YEAR", "END_YEAR"), help="Get all tickers for all live assets (optional: START_YEAR END_YEAR to expand BBGfc)")
    p.add_argument("--chain-csv", metavar="CSV_PATH", help="CSV file with normalized_future,actual_future columns for ticker lookup (use with --live-tickers)")
    p.add_argument("--schedules-raw", action="store_true", help="List all live assets with their schedules")
    p.add_argument("--schedules", action="store_true", help="List all live assets with schedules, expiry codes fixed")
    p.add_argument("--expand-raw", nargs=2, type=int, metavar=("START_YEAR", "END_YEAR"), help="Expand live schedules across year/month range")
    p.add_argument("--expand", nargs=2, type=int, metavar=("START_YEAR", "END_YEAR"), help="Expand live schedules with fixed expiry codes")
    p.add_argument("--pack", nargs=2, type=int, metavar=("START_YEAR", "END_YEAR"), help="Expand and pack into straddle strings")
    p.add_argument("--value", "-v", metavar="KEY_PATH", help="Get value by dot-separated key path (e.g., backtest.aum)")
    p.add_argument("--aum", action="store_true", help="Get AUM value")
    p.add_argument("--leverage", action="store_true", help="Get leverage value")
    p.add_argument("--fut", nargs=3, metavar=("SPEC", "YEAR", "MONTH"), help="Compute futures ticker from spec string, year, and month")
    p.add_argument("--straddle", nargs=2, metavar=("UNDERLYING", "STRADDLE"), help="Get straddle info with tickers for an asset")
    args = p.parse_args()

    if args.get:
        asset = get_asset(args.path, args.get)
        if asset:
            print(yaml.dump(asset, default_flow_style=False))
        else:
            print(f"Asset not found: {args.get}")
            return 1
    elif args.find:
        matches = find_by_underlying(args.path, args.find)
        if matches:
            for name, asset in matches:
                print(f"--- {name} ---")
                print(yaml.dump(asset, default_flow_style=False))
        else:
            print(f"No assets with Underlying: {args.find}")
            return 1
    elif args.schedule:
        matches = find_by_underlying(args.path, args.schedule)
        if not matches:
            print(f"No asset with Underlying: {args.schedule}")
            return 1
        # Use the first match
        name, _ = matches[0]
        schedule = get_schedule(args.path, name)
        if schedule:
            for entry in schedule:
                parts = entry.split("_")
                print("\t".join(parts))
        else:
            print(f"No schedule found for: {args.schedule}")
            return 1
    elif args.table:
        try:
            table = get_table(args.path, args.table)
            print(format_table(table))
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    elif args.list:
        for name in list_assets(args.path):
            print(name)
    elif args.all:
        table = assets(args.path)
        print_table(table)
    elif args.live:
        table = live_assets(args.path)
        print_table(table)
    elif args.live_class:
        table = live_class(args.path)
        print_table(table)
    elif args.live_group:
        table = live_group(args.path)
        print_table(table)
    elif args.live_table:
        try:
            table = live_table(args.path, args.live_table)
            print_table(table)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    elif args.asset_tickers:
        table = asset_tickers(args.path, args.asset_tickers)
        if table["rows"]:
            print_table(table)
        else:
            print(f"No asset found with Underlying: {args.asset_tickers}")
            return 1
    elif args.live_tickers is not None:
        if len(args.live_tickers) == 0:
            # No years provided - just list tickers without expansion
            table = live_tickers(args.path, chain_csv=args.chain_csv)
        elif len(args.live_tickers) == 2:
            # Years provided - expand BBGfc to monthly tickers
            start_year, end_year = args.live_tickers
            table = live_tickers(args.path, start_year, end_year, chain_csv=args.chain_csv)
        else:
            print("Error: --live-tickers requires either no arguments or exactly 2 (START_YEAR END_YEAR)")
            return 1
        print_table(table)
    elif args.schedules_raw:
        table = live_schedules(args.path)
        print_table(table)
    elif args.schedules:
        table = live_schedules(args.path)
        table = fix_expiry(table)
        print_table(table)
    elif args.expand_raw:
        start_year, end_year = args.expand_raw
        table = expand_schedules(args.path, start_year, end_year)
        print_table(table)
    elif args.expand:
        start_year, end_year = args.expand
        table = expand_schedules_fixed(args.path, start_year, end_year)
        print_table(table)
    elif args.pack:
        start_year, end_year = args.pack
        table = expand_schedules_fixed(args.path, start_year, end_year)
        table = pack_straddle(table)
        print_table(table)
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
    elif args.fut:
        spec, year_str, month_str = args.fut
        try:
            year = int(year_str)
            month = int(month_str)
            ticker = fut_ticker(spec, year, month)
            print(ticker)
        except (ValueError, IndexError) as e:
            print(f"Error computing futures ticker: {e}")
            return 1
    elif args.straddle:
        underlying, straddle_str = args.straddle
        try:
            table = asset_straddle(args.path, underlying, straddle_str, chain_csv=args.chain_csv)
            print_table(table)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        p.print_help()

    return 0


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(_main())
