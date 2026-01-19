# -------------------------------------
# AMT tickers - Ticker extraction
# -------------------------------------
"""
Ticker extraction and transformation utilities.

Handles extracting tickers from assets, expanding futures specs,
split ticker handling, and straddle info building.
"""
import re
from pathlib import Path
from typing import Any
import csv
import calendar
import duckdb

from . import loader
from . import schedules

def _split_ticker(ticker: str, param: str) -> list[tuple[str, str]]:
    """Split a ticker if it's a "split ticker" format: ticker1:YYYY-MM:ticker2"""
    # Match pattern: something:YYYY-MM:something
    # The date part is 4 digits, dash, 2 digits
    match = re.match(r'^(.+):(\d{4}-\d{2}):(.+)$', ticker)
    if match:
        ticker1, date, ticker2 = match.group(1), match.group(2), match.group(3)
        return [(ticker1, f"{param}<{date}"), (ticker2, f"{param}>{date}")]
    return [(ticker, param)]


def _split_ticker_specstr(spec: str) -> tuple[str, str, str]:
    """Parse a source:ticker:field spec string."""
    parts = spec.split(":")
    if len(parts) < 3: return ("", "", "")
    source, ticker, field = parts[0], parts[1], parts[2]
    if source == "CV": field = "none"
    return (source, ticker, field)

def _make_ticker_specstr(spec: dict) -> str:
    """Format a ticker row as source:ticker:field spec string."""
    if spec["source"] == "CV": return f"{spec['source']}:{spec['field']}:none"
    return f"{spec['source']}:{spec['ticker']}:{spec['field']}"

def _parse_date_constraint(param: str, xpry: int, xprm: int) -> tuple[str, bool]:
    """Parse a param string with optional date constraint and check if included.

    Returns (clean_param, include) where include is True if the expiry date
    passes the constraint check.

    Constraints:
    - paramXYYYY-MM: valid for exactly that date (include if expiry == limit)
    - param<YYYY-MM: valid before that date (include if expiry < limit)
    - param>YYYY-MM: valid after that date (include if expiry > limit)
    """
    # Check for X (equality) first - must come before < and >
    if "X" in param:
        clean_param, date_str = param.split("X", 1)
        try:
            limit_year, limit_month = map(int, date_str.split("-"))
            include = (xpry, xprm) == (limit_year, limit_month)
            return (clean_param, include)
        except ValueError:
            return (param, True)
    elif "<" in param:
        clean_param, date_str = param.split("<", 1)
        try:
            limit_year, limit_month = map(int, date_str.split("-"))
            include = (xpry, xprm) < (limit_year, limit_month)
            return (clean_param, include)
        except ValueError:
            return (param, True)
    elif ">" in param:
        clean_param, date_str = param.split(">", 1)
        try:
            limit_year, limit_month = map(int, date_str.split("-"))
            include = (xpry, xprm) > (limit_year, limit_month)
            return (clean_param, include)
        except ValueError:
            return (param, True)
    return (param, True)


# -------------------------------------
# Ticker schemas
# -------------------------------------


def _market_tickers(market: dict, underlying: str, cls: str) -> list[list]:
    """Handle Market tickers - BBG source with Tickers list."""
    field = market.get("Field", "")
    tickers = market.get("Tickers", "")
    if isinstance(tickers, str):
        ticker_list = [tickers] if tickers else []
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        ticker_list = []
    return [[underlying, cls, "Market", "-", "BBG", t, field] for t in ticker_list]


def _vol_tickers(vol: dict, underlying: str, cls: str) -> list[list]:
    """Handle Vol tickers - Near and Far fields with deduplication."""
    source = vol.get("Source", "")
    ticker = vol.get("Ticker", "")
    near = vol.get("Near", "")
    far = vol.get("Far", "")

    rows = []
    seen = set()
    for param, field in [("Near", near), ("Far", far)]:
        if field and field != "NONE":
            key = (ticker, field)
            if key not in seen:
                seen.add(key)
                rows.append([underlying, cls, "Vol", param, source, ticker, field])
    return rows


def _hedge_nonfut(hedge: dict, underlying: str, cls: str) -> list[list]:
    """Handle 'nonfut' hedge - simple BBG ticker."""
    ticker = hedge.get("Ticker", "")
    field = hedge.get("Field", "")
    return [[underlying, cls, "Hedge", "hedge", "BBG", ticker, field]]


def _hedge_cds(hedge: dict, underlying: str, cls: str) -> list[list]:
    """Handle 'cds' hedge - two BBG tickers with PX_LAST."""
    rows = []
    for param in ["hedge", "hedge1"]:
        ticker = hedge.get(param, "")
        if ticker:
            rows.append([underlying, cls, "Hedge", param, "BBG", ticker, "PX_LAST"])
    return rows


def _hedge_fut(hedge: dict, underlying: str, cls: str) -> list[list]:
    """Handle 'fut' hedge - BBGfc source with spec string."""
    spec_parts = [f"{k}:{v}" for k, v in hedge.items() if k != "Source"]
    spec_str = ",".join(spec_parts)
    return [[underlying, cls, "Hedge", "fut", "BBGfc", spec_str, "PX_LAST"]]


def _hedge_calc(hedge: dict, underlying: str, cls: str) -> list[list]:
    """Handle 'calc' hedge - 4 calculated tickers."""
    ccy_list = hedge.get("ccy", [])
    tenor_list = hedge.get("tenor", [])
    ccy = ccy_list[0] if isinstance(ccy_list, list) and ccy_list else str(ccy_list)
    tenor = tenor_list[0] if isinstance(tenor_list, list) and tenor_list else str(tenor_list)
    return [
        [underlying, cls, "Hedge", "hedge", "calc", f"{ccy}_fsw0m_{tenor}", ""],
        [underlying, cls, "Hedge", "hedge1", "calc", f"{ccy}_fsw6m_{tenor}", ""],
        [underlying, cls, "Hedge", "hedge2", "calc", f"{ccy}_pva0m_{tenor}", ""],
        [underlying, cls, "Hedge", "hedge3", "calc", f"{ccy}_pva6m_{tenor}", ""],
    ]


def _hedge_default(hedge: dict, underlying: str, cls: str, source: str) -> list[list]:
    """Handle default hedge - condensed spec string."""
    spec_parts = [f"{k}:{v}" for k, v in hedge.items() if k != "Source"]
    spec_str = ",".join(spec_parts)
    return [[underlying, cls, "Hedge", source, source, "", spec_str]]


# Dispatch table for hedge source handlers
_HEDGE_HANDLERS = {
    "nonfut": _hedge_nonfut,
    "cds": _hedge_cds,
    "fut": _hedge_fut,
    "calc": _hedge_calc,
}


def get_tschemas(path: str | Path, underlying: str) -> dict[str, Any]:
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
    asset_data = loader.get_asset(path, underlying)
    if not asset_data:
        return {"columns": ["asset", "cls", "type", "param", "source", "ticker", "field"], "rows": []}

    rows = []
    asset_underlying = asset_data.get("Underlying", "")
    asset_class = asset_data.get("Class", "")

    # Market tickers
    market = asset_data.get("Market", {})
    if isinstance(market, dict):
        rows.extend(_market_tickers(market, asset_underlying, asset_class))

    # Vol tickers
    vol = asset_data.get("Vol", {})
    if isinstance(vol, dict):
        rows.extend(_vol_tickers(vol, asset_underlying, asset_class))

    # Hedge tickers
    hedge = asset_data.get("Hedge", {})
    if isinstance(hedge, dict):
        source = hedge.get("Source", "")
        handler = _HEDGE_HANDLERS.get(source)
        if handler:
            rows.extend(handler(hedge, asset_underlying, asset_class))
        else:
            rows.extend(_hedge_default(hedge, asset_underlying, asset_class, source))

    return {
        "columns": ["asset", "cls", "type", "param", "source", "ticker", "field"],
        "rows": rows,
    }


def find_tschemas(path: str | Path, pattern: str, live_only: bool = False) -> dict[str, Any]:
    """Find all tickers for assets matching a regex pattern on Underlying."""
    columns = ["asset", "cls", "type", "param", "source", "ticker", "field"]
    rows = []
    assets_table = loader.find_assets(path, pattern=pattern,live_only=live_only)
    for underlying in loader.table_column(assets_table, "asset"):
        table = get_tschemas(path, underlying)
        rows.extend(table["rows"])
    return {"columns": columns, "rows": rows}

# -------------------------------------
# Compute Tickers from Ticker schemas
# -------------------------------------

def fut_spec2ticker(spec: str, year: int, month: int) -> str:
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


def fut_norm2act(csv_path: str | Path, ticker: str) -> str | None:
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

        mapping = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = row.get("normalized_future", "")
                actual = row.get("actual_future", "")
                if normalized and actual: mapping[normalized] = actual
        _NORMALIZED_CACHE[csv_path] = mapping

    return _NORMALIZED_CACHE[csv_path].get(ticker)


def clear_normalized_cache() -> None:
    """Clear the normalized to actual futures cache."""
    _NORMALIZED_CACHE.clear()

def _tschma_dict_bbgfc_ym(
    tschema_dict: dict,
    year: int,
    month: int,
    chain_csv: str | Path | None = None
) -> list[dict]:
    spec = tschema_dict["ticker"]

    ticker = fut_spec2ticker(spec, year, month)
    new_tschema_dict = tschema_dict.copy()
    new_tschema_dict["param"] = f"hedgeX{year}-{month:02d}"
    # Try to look up actual ticker if chain_csv provided
    if chain_csv is not None:
        actual = fut_norm2act(chain_csv, ticker)
        if actual is not None:
            new_tschema_dict["source"] = "BBG"
            new_tschema_dict["ticker"] = actual
        else:
            new_tschema_dict["source"] = "nBBG"
            new_tschema_dict["ticker"] = ticker
    else:
        new_tschema_dict["source"] = "nBBG"
        new_tschema_dict["ticker"] = ticker

    return new_tschema_dict

def _tschma_dict_expand_bbgfc(
    tschema_dict: dict,
    start_year: int,
    end_year: int,
    chain_csv: str | Path | None = None
) -> list[dict]:
    """Expand a BBGfc row into monthly ticker rows."""
    expanded_tschema_dict = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            expanded_tschema_dict.append(_tschma_dict_bbgfc_ym(tschema_dict,year,month,chain_csv))
    return expanded_tschema_dict


def _tschma_dict_expand_split(ticker_dict: dict) -> list[dict]:
    """Expand a BBG row if it contains a split ticker."""
    split_result = _split_ticker(ticker_dict["ticker"], ticker_dict["param"])
    if len(split_result) == 1: return [ticker_dict]
    expanded_ticker_dict = []
    for new_ticker, new_param in split_result:
        new_row = ticker_dict.copy()
        new_row["param"] = new_param
        new_row["ticker"] = new_ticker
        expanded_ticker_dict.append(new_row)
    return expanded_ticker_dict

def find_tickers_ym(
    path: str | Path, 
    pattern: str,
    live_only: bool,
    year: int , 
    month: int , 
    chain_csv: str | Path | None = None
) -> dict[str, Any]:
    """Get all tickers for assets."""
    rows = []
    for _, underlying in loader._iter_assets(path, pattern=pattern, live_only=live_only):
        tschemas_table = get_tschemas(path, underlying)
        for tschema_row in tschemas_table["rows"]:
            tschema_dict = dict(zip(tschemas_table["columns"], tschema_row))
            source = tschema_dict["source"]
            if source == "BBGfc" and year is not None and month is not None:
                rows.append(_tschma_dict_bbgfc_ym(tschema_dict, year, month, chain_csv))
            elif source == "BBG":
                rows.extend(_tschma_dict_expand_split(tschema_dict))
            else:
                rows.append(tschema_dict)

    # Convert dict rows back to list rows
    ticker_rows = [[row[col] for col in tschemas_table["columns"]] for row in rows]

    return {
        "columns": tschemas_table["columns"],
        "rows": ticker_rows,
    }

def find_tickers(
    path: str | Path, 
    pattern: str,
    live_only: bool = True,
    start_year: int | None = None, 
    end_year: int | None = None, 
    chain_csv: str | Path | None = None
) -> dict[str, Any]:
    """Get all tickers for assets."""
    rows = []
    for _, underlying in loader._iter_assets(path, pattern=pattern, live_only=live_only):
        tschemas_table = get_tschemas(path, underlying)
        for tschema_row in tschemas_table["rows"]:
            tschema_dict = dict(zip(tschemas_table["columns"], tschema_row))
            source = tschema_dict["source"]
            if source == "BBGfc" and start_year is not None and end_year is not None:
                rows.extend(_tschma_dict_expand_bbgfc(tschema_dict, start_year, end_year, chain_csv))
            elif source == "BBG":
                rows.extend(_tschma_dict_expand_split(tschema_dict))
            else:
                rows.append(tschema_dict)

    # Convert dict rows back to list rows
    ticker_rows = [[row[col] for col in tschemas_table["columns"]] for row in rows]

    return {
        "columns": tschemas_table["columns"],
        "rows": ticker_rows,
    }

# -------------------------------------
# Compute Tickers from Ticker schemas
# straddle strings are inputs
# -------------------------------------

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

    # Get near/far code - determines which Vol field to use
    ntrc = parts[2]  # "N" or "F"
    vol_param = "Near" if ntrc == "N" else "Far"

    # Get tickers for the asset
    ticker_table = get_tschemas(path, underlying)
    if not ticker_table["rows"]:
        raise ValueError(f"No asset found with Underlying: {underlying}")

    ticker_columns = ticker_table["columns"]
    rows = []

    # Add asset and straddle info
    rows.append(["asset", underlying])
    rows.append(["straddle", straddle])

    # Add valuation info - comma-delimited name=value pairs
    asset_data = loader.get_asset(path, underlying)
    if asset_data:
        valuation = asset_data.get("Valuation", {})
        if isinstance(valuation, dict) and valuation:
            val_parts = [f"{k}={v}" for k, v in valuation.items()]
            rows.append(["valuation", ",".join(val_parts)])

    # Process tickers: expand BBGfc and split tickers for the expiry month
    for list_row in ticker_table["rows"]:
        row = dict(zip(ticker_columns, list_row))
        ticker_type = row["type"]
        source = row["source"]
        param = row["param"]

        if ticker_type == "Vol":
            # Vol tickers - only include the one matching ntrc (Near for N, Far for F)
            if param != vol_param:
                continue
            rows.append(["vol", _make_ticker_specstr(row)])

        elif ticker_type == "Hedge":
            if source == "BBGfc":
                # Expand for the specific expiry month
                expanded = _tschma_dict_expand_bbgfc(row, xpry, xpry, chain_csv)
                for exp_row in expanded:
                    clean_param, include = _parse_date_constraint(exp_row["param"], xpry, xprm)
                    if include: rows.append([clean_param, _make_ticker_specstr(exp_row)])
            elif source == "BBG":
                # Check for split tickers and apply date filter
                expanded = _tschma_dict_expand_split(row)
                for exp_row in expanded:
                    clean_param, include = _parse_date_constraint(exp_row["param"], xpry, xprm)
                    if include: rows.append([clean_param, _make_ticker_specstr(exp_row)])
            else:
                # Other hedge sources - include as-is
                rows.append([param, _make_ticker_specstr(row)])

    return {
        "columns": ["name", "value"],
        "rows": rows,
    }


# -------------------------------------
# Finally, Prices
# -------------------------------------

def straddle_days(table: dict[str, Any], prices_parquet: str | Path) -> dict[str, Any]:
    """
    Add daily price values to a straddle table for the entry month.

    Takes the output from asset_straddle() and adds rows for each calendar day
    in the entry month. Each date row has tab-separated values for vol and each
    hedge ticker.

    The vol ticker is parsed from the "vol" row (format: source:ticker:field).
    Hedge tickers are parsed from "hedge", "hedge1", etc. rows.
    For CV source, field "none" is used in the query.

    Args:
        table: Output from asset_straddle() with columns ['name', 'value']
        prices_parquet: Path to the prices parquet file

    Returns:
        New table with daily value rows inserted after the last ticker row.
        Daily rows have format: YYYY-MM-DD -> vol_value\\thedge_value\\thedge1_value...

    Example:
        >>> straddle = asset_straddle("data/amt.yml", "C Comdty", "|2023-12|2024-01|N|0|OVERRIDE||33.3|")
        >>> result = straddle_days(straddle, "data/prices.parquet")
        >>> # Date rows have tab-separated values for each ticker
    """


    # Copy existing rows
    rows = [row[:] for row in table["rows"]]

    # Find straddle, vol, and hedge rows
    straddle_str = None
    vol_spec = None
    hedge_specs = []  # List of (name, spec) tuples in order
    for name, value in table["rows"]:
        if name == "straddle":
            straddle_str = value
        elif name == "vol":
            vol_spec = value
        elif name.startswith("hedge") or name == "calc":
            hedge_specs.append((name, value))

    if not straddle_str or not vol_spec:
        return table  # Can't process without straddle and vol

    # Parse straddle to get entry month: |ntry-ntrm|xpry-xprm|ntrc|...|
    parts = straddle_str.strip("|").split("|")
    if len(parts) < 1:
        return table

    entry_part = parts[0]  # "2023-12"
    try:
        ntry, ntrm = entry_part.split("-")
        ntry = int(ntry)
        ntrm = int(ntrm)
    except ValueError:
        return table

    # Get number of days in the entry month
    _, num_days = calendar.monthrange(ntry, ntrm)
    start_date = f"{ntry}-{ntrm:02d}-01"
    end_date = f"{ntry}-{ntrm:02d}-{num_days:02d}"

    # Collect all ticker/field pairs to query
    ticker_field_pairs = []  # List of (ticker, field) tuples
    _, vol_ticker, vol_field = _split_ticker_specstr(vol_spec)
    if vol_ticker:
        ticker_field_pairs.append((vol_ticker, vol_field))

    hedge_ticker_fields = []  # Keep track of hedge ticker/field for result mapping
    for _, hedge_spec in hedge_specs:
        _, hedge_ticker, hedge_field = _split_ticker_specstr(hedge_spec)
        hedge_ticker_fields.append((hedge_ticker, hedge_field))
        if hedge_ticker:
            ticker_field_pairs.append((hedge_ticker, hedge_field))

    # Query all prices in one go
    prices_parquet = Path(prices_parquet)
    con = duckdb.connect()
    table_name = prices_parquet.stem
    con.execute(f"CREATE VIEW {table_name} AS SELECT * FROM '{prices_parquet}'")

    # Build a single query for all ticker/field pairs
    all_prices: dict[tuple[str, str], dict[str, str]] = {}
    if ticker_field_pairs:
        # Build WHERE clause with OR conditions for each ticker/field pair
        conditions = " OR ".join(
            f"(ticker = '{t}' AND field = '{f}')" for t, f in ticker_field_pairs
        )
        query = f"""
            SELECT ticker, field, date, value
            FROM {table_name}
            WHERE ({conditions})
            AND date >= '{start_date}'
            AND date <= '{end_date}'
            ORDER BY date
        """
        result = con.execute(query).fetchall()

        # Organize results by (ticker, field) -> {date -> value}
        for ticker, field, date, value in result:
            key = (ticker, field)
            if key not in all_prices:
                all_prices[key] = {}
            all_prices[key][str(date)] = str(value)

    con.close()

    # Extract prices for vol and each hedge
    vol_prices = all_prices.get((vol_ticker, vol_field), {}) if vol_ticker else {}
    hedge_prices_list = []
    for hedge_ticker, hedge_field in hedge_ticker_fields:
        if hedge_ticker:
            hedge_prices_list.append(all_prices.get((hedge_ticker, hedge_field), {}))
        else:
            hedge_prices_list.append({})

    # Find where to insert daily rows (after the last ticker row: vol, hedge, hedge1, calc)
    insert_idx = None
    ticker_names = {"vol", "calc"} | {name for name, _ in hedge_specs}
    for i, (name, _) in enumerate(rows):
        if name in ticker_names:
            insert_idx = i

    if insert_idx is not None:
        insert_idx += 1  # Insert after the last ticker row

    # Build daily rows with tab-separated values
    daily_rows = []
    for day in range(1, num_days + 1):
        date_str = f"{ntry}-{ntrm:02d}-{day:02d}"
        values = [vol_prices.get(date_str, "none")]
        for hedge_prices in hedge_prices_list:
            values.append(hedge_prices.get(date_str, "none"))
        # Join with tabs
        combined_value = "\t".join(values)
        daily_rows.append([date_str, combined_value])

    # Insert daily rows
    if insert_idx is not None:
        rows = rows[:insert_idx] + daily_rows + rows[insert_idx:]
    else:
        rows.extend(daily_rows)

    return {
        "columns": ["name", "value"],
        "rows": rows,
    }



# -------------------------------------
# CLI
# -------------------------------------


def _main() -> int:
    import argparse
    from .loader import print_table

    p = argparse.ArgumentParser(
        description="Ticker extraction and transformation utilities.",
    )
    p.add_argument("path", help="Path to AMT YAML file")
    p.add_argument("--chain-csv", default="data/futs.csv",
                   help="CSV file with normalized_future,actual_future columns (default: data/futs.csv)")
    p.add_argument("--prices", default="data/prices.parquet",
                   help="Prices parquet file (default: data/prices.parquet)")

    # Commands
    p.add_argument("--asset-tschemas", metavar="UNDERLYING", type=str,
                   help="Get all ticker schemas for an asset by Underlying value.")
    
    p.add_argument("--find-tschemas", nargs=2, type=str, metavar=("PATTERN", "LIVE"),  
                   help="Get all ticker schemas for assets matching pattern.")
    
    p.add_argument("--find-tickers", nargs=4, type=str, metavar=( "PATTERN", "LIVE", "START_YEAR", "END_YEAR"),
                   help="Get all tickers for assets in period.")
    
    p.add_argument("--find-tickers-ym", nargs=4, type=str, metavar=( "PATTERN", "LIVE", "YEAR", "MONTH"),
                   help="Get all tickers for assets on specific month.")

    p.add_argument("--expand-ym", nargs=4, type=str, metavar=("PATTERN", "LIVE","YEAR", "MONTH"),
                   help="Get straddles for assets matching patterns on month.")
    
    p.add_argument("--get-expand-ym", nargs=3, type=str, metavar=("ASSET", "YEAR", "MONTH"),
                   help="Get straddles for asset on month.")
    
    p.add_argument("--straddle-days", nargs=2, metavar=("UNDERLYING", "STRADDLE"),
                   help="Get straddle info with daily prices for entry month.")
    
    p.add_argument("--fut", nargs=3, metavar=("SPEC", "YEAR", "MONTH"),
                   help="Compute futures ticker from spec string, year, and month.")
    
    args = p.parse_args()

    def str2bool(s: str) -> bool:
        s = s.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Expected a boolean, got {s!r}")

    if args.asset_tschemas:
        table = get_tschemas(args.path, args.asset_tschemas)
        if not table["rows"]:
            print(f"No asset found with Underlying: {args.asset_tickers}")
            return 1
        print_table(table)

    elif args.find_tschemas is not None:
        table = find_tschemas(args.path, args.find_tschemas[0], str2bool(args.find_tschemas[1]) )
        loader.print_table(table)

    elif args.find_tickers is not None:
        table = find_tickers(
            args.path, 
            args.find_tickers[0], 
            str2bool(args.find_tickers[1]),
            int(args.find_tickers[2]),
            int(args.find_tickers[3]),
            args.chain_csv
        )
        loader.print_table(table)

    elif args.find_tickers_ym is not None:
        table = find_tickers_ym(
            args.path, 
            args.find_tickers_ym[0], 
            str2bool(args.find_tickers_ym[1]),
            int(args.find_tickers_ym[2]),
            int(args.find_tickers_ym[3]),
            args.chain_csv
        )
        loader.print_table(table)

    elif args.fut:
        spec, year, month = args.fut
        ticker = fut_spec2ticker(spec, int(year), int(month))
        print(ticker)

    elif args.expand_ym:
        pattern, live, year, month = args.expand_ym
        table = schedules.expand_ym( args.path, int(year), int(month), pattern, str2bool(live) )
        loader.print_table(table)

    elif args.get_expand_ym:
        asset, year, month = args.get_expand_ym
        table = schedules.get_expand_ym( args.path, asset, int(year), int(month) )
        loader.print_table(table)

    elif args.straddle_days:
        underlying, straddle_str = args.straddle_days
        table = asset_straddle(args.path, underlying, straddle_str, args.chain_csv)
        table = straddle_days(table, args.prices)
        loader.print_table(table)

    else:
        p.print_help()

    return 0


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(_main())