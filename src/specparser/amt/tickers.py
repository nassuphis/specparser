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

from .loader import get_asset, _iter_assets


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
    asset_data = get_asset(path, underlying)
    if not asset_data:
        return {"columns": ["asset", "cls", "type", "param", "source", "ticker", "field"], "rows": []}

    rows = []
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
            rows.append([asset_underlying, asset_class, "Hedge", "hedge", "BBG", ticker, field])
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
            # calc: generate 4 hedge tickers from ccy and tenor
            # ccy and tenor are lists, take first element
            ccy_list = hedge.get("ccy", [])
            tenor_list = hedge.get("tenor", [])
            ccy = ccy_list[0] if isinstance(ccy_list, list) and ccy_list else str(ccy_list)
            tenor = tenor_list[0] if isinstance(tenor_list, list) and tenor_list else str(tenor_list)
            # Generate 4 tickers: fsw0m, fsw6m, pva0m, pva6m
            # Field is empty string for calc tickers (as stored in prices parquet)
            rows.append([asset_underlying, asset_class, "Hedge", "hedge", "calc", f"{ccy}_fsw0m_{tenor}", ""])
            rows.append([asset_underlying, asset_class, "Hedge", "hedge1", "calc", f"{ccy}_fsw6m_{tenor}", ""])
            rows.append([asset_underlying, asset_class, "Hedge", "hedge2", "calc", f"{ccy}_pva0m_{tenor}", ""])
            rows.append([asset_underlying, asset_class, "Hedge", "hedge3", "calc", f"{ccy}_pva6m_{tenor}", ""])
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

    # Get near/far code - determines which Vol field to use
    ntrc = parts[2]  # "N" or "F"
    vol_param = "Near" if ntrc == "N" else "Far"

    # Get tickers for the asset
    ticker_table = asset_tickers(path, underlying)
    if not ticker_table["rows"]:
        raise ValueError(f"No asset found with Underlying: {underlying}")

    ticker_columns = ticker_table["columns"]
    rows = []

    # Add asset and straddle info
    rows.append(["asset", underlying])
    rows.append(["straddle", straddle])

    # Add valuation info - comma-delimited name=value pairs
    asset_data = get_asset(path, underlying)
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
            # Format: source:ticker:field
            # For CV source, the ticker is in the field column and field should be "none"
            if row['source'] == "CV":
                value = f"{row['source']}:{row['field']}:none"
            else:
                value = f"{row['source']}:{row['ticker']}:{row['field']}"
            rows.append(["vol", value])

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
                        rows.append([clean_param, value])
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
                        rows.append([clean_param, value])
            else:
                # Other hedge sources - include as-is
                value = f"{row['source']}:{row['ticker']}:{row['field']}"
                rows.append([param, value])

    return {
        "columns": ["name", "value"],
        "rows": rows,
    }


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
    import calendar
    import duckdb

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

    def parse_ticker_spec(spec: str) -> tuple[str, str]:
        """Parse source:ticker:field spec and return (ticker, field)."""
        parts = spec.split(":")
        if len(parts) < 3:
            return ("", "")
        source = parts[0]
        ticker = parts[1]
        field = parts[2]
        # For CV source, use "none" as field
        if source == "CV":
            field = "none"
        return (ticker, field)

    # Collect all ticker/field pairs to query
    ticker_field_pairs = []  # List of (ticker, field) tuples
    vol_ticker, vol_field = parse_ticker_spec(vol_spec)
    if vol_ticker:
        ticker_field_pairs.append((vol_ticker, vol_field))

    hedge_ticker_fields = []  # Keep track of hedge ticker/field for result mapping
    for _, hedge_spec in hedge_specs:
        hedge_ticker, hedge_field = parse_ticker_spec(hedge_spec)
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
