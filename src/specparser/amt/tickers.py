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
from . import chain
from . import asset_straddle_tickers

# -------------------------------------
# Tschemas cache
# -------------------------------------

_TSCHEMAS_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_MEMOIZE_ENABLED: bool = True


def set_memoize_enabled(enabled: bool) -> None:
    """Enable or disable memoization for ticker functions."""
    global _MEMOIZE_ENABLED
    _MEMOIZE_ENABLED = enabled


def clear_ticker_caches() -> None:
    """Clear all ticker-related caches."""
    _TSCHEMAS_CACHE.clear()
    _TICKERS_YM_CACHE.clear()


# -------------------------------------
# Fetch tickers
# -------------------------------------




# -------------------------------------
# Tschemas 
# -------------------------------------

def _split_ticker(ticker: str, param: str) -> list[tuple[str, str]]:
    """Split a ticker if it's a "split ticker" format: ticker1:YYYY-MM:ticker2"""
    # Match pattern: something:YYYY-MM:something
    # The date part is 4 digits, dash, 2 digits
    match = re.match(r'^(.+):(\d{4}-\d{2}):(.+)$', ticker)
    if match:
        ticker1, date, ticker2 = match.group(1), match.group(2), match.group(3)
        return [(ticker1, f"{param}<{date}"), (ticker2, f"{param}>{date}")]
    return [(ticker, param)]


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


_TSCHEMA_COLUMNS = ["asset", "cls", "type", "param", "source", "ticker", "field"]

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
    path_str = str(Path(path).resolve())
    cache_key = (path_str, underlying)
    if _MEMOIZE_ENABLED and cache_key in _TSCHEMAS_CACHE:
        return _TSCHEMAS_CACHE[cache_key]

    asset_data = loader.get_asset(path, underlying)
    if not asset_data:
        result = {"orientation": "row", "columns": _TSCHEMA_COLUMNS, "rows": []}
        if _MEMOIZE_ENABLED:
            _TSCHEMAS_CACHE[cache_key] = result
        return result

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

    result = {
        "orientation": "row",
        "columns": _TSCHEMA_COLUMNS,
        "rows": rows,
    }
    if _MEMOIZE_ENABLED:
        _TSCHEMAS_CACHE[cache_key] = result
    return result


def find_tschemas(path: str | Path, pattern: str, live_only: bool = False) -> dict[str, Any]:
    """Find all tickers for assets matching a regex pattern on Underlying."""
    rows = []
    assets_table = loader.find_assets(path, pattern=pattern,live_only=live_only)
    for underlying in loader.table_column(assets_table, "asset"):
        table = get_tschemas(path, underlying)
        rows.extend(table["rows"])
    return {"orientation": "row", "columns": _TSCHEMA_COLUMNS, "rows": rows}

# -------------------------------------
# Compute Tickers from Ticker schemas
# -------------------------------------

def fut_spec2ticker(spec: str, year: int, month: int) -> str:
    """Compute the actual futures ticker from a spec string and year/month."""
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


def _tschema_dict_bbgfc_ym(
    tschema_dict: dict,
    year: int,
    month: int,
    chain_csv: str | Path | None = None
) -> dict:
    spec = tschema_dict["ticker"]

    ticker = fut_spec2ticker(spec, year, month)
    new_tschema_dict = tschema_dict.copy()
    new_tschema_dict["param"] = f"hedgeX{year}-{month:02d}"
    # Try to look up actual ticker if chain_csv provided
    if chain_csv is not None:
        actual = chain.fut_norm2act(chain_csv, ticker)
        if actual is not None:
            new_tschema_dict["source"] = "BBG"
            new_tschema_dict["ticker"] = actual
        else:
            new_tschema_dict["source"] = "nBBG"
            new_tschema_dict["ticker"] = ticker
    else:
        new_tschema_dict["source"] = "BBG"
        new_tschema_dict["ticker"] = ticker

    return new_tschema_dict

def _tschema_dict_expand_bbgfc(
    tschema_dict: dict,
    start_year: int,
    end_year: int,
    chain_csv: str | Path | None = None
) -> list[dict]:
    """Expand a BBGfc row into monthly ticker rows."""
    expanded_tschema_dict = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            expanded_tschema_dict.append(_tschema_dict_bbgfc_ym(tschema_dict,year,month,chain_csv))
    return expanded_tschema_dict


def _tschema_dict_expand_split(ticker_dict: dict) -> list[dict]:
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

# -------------------------------------
# Tickers YM cache
# -------------------------------------

_TICKERS_YM_CACHE: dict[tuple[str, str, int, int, str | None], dict[str, Any]] = {}


def get_tickers_ym(
    path: str | Path,
    asset: str,
    year: int ,
    month: int ,
    chain_csv: str | Path | None = None
) -> dict[str, Any]:
    path_str = str(Path(path).resolve())
    chain_str = str(Path(chain_csv).resolve()) if chain_csv else None
    cache_key = (path_str, asset, year, month, chain_str)
    if _MEMOIZE_ENABLED and cache_key in _TICKERS_YM_CACHE:
        return _TICKERS_YM_CACHE[cache_key]

    tschemas_table = get_tschemas(path, asset)
    ticker_dicts = []
    for tschema_row in tschemas_table["rows"]:
        tschema_dict = dict(zip(tschemas_table["columns"], tschema_row))
        source = tschema_dict["source"]
        if source == "BBGfc" or source == "BBG":
            if source == "BBGfc" and year is not None and month is not None:
                expanded_tschema_dict = [_tschema_dict_bbgfc_ym(tschema_dict, year, month, chain_csv)]
            elif source == "BBG":
                expanded_tschema_dict = _tschema_dict_expand_split(tschema_dict)
            for expanded_row in expanded_tschema_dict:
                    clean_param, include = _parse_date_constraint(expanded_row["param"], year, month)
                    if include:
                        new_row = expanded_row.copy()
                        new_row["param"]=clean_param
                        ticker_dicts.append(new_row)
        else:
            ticker_dicts.append(tschema_dict)
    ticker_rows = [[row[col] for col in tschemas_table["columns"]] for row in ticker_dicts]
    result = { "orientation": "row", "columns": tschemas_table["columns"], "rows": ticker_rows }
    if _MEMOIZE_ENABLED:
        _TICKERS_YM_CACHE[cache_key] = result
    return result

def find_tickers_ym(
    path: str | Path,
    pattern: str,
    live_only: bool,
    year: int ,
    month: int ,
    chain_csv: str | Path | None = None
) -> dict[str, Any]:
    """Get all tickers for assets."""
    tables = []
    for _, underlying in loader._iter_assets(path, pattern=pattern, live_only=live_only):
        table = get_tickers_ym(path, underlying, year, month, chain_csv)
        tables.append(table)
    return loader.bind_rows(*tables)

def find_tickers(
    path: str | Path,
    pattern: str,
    live_only: bool = True,
    start_year: int | None = None,
    end_year: int | None = None,
    chain_csv: str | Path | None = None
) -> dict[str, Any]:
    """Get all tickers for assets across a year range."""
    if start_year is None or end_year is None:
        # No year range - just get tschemas without expansion
        return find_tschemas(path, pattern, live_only)

    tables = None
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            table = find_tickers_ym(path, pattern, live_only, year, month, chain_csv)
            if tables is None:
                tables=table
            else:
                tables = loader.table_unique_rows(loader.bind_rows(tables,table))

    # Return empty table if no iterations (start_year > end_year)
    if tables is None:
        return {"orientation": "row", "columns": ["asset", "source", "type", "param", "tschema", "ticker", "field"], "rows": []}
    return tables

# -------------------------------------
# Compute Tickers from Ticker schemas
# straddle strings are inputs
# -------------------------------------

def _filter_straddle_tickers(rows: list[list], columns: list[str], ntrc: str) -> list[list]:
    """Filter ticker rows based on straddle rules.

    Rules:
    1. Market rows are excluded
    2. Vol/Near rows kept only if ntrc == "N", param changed to "vol"
    3. Vol/Far rows kept only if ntrc == "F", param changed to "vol"
    4. Hedge rows are always kept
    """
    type_idx = columns.index("type")
    param_idx = columns.index("param")
    source_idx = columns.index("source")
    ticker_idx = columns.index("ticker")
    field_idx = columns.index("field")

    filtered = []
    for row in rows:
        row_type = row[type_idx]
        row_param = row[param_idx]
        row_source = row[source_idx]
        row_ticker = row[ticker_idx]
        row_field = row[field_idx]

        if row_type == "Market":
            continue  # Exclude all Market rows
        elif row_type == "Vol":
            
            new_row = row[:]

            if row_source == "CV":
                new_row[ticker_idx] = row_field
                new_row[field_idx] = "none"

            if row_source == "calc":
                new_row[field_idx] = ""

            if row_param == "Near" and ntrc == "N":
                new_row[param_idx] = "vol"
            elif row_param == "Far" and ntrc == "F":
                new_row[param_idx] = "vol"
            else:
                # Other Vol rows are excluded
                continue

            filtered.append(new_row) # These are kept

        elif row_type == "Hedge":
            filtered.append(row)  # Keep all Hedge rows
        else:
            filtered.append(row)  # Keep other types (calc, etc.)

    return filtered


def filter_tickers(
    asset: str,
    year: int,
    month: int,
    i: int,
    amt_path: str | Path,
    chain_path: str | Path | None = None,
) -> dict[str, Any]:
    """Get tickers for an asset's straddle with the straddle string included.

    Columns: ['asset', 'param', 'source', 'ticker', 'field', 'straddle']

    Filtering rules:
    - Market rows are excluded
    - Vol/Near rows kept only if straddle ntrc == "N", param changed to "vol"
    - Vol/Far rows kept only if straddle ntrc == "F", param changed to "vol"
    - Hedge rows are always kept

    Args:
        asset: Asset underlying value
        year: Expiry year
        month: Expiry month
        i: Straddle selector index (i % len(straddles))
        amt_path: Path to AMT YAML file
        chain_path: Optional CSV for futures ticker lookup

    Returns:
        Table with ticker rows plus straddle column
    """
    # Check if underlying exists first
    if loader.get_asset(amt_path, asset) is None:
        raise ValueError(f"Asset '{asset}' not found")

    straddles = loader.table_column(schedules.get_expand_ym(amt_path, asset, year, month), "straddle")
    if len(straddles) < 1:
        raise ValueError(f"'{asset}' has no straddles in {year}-{month:02d}")
    straddle = straddles[i % len(straddles)]
    xpry, xprm = schedules.xpry(straddle), schedules.xprm(straddle)
    ntrc = schedules.ntrc(straddle)
    ticker_table = get_tickers_ym(amt_path, asset, xpry, xprm, chain_path)
    # Filter rows based on straddle rules
    filtered_rows = _filter_straddle_tickers(ticker_table["rows"], ticker_table["columns"], ntrc)
    # Build filtered table, then transform using table utilities
    filtered_table = {"orientation": "row", "columns": ticker_table["columns"], "rows": filtered_rows}
    # Drop cls and type columns
    filtered_table = loader.table_drop_columns(filtered_table, ["cls", "type"])
    # Add straddle column after asset
    filtered_table = loader.table_add_column(filtered_table, "straddle", value=straddle, position=1)
    return filtered_table

# -------------------------------------
# Finally, Prices
# -------------------------------------


import math


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# -------------------------------------
# Valuation Models
# -------------------------------------


def model_ES(row: dict[str, Any]) -> dict[str, str]:
    """European Straddle pricing model.

    Formula: mv = S * 2*N(d1) - X * 2*N(d2) + X - S
             delta = N_d1 - 1

    Inputs from row:
        - hedge: current underlying price (S)
        - strike: strike price captured at entry (X)
        - vol: current implied vol in percent
        - date: current date
        - expiry: expiry date

    Returns dict with "mv" and "delta" keys. Values are "-" for any inadequate
    inputs (missing, non-numeric, invalid dates, t < 0, zero/negative prices or vol, etc.)
    """
    try:
        from datetime import date as date_type

        S = float(row["hedge"])
        X = float(row["strike"])
        v = float(row["vol"])

        # Validate positive values
        if S <= 0 or X <= 0 or v <= 0:
            return {"mv": "-", "delta": "-"}

        # Calculate days to expiry
        current_date = date_type.fromisoformat(row["date"])
        expiry_date = date_type.fromisoformat(row["expiry"])
        t = (expiry_date - current_date).days

        if t < 0:
            return {"mv": "-", "delta": "-"}  # Past expiry - inadequate input

        if t == 0:
            # At expiry - intrinsic value, delta is +1 or -1 depending on S vs X
            mv = abs(S - X) / X
            delta = 1.0 if S >= X else -1.0
            return {"mv": str(mv), "delta": str(delta)}

        # Total volatility
        tv = (v / 100) * math.sqrt(t / 365)

        d1 = math.log(S / X) / tv + 0.5 * tv
        d2 = d1 - tv

        N_d1 = 2 * _norm_cdf(d1)
        N_d2 = 2 * _norm_cdf(d2)

        mv = S * N_d1 - X * N_d2 + X - S
        delta = N_d1 - 1

        # Return option value divided by strike, and delta
        return {"mv": str(mv / X), "delta": str(delta)}
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return {"mv": "-", "delta": "-"}


def model_NS(row: dict[str, Any]) -> dict[str, str]:
    """Normal Straddle model - placeholder."""
    return {"mv": "-", "delta": "-"}


def model_BS(row: dict[str, Any]) -> dict[str, str]:
    """Black-Scholes model - placeholder."""
    return {"mv": "-", "delta": "-"}


def model_default(row: dict[str, Any]) -> dict[str, str]:
    """Default model for unknown model names."""
    return {"mv": "-", "delta": "-"}


MODEL_DISPATCH = {
    "ES": model_ES,
    "NS": model_NS,
    "BS": model_BS,
    "CDS_ES": model_ES,  # CDS_ES uses ES model
}


def prices_last(prices_parquet: str | Path, pattern: str) -> dict[str, Any]:
    """Get last date for each ticker/field matching regex pattern.

    Args:
        prices_parquet: Path to prices parquet file
        pattern: Regex pattern to match tickers

    Returns:
        Table with columns: [ticker, field, last_date]
    """
    con = duckdb.connect()
    table_name = Path(prices_parquet).stem
    con.execute(f"CREATE VIEW {table_name} AS SELECT * FROM '{prices_parquet}'")

    query = f"""
        SELECT ticker, field, MAX(date) AS last_date
        FROM {table_name}
        WHERE regexp_matches(ticker, ?)
        GROUP BY ticker, field
        ORDER BY ticker, field
    """
    result = con.execute(query, [pattern]).fetchall()
    con.close()

    rows = [[str(ticker), str(field), str(last_date)] for ticker, field, last_date in result]
    return {"orientation": "row", "columns": ["ticker", "field", "last_date"], "rows": rows}


def prices_query(prices_parquet: str | Path, sql: str) -> dict[str, Any]:
    """Run arbitrary SQL query against prices parquet.

    The parquet file is exposed as table 'prices'.

    Args:
        prices_parquet: Path to prices parquet file
        sql: SQL query to execute

    Returns:
        Table with query results
    """
    con = duckdb.connect()
    con.execute(f"CREATE VIEW prices AS SELECT * FROM '{prices_parquet}'")

    result = con.execute(sql)
    columns = [desc[0] for desc in result.description]
    rows = [[str(v) for v in row] for row in result.fetchall()]
    con.close()

    return {"orientation": "row", "columns": columns, "rows": rows}


def _add_calendar_days(date_str: str, days: int) -> str:
    """Add calendar days to a date string.

    Args:
        date_str: ISO date string like "2024-01-19"
        days: Number of calendar days to add (can be 0)

    Returns:
        New date string
    """
    from datetime import date, timedelta
    d = date.fromisoformat(date_str)
    return (d + timedelta(days=days)).isoformat()


def _last_good_day_in_month(
    rows: list[list],
    vol_idx: int,
    hedge_indices: list[int],
    date_idx: int,
    year: int,
    month: int
) -> int | None:
    """Find the last good day in a given month.

    A "good day" is one where vol and all hedge columns are not "none".

    Args:
        rows: Data rows from get_straddle_actions output
        vol_idx: Index of vol column
        hedge_indices: List of hedge column indices
        date_idx: Index of date column
        year: Year
        month: Month (1-12)

    Returns:
        Row index of last good day, or None if no good days exist.
    """
    month_start = f"{year}-{month:02d}-01"
    _, num_days = calendar.monthrange(year, month)
    month_end = f"{year}-{month:02d}-{num_days:02d}"

    def is_good_day(row: list) -> bool:
        if row[vol_idx] == "none":
            return False
        return all(row[idx] != "none" for idx in hedge_indices)

    last_good_idx = None
    for i, row in enumerate(rows):
        row_date = row[date_idx]
        if row_date < month_start:
            continue
        if row_date > month_end:
            break
        if is_good_day(row):
            last_good_idx = i

    return last_good_idx


# -------------------------------------
# Preloaded prices dict (for backtest)
# -------------------------------------

_PRICES_DICT: dict[str, str] | None = None


def load_all_prices(
    prices_parquet: str | Path,
    start_date: str | None = None,
    end_date: str | None = None
) -> dict[str, str]:
    """Load all prices into a flat dict with composite key.

    Args:
        prices_parquet: Path to parquet file
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Dict mapping "ticker|field|date" -> value
    """
    global _PRICES_DICT
    if _PRICES_DICT is not None:
        return _PRICES_DICT

    con = duckdb.connect()

    # Build query with optional date filters
    query = f"SELECT ticker, field, date, value FROM '{prices_parquet}'"
    conditions = []
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    if end_date:
        conditions.append(f"date <= '{end_date}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    result = con.execute(query).fetchall()
    con.close()

    # Build flat dict with composite string key
    _PRICES_DICT = {}
    for ticker, field, dt, value in result:
        key = f"{ticker}|{field}|{dt}"
        _PRICES_DICT[key] = str(value)

    return _PRICES_DICT


def set_prices_dict(prices_dict: dict[str, str] | None) -> None:
    """Set the global prices dict (used by workers)."""
    global _PRICES_DICT
    _PRICES_DICT = prices_dict


def get_price(prices_dict: dict[str, str], ticker: str, field: str, date_str: str) -> str:
    """Look up a price from the preloaded dict.

    Returns "none" if not found.
    """
    key = f"{ticker}|{field}|{date_str}"
    return prices_dict.get(key, "none")


def clear_prices_dict() -> None:
    """Clear the global prices dict."""
    global _PRICES_DICT
    _PRICES_DICT = None


# -------------------------------------
# DuckDB connection cache
# -------------------------------------

_DUCKDB_CACHE: dict[str, "duckdb.DuckDBPyConnection"] = {}


def _get_prices_connection(prices_parquet: str | Path) -> "duckdb.DuckDBPyConnection":
    """Get or create a cached DuckDB connection for the given parquet file.

    The connection is cached by the resolved path string. The parquet file
    is exposed as the 'prices' view.

    Args:
        prices_parquet: Path to prices parquet file

    Returns:
        DuckDB connection with 'prices' view created
    """
    path_str = str(Path(prices_parquet).resolve())
    if path_str not in _DUCKDB_CACHE:
        con = duckdb.connect()
        con.execute(f"CREATE VIEW prices AS SELECT * FROM '{prices_parquet}'")
        _DUCKDB_CACHE[path_str] = con
    return _DUCKDB_CACHE[path_str]


def _clear_prices_cache():
    """Clear the DuckDB connection cache."""
    for con in _DUCKDB_CACHE.values():
        try:
            con.close()
        except Exception:
            pass
    _DUCKDB_CACHE.clear()


def clear_prices_connection_cache():
    """Clear the cached DuckDB connections for prices parquet files.

    Call this to release database connections and free memory,
    especially in long-running processes or after processing many
    different parquet files.
    """
    _clear_prices_cache()


# -------------------------------------
# Override expiry lookup
# -------------------------------------

_OVERRIDE_CACHE: dict[tuple[str, str], str] | None = None


def _load_overrides(path: str | Path = "data/overrides.csv") -> dict[tuple[str, str], str]:
    """Load and cache override expiry dates.

    Args:
        path: Path to overrides CSV file

    Returns:
        Dict mapping (ticker, "YYYY-MM") -> "YYYY-MM-DD"
    """
    global _OVERRIDE_CACHE
    if _OVERRIDE_CACHE is not None:
        return _OVERRIDE_CACHE

    _OVERRIDE_CACHE = {}
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row["ticker"]
                expiry = row["expiry"]  # "YYYY-MM-DD"
                year_month = expiry[:7]  # "YYYY-MM"
                key = (ticker, year_month)
                _OVERRIDE_CACHE[key] = expiry
    except (FileNotFoundError, KeyError):
        pass  # Return empty cache if file not found or malformed

    return _OVERRIDE_CACHE


def clear_override_cache():
    """Clear the cached override expiry dates.

    Call this to force reloading the overrides CSV file,
    useful after modifying the file or in long-running processes.
    """
    global _OVERRIDE_CACHE
    _OVERRIDE_CACHE = None


def _override_expiry(
    underlying: str,
    year: int,
    month: int,
    overrides_path: str | Path = "data/overrides.csv"
) -> str | None:
    """Look up override expiry date for an asset/month.

    Args:
        underlying: Asset identifier (e.g., "0R Comdty")
        year: Expiry year
        month: Expiry month (1-12)
        overrides_path: Path to overrides CSV

    Returns:
        Expiry date string "YYYY-MM-DD" or None if not found
    """
    overrides = _load_overrides(overrides_path)
    key = (underlying, f"{year}-{month:02d}")
    return overrides.get(key)


def _anchor_day(
    xprc: str,
    xprv: str,
    year: int,
    month: int,
    underlying: str | None = None,
    overrides_path: str | Path | None = None
) -> str | None:
    """
    Calculate the anchor day for a given month.

    Args:
        xprc: Code ("F", "R", "W", "BD", or "OVERRIDE")
        xprv: Value (string, the Nth occurrence; ignored for OVERRIDE)
        year: Year (int)
        month: Month (int, 1-12)
        underlying: Asset name (required for OVERRIDE)
        overrides_path: Path to overrides CSV (for OVERRIDE)

    Returns:
        Date string in ISO format (e.g., "2024-06-21"), or None if:
        - xprc is not in ["F", "R", "W", "BD", "OVERRIDE"]
        - xprv is not a valid positive integer (for non-OVERRIDE)
        - The Nth weekday/business day doesn't exist in the month
        - OVERRIDE lookup fails (no entry for asset/month)
    """
    # Handle OVERRIDE first - no xprv validation needed
    if xprc == "OVERRIDE":
        if underlying is None:
            return None
        return _override_expiry(underlying, year, month,
                                overrides_path or "data/overrides.csv")

    WEEKDAY_MAP = {"F": 4, "R": 3, "W": 2}  # Friday, Thursday, Wednesday

    try:
        n = int(xprv)
        if n < 1:
            return None
    except (ValueError, TypeError):
        return None

    _, num_days = calendar.monthrange(year, month)

    if xprc == "BD":
        # Find Nth business day (Mon-Fri) in the month
        from datetime import date
        bd_count = 0
        for day in range(1, num_days + 1):
            d = date(year, month, day)
            if d.weekday() < 5:  # Mon=0, Fri=4
                bd_count += 1
                if bd_count == n:
                    return d.isoformat()
        return None  # Not enough business days

    elif xprc in WEEKDAY_MAP:
        target_weekday = WEEKDAY_MAP[xprc]

        # Find all occurrences of target weekday in the month
        from datetime import date
        weekday_dates = []
        for day in range(1, num_days + 1):
            d = date(year, month, day)
            if d.weekday() == target_weekday:
                weekday_dates.append(d)

        # Return Nth occurrence (1-indexed)
        if n > len(weekday_dates):
            return None  # e.g., 5th Friday doesn't exist

        return weekday_dates[n - 1].isoformat()

    return None


def _nth_good_day_after(
    rows: list[list],
    vol_idx: int,
    hedge_indices: list[int],
    date_idx: int,
    anchor_date: str,
    n: int,
    month_limit: str | None = None
) -> int | None:
    """
    Find the Nth good day after (or at) an anchor date.

    A "good day" is one where vol and all hedge columns are not "none".

    Semantics for n:
    - n = 0: Return anchor row if anchor is good, else first good day after anchor
    - n > 0: Return Nth good day after Day 0

    Args:
        rows: Data rows from get_straddle_actions output
        vol_idx: Index of vol column
        hedge_indices: List of hedge column indices
        date_idx: Index of date column
        anchor_date: Date string like "2024-05-17" (the anchor)
        n: Offset from anchor (0 = anchor or first good after, 1 = first good after Day 0, etc.)
        month_limit: Optional date string like "2024-06-30" - stop searching after this date

    Returns:
        Row index of the target good day, or None if not found
    """
    if n < 0:
        return None

    def is_good_day(row: list) -> bool:
        if row[vol_idx] == "none":
            return False
        return all(row[idx] != "none" for idx in hedge_indices)

    # First, find Day 0 (anchor if good, else first good day after anchor)
    day_0_idx = None
    for i, row in enumerate(rows):
        row_date = row[date_idx]

        # Skip rows before anchor
        if row_date < anchor_date:
            continue

        # Stop if past month_limit
        if month_limit is not None and row_date > month_limit:
            break

        if is_good_day(row):
            day_0_idx = i
            break

    if day_0_idx is None:
        return None  # No good day found at or after anchor

    if n == 0:
        return day_0_idx

    # For n > 0, count n good days after Day 0
    count = 0
    for i in range(day_0_idx + 1, len(rows)):
        row = rows[i]
        row_date = row[date_idx]

        # Stop if past month_limit
        if month_limit is not None and row_date > month_limit:
            break

        if is_good_day(row):
            count += 1
            if count == n:
                return i

    return None  # Not enough good days after Day 0


def _compute_actions(
    rows: list[list],
    columns: list[str],
    ntrc: str,
    ntrv: str,
    xprc: str,
    xprv: str,
    xpry: int | None = None,
    xprm: int | None = None,
    ntry: int | None = None,
    ntrm: int | None = None,
    underlying: str | None = None,
    overrides_path: str | Path | None = None
) -> list[str]:
    """
    Compute action values for each row in get_straddle_actions output.

    Args:
        rows: Output rows from get_straddle_actions (before action column added)
        columns: Column names (to find vol/hedge indices)
        ntrc: Entry code from straddle
        ntrv: Entry value from straddle
        xprc: Expiry code from straddle
        xprv: Expiry value from straddle
        xpry: Expiry year from straddle
        xprm: Expiry month from straddle
        ntry: Entry year from straddle
        ntrm: Entry month from straddle
        underlying: Asset name (required for OVERRIDE code)
        overrides_path: Path to overrides CSV (for OVERRIDE code)

    Returns:
        List of action strings, one per row ("-" for no action, "ntry" for entry trigger, "xpry" for expiry trigger)
    """
    actions = ["-"] * len(rows)

    # Find vol index
    vol_idx = columns.index("vol") if "vol" in columns else None
    if vol_idx is None:
        return actions  # Missing vol column

    # Find all hedge column indices (hedge, hedge1, hedge2, ...)
    hedge_indices = []
    for i, col in enumerate(columns):
        if col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
            hedge_indices.append(i)

    if not hedge_indices:
        return actions  # No hedge columns

    # Find date column index
    date_idx = columns.index("date") if "date" in columns else None
    if date_idx is None:
        return actions  # Missing date column

    # Unified rules for F/R/W/BD/OVERRIDE codes
    # Entry: anchor + ntrv calendar days -> first good day at or after -> fallback to last good day
    # Expiry: anchor -> first good day at or after
    if xprc in ["F", "R", "W", "BD", "OVERRIDE"]:
        # Entry trigger ("ntry")
        if ntry is not None and ntrm is not None:
            entry_anchor = _anchor_day(xprc, xprv, ntry, ntrm, underlying, overrides_path)
            if entry_anchor is not None:
                try:
                    ntrv_int = int(ntrv) if ntrv else 0
                    _, entry_num_days = calendar.monthrange(ntry, ntrm)
                    entry_month_end = f"{ntry}-{ntrm:02d}-{entry_num_days:02d}"

                    # Add calendar days to anchor
                    target_date = _add_calendar_days(entry_anchor, ntrv_int)

                    # If target is past entry month, use last good day of month
                    if target_date > entry_month_end:
                        idx = _last_good_day_in_month(rows, vol_idx, hedge_indices, date_idx, ntry, ntrm)
                    else:
                        # Find first good day at or after target
                        idx = _nth_good_day_after(rows, vol_idx, hedge_indices, date_idx,
                                                  target_date, 0, entry_month_end)
                        # If no good day found at or after target, use last good day of month
                        if idx is None:
                            idx = _last_good_day_in_month(rows, vol_idx, hedge_indices, date_idx, ntry, ntrm)

                    if idx is not None:
                        actions[idx] = "ntry"
                except (ValueError, TypeError):
                    pass

        # Expiry trigger ("xpry")
        # Anchor -> first good day at or after
        if xpry is not None and xprm is not None:
            expiry_anchor = _anchor_day(xprc, xprv, xpry, xprm, underlying, overrides_path)
            if expiry_anchor is not None:
                _, expiry_num_days = calendar.monthrange(xpry, xprm)
                expiry_month_end = f"{xpry}-{xprm:02d}-{expiry_num_days:02d}"

                idx = _nth_good_day_after(rows, vol_idx, hedge_indices, date_idx,
                                          expiry_anchor, 0, expiry_month_end)
                if idx is not None:
                    actions[idx] = "xpry"

    return actions


# -------------------------------------
# Refactored straddle price/action helpers
# -------------------------------------


def _build_ticker_map(
    ticker_table: dict[str, Any],
    chain_csv: str | Path | None = None,
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Build ticker map from ticker table.

    Args:
        ticker_table: Output from filter_tickers()
        chain_csv: Optional CSV for futures ticker normalization

    Returns:
        (ticker_map, params_ordered) where:
        - ticker_map: param -> (ticker, field) for price lookup
        - params_ordered: list of params in order for output columns
    """
    param_idx = ticker_table["columns"].index("param")
    ticker_idx = ticker_table["columns"].index("ticker")
    field_idx = ticker_table["columns"].index("field")

    raw_ticker_map = {}  # param -> (ticker, field)
    params_ordered = []  # preserve order for output columns
    for row in ticker_table["rows"]:
        param = row[param_idx]
        if param not in raw_ticker_map:
            params_ordered.append(param)
            raw_ticker_map[param] = (row[ticker_idx], row[field_idx])

    # Normalize tickers for price lookup (prices DB uses normalized tickers)
    ticker_map = {}
    if chain_csv is not None:
        for param, (ticker, field) in raw_ticker_map.items():
            normalized = chain.fut_act2norm(chain_csv, ticker)
            if normalized is not None:
                ticker_map[param] = (normalized, field)
            else:
                ticker_map[param] = (ticker, field)
    else:
        ticker_map = raw_ticker_map

    return ticker_map, params_ordered


def _lookup_straddle_prices(
    dates: list,
    ticker_map: dict[str, tuple[str, str]],
    prices_parquet: str | Path | None = None,
) -> dict[tuple[str, str], dict[str, str]]:
    """Lookup prices for all dates and ticker/field pairs.

    Args:
        dates: List of date objects to lookup
        ticker_map: Mapping from param name to (ticker, field)
        prices_parquet: Path to parquet for DuckDB fallback

    Returns:
        Dict mapping (ticker, field) -> {date_str -> value}
    """
    ticker_field_pairs = list(ticker_map.values())
    prices: dict[tuple[str, str], dict[str, str]] = {}

    if _PRICES_DICT is not None:
        # Use preloaded dict - O(1) lookups
        for ticker, field in ticker_field_pairs:
            prices[(ticker, field)] = {}
            for dt in dates:
                date_str = dt.isoformat()
                value = get_price(_PRICES_DICT, ticker, field, date_str)
                if value != "none":
                    prices[(ticker, field)][date_str] = value
    elif prices_parquet is not None:
        # Fallback to DuckDB query
        start_date = dates[0].isoformat()
        end_date = dates[-1].isoformat()

        con = _get_prices_connection(prices_parquet)

        # Build parameterized query with placeholders
        conditions = " OR ".join(
            "(ticker = ? AND field = ?)" for _ in ticker_field_pairs
        )
        # Flatten ticker/field pairs into params list
        params: list[str] = []
        for t, f in ticker_field_pairs:
            params.extend([t, f])
        params.extend([start_date, end_date])

        query = f"""
            SELECT ticker, field, date, value
            FROM prices
            WHERE ({conditions})
            AND date >= ?
            AND date <= ?
        """
        result = con.execute(query, params).fetchall()

        for ticker, field, dt, value in result:
            key = (ticker, field)
            if key not in prices:
                prices[key] = {}
            prices[key][str(dt)] = str(value)
    else:
        raise ValueError("No prices available: call load_all_prices() or set_prices_dict() first, or provide prices_parquet")

    return prices


def _build_prices_table(
    asset: str,
    straddle: str,
    dates: list,
    params_ordered: list[str],
    ticker_map: dict[str, tuple[str, str]],
    prices: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    """Build the prices table with one row per day.

    Args:
        asset: Asset identifier
        straddle: Straddle string
        dates: List of date objects
        params_ordered: List of param names in column order
        ticker_map: param -> (ticker, field)
        prices: (ticker, field) -> {date_str -> value}

    Returns:
        Table with columns: [asset, straddle, date, <params...>]
    """
    out_columns = ["asset", "straddle", "date"] + params_ordered
    out_rows = []

    for dt in dates:
        date_str = dt.isoformat()
        row = [asset, straddle, date_str]
        for param in params_ordered:
            ticker, field = ticker_map[param]
            value = prices.get((ticker, field), {}).get(date_str, "none")
            row.append(value)
        out_rows.append(row)

    return {"orientation": "row", "columns": out_columns, "rows": out_rows}


def _find_action_indices(table: dict[str, Any]) -> tuple[int | None, int | None]:
    """Find ntry and xpry row indices from action column.

    Args:
        table: Table with 'action' column

    Returns:
        (ntry_idx, xpry_idx) - row indices or None if not found
    """
    if "action" not in table["columns"]:
        return None, None

    action_idx = table["columns"].index("action")
    ntry_idx = None
    xpry_idx = None

    for i, row in enumerate(table["rows"]):
        if row[action_idx] == "ntry":
            ntry_idx = i
        elif row[action_idx] == "xpry":
            xpry_idx = i

    return ntry_idx, xpry_idx


def _add_action_column(
    table: dict[str, Any],
    straddle: str,
    underlying: str,
    overrides_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Add action column to prices table.

    Args:
        table: Prices table with columns [asset, straddle, date, <params...>]
        straddle: Straddle string for parsing entry/expiry info
        underlying: Asset underlying for override lookups
        overrides_csv: Path to overrides CSV (for OVERRIDE code)

    Returns:
        Table with 'action' column added
    """
    ntrc_val = schedules.ntrc(straddle)
    ntrv_val = schedules.ntrv(straddle)
    xprc_val = schedules.xprc(straddle)
    xprv_val = schedules.xprv(straddle)
    entry_year, entry_month = schedules.ntry(straddle), schedules.ntrm(straddle)
    expiry_year, expiry_month = schedules.xpry(straddle), schedules.xprm(straddle)

    actions = _compute_actions(
        table["rows"], table["columns"],
        ntrc_val, ntrv_val, xprc_val, xprv_val,
        expiry_year, expiry_month, entry_year, entry_month,
        underlying, overrides_csv
    )

    # Add action to each row
    new_rows = [row + [action] for row, action in zip(table["rows"], actions)]
    new_columns = table["columns"] + ["action"]

    return {"orientation": "row", "columns": new_columns, "rows": new_rows}


def _add_model_column(
    table: dict[str, Any],
    underlying: str,
    path: str | Path,
) -> dict[str, Any]:
    """Add model column to table.

    Args:
        table: Input table
        underlying: Asset underlying
        path: Path to AMT YAML file

    Returns:
        Table with 'model' column added
    """
    asset_data = loader.get_asset(path, underlying)
    if asset_data is not None:
        valuation = asset_data.get("Valuation", {})
        model = valuation.get("Model", "") if isinstance(valuation, dict) else ""
    else:
        model = ""

    new_rows = [row + [model] for row in table["rows"]]
    new_columns = table["columns"] + ["model"]

    return {"orientation": "row", "columns": new_columns, "rows": new_rows}


def _add_strike_columns(
    table: dict[str, Any],
    ntry_idx: int | None,
    xpry_idx: int | None,
) -> dict[str, Any]:
    """Add strike_vol, strike, strike1..., and expiry columns.

    Values come from ntry row, shown only between ntry and xpry.

    Args:
        table: Table with prices, action, model columns
        ntry_idx: Row index of entry trigger (or None)
        xpry_idx: Row index of expiry trigger (or None)

    Returns:
        Table with strike and expiry columns added
    """
    columns = table["columns"]
    rows = [row[:] for row in table["rows"]]  # Copy rows

    # Find vol column index
    vol_col_idx = columns.index("vol") if "vol" in columns else None

    # Find hedge column indices
    hedge_col_indices = []
    for i, col in enumerate(columns):
        if col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
            hedge_col_indices.append(i)

    # Get strike values from ntry row
    if ntry_idx is not None and vol_col_idx is not None:
        strike_vol_value = rows[ntry_idx][vol_col_idx]
        strike_values = [rows[ntry_idx][idx] for idx in hedge_col_indices]
    else:
        strike_vol_value = "-"
        strike_values = ["-"] * len(hedge_col_indices)

    # Add strike_vol column
    new_columns = columns + ["strike_vol"]
    for i, row in enumerate(rows):
        in_range = (ntry_idx is not None and i >= ntry_idx and
                    (xpry_idx is None or i <= xpry_idx))
        row.append(strike_vol_value if in_range else "-")

    # Add strike columns (one per hedge)
    for j in range(len(hedge_col_indices)):
        strike_col_name = "strike" if j == 0 else f"strike{j}"
        for i, row in enumerate(rows):
            in_range = (ntry_idx is not None and i >= ntry_idx and
                        (xpry_idx is None or i <= xpry_idx))
            row.append(strike_values[j] if in_range else "-")
        new_columns.append(strike_col_name)

    # Add expiry column
    date_col_idx = columns.index("date") if "date" in columns else None
    if xpry_idx is not None and date_col_idx is not None:
        expiry_value = rows[xpry_idx][date_col_idx]
    else:
        expiry_value = "-"

    for i, row in enumerate(rows):
        in_range = (ntry_idx is not None and i >= ntry_idx and
                    (xpry_idx is None or i <= xpry_idx))
        row.append(expiry_value if in_range else "-")
    new_columns.append("expiry")

    return {"orientation": "row", "columns": new_columns, "rows": rows}


# -------------------------------------
# Public API: Decomposed straddle functions
# -------------------------------------


def get_prices(
    underlying: str,
    year: int,
    month: int,
    i: int,
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
) -> dict[str, Any]:
    """Get daily prices for a straddle from entry to expiry.

    This is a focused function that ONLY handles price lookup.
    Does NOT compute actions, strikes, or model columns.

    Args:
        underlying: Asset underlying value
        year: Expiry year
        month: Expiry month
        i: Straddle selector index (i % len(straddles))
        path: Path to AMT YAML file
        chain_csv: Optional CSV for futures ticker lookup
        prices_parquet: Path to prices parquet file (for DuckDB fallback)

    Returns:
        Table with columns: [asset, straddle, date, vol, hedge, ...]
    """
    # 1. Get ticker table
    ticker_table = filter_tickers(underlying, year, month, i, path, chain_csv)

    if not ticker_table["rows"]:
        return {"orientation": "row", "columns": ["asset", "straddle", "date"], "rows": []}

    # 2. Extract asset and straddle
    asset = ticker_table["rows"][0][0]
    straddle = ticker_table["rows"][0][1]

    # 3. Generate dates using existing schedules.straddle_days()
    dates = schedules.straddle_days(straddle)

    # 4. Build ticker map
    ticker_map, params_ordered = _build_ticker_map(ticker_table, chain_csv)

    # 5. Lookup prices
    prices = _lookup_straddle_prices(dates, ticker_map, prices_parquet)

    # 6. Build and return table
    return _build_prices_table(asset, straddle, dates, params_ordered, ticker_map, prices)


def actions(
    prices_table: dict[str, Any],
    path: str | Path,
    overrides_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Add action, model, and strike columns to a prices table.

    Args:
        prices_table: Output from get_prices() - must have 'asset' and 'straddle' columns
        path: Path to AMT YAML
        overrides_csv: Path to overrides CSV (for OVERRIDE code)

    Returns:
        Table with added columns: action, model, strike_vol, strike, expiry
    """
    if not prices_table["rows"]:
        return prices_table

    # Extract asset (underlying) and straddle from first row
    asset_idx = prices_table["columns"].index("asset")
    straddle_idx = prices_table["columns"].index("straddle")
    underlying = prices_table["rows"][0][asset_idx]
    straddle = prices_table["rows"][0][straddle_idx]

    # 1. Add action column
    table = _add_action_column(prices_table, straddle, underlying, overrides_csv)

    # 2. Add model column
    table = _add_model_column(table, underlying, path)

    # 3. Add strike columns
    ntry_idx, xpry_idx = _find_action_indices(table)
    table = _add_strike_columns(table, ntry_idx, xpry_idx)

    return table


def get_straddle_actions(
    underlying: str,
    year: int,
    month: int,
    i: int,
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
    overrides_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Get daily prices for a straddle from entry to expiry month.

    Columns: ['asset', 'straddle', 'date', <param1>, <param2>, ..., 'action', 'model', 'strike_vol', 'strike', 'expiry']
    Where params are 'vol', 'hedge', 'hedge1', etc.

    This is a convenience function that composes:
    1. get_prices() - price lookup
    2. actions() - action, model, and strike columns

    For more control, use the individual functions directly.

    Uses the module-level _PRICES_DICT if set (via set_prices_dict or load_all_prices),
    otherwise falls back to DuckDB queries on prices_parquet.

    Args:
        underlying: Asset underlying value
        year: Expiry year
        month: Expiry month
        i: Straddle selector index (i % len(straddles))
        path: Path to AMT YAML file
        chain_csv: Optional CSV for futures ticker lookup
        prices_parquet: Path to prices parquet file (for DuckDB fallback)
        overrides_csv: Path to overrides CSV (for OVERRIDE code)

    Returns:
        Table with one row per day, columns for each param's price plus action/strike columns
    """
    # Get prices table
    prices_table = get_prices(
        underlying, year, month, i, path, chain_csv, prices_parquet
    )

    if not prices_table["rows"]:
        return prices_table

    # Add actions, model, and strikes
    return actions(prices_table, path, overrides_csv)


def _get_rollforward_fields(columns: list[str]) -> set[str]:
    """Get fields that should be rolled forward (vol and hedge columns)."""
    fields = set()
    for col in columns:
        if col == "vol":
            fields.add(col)
        elif col == "hedge" or (col.startswith("hedge") and col[5:].isdigit()):
            fields.add(col)
    return fields


def get_straddle_valuation(
    underlying: str,
    year: int,
    month: int,
    i: int,
    path: str | Path,
    chain_csv: str | Path | None = None,
    prices_parquet: str | Path | None = None,
    overrides_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Get straddle valuation with mv column.

    Calls get_straddle_actions and adds mv (mark-to-market value) column
    computed using the asset's valuation model.

    Args:
        Same as get_straddle_actions

    Returns:
        Table with additional mv column
    """
    # Get base table
    table = get_straddle_actions(underlying, year, month, i, path, chain_csv, prices_parquet, overrides_csv)

    columns = table["columns"]
    rows = table["rows"]

    # Find action column and check for ntry/xpry
    if "action" not in columns:
        # No action column, add all valuation columns as "-"
        for row in rows:
            row.extend(["-", "-", "-", "-", "-"])
        columns.extend(["mv", "delta", "opnl", "hpnl", "pnl"])
        return {"orientation": "row", "columns": columns, "rows": rows}

    action_idx = columns.index("action")

    ntry_idx = None
    xpry_idx = None
    for idx, row in enumerate(rows):
        if row[action_idx] == "ntry":
            ntry_idx = idx
        elif row[action_idx] == "xpry":
            xpry_idx = idx

    if ntry_idx is None or xpry_idx is None:
        # Missing ntry or xpry, add all valuation columns as "-"
        for row in rows:
            row.extend(["-", "-", "-", "-", "-"])
        columns.extend(["mv", "delta", "opnl", "hpnl", "pnl"])
        return {"orientation": "row", "columns": columns, "rows": rows}

    # Get model from first row
    model_idx = columns.index("model") if "model" in columns else None
    if model_idx is not None and rows:
        model_name = rows[0][model_idx]
    else:
        model_name = ""

    # Get model function
    model_fn = MODEL_DISPATCH.get(model_name, model_default)

    # Get fields to roll forward (vol and hedge columns)
    rollforward_fields = _get_rollforward_fields(columns)

    # Initialize rolled-forward data from ntry row
    rolled_data = {}
    ntry_row_dict = dict(zip(columns, rows[ntry_idx]))
    for key in rollforward_fields:
        if key in ntry_row_dict:
            rolled_data[key] = ntry_row_dict[key]

    # Get strike price for hpnl calculation (hedge at entry)
    strike_col_idx = columns.index("strike") if "strike" in columns else None
    strike_price = None
    if strike_col_idx is not None:
        try:
            strike_price = float(rows[ntry_idx][strike_col_idx])
        except (ValueError, TypeError):
            pass

    # Track previous day's values for PnL calculations
    prev_mv = None
    prev_delta = None
    prev_hedge = None

    # Compute mv, delta, opnl, hpnl, pnl for each row
    for idx, row in enumerate(rows):
        if idx < ntry_idx or idx > xpry_idx:
            row.append("-")  # mv
            row.append("-")  # delta
            row.append("-")  # opnl
            row.append("-")  # hpnl
            row.append("-")  # pnl
        else:
            # Update rolled_data with any non-missing market values
            row_dict = dict(zip(columns, row))
            for key in rollforward_fields:
                if key in row_dict and row_dict[key] != "none":
                    rolled_data[key] = row_dict[key]

            # Build model input: current row data + rolled-forward market data
            model_input = row_dict.copy()
            model_input.update(rolled_data)

            result = model_fn(model_input)
            mv_str = result["mv"]
            delta_str = result["delta"]
            row.append(mv_str)
            row.append(delta_str)

            # Get current hedge (rolled forward)
            current_hedge = None
            try:
                current_hedge = float(rolled_data.get("hedge", ""))
            except (ValueError, TypeError):
                pass

            # Compute PnL columns
            if idx == ntry_idx:
                # First day: opnl = 0, hpnl = 0, pnl = 0
                row.append("0")  # opnl
                row.append("0")  # hpnl
                row.append("0")  # pnl
            else:
                # opnl = mv[today] - mv[yesterday]
                opnl = "-"
                if mv_str != "-" and prev_mv is not None:
                    try:
                        opnl = str(float(mv_str) - prev_mv)
                    except (ValueError, TypeError):
                        pass

                # hpnl = -delta[yesterday] * (hedge[today] - hedge[yesterday]) / strike
                hpnl = "-"
                if (prev_delta is not None and current_hedge is not None and
                    prev_hedge is not None and strike_price is not None and strike_price != 0):
                    try:
                        hpnl = str(-prev_delta * (current_hedge - prev_hedge) / strike_price)
                    except (ValueError, TypeError):
                        pass

                # pnl = opnl + hpnl
                pnl = "-"
                if opnl != "-" and hpnl != "-":
                    try:
                        pnl = str(float(opnl) + float(hpnl))
                    except (ValueError, TypeError):
                        pass

                row.append(opnl)
                row.append(hpnl)
                row.append(pnl)

            # Update previous values for next iteration
            try:
                prev_mv = float(mv_str) if mv_str != "-" else None
            except (ValueError, TypeError):
                prev_mv = None
            try:
                prev_delta = float(delta_str) if delta_str != "-" else None
            except (ValueError, TypeError):
                prev_delta = None
            prev_hedge = current_hedge

    columns.append("mv")
    columns.append("delta")
    columns.append("opnl")
    columns.append("hpnl")
    columns.append("pnl")
    return {"orientation": "row", "columns": columns, "rows": rows}


# -------------------------------------
# CLI
# -------------------------------------


def _main() -> int:
    import argparse
    from .loader import print_table

    p = argparse.ArgumentParser(
        description="Ticker extraction and transformation utilities.",
        allow_abbrev=False,  # Disable prefix matching
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
    
    p.add_argument("--get-tickers-ym", nargs=3, type=str, metavar=( "ASSET", "YEAR", "MONTH"),
                   help="Get all tickers for assets on specific month.")

    p.add_argument("--expand-ym", nargs=4, type=str, metavar=("PATTERN", "LIVE","YEAR", "MONTH"),
                   help="Get straddles for assets matching patterns on month.")
    
    p.add_argument("--get-expand-ym", nargs=3, type=str, metavar=("ASSET", "YEAR", "MONTH"),
                   help="Get straddles for asset on month.")
    
    p.add_argument("--asset-tickers", nargs=4, type=str, metavar=("UNDERLYING", "YEAR", "MONTH","NDX"),
                   help="Get straddle info with daily prices for entry month.")

    p.add_argument("--asset-days", nargs=4, type=str, metavar=("UNDERLYING", "YEAR", "MONTH", "NDX"),
                   help="Get daily prices for a straddle from entry to expiry month.")

    p.add_argument("--fut", nargs=3, metavar=("SPEC", "YEAR", "MONTH"),
                   help="Compute futures ticker from spec string, year, and month.")

    p.add_argument("--prices-last", metavar="REGEX",
                   help="Show last date for each ticker/field matching regex")

    p.add_argument("--prices-query", metavar="SQL",
                   help="Run arbitrary SQL query against prices parquet (table: prices)")

    p.add_argument("--straddle-valuation", nargs=4, type=str, metavar=("UNDERLYING", "YEAR", "MONTH", "NDX"),
                   help="Get straddle valuation, delta, pnls.")

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
        print("Find Tickes YM")
        table = find_tickers_ym(
            path=args.path, 
            pattern=args.find_tickers_ym[0], 
            live_only=str2bool(args.find_tickers_ym[1]),
            year=int(args.find_tickers_ym[2]),
            month=int(args.find_tickers_ym[3]),
            chain_csv=args.chain_csv
        )
        loader.print_table(table)

    elif args.get_tickers_ym is not None:
        table = get_tickers_ym(
            args.path, 
            args.get_tickers_ym[0], 
            int(args.get_tickers_ym[1]),
            int(args.get_tickers_ym[2]),
            args.chain_csv
        )
        loader.print_table(table)

    elif args.fut:
        spec, year, month = args.fut
        ticker = fut_spec2ticker(spec, int(year), int(month))
        print(ticker)

    elif args.expand_ym:
        pattern, live, year, month = args.expand_ym
        table = schedules.find_straddle_ym( args.path, int(year), int(month), pattern, str2bool(live) )
        loader.print_table(table)

    elif args.get_expand_ym:
        asset, year, month = args.get_expand_ym
        table = schedules.get_expand_ym( args.path, asset, int(year), int(month) )
        loader.print_table(table)

    elif args.asset_tickers:
        underlying, year, month, i = args.asset_tickers
        table = filter_tickers(underlying, int(year), int(month), int(i), args.path, args.chain_csv)
        loader.print_table(table)

    elif args.asset_days:
        underlying, year, month, i = args.asset_days
        table = get_straddle_actions(underlying, int(year), int(month), int(i), args.path, args.chain_csv, args.prices)
        loader.print_table(table)

    elif args.prices_last:
        table = prices_last(args.prices, args.prices_last)
        loader.print_table(table)

    elif args.prices_query:
        table = prices_query(args.prices, args.prices_query)
        loader.print_table(table)

    elif args.straddle_valuation:
        underlying, year, month, i = args.straddle_valuation
        table = get_straddle_valuation(underlying, int(year), int(month), int(i), args.path, args.chain_csv, args.prices)
        loader.print_table(table)

    else:
        p.print_help()

    return 0


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(_main())