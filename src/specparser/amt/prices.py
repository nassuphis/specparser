# -------------------------------------
# AMT prices - Price data access
# -------------------------------------
"""
Price data access utilities.

Handles fetching, caching, and looking up price data from parquet files.
Includes DuckDB connection caching and preloaded dict-based lookups.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import date

import duckdb
import numpy as np

from . import chain
from . import tickers
from . import schedules


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
# Numeric price arrays (for vectorized batch valuation)
# -------------------------------------

# Days since epoch for date conversion
_EPOCH = date(1970, 1, 1)


def _date32_to_isoformat(d: int) -> str:
    """Convert date32 (days since 1970-01-01) to ISO format string."""
    return (date.fromordinal(_EPOCH.toordinal() + d)).isoformat()


def build_price_arrays(
    prices_dict: dict[str, str],
    ticker_fields: list[tuple[str, str]],
    dates: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    """Build numeric price arrays for bulk lookup.

    Converts string prices to float64 arrays with NaN for missing values.
    This enables vectorized operations without per-lookup string overhead.

    Args:
        prices_dict: The preloaded prices dict ("ticker|field|date" -> "value")
        ticker_fields: List of (ticker, field) pairs to extract
        dates: int32 array of date32 values (days since 1970-01-01)

    Returns:
        Dict mapping (ticker, field) -> float64 array with same length as dates.
        Missing values are NaN.

    Example:
        >>> prices_dict = load_all_prices("data/prices.parquet")
        >>> tickers = [("CL1 Comdty", "PX_LAST"), ("CO1 Comdty", "PX_LAST")]
        >>> dates = np.array([19724, 19725, 19726], dtype=np.int32)  # 2024-01-01 to 2024-01-03
        >>> arrays = build_price_arrays(prices_dict, tickers, dates)
        >>> arrays[("CL1 Comdty", "PX_LAST")]
        array([72.31, 73.15, 72.89])
    """
    n_dates = len(dates)
    result: dict[tuple[str, str], np.ndarray] = {}

    for ticker, field in ticker_fields:
        arr = np.full(n_dates, np.nan, dtype=np.float64)
        for i, d in enumerate(dates):
            date_str = _date32_to_isoformat(int(d))
            key = f"{ticker}|{field}|{date_str}"
            value_str = prices_dict.get(key)
            if value_str is not None and value_str != "none":
                try:
                    arr[i] = float(value_str)
                except (ValueError, TypeError):
                    pass  # Keep as NaN
        result[(ticker, field)] = arr

    return result


def build_price_matrix(
    prices_dict: dict[str, str],
    ticker_fields: list[tuple[str, str]],
    dates: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[str, str], int]]:
    """Build a price matrix for multiple ticker/fields.

    More memory-efficient than build_price_arrays when working with many
    ticker/field pairs, as it stores everything in a single 2D array.

    Args:
        prices_dict: The preloaded prices dict
        ticker_fields: List of (ticker, field) pairs
        dates: int32 array of date32 values

    Returns:
        (price_matrix, ticker_field_to_idx) where:
        - price_matrix: float64 array shape (n_ticker_fields, n_dates)
        - ticker_field_to_idx: dict mapping (ticker, field) -> row index
    """
    n_dates = len(dates)
    n_tickers = len(ticker_fields)

    price_matrix = np.full((n_tickers, n_dates), np.nan, dtype=np.float64)
    ticker_field_to_idx: dict[tuple[str, str], int] = {}

    for row_idx, (ticker, field) in enumerate(ticker_fields):
        ticker_field_to_idx[(ticker, field)] = row_idx
        for col_idx, d in enumerate(dates):
            date_str = _date32_to_isoformat(int(d))
            key = f"{ticker}|{field}|{date_str}"
            value_str = prices_dict.get(key)
            if value_str is not None and value_str != "none":
                try:
                    price_matrix[row_idx, col_idx] = float(value_str)
                except (ValueError, TypeError):
                    pass

    return price_matrix, ticker_field_to_idx


# -------------------------------------
# PriceMatrix: Numba-friendly price storage
# -------------------------------------


class PriceMatrix:
    """Numba-friendly price storage structure.

    Stores prices in a 2D numpy array indexed by:
    - Row: ticker|field combination
    - Column: date (as offset from min_date32)

    This allows O(1) array lookups in Numba kernels instead of
    Python dict lookups with string keys.

    Attributes:
        price_matrix: float64[n_ticker_fields, n_dates] - the price data
        ticker_field_to_row: dict[str, int] - "ticker|field" -> row index
        date32_to_col: int32[n_calendar_days] - date32 offset -> col index (-1 if not a business day)
        min_date32: int - minimum date32 value (for offset calculation)
        n_rows: int - number of ticker|field combinations
        n_cols: int - number of business days with prices
    """

    def __init__(
        self,
        price_matrix: np.ndarray,
        ticker_field_to_row: dict[str, int],
        date32_to_col: np.ndarray,
        min_date32: int,
    ):
        self.price_matrix = price_matrix
        self.ticker_field_to_row = ticker_field_to_row
        self.date32_to_col = date32_to_col
        self.min_date32 = min_date32
        self.n_rows = price_matrix.shape[0]
        self.n_cols = price_matrix.shape[1]

    def get_row_idx(self, ticker: str, field: str) -> int:
        """Get row index for a ticker|field combination.

        Returns -1 if not found.
        """
        key = f"{ticker}|{field}"
        return self.ticker_field_to_row.get(key, -1)

    def get_col_idx(self, date32: int) -> int:
        """Get column index for a date32 value.

        Returns -1 if date is outside range or not a business day.
        """
        offset = date32 - self.min_date32
        if offset < 0 or offset >= len(self.date32_to_col):
            return -1
        return int(self.date32_to_col[offset])

    def lookup(self, ticker: str, field: str, date32: int) -> float:
        """Look up a single price (Python API).

        Returns np.nan if not found.
        """
        row = self.get_row_idx(ticker, field)
        col = self.get_col_idx(date32)
        if row < 0 or col < 0:
            return np.nan
        return self.price_matrix[row, col]


# Global cache for PriceMatrix
_PRICE_MATRIX: PriceMatrix | None = None


def load_prices_matrix(
    prices_parquet: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> PriceMatrix:
    """Load prices into a Numba-friendly matrix structure.

    This builds a 2D array where:
    - Rows are unique ticker|field combinations
    - Columns are business days (dates with at least one price)

    The matrix is cached globally for reuse.

    Args:
        prices_parquet: Path to parquet file
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        PriceMatrix with all prices loaded
    """
    global _PRICE_MATRIX
    if _PRICE_MATRIX is not None:
        return _PRICE_MATRIX

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

    # Order by date for efficient column building
    query += " ORDER BY ticker, field, date"

    result = con.execute(query).fetchall()
    con.close()

    if not result:
        # Return empty matrix
        _PRICE_MATRIX = PriceMatrix(
            price_matrix=np.empty((0, 0), dtype=np.float64),
            ticker_field_to_row={},
            date32_to_col=np.array([], dtype=np.int32),
            min_date32=0,
        )
        return _PRICE_MATRIX

    # Pass 1: Collect unique ticker|field and dates
    ticker_field_set: set[str] = set()
    date32_set: set[int] = set()
    epoch_ordinal = _EPOCH.toordinal()

    for ticker, field, dt, value in result:
        ticker_field_set.add(f"{ticker}|{field}")
        # Convert date to date32
        if isinstance(dt, date):
            date32 = dt.toordinal() - epoch_ordinal
        else:
            # String date
            dt_obj = date.fromisoformat(str(dt))
            date32 = dt_obj.toordinal() - epoch_ordinal
        date32_set.add(date32)

    # Build row index mapping
    ticker_field_list = sorted(ticker_field_set)
    ticker_field_to_row = {tf: i for i, tf in enumerate(ticker_field_list)}
    n_rows = len(ticker_field_list)

    # Build column index mapping (dense array for date32 -> col)
    date32_sorted = sorted(date32_set)
    date32_to_col_dict = {d: i for i, d in enumerate(date32_sorted)}
    n_cols = len(date32_sorted)

    min_date32 = date32_sorted[0]
    max_date32 = date32_sorted[-1]
    n_calendar_days = max_date32 - min_date32 + 1

    # Build dense date32_to_col array (-1 for non-business days)
    date32_to_col = np.full(n_calendar_days, -1, dtype=np.int32)
    for d32, col in date32_to_col_dict.items():
        date32_to_col[d32 - min_date32] = col

    # Pass 2: Fill price matrix
    price_matrix = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    for ticker, field, dt, value in result:
        key = f"{ticker}|{field}"
        row = ticker_field_to_row[key]

        # Convert date to date32
        if isinstance(dt, date):
            date32 = dt.toordinal() - epoch_ordinal
        else:
            dt_obj = date.fromisoformat(str(dt))
            date32 = dt_obj.toordinal() - epoch_ordinal

        col = date32_to_col_dict[date32]

        # Convert value to float
        try:
            price_matrix[row, col] = float(value)
        except (ValueError, TypeError):
            pass  # Keep as NaN

    _PRICE_MATRIX = PriceMatrix(
        price_matrix=price_matrix,
        ticker_field_to_row=ticker_field_to_row,
        date32_to_col=date32_to_col,
        min_date32=min_date32,
    )

    return _PRICE_MATRIX


def get_prices_matrix() -> PriceMatrix | None:
    """Get the cached PriceMatrix (or None if not loaded)."""
    return _PRICE_MATRIX


def set_prices_matrix(pm: PriceMatrix | None) -> None:
    """Set the global PriceMatrix (used by workers)."""
    global _PRICE_MATRIX
    _PRICE_MATRIX = pm


def clear_prices_matrix() -> None:
    """Clear the cached PriceMatrix."""
    global _PRICE_MATRIX
    _PRICE_MATRIX = None


# -------------------------------------
# PricesNumba: Fast PyArrow-based price storage
# -------------------------------------


@dataclass
class PricesNumba:
    """Numba-compatible price storage using sorted arrays with binary search.

    This structure is 5x faster to load than PriceMatrix (uses PyArrow + to_numpy()
    instead of DuckDB + Python loops) and supports 5x faster lookups via Numba
    binary search.

    The key insight is that string columns are dictionary-encoded to integer indices,
    which Numba can work with directly. The actual string values are stored in small
    lookup dicts for display/debugging only.

    Attributes:
        sorted_keys: int64[N] - composite keys for binary search
            Key formula: (ticker_idx * n_fields + field_idx) * n_dates + date_offset
        sorted_values: float64[N] - price values corresponding to sorted_keys
        ticker_to_idx: dict[str, int] - ticker string -> integer index
        field_to_idx: dict[str, int] - field string -> integer index
        n_tickers: int - number of unique tickers
        n_fields: int - number of unique fields
        n_dates: int - date range span (max_date - min_date + 1)
        min_date32: int - minimum date as days since 1970-01-01
    """
    sorted_keys: np.ndarray
    sorted_values: np.ndarray
    ticker_to_idx: dict[str, int]
    field_to_idx: dict[str, int]
    n_tickers: int
    n_fields: int
    n_dates: int
    min_date32: int

    def get_ticker_idx(self, ticker: str) -> int:
        """Get integer index for a ticker string. Returns -1 if not found."""
        return self.ticker_to_idx.get(ticker, -1)

    def get_field_idx(self, field: str) -> int:
        """Get integer index for a field string. Returns -1 if not found."""
        return self.field_to_idx.get(field, -1)


# Global cache for PricesNumba
_PRICES_NUMBA: PricesNumba | None = None


def load_prices_numba(
    prices_parquet: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> PricesNumba:
    """Load prices into Numba-compatible sorted arrays.

    This is 5x faster than load_all_prices() and supports 5x faster lookups
    using Numba binary search instead of Python dict.get().

    Uses PyArrow for fast parquet loading with dictionary encoding for strings,
    avoiding the expensive Python object creation that makes to_pylist() slow.

    Args:
        prices_parquet: Path to parquet file
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        PricesNumba with sorted arrays for binary search lookup
    """
    global _PRICES_NUMBA
    if _PRICES_NUMBA is not None:
        return _PRICES_NUMBA

    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    # 1. Load with PyArrow (0.2s vs 2.3s for DuckDB)
    table = pq.read_table(str(prices_parquet))

    # 2. Optional date filter using Arrow predicates
    if start_date or end_date:
        import pyarrow as pa
        filters = []
        if start_date:
            start_scalar = pa.scalar(date.fromisoformat(start_date), type=pa.date32())
            filters.append(pc.greater_equal(table['date'], start_scalar))
        if end_date:
            end_scalar = pa.scalar(date.fromisoformat(end_date), type=pa.date32())
            filters.append(pc.less_equal(table['date'], end_scalar))

        if len(filters) == 1:
            mask = filters[0]
        else:
            mask = pc.and_(filters[0], filters[1])
        table = table.filter(mask)

    if table.num_rows == 0:
        # Return empty structure
        _PRICES_NUMBA = PricesNumba(
            sorted_keys=np.array([], dtype=np.int64),
            sorted_values=np.array([], dtype=np.float64),
            ticker_to_idx={},
            field_to_idx={},
            n_tickers=0,
            n_fields=0,
            n_dates=0,
            min_date32=0,
        )
        return _PRICES_NUMBA

    # 3. Dictionary-encode strings (0.13s) - converts to integer indices
    ticker_dict = pc.dictionary_encode(table['ticker']).combine_chunks()
    field_dict = pc.dictionary_encode(table['field']).combine_chunks()

    # 4. Extract NumPy arrays (0.03s) - near zero-copy for numeric types
    ticker_idx = ticker_dict.indices.to_numpy().astype(np.int32)
    field_idx = field_dict.indices.to_numpy().astype(np.int32)
    date_arr = table['date'].to_numpy()
    value_arr = table['value'].to_numpy().astype(np.float64)

    # 5. Get string dictionaries and remap to alphabetical order (~0.02s)
    # This ensures composite keys are monotonic if parquet is sorted by (ticker, field, date)
    ticker_strings = ticker_dict.dictionary.to_pylist()
    field_strings = field_dict.dictionary.to_pylist()

    # Build alphabetical order remapping for tickers
    ticker_alpha_order = np.argsort(ticker_strings)
    ticker_remap = np.empty(len(ticker_alpha_order), dtype=np.int32)
    ticker_remap[ticker_alpha_order] = np.arange(len(ticker_alpha_order), dtype=np.int32)
    ticker_idx = ticker_remap[ticker_idx]

    # Build alphabetical order remapping for fields
    field_alpha_order = np.argsort(field_strings)
    field_remap = np.empty(len(field_alpha_order), dtype=np.int32)
    field_remap[field_alpha_order] = np.arange(len(field_alpha_order), dtype=np.int32)
    field_idx = field_remap[field_idx]

    # Build lookup dicts with alphabetical indices
    ticker_to_idx = {ticker_strings[i]: int(ticker_remap[i]) for i in range(len(ticker_strings))}
    field_to_idx = {field_strings[i]: int(field_remap[i]) for i in range(len(field_strings))}

    # 6. Convert dates to int32 offset from epoch
    epoch = np.datetime64('1970-01-01', 'D')
    date_int32 = (date_arr - epoch).astype(np.int32)
    min_date32 = int(date_int32.min())
    max_date32 = int(date_int32.max())
    n_dates = max_date32 - min_date32 + 1

    # 7. Build composite keys and sort (0.13s if unsorted, skipped if pre-sorted)
    n_tickers = len(ticker_strings)
    n_fields = len(field_strings)
    date_offset = (date_int32 - min_date32).astype(np.int64)

    # Composite key: (ticker_idx * n_fields + field_idx) * n_dates + date_offset
    composite_key = (ticker_idx.astype(np.int64) * n_fields + field_idx) * n_dates + date_offset

    # Check if already sorted (cheap O(n) check, ~0.002s)
    # If parquet was pre-sorted by (ticker, field, date) and we've remapped indices
    # to alphabetical order, the composite key will be monotonic
    is_sorted = len(composite_key) <= 1 or bool(np.all(composite_key[:-1] <= composite_key[1:]))

    if is_sorted:
        # Skip expensive argsort (~0.11s) and reorder (~0.02s)
        sorted_keys = composite_key
        sorted_values = value_arr
    else:
        sort_idx = np.argsort(composite_key)
        sorted_keys = composite_key[sort_idx]
        sorted_values = value_arr[sort_idx]

    _PRICES_NUMBA = PricesNumba(
        sorted_keys=sorted_keys,
        sorted_values=sorted_values,
        ticker_to_idx=ticker_to_idx,
        field_to_idx=field_to_idx,
        n_tickers=n_tickers,
        n_fields=n_fields,
        n_dates=n_dates,
        min_date32=min_date32,
    )

    return _PRICES_NUMBA


def get_prices_numba() -> PricesNumba | None:
    """Get the cached PricesNumba (or None if not loaded)."""
    return _PRICES_NUMBA


def set_prices_numba(pn: PricesNumba | None) -> None:
    """Set the global PricesNumba (used by workers)."""
    global _PRICES_NUMBA
    _PRICES_NUMBA = pn


def clear_prices_numba() -> None:
    """Clear the cached PricesNumba."""
    global _PRICES_NUMBA
    _PRICES_NUMBA = None


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
# Query-based price access
# -------------------------------------


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


# -------------------------------------
# Straddle price building helpers
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


# -------------------------------------
# Public API: get_prices
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
    ticker_table = tickers.filter_tickers(underlying, year, month, i, path, chain_csv)

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
