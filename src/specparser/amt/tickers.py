# -------------------------------------
# AMT tickers - Ticker extraction
# -------------------------------------
"""
Ticker extraction for straddle pricing.

This module provides `filter_tickers()` which returns tickers needed for
pricing a specific straddle. The actual ticker computation is delegated to
`asset_straddle_tickers.py` which handles all source types and smart caching.
"""
from pathlib import Path
from typing import Any

from . import loader
from . import schedules
from . import asset_straddle_tickers
from .table import table_column


# -------------------------------------
# Memoization control
# -------------------------------------

def set_memoize_enabled(enabled: bool) -> None:
    """Enable or disable memoization for ticker functions."""
    asset_straddle_tickers.set_memoize_enabled(enabled)


def clear_ticker_caches() -> None:
    """Clear all ticker-related caches."""
    asset_straddle_tickers.clear_ticker_caches()


# -------------------------------------
# Source inference
# -------------------------------------

def _infer_source(field: str) -> str:
    """Infer source from field pattern."""
    if field == "none":  # CV vol
        return "CV"
    if field == "":  # calc tickers
        return "calc"
    return "BBG"  # Default for everything with a field


# -------------------------------------
# Main API
# -------------------------------------

def filter_tickers(
    asset: str,
    year: int,
    month: int,
    i: int,
    amt_path: str | Path,
    chain_path: str | Path | None = None,
) -> dict[str, Any]:
    """Get tickers for an asset's straddle.

    Columns: ['asset', 'straddle', 'param', 'source', 'ticker', 'field']

    Args:
        asset: Asset underlying value
        year: Entry year
        month: Entry month
        i: Straddle selector index (i % len(straddles))
        amt_path: Path to AMT YAML file
        chain_path: Optional CSV for futures ticker lookup (unused, kept for API compatibility)

    Returns:
        Table with ticker rows for the straddle
    """
    # Validate asset exists
    if loader.get_asset(amt_path, asset) is None:
        raise ValueError(f"Asset '{asset}' not found")

    # Get straddle info
    straddles = table_column(schedules.get_expand_ym(amt_path, asset, year, month), "straddle")
    if len(straddles) < 1:
        raise ValueError(f"'{asset}' has no straddles in {year}-{month:02d}")
    straddle = straddles[i % len(straddles)]

    xpry, xprm = schedules.xpry(straddle), schedules.xprm(straddle)
    ntrc = schedules.ntrc(straddle)
    strym = f"{xpry}-{xprm:02d}"

    # Call into asset_straddle_tickers (the source of truth)
    ticker_table = asset_straddle_tickers.get_asset_straddle_tickers(asset, strym, ntrc, amt_path)

    # Transform output to expected format
    # Input columns: ['name', 'ticker', 'field']
    # Output columns: ['asset', 'straddle', 'param', 'source', 'ticker', 'field']
    result = {
        "orientation": "row",
        "columns": ["asset", "straddle", "param", "source", "ticker", "field"],
        "rows": []
    }

    for row in ticker_table["rows"]:
        name, ticker, field = row
        source = _infer_source(field)
        result["rows"].append([asset, straddle, name, source, ticker, field])

    return result


# -------------------------------------
# CLI
# -------------------------------------


def _main() -> int:
    import argparse
    from .table import print_table
    from . import prices as prices_module
    from . import valuation as valuation_module

    p = argparse.ArgumentParser(
        description="Ticker extraction and transformation utilities.",
        allow_abbrev=False,
    )
    p.add_argument("path", help="Path to AMT YAML file")

    p.add_argument("--chain-csv", default="data/futs.csv",
                   help="CSV file with normalized_future,actual_future columns (default: data/futs.csv)")

    p.add_argument("--prices", default="data/prices.parquet",
                   help="Prices parquet file (default: data/prices.parquet)")

    # Commands
    p.add_argument("--expand-ym", nargs=4, type=str, metavar=("PATTERN", "LIVE", "YEAR", "MONTH"),
                   help="Get straddles for assets matching patterns on month.")

    p.add_argument("--get-expand-ym", nargs=3, type=str, metavar=("ASSET", "YEAR", "MONTH"),
                   help="Get straddles for asset on month.")

    p.add_argument("--asset-tickers", nargs=4, type=str, metavar=("UNDERLYING", "YEAR", "MONTH", "NDX"),
                   help="Get tickers for a straddle.")

    p.add_argument("--straddle-tickers", nargs=3, type=str, metavar=("ASSET", "YYYY-MM", "NTRC"),
                   help="Get tickers directly via get_asset_straddle_tickers().")

    p.add_argument("--asset-days", nargs=4, type=str, metavar=("UNDERLYING", "YEAR", "MONTH", "NDX"),
                   help="Get daily prices for a straddle from entry to expiry month.")

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

    if args.expand_ym:
        pattern, live, year, month = args.expand_ym
        table = schedules.find_straddle_ym(args.path, int(year), int(month), pattern, str2bool(live))
        print_table(table)

    elif args.get_expand_ym:
        asset, year, month = args.get_expand_ym
        table = schedules.get_expand_ym(args.path, asset, int(year), int(month))
        print_table(table)

    elif args.asset_tickers:
        underlying, year, month, i = args.asset_tickers
        table = filter_tickers(underlying, int(year), int(month), int(i), args.path, args.chain_csv)
        print_table(table)

    elif args.straddle_tickers:
        asset, ym, ntrc = args.straddle_tickers
        table = asset_straddle_tickers.get_asset_straddle_tickers(asset, ym, ntrc, args.path)
        print_table(table)

    elif args.asset_days:
        underlying, year, month, i = args.asset_days
        table = valuation_module.get_straddle_actions(underlying, int(year), int(month), int(i), args.path, args.chain_csv, args.prices)
        print_table(table)

    elif args.prices_last:
        table = prices_module.prices_last(args.prices, args.prices_last)
        print_table(table)

    elif args.prices_query:
        table = prices_module.prices_query(args.prices, args.prices_query)
        print_table(table)

    elif args.straddle_valuation:
        underlying, year, month, i = args.straddle_valuation
        table = valuation_module.get_straddle_valuation(underlying, int(year), int(month), int(i), args.path, args.chain_csv, args.prices)
        print_table(table)

    else:
        p.print_help()

    return 0


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(_main())
