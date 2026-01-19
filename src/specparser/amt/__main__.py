# -------------------------------------
# AMT CLI entry point
# -------------------------------------
"""
CLI entry point for the AMT module (ticker operations).

For asset queries and table utilities, use:
    python -m specparser.amt.loader

Usage:
    python -m specparser.amt data/amt.yml --asset-tickers "LA Comdty"
    python -m specparser.amt data/amt.yml --live-tickers 2024 2025
"""
import argparse

from .loader import print_table
from .tickers import (
    get_tschemas,
    find_tschemas,
    live_tickers,
    fut_spec2ticker,
    asset_straddle,
    straddle_days,
)


def _main() -> int:
    p = argparse.ArgumentParser(
        description="AMT ticker utilities. For asset/table queries, use: python -m specparser.amt.loader",
    )
    p.add_argument("path", help="Path to AMT YAML file")
    p.add_argument("--asset-tickers", "-g", metavar="UNDERLYING", help="Get all tickers for an asset by Underlying value")
    p.add_argument("--find-tickers", "-f", metavar="PATTERN", help="Find tickers for assets matching regex pattern")
    p.add_argument("--live-tickers", nargs="*", type=int, metavar=("START_YEAR", "END_YEAR"), help="Get all tickers for live assets (optional: START_YEAR END_YEAR to expand BBGfc)")
    p.add_argument("--chain-csv", metavar="CSV_PATH", help="CSV file with normalized_future,actual_future columns for ticker lookup")
    p.add_argument("--fut", nargs=3, metavar=("SPEC", "YEAR", "MONTH"), help="Compute futures ticker from spec string, year, and month")
    p.add_argument("--straddle", nargs=2, metavar=("UNDERLYING", "STRADDLE"), help="Get straddle info with tickers for an asset")
    p.add_argument("--prices", metavar="PARQUET_PATH", help="Prices parquet file for straddle_days lookup (use with --straddle)")
    args = p.parse_args()

    if args.asset_tickers:
        table = get_tschemas(args.path, args.asset_tickers)
        if table["rows"]:
            print_table(table)
        else:
            print(f"No asset found with Underlying: {args.asset_tickers}")
            return 1
    elif args.find_tickers:
        table = find_tschemas(args.path, args.find_tickers)
        if table["rows"]:
            print_table(table)
        else:
            print(f"No assets found matching: {args.find_tickers}")
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
    elif args.fut:
        spec, year_str, month_str = args.fut
        try:
            year = int(year_str)
            month = int(month_str)
            ticker = fut_spec2ticker(spec, year, month)
            print(ticker)
        except (ValueError, IndexError) as e:
            print(f"Error computing futures ticker: {e}")
            return 1
    elif args.straddle:
        underlying, straddle_str = args.straddle
        try:
            table = asset_straddle(args.path, underlying, straddle_str, chain_csv=args.chain_csv)
            if args.prices:
                table = straddle_days(table, args.prices)
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
