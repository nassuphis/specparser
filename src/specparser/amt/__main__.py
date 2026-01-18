# -------------------------------------
# AMT CLI entry point
# -------------------------------------
"""
CLI entry point for the AMT module.

Usage:
    python -m specparser.amt data/amt.yml --expand 2024 2025
"""
import argparse

import yaml

from .loader import (
    get_asset,
    find_underlyings,
    get_table,
    format_table,
    print_table,
    list_assets,
    assets,
    live_assets,
    live_class,
    live_group,
    live_table,
    get_value,
    get_aum,
    get_leverage,
)
from .tickers import (
    asset_tickers,
    live_tickers,
    fut_ticker,
    asset_straddle,
    straddle_days,
)
from .schedules import (
    get_schedule,
    find_schedules,
    live_schedules,
    expand,
    find_expand,
)


def _main() -> int:
    p = argparse.ArgumentParser(
        description="AMT (Asset Management Table) utilities.",
    )
    p.add_argument("path", help="Path to AMT YAML file")
    p.add_argument("--get", "-g", metavar="UNDERLYING", help="Get asset by Underlying value")
    p.add_argument("--find", "-f", metavar="PATTERN", help="Find assets by regex pattern on Underlying")
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
    p.add_argument("--schedules", action="store_true", help="List all live assets with their schedules")
    p.add_argument("--find-schedules", metavar="PATTERN", help="Find schedules by regex pattern on Underlying")
    p.add_argument("--expand", nargs=2, type=int, metavar=("START_YEAR", "END_YEAR"), help="Expand live schedules into straddle strings")
    p.add_argument("--find-expand", nargs=3, metavar=("PATTERN", "START_YEAR", "END_YEAR"), help="Expand schedules matching pattern into straddle strings")
    p.add_argument("--value", "-v", metavar="KEY_PATH", help="Get value by dot-separated key path (e.g., backtest.aum)")
    p.add_argument("--aum", action="store_true", help="Get AUM value")
    p.add_argument("--leverage", action="store_true", help="Get leverage value")
    p.add_argument("--fut", nargs=3, metavar=("SPEC", "YEAR", "MONTH"), help="Compute futures ticker from spec string, year, and month")
    p.add_argument("--straddle", nargs=2, metavar=("UNDERLYING", "STRADDLE"), help="Get straddle info with tickers for an asset")
    p.add_argument("--prices", metavar="PARQUET_PATH", help="Prices parquet file for straddle_days lookup (use with --straddle)")
    args = p.parse_args()

    if args.get:
        asset = get_asset(args.path, args.get)
        if asset:
            print(yaml.dump(asset, default_flow_style=False))
        else:
            print(f"Asset not found: {args.get}")
            return 1
    elif args.find:
        underlyings = find_underlyings(args.path, args.find)
        if not underlyings:
            print(f"No assets found matching: {args.find}")
            return 1
        rows = []
        for u in underlyings:
            asset = get_asset(args.path, u)
            if asset:
                rows.append([u, asset.get("Class", ""), asset.get("WeightCap", "")])
        table = {"columns": ["Underlying", "Class", "WeightCap"], "rows": rows}
        print_table(table)
    elif args.schedule:
        asset = get_asset(args.path, args.schedule)
        if not asset:
            print(f"No asset with Underlying: {args.schedule}")
            return 1
        schedule = get_schedule(args.path, args.schedule)
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
    elif args.schedules:
        table = live_schedules(args.path)
        print_table(table)
    elif args.find_schedules:
        table = find_schedules(args.path, args.find_schedules)
        if not table["rows"]:
            print(f"No assets found matching: {args.find_schedules}")
            return 1
        print_table(table)
    elif args.expand:
        start_year, end_year = args.expand
        table = expand(args.path, start_year, end_year)
        print_table(table)
    elif args.find_expand:
        pattern, start_year, end_year = args.find_expand
        table = find_expand(args.path, pattern, int(start_year), int(end_year))
        if not table["rows"]:
            print(f"No assets found matching: {pattern}")
            return 1
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
