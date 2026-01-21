#!/usr/bin/env python3
"""
Straddle valuation script.

Displays straddle valuation data (mv, delta, pnl) for debugging.
"""
import argparse
import sys
from pathlib import Path

# Add src to path so we can import specparser
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from specparser.amt import loader
from specparser.amt.tickers import get_straddle_valuation


def main():
    parser = argparse.ArgumentParser(
        description="Display straddle valuation data"
    )
    parser.add_argument("underlying", help="Asset underlying name")
    parser.add_argument("year", type=int, help="Expiry year")
    parser.add_argument("month", type=int, help="Expiry month")
    parser.add_argument("straddle_idx", type=int, help="Straddle index (0-based)")

    parser.add_argument("--amt", default="data/amt.yml",
                        help="Path to AMT YAML file (default: data/amt.yml)")
    parser.add_argument("--chain", default="data/futs.csv",
                        help="Path to futures chain CSV (default: data/futs.csv)")
    parser.add_argument("--prices", default="data/prices.parquet",
                        help="Path to prices parquet file (default: data/prices.parquet)")

    args = parser.parse_args()

    try:
        table = get_straddle_valuation(
            args.underlying,
            args.year,
            args.month,
            args.straddle_idx,
            args.amt,
            args.chain,
            args.prices
        )

        # Filter out rows where mv is "-"
        mv_idx = table["columns"].index("mv") if "mv" in table["columns"] else None
        if mv_idx is not None:
            table["rows"] = [row for row in table["rows"] if row[mv_idx] != "-"]

        loader.print_table(table)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
