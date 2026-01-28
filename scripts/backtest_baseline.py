#!/usr/bin/env python
"""
Baseline backtest script.

Runs straddle valuations for all straddles matching a pattern across a year range.
Uses the per-straddle get_straddle_valuation() approach with dict lookups.

THIS IS THE BASELINE - DO NOT OPTIMIZE.
Use backtest.py or backtest_new.py for optimized batch processing.
"""
import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

# Add src to path so we can import specparser
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from specparser.amt import loader, schedules
from specparser.amt.prices import load_all_prices, set_prices_dict
from specparser.amt.valuation import get_straddle_valuation
from specparser.amt import tickers as tickers_module
from specparser.amt import schedules as schedules_module


# -------------------------------------
# Timing utilities
# -------------------------------------

class Timer:
    """Simple checkpoint timer for performance analysis."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.start_time = time.perf_counter()
        self.checkpoints = []

    def checkpoint(self, name: str):
        """Record a checkpoint with elapsed time since start."""
        if self.enabled:
            elapsed = time.perf_counter() - self.start_time
            self.checkpoints.append((name, elapsed))

    def report(self, file=sys.stderr):
        """Print timing report."""
        if not self.enabled or not self.checkpoints:
            return

        print("\n" + "=" * 60, file=file)
        print("TIMING BREAKDOWN", file=file)
        print("=" * 60, file=file)

        prev_time = 0
        for name, elapsed in self.checkpoints:
            delta = elapsed - prev_time
            pct = (delta / self.checkpoints[-1][1] * 100) if self.checkpoints[-1][1] > 0 else 0
            print(f"  {name:30s} {delta:8.3f}s  ({pct:5.1f}%)", file=file)
            prev_time = elapsed

        print("-" * 60, file=file)
        print(f"  {'TOTAL':30s} {self.checkpoints[-1][1]:8.3f}s", file=file)
        print("=" * 60 + "\n", file=file)


def process_straddle(asset, year, month, i, amt_path, chain_path, prices_path):
    """Process a single straddle valuation."""
    try:
        val_table = get_straddle_valuation(
            asset, year, month, i, amt_path, chain_path, prices_path
        )

        # Filter out rows where mv is "-"
        columns = val_table["columns"]
        mv_idx = columns.index("mv") if "mv" in columns else None
        if mv_idx is not None:
            filtered_rows = [row for row in val_table["rows"] if row[mv_idx] != "-"]
        else:
            filtered_rows = val_table["rows"]

        return columns, filtered_rows, None
    except ValueError as e:
        return None, [], str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Run straddle valuations for all straddles matching a pattern (fast version)"
    )
    parser.add_argument("pattern", help="Regex pattern to match assets (e.g., '^LA Comdty')")
    parser.add_argument("start_year", type=int, help="Start year")
    parser.add_argument("end_year", type=int, help="End year")

    parser.add_argument("--amt", default="data/amt.yml",
                        help="Path to AMT YAML file (default: data/amt.yml)")
    parser.add_argument("--chain", default="data/futs.csv",
                        help="Path to futures chain CSV (default: data/futs.csv)")
    parser.add_argument("--prices", default="data/prices.parquet",
                        help="Path to prices parquet file (default: data/prices.parquet)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress to stderr")
    parser.add_argument("--timing", "-t", action="store_true",
                        help="Print detailed timing breakdown")
    parser.add_argument("--benchmark", "-b", nargs="?", const=1, type=int, metavar="N",
                        help="Benchmark mode: run N iterations and report timing (default: 1)")

    args = parser.parse_args()

    # Initialize timer
    timer = Timer(enabled=args.timing or args.benchmark)

    # === Find all straddles ===
    table = schedules.find_straddle_yrs(args.amt, args.start_year, args.end_year, args.pattern, True)
    timer.checkpoint(f"Find straddles ({len(table['rows']):,})")

    if not table["rows"]:
        print(f"No straddles found for pattern '{args.pattern}' in {args.start_year}-{args.end_year}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(table['rows']):,} straddles", file=sys.stderr)

    # === Group by (asset, expiry_year, expiry_month) ===
    asset_idx = table["columns"].index("asset")
    straddle_idx = table["columns"].index("straddle")

    straddle_counts = defaultdict(int)
    for row in table["rows"]:
        asset = row[asset_idx]
        straddle = row[straddle_idx]
        xpry = schedules.xpry(straddle)
        xprm = schedules.xprm(straddle)
        key = (asset, xpry, xprm)
        straddle_counts[key] += 1
    timer.checkpoint("Group straddles by month")

    # === Build task list ===
    sorted_keys = sorted(straddle_counts.keys())
    total_straddles = sum(straddle_counts[k] for k in sorted_keys)

    tasks = []
    for (asset, year, month) in sorted_keys:
        count = straddle_counts[(asset, year, month)]
        for i in range(count):
            tasks.append((asset, year, month, i))
    timer.checkpoint(f"Build task list ({len(tasks):,} tasks)")

    # === Load prices ===
    start_date = f"{args.start_year - 1}-01-01"
    end_date = f"{args.end_year}-12-31"

    if args.verbose:
        print(f"Loading prices from {start_date} to {end_date}...", file=sys.stderr)

    prices_dict = load_all_prices(args.prices, start_date, end_date)
    set_prices_dict(prices_dict)
    timer.checkpoint(f"Load prices ({len(prices_dict):,} entries)")

    if args.verbose:
        print(f"Loaded {len(prices_dict):,} price entries", file=sys.stderr)
        print(f"Processing {total_straddles} straddles (single-threaded)...", file=sys.stderr)

    # === Enable memoization ===
    tickers_module.set_memoize_enabled(True)
    schedules_module.set_memoize_enabled(True)

    # === Process straddles (single-threaded) ===
    all_rows = []
    columns = None
    completed = 0
    errors = 0

    for asset, year, month, i in tasks:
        cols, rows, error = process_straddle(
            asset, year, month, i, args.amt, args.chain, args.prices
        )
        completed += 1

        if args.verbose and completed % 100 == 0:
            print(f"[{completed}/{total_straddles}] {asset} {year}-{month:02d} straddle {i}", file=sys.stderr)

        if error:
            errors += 1
            if args.verbose:
                print(f"# Error for {asset} {year}-{month:02d} straddle {i}: {error}", file=sys.stderr)
        else:
            if columns is None and cols:
                columns = cols
            all_rows.extend(rows)

    timer.checkpoint(f"Process straddles ({completed:,} done)")

    # Calculate processing rate
    process_time = timer.checkpoints[-1][1] - timer.checkpoints[-2][1] if len(timer.checkpoints) >= 2 else 0
    rate = completed / process_time if process_time > 0 else 0

    if args.verbose:
        print(f"Processed {completed} straddles ({rate:.1f} straddles/s)", file=sys.stderr)
        if errors:
            print(f"Errors: {errors}", file=sys.stderr)

    # === Sort results ===
    if columns and all_rows:
        asset_col = columns.index("asset") if "asset" in columns else None
        straddle_col = columns.index("straddle") if "straddle" in columns else None
        date_col = columns.index("date") if "date" in columns else None

        if asset_col is not None and straddle_col is not None and date_col is not None:
            all_rows.sort(key=lambda r: (r[asset_col], r[straddle_col], r[date_col]))
        timer.checkpoint(f"Sort results ({len(all_rows):,} rows)")

        # === Output (skip in benchmark mode) ===
        if not args.benchmark:
            loader.print_table({"columns": columns, "rows": all_rows})
            timer.checkpoint("Print output")

    # Print timing report (only if --timing is set)
    if args.timing:
        timer.report()

    # Benchmark summary (only if --benchmark is set)
    if args.benchmark:
        total_time = timer.checkpoints[-1][1] if timer.checkpoints else 0
        print(f"\n{'='*60}", file=sys.stderr)
        print("BENCHMARK RESULTS", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"  Straddles:       {completed:,}", file=sys.stderr)
        print(f"  Total time:      {total_time:.2f}s", file=sys.stderr)
        print(f"  Rate:            {rate:,.1f} straddles/sec", file=sys.stderr)
        print(f"  Output rows:     {len(all_rows):,}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
