#!/usr/bin/env python
"""
Fast backtest script using PyArrow-based price loading with unified Numba kernel.

Uses load_prices_numba() for 8x faster price loading compared to DuckDB,
combined with full_backtest_kernel_sorted() - a unified Numba kernel that
performs binary search price lookups, entry/expiry detection, model computation,
and PnL calculation all in a single parallel kernel.

Compare with backtest.py which uses load_prices_matrix() (DuckDB-based) and
full_backtest_kernel() (matrix-based price lookup).
"""
import argparse
import sys
import time
from pathlib import Path

# Add src to path so we can import specparser
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from specparser.amt.prices import load_prices_numba, clear_prices_numba
from specparser.amt.valuation import get_straddle_backtests


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


def main():
    parser = argparse.ArgumentParser(
        description="Run straddle valuations using PyArrow + Numba pipeline (fast)"
    )
    parser.add_argument("pattern", help="Regex pattern to match assets (e.g., '^LA Comdty', '.')")
    parser.add_argument("start_year", type=int, help="Start year")
    parser.add_argument("end_year", type=int, help="End year")

    parser.add_argument("--amt", default="data/amt.yml",
                        help="Path to AMT YAML file (default: data/amt.yml)")
    parser.add_argument("--chain", default="data/futs.csv",
                        help="Path to futures chain CSV (default: data/futs.csv)")
    parser.add_argument("--prices", default="data/prices.parquet",
                        help="Path to prices parquet file (default: data/prices.parquet)")
    parser.add_argument("--overrides", default="data/overrides.csv",
                        help="Path to overrides CSV file (default: data/overrides.csv)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress to stderr")
    parser.add_argument("--timing", "-t", action="store_true",
                        help="Print detailed timing breakdown")
    parser.add_argument("--benchmark", "-b", nargs="?", const=1, type=int, metavar="N",
                        help="Benchmark mode: run N iterations and report timing (default: 1)")

    args = parser.parse_args()

    # Initialize timer
    timer = Timer(enabled=args.timing or args.benchmark)

    # === Load prices using PyArrow (faster than DuckDB) ===
    if args.verbose:
        print("Loading prices with PyArrow...", file=sys.stderr)

    # Calculate date range for filtering (add buffer for entry dates)
    start_date = f"{args.start_year - 1}-01-01"
    end_date = f"{args.end_year + 1}-12-31"

    import time as time_module
    load_start = time_module.perf_counter()
    load_prices_numba(args.prices, start_date, end_date)
    load_elapsed = time_module.perf_counter() - load_start
    timer.checkpoint("Load prices (PyArrow)")

    # Store load time for benchmark output
    prices_load_time = load_elapsed

    # === Warmup run (for JIT compilation) ===
    if args.verbose:
        print("Warming up Numba JIT...", file=sys.stderr)

    # Small warmup run to compile the kernel
    _ = get_straddle_backtests(
        args.pattern, args.start_year, args.start_year,
        args.amt, args.chain, args.prices,
        price_lookup='numba_sorted_kernel', valid_only=True,
        overrides_path=args.overrides,
    )
    timer.checkpoint("JIT warmup")

    # === Run backtest with numba_sorted_kernel (full unified kernel) ===
    if args.verbose:
        print(f"Running backtest for pattern '{args.pattern}' {args.start_year}-{args.end_year}...", file=sys.stderr)

    result = get_straddle_backtests(
        args.pattern, args.start_year, args.end_year,
        args.amt, args.chain, args.prices,
        price_lookup='numba_sorted_kernel', valid_only=True,
        overrides_path=args.overrides,
    )
    # Get row count for checkpoint message
    if hasattr(result, 'num_rows'):
        checkpoint_rows = result.num_rows
    else:
        checkpoint_rows = len(result.get('rows', []))
    timer.checkpoint(f"Backtest ({checkpoint_rows:,} rows)")

    # Get row count (works for both dict and Arrow table)
    if hasattr(result, 'num_rows'):
        n_rows = result.num_rows
    else:
        n_rows = len(result.get('rows', []))

    # === Convert and sort ===
    if args.verbose:
        print("Converting to pandas and sorting...", file=sys.stderr)

    import pandas as pd

    # Handle both PyArrow table and dict output
    if hasattr(result, 'to_pandas'):
        df = result.to_pandas()
    else:
        # Dict-based output - convert to DataFrame
        df = pd.DataFrame(result['rows'], columns=result['columns'])

    # Convert date columns to string for consistent output
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)
    if 'expiry' in df.columns:
        df['expiry'] = df['expiry'].astype(str)

    # Sort by asset, straddle, date
    df = df.sort_values(['asset', 'straddle', 'date'])
    timer.checkpoint("Convert and sort")

    # === Output (skip in benchmark mode) ===
    if not args.benchmark:
        # Output as tab-separated (matching print_table format)
        print('\t'.join(df.columns))
        for _, row in df.iterrows():
            print('\t'.join(str(v) for v in row.values))
        timer.checkpoint("Print output")

    # Print timing report (only if --timing is set)
    if args.timing:
        timer.report()

    # Benchmark summary (only if --benchmark is set)
    if args.benchmark:
        # Get processing time (excluding price loading and JIT warmup)
        process_time = 0
        for i, (name, elapsed) in enumerate(timer.checkpoints):
            if "Backtest" in name:
                prev_elapsed = timer.checkpoints[i-1][1] if i > 0 else 0
                process_time = elapsed - prev_elapsed
                break

        # Count straddles (approximate from pattern)
        from specparser.amt import schedules
        straddles_table = schedules.find_straddle_yrs(
            args.amt, args.start_year, args.end_year, args.pattern, live_only=True
        )
        n_straddles = len(straddles_table["rows"])

        rate = n_straddles / process_time if process_time > 0 else 0

        print(f"\n{'='*60}", file=sys.stderr)
        print("BENCHMARK RESULTS (PyArrow + numba_sorted_kernel)", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"  Prices load:     {prices_load_time:.2f}s", file=sys.stderr)
        print(f"  Straddles:       {n_straddles:,}", file=sys.stderr)
        print(f"  Backtest time:   {process_time:.2f}s", file=sys.stderr)
        print(f"  Rate:            {rate:,.1f} straddles/sec", file=sys.stderr)
        print(f"  Output rows:     {n_rows:,}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    # Clean up
    clear_prices_numba()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
