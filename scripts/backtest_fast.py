#!/usr/bin/env python3
"""
Fast backtest script.

Runs straddle valuations for all straddles matching a pattern across a year range.
Uses fast straddle expansion from strings.py (~12x faster than schedules.expand).
"""
import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# Add src to path so we can import specparser
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from specparser.amt import loader, schedules
from specparser.amt.tickers import get_straddle_valuation, load_all_prices, set_prices_dict
from specparser.amt import tickers as tickers_module
from specparser.amt import schedules as schedules_module
from specparser.amt.strings import precompute_templates, expand_fast


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


# Global vars set by worker initializer
_worker_amt = None
_worker_prices = None
_worker_chain = None
_worker_memoize = True
_worker_prices_dict = None


def init_worker(amt, prices, chain, memoize, prices_dict):
    """Initialize worker process by warming up caches."""
    global _worker_amt, _worker_prices, _worker_chain, _worker_memoize, _worker_prices_dict
    _worker_amt = amt
    _worker_prices = prices
    _worker_chain = chain
    _worker_memoize = memoize
    _worker_prices_dict = prices_dict

    # Set memoization state
    tickers_module.set_memoize_enabled(memoize)
    schedules_module.set_memoize_enabled(memoize)

    # Set the prices dict in tickers module (for workers to use)
    set_prices_dict(prices_dict)

    # Warm up other caches
    loader.load_amt(amt)  # Cache AMT file


def process_straddle(args_tuple):
    """Process a single straddle valuation. Used by multiprocessing pool."""
    asset, year, month, i, task_id, total = args_tuple

    try:
        val_table = get_straddle_valuation(
            asset, year, month, i, _worker_amt, _worker_chain, _worker_prices
        )

        # Filter out rows where mv is "-"
        columns = val_table["columns"]
        mv_idx = columns.index("mv") if "mv" in columns else None
        if mv_idx is not None:
            filtered_rows = [row for row in val_table["rows"] if row[mv_idx] != "-"]
        else:
            filtered_rows = val_table["rows"]

        return {
            "task_id": task_id,
            "total": total,
            "asset": asset,
            "year": year,
            "month": month,
            "i": i,
            "columns": columns,
            "rows": filtered_rows,
            "error": None
        }
    except ValueError as e:
        return {
            "task_id": task_id,
            "total": total,
            "asset": asset,
            "year": year,
            "month": month,
            "i": i,
            "columns": None,
            "rows": [],
            "error": str(e)
        }


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
    parser.add_argument("--workers", "-j", type=int, default=None,
                        help="Number of worker processes (default: CPU count)")
    parser.add_argument("--no-memoize", action="store_true",
                        help="Disable memoization/caching (for performance comparison)")

    args = parser.parse_args()
    memoize = not args.no_memoize

    # Initialize timer
    timer = Timer(enabled=args.timing)

    # === Load AMT ===
    amt = loader.load_amt(args.amt)
    timer.checkpoint("Load AMT YAML")

    # === Get assets matching pattern ===
    schedules_dict = amt.get("expiry_schedules", {})
    assets = list(loader._iter_assets(args.amt, live_only=True, pattern=args.pattern))
    timer.checkpoint("Find matching assets")

    if not assets:
        print(f"No assets found for pattern '{args.pattern}'", file=sys.stderr)
        return 1

    # === Pre-compute templates ===
    templates = precompute_templates(
        schedules_dict, assets,
        schedules._underlying_hash, schedules._fix_value,
    )
    timer.checkpoint("Precompute templates")

    if not templates:
        print(f"No schedules found for pattern '{args.pattern}'", file=sys.stderr)
        return 1

    # === Fast expand ===
    table = expand_fast(templates, args.start_year, args.end_year)
    timer.checkpoint(f"Expand straddles ({len(table['rows']):,})")

    if args.verbose:
        print(f"Expanded {len(table['rows']):,} straddles", file=sys.stderr)

    if not table["rows"]:
        print(f"No straddles found for pattern '{args.pattern}' in {args.start_year}-{args.end_year}", file=sys.stderr)
        return 1

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
    task_id = 0
    for (asset, year, month) in sorted_keys:
        count = straddle_counts[(asset, year, month)]
        for i in range(count):
            task_id += 1
            tasks.append((asset, year, month, i, task_id, total_straddles))
    timer.checkpoint(f"Build task list ({len(tasks):,} tasks)")

    # === Load prices ===
    num_workers = args.workers or cpu_count()
    start_date = f"{args.start_year - 1}-01-01"
    end_date = f"{args.end_year}-12-31"

    if args.verbose:
        print(f"Loading prices from {start_date} to {end_date}...", file=sys.stderr)

    prices_dict = load_all_prices(args.prices, start_date, end_date)
    timer.checkpoint(f"Load prices ({len(prices_dict):,} entries)")

    if args.verbose:
        print(f"Loaded {len(prices_dict):,} price entries", file=sys.stderr)
        print(f"Processing {total_straddles} straddles with {num_workers} workers...", file=sys.stderr)

    # === Process straddles with multiprocessing ===
    all_rows = []
    columns = None
    completed = 0

    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(args.amt, args.prices, args.chain, memoize, prices_dict)
    ) as pool:
        for result in pool.imap_unordered(process_straddle, tasks):
            completed += 1
            if args.verbose:
                if completed % 100 == 0:
                    print(f"[{completed}/{total_straddles}] {result['asset']} {result['year']}-{result['month']:02d} straddle {result['i']}", file=sys.stderr)

            if result["error"]:
                print(f"# Error for {result['asset']} {result['year']}-{result['month']:02d} straddle {result['i']}: {result['error']}", file=sys.stderr)
            else:
                if columns is None and result["columns"]:
                    columns = result["columns"]
                all_rows.extend(result["rows"])

    timer.checkpoint(f"Process straddles ({completed:,} done)")

    if args.verbose:
        rate = completed / (timer.checkpoints[-1][1] - timer.checkpoints[-2][1]) if len(timer.checkpoints) >= 2 else 0
        print(f"Processed {completed} straddles ({rate:.1f} straddles/s)", file=sys.stderr)

    # === Sort results ===
    if columns and all_rows:
        asset_col = columns.index("asset") if "asset" in columns else None
        straddle_col = columns.index("straddle") if "straddle" in columns else None
        date_col = columns.index("date") if "date" in columns else None

        if asset_col is not None and straddle_col is not None and date_col is not None:
            all_rows.sort(key=lambda r: (r[asset_col], r[straddle_col], r[date_col]))
        timer.checkpoint(f"Sort results ({len(all_rows):,} rows)")

        # === Output ===
        loader.print_table({"columns": columns, "rows": all_rows})
        timer.checkpoint("Print output")

    # Print timing report
    timer.report()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
