#!/usr/bin/env python3
"""
Profile multiprocessing overhead in backtest.

Compares:
1. Single-threaded processing
2. Multiprocessing with varying worker counts
"""
import sys
import time
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from specparser.amt import loader, schedules
from specparser.amt.prices import load_all_prices, set_prices_dict
from specparser.amt.valuation import get_straddle_valuation
from specparser.amt import tickers as tickers_module
from specparser.amt import schedules as schedules_module


# Global vars for workers
_worker_amt = None
_worker_prices = None
_worker_chain = None
_worker_memoize = True
_worker_prices_dict = None


def init_worker(amt, prices, chain, memoize, prices_dict):
    """Initialize worker process."""
    global _worker_amt, _worker_prices, _worker_chain, _worker_memoize, _worker_prices_dict
    _worker_amt = amt
    _worker_prices = prices
    _worker_chain = chain
    _worker_memoize = memoize
    _worker_prices_dict = prices_dict

    tickers_module.set_memoize_enabled(memoize)
    schedules_module.set_memoize_enabled(memoize)
    set_prices_dict(prices_dict)
    loader.load_amt(amt)


def process_straddle(args_tuple):
    """Process a single straddle."""
    asset, year, month, i = args_tuple
    try:
        val_table = get_straddle_valuation(
            asset, year, month, i, _worker_amt, _worker_chain, _worker_prices
        )
        return len(val_table["rows"])
    except Exception as e:
        return 0


def process_straddle_singlethread(args_tuple, amt, chain, prices):
    """Process single straddle without multiprocessing globals."""
    asset, year, month, i = args_tuple
    try:
        val_table = get_straddle_valuation(asset, year, month, i, amt, chain, prices)
        return len(val_table["rows"])
    except Exception:
        return 0


def benchmark_singlethread(tasks, amt, chain, prices, prices_dict):
    """Run all tasks in a single thread."""
    set_prices_dict(prices_dict)
    loader.load_amt(amt)
    tickers_module.set_memoize_enabled(True)
    schedules_module.set_memoize_enabled(True)

    total_rows = 0
    for task in tasks:
        total_rows += process_straddle_singlethread(task, amt, chain, prices)
    return total_rows


def benchmark_multiprocess(tasks, num_workers, amt, prices, chain, prices_dict):
    """Run tasks with multiprocessing."""
    total_rows = 0
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(amt, prices, chain, True, prices_dict)
    ) as pool:
        for rows in pool.imap_unordered(process_straddle, tasks):
            total_rows += rows
    return total_rows


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile multiprocessing overhead")
    parser.add_argument("--pattern", default="^LA Comdty", help="Asset pattern")
    parser.add_argument("--start-year", type=int, default=2001, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument("--amt", default="data/amt.yml", help="AMT YAML file")
    parser.add_argument("--chain", default="data/futs.csv", help="Chain CSV file")
    parser.add_argument("--prices", default="data/prices.parquet", help="Prices parquet file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of straddles")

    args = parser.parse_args()

    # Load prices
    print("Loading prices...")
    start_date = f"{args.start_year - 1}-01-01"
    end_date = f"{args.end_year}-12-31"
    prices_dict = load_all_prices(args.prices, start_date, end_date)
    print(f"Loaded {len(prices_dict):,} prices ({len(prices_dict) * 50 / 1024 / 1024:.1f} MB approx)")

    # Get all straddles
    print(f"\nFinding straddles for pattern '{args.pattern}'...")
    set_prices_dict(prices_dict)
    table = schedules.find_straddle_yrs(args.amt, args.start_year, args.end_year, args.pattern, True)
    print(f"Found {len(table['rows'])} straddles")

    # Build task list
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

    tasks = []
    for (asset, year, month) in sorted(straddle_counts.keys()):
        count = straddle_counts[(asset, year, month)]
        for i in range(count):
            tasks.append((asset, year, month, i))

    if args.limit:
        tasks = tasks[:args.limit]

    print(f"Processing {len(tasks)} straddles")

    results = []

    # Single-threaded benchmark
    print(f"\n--- Single-threaded ---")
    # Clear caches
    tickers_module.clear_ticker_caches()
    schedules_module.clear_schedule_caches()

    t0 = time.perf_counter()
    rows = benchmark_singlethread(tasks, args.amt, args.chain, args.prices, prices_dict)
    elapsed = time.perf_counter() - t0
    results.append(("Single-thread", 1, elapsed))
    print(f"Time: {elapsed:.2f}s ({len(tasks)/elapsed:.1f} straddles/sec)")
    print(f"Total rows: {rows:,}")

    # Multiprocessing benchmarks
    for num_workers in [2, 4, 8, 16]:
        if num_workers > cpu_count():
            continue

        print(f"\n--- {num_workers} workers ---")

        t0 = time.perf_counter()
        rows = benchmark_multiprocess(tasks, num_workers, args.amt, args.prices, args.chain, prices_dict)
        elapsed = time.perf_counter() - t0
        results.append((f"{num_workers} workers", num_workers, elapsed))
        print(f"Time: {elapsed:.2f}s ({len(tasks)/elapsed:.1f} straddles/sec)")
        print(f"Total rows: {rows:,}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<20} {'Workers':<10} {'Time':>10} {'Speedup':>10} {'Efficiency':>10}")
    print("-" * 60)

    baseline = results[0][2]
    for name, workers, elapsed in results:
        speedup = baseline / elapsed
        efficiency = speedup / workers * 100 if workers > 0 else 100
        print(f"{name:<20} {workers:<10} {elapsed:>10.2f}s {speedup:>10.2f}x {efficiency:>9.1f}%")


if __name__ == "__main__":
    main()
