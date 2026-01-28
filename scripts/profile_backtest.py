#!/usr/bin/env python3
"""
Profile the backtest workflow to identify bottlenecks.

This script instruments key functions and measures their execution time.
"""
import sys
import time
from pathlib import Path
from collections import defaultdict
from functools import wraps

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import modules
from specparser.amt import loader, schedules
from specparser.amt.prices import load_all_prices, set_prices_dict, get_price, _lookup_straddle_prices, _build_ticker_map, _build_prices_table
from specparser.amt.valuation import get_straddle_valuation, get_straddle_actions, actions, _add_action_column, _add_model_column, _add_strike_columns, _compute_actions, model_ES
from specparser.amt import tickers as tickers_module
from specparser.amt import prices as prices_module


# Timing storage
TIMINGS = defaultdict(list)


def timed(name):
    """Decorator to measure function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            TIMINGS[name].append(elapsed)
            return result
        return wrapper
    return decorator


def print_timing_report():
    """Print timing report."""
    print("\n" + "=" * 70)
    print("TIMING REPORT")
    print("=" * 70)

    # Sort by total time
    sorted_items = sorted(TIMINGS.items(), key=lambda x: sum(x[1]), reverse=True)

    total_all = sum(sum(times) for _, times in sorted_items)

    for name, times in sorted_items:
        total = sum(times)
        count = len(times)
        avg = total / count if count > 0 else 0
        min_t = min(times) if times else 0
        max_t = max(times) if times else 0
        pct = (total / total_all * 100) if total_all > 0 else 0

        print(f"\n{name}:")
        print(f"  Calls: {count:,}")
        print(f"  Total: {total:.3f}s ({pct:.1f}%)")
        print(f"  Avg:   {avg*1000:.3f}ms")
        print(f"  Min:   {min_t*1000:.3f}ms")
        print(f"  Max:   {max_t*1000:.3f}ms")


def profile_single_straddle(asset, year, month, i, amt_path, chain_path, prices_path, prices_dict):
    """Profile a single straddle valuation with detailed timing."""

    timings = {}

    # 1. filter_tickers
    t0 = time.perf_counter()
    ticker_table = tickers_module.filter_tickers(asset, year, month, i, amt_path, chain_path)
    timings['filter_tickers'] = time.perf_counter() - t0

    if not ticker_table["rows"]:
        return timings

    asset_name = ticker_table["rows"][0][0]
    straddle = ticker_table["rows"][0][1]

    # 2. straddle_days
    t0 = time.perf_counter()
    dates = schedules.straddle_days(straddle)
    timings['straddle_days'] = time.perf_counter() - t0

    # 3. _build_ticker_map
    t0 = time.perf_counter()
    ticker_map, params_ordered = _build_ticker_map(ticker_table, chain_path)
    timings['build_ticker_map'] = time.perf_counter() - t0

    # 4. _lookup_straddle_prices
    t0 = time.perf_counter()
    prices = _lookup_straddle_prices(dates, ticker_map, prices_path)
    timings['lookup_straddle_prices'] = time.perf_counter() - t0

    # 5. _build_prices_table
    t0 = time.perf_counter()
    prices_table = _build_prices_table(asset_name, straddle, dates, params_ordered, ticker_map, prices)
    timings['build_prices_table'] = time.perf_counter() - t0

    # 6. actions() - which includes _add_action_column, _add_model_column, _add_strike_columns
    t0 = time.perf_counter()
    actions_table = actions(prices_table, amt_path, None)
    timings['actions'] = time.perf_counter() - t0

    # 7. valuation (from get_straddle_valuation after get_straddle_actions)
    # This is the MODEL computation part
    t0 = time.perf_counter()
    val_table = get_straddle_valuation(asset, year, month, i, amt_path, chain_path, prices_path)
    timings['get_straddle_valuation_full'] = time.perf_counter() - t0

    return timings


def profile_batch(pattern, start_year, end_year, amt_path, chain_path, prices_path, sample_size=100):
    """Profile a batch of straddles."""

    print(f"Loading prices...")
    start_date = f"{start_year - 1}-01-01"
    end_date = f"{end_year}-12-31"
    prices_dict = load_all_prices(prices_path, start_date, end_date)
    set_prices_dict(prices_dict)
    print(f"Loaded {len(prices_dict):,} prices")

    # Get all straddles
    print(f"\nFinding straddles for pattern '{pattern}'...")
    table = schedules.find_straddle_yrs(amt_path, start_year, end_year, pattern, True)
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

    # Sample tasks
    import random
    sample_tasks = random.sample(tasks, min(sample_size, len(tasks)))

    print(f"\nProfiling {len(sample_tasks)} straddles (sampled from {len(tasks)} total)...")

    all_timings = defaultdict(list)

    for j, (asset, year, month, i) in enumerate(sample_tasks):
        if j % 10 == 0:
            print(f"  [{j+1}/{len(sample_tasks)}] {asset} {year}-{month:02d}")

        try:
            timings = profile_single_straddle(asset, year, month, i, amt_path, chain_path, prices_path, prices_dict)
            for key, value in timings.items():
                all_timings[key].append(value)
        except Exception as e:
            print(f"  Error: {e}")

    # Print report
    print("\n" + "=" * 70)
    print("DETAILED TIMING REPORT (per straddle)")
    print("=" * 70)

    total_time = sum(sum(times) for times in all_timings.values())

    sorted_items = sorted(all_timings.items(), key=lambda x: sum(x[1]), reverse=True)

    for name, times in sorted_items:
        total = sum(times)
        count = len(times)
        avg = total / count if count > 0 else 0
        min_t = min(times) if times else 0
        max_t = max(times) if times else 0
        pct = (total / total_time * 100) if total_time > 0 else 0

        print(f"\n{name}:")
        print(f"  Calls: {count:,}")
        print(f"  Total: {total:.3f}s ({pct:.1f}%)")
        print(f"  Avg:   {avg*1000:.3f}ms")
        print(f"  Min:   {min_t*1000:.3f}ms")
        print(f"  Max:   {max_t*1000:.3f}ms")

    # Estimate total time for all straddles
    avg_per_straddle = total_time / len(sample_tasks) if sample_tasks else 0
    estimated_total = avg_per_straddle * len(tasks)

    print(f"\n" + "=" * 70)
    print(f"SUMMARY")
    print(f"=" * 70)
    print(f"Sampled straddles:  {len(sample_tasks)}")
    print(f"Total straddles:    {len(tasks)}")
    print(f"Avg per straddle:   {avg_per_straddle*1000:.2f}ms")
    print(f"Estimated total:    {estimated_total:.1f}s (single-threaded)")
    print(f"With 16 workers:    ~{estimated_total/16:.1f}s (ideal)")


def profile_internals(asset, year, month, i, amt_path, chain_path, prices_path):
    """Profile internal function calls in detail."""

    print(f"\nProfiling internals for {asset} {year}-{month:02d} straddle {i}")
    print("=" * 70)

    # Load prices first
    start_date = f"{year - 1}-01-01"
    end_date = f"{year}-12-31"
    prices_dict = load_all_prices(prices_path, start_date, end_date)
    set_prices_dict(prices_dict)

    # Profile filter_tickers internals
    print("\n1. filter_tickers() breakdown:")

    t0 = time.perf_counter()
    asset_data = loader.get_asset(amt_path, asset)
    print(f"   get_asset: {(time.perf_counter()-t0)*1000:.3f}ms")

    t0 = time.perf_counter()
    expand_table = schedules.get_expand_ym(amt_path, asset, year, month)
    print(f"   get_expand_ym: {(time.perf_counter()-t0)*1000:.3f}ms")

    straddles = loader.table_column(expand_table, "straddle")
    straddle = straddles[i % len(straddles)]
    xpry, xprm = schedules.xpry(straddle), schedules.xprm(straddle)

    t0 = time.perf_counter()
    ticker_table = tickers_module.get_tickers_ym(amt_path, asset, xpry, xprm, chain_path)
    print(f"   get_tickers_ym: {(time.perf_counter()-t0)*1000:.3f}ms")

    t0 = time.perf_counter()
    filtered_rows = tickers_module._filter_straddle_tickers(
        ticker_table["rows"], ticker_table["columns"], schedules.ntrc(straddle)
    )
    print(f"   _filter_straddle_tickers: {(time.perf_counter()-t0)*1000:.3f}ms")

    # Profile get_prices internals
    print("\n2. get_prices() breakdown:")

    t0 = time.perf_counter()
    full_ticker_table = tickers_module.filter_tickers(asset, year, month, i, amt_path, chain_path)
    print(f"   filter_tickers (full): {(time.perf_counter()-t0)*1000:.3f}ms")

    asset_name = full_ticker_table["rows"][0][0]
    straddle = full_ticker_table["rows"][0][1]

    t0 = time.perf_counter()
    dates = schedules.straddle_days(straddle)
    print(f"   straddle_days: {(time.perf_counter()-t0)*1000:.3f}ms ({len(dates)} days)")

    t0 = time.perf_counter()
    ticker_map, params_ordered = _build_ticker_map(full_ticker_table, chain_path)
    print(f"   _build_ticker_map: {(time.perf_counter()-t0)*1000:.3f}ms ({len(ticker_map)} params)")

    t0 = time.perf_counter()
    prices = _lookup_straddle_prices(dates, ticker_map, prices_path)
    print(f"   _lookup_straddle_prices: {(time.perf_counter()-t0)*1000:.3f}ms")

    # Count price lookups
    total_lookups = len(dates) * len(ticker_map)
    print(f"      (lookups: {total_lookups} = {len(dates)} days × {len(ticker_map)} params)")

    t0 = time.perf_counter()
    prices_table = _build_prices_table(asset_name, straddle, dates, params_ordered, ticker_map, prices)
    print(f"   _build_prices_table: {(time.perf_counter()-t0)*1000:.3f}ms ({len(prices_table['rows'])} rows)")

    # Profile actions internals
    print("\n3. actions() breakdown:")

    t0 = time.perf_counter()
    table_with_action = _add_action_column(prices_table, straddle, asset, None)
    print(f"   _add_action_column: {(time.perf_counter()-t0)*1000:.3f}ms")

    t0 = time.perf_counter()
    table_with_model = _add_model_column(table_with_action, asset, amt_path)
    print(f"   _add_model_column: {(time.perf_counter()-t0)*1000:.3f}ms")

    # Find ntry/xpry indices
    action_idx = table_with_model["columns"].index("action")
    ntry_idx = xpry_idx = None
    for idx, row in enumerate(table_with_model["rows"]):
        if row[action_idx] == "ntry":
            ntry_idx = idx
        elif row[action_idx] == "xpry":
            xpry_idx = idx

    t0 = time.perf_counter()
    table_with_strikes = _add_strike_columns(table_with_model, ntry_idx, xpry_idx)
    print(f"   _add_strike_columns: {(time.perf_counter()-t0)*1000:.3f}ms")

    # Profile valuation
    print("\n4. valuation model breakdown:")

    t0 = time.perf_counter()
    val_table = get_straddle_valuation(asset, year, month, i, amt_path, chain_path, prices_path)
    total_val = time.perf_counter() - t0
    print(f"   get_straddle_valuation (total): {total_val*1000:.3f}ms")

    # Count model calls
    if ntry_idx is not None and xpry_idx is not None:
        model_calls = xpry_idx - ntry_idx + 1
        print(f"      (model calls: {model_calls})")

    # Profile model_ES directly
    if val_table["rows"] and "strike" in val_table["columns"]:
        sample_row = dict(zip(val_table["columns"], val_table["rows"][ntry_idx if ntry_idx else 0]))

        # Time 1000 model calls
        t0 = time.perf_counter()
        for _ in range(1000):
            model_ES(sample_row)
        model_time = time.perf_counter() - t0
        print(f"   model_ES (1000 calls): {model_time*1000:.3f}ms ({model_time:.6f}s per call)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile backtest workflow")
    parser.add_argument("--pattern", default="^LA Comdty", help="Asset pattern")
    parser.add_argument("--start-year", type=int, default=2024, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument("--amt", default="data/amt.yml", help="AMT YAML file")
    parser.add_argument("--chain", default="data/futs.csv", help="Chain CSV file")
    parser.add_argument("--prices", default="data/prices.parquet", help="Prices parquet file")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of straddles to sample")
    parser.add_argument("--single", action="store_true", help="Profile single straddle in detail")

    args = parser.parse_args()

    if args.single:
        # Find first matching straddle
        prices_dict = load_all_prices(args.prices, f"{args.start_year-1}-01-01", f"{args.end_year}-12-31")
        set_prices_dict(prices_dict)

        table = schedules.find_straddle_yrs(args.amt, args.start_year, args.end_year, args.pattern, True)
        if table["rows"]:
            asset = table["rows"][0][table["columns"].index("asset")]
            straddle = table["rows"][0][table["columns"].index("straddle")]
            xpry = schedules.xpry(straddle)
            xprm = schedules.xprm(straddle)
            profile_internals(asset, xpry, xprm, 0, args.amt, args.chain, args.prices)
    else:
        profile_batch(args.pattern, args.start_year, args.end_year,
                     args.amt, args.chain, args.prices, args.sample_size)


if __name__ == "__main__":
    main()
