#!/usr/bin/env python
"""
Detailed performance profiling for backtest_new.py pipeline.

Breaks down the pipeline into phases and sub-phases to identify
optimization opportunities in the numba_sorted_kernel code path.

Usage:
    uv run python scripts/profile_backtest_new.py '.' 2001 2026
    uv run python scripts/profile_backtest_new.py '^LA Comdty' 2022 2024
"""
import argparse
import sys
import time
from datetime import date
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np


# -------------------------------------
# Timing utilities
# -------------------------------------

class DetailedTimer:
    """Hierarchical timer for nested phase/sub-phase tracking."""

    def __init__(self):
        self.start_time = time.perf_counter()
        self.phases = []  # [(name, start, end, sub_phases), ...]
        self._current_phase = None
        self._phase_start = None
        self._sub_phases = []

    def start_phase(self, name: str):
        """Start a new phase."""
        if self._current_phase is not None:
            self.end_phase()
        self._current_phase = name
        self._phase_start = time.perf_counter()
        self._sub_phases = []

    def sub_phase(self, name: str):
        """Record a sub-phase checkpoint within current phase."""
        now = time.perf_counter()
        self._sub_phases.append((name, now))

    def end_phase(self):
        """End the current phase."""
        if self._current_phase is None:
            return
        end_time = time.perf_counter()
        self.phases.append((self._current_phase, self._phase_start, end_time, self._sub_phases.copy()))
        self._current_phase = None
        self._phase_start = None
        self._sub_phases = []

    def report(self, file=sys.stderr):
        """Print detailed timing report."""
        if self._current_phase is not None:
            self.end_phase()

        total_time = self.phases[-1][2] - self.start_time if self.phases else 0

        print("\n" + "=" * 80, file=file)
        print("DETAILED PERFORMANCE BREAKDOWN", file=file)
        print("=" * 80, file=file)

        for phase_name, phase_start, phase_end, sub_phases in self.phases:
            phase_time = phase_end - phase_start
            pct = (phase_time / total_time * 100) if total_time > 0 else 0
            print(f"\n{phase_name}: {phase_time:.3f}s ({pct:.1f}%)", file=file)

            if sub_phases:
                prev_time = phase_start
                for i, (sub_name, sub_time) in enumerate(sub_phases):
                    delta = sub_time - prev_time
                    sub_pct = (delta / phase_time * 100) if phase_time > 0 else 0
                    print(f"  ├─ {sub_name:50s} {delta:8.4f}s  ({sub_pct:5.1f}%)", file=file)
                    prev_time = sub_time
                # Final segment to phase end
                final_delta = phase_end - prev_time
                if final_delta > 0.0001:  # Only show if significant
                    final_pct = (final_delta / phase_time * 100) if phase_time > 0 else 0
                    print(f"  └─ {'(remaining)':50s} {final_delta:8.4f}s  ({final_pct:5.1f}%)", file=file)

        print("\n" + "-" * 80, file=file)
        print(f"TOTAL: {total_time:.3f}s", file=file)
        print("=" * 80 + "\n", file=file)


def main():
    parser = argparse.ArgumentParser(description="Profile backtest_new.py pipeline")
    parser.add_argument("pattern", help="Regex pattern to match assets")
    parser.add_argument("start_year", type=int, help="Start year")
    parser.add_argument("end_year", type=int, help="End year")
    parser.add_argument("--amt", default="data/amt.yml")
    parser.add_argument("--prices", default="data/prices.parquet")
    parser.add_argument("--overrides", default="data/overrides.csv")
    args = parser.parse_args()

    timer = DetailedTimer()

    # =========================================================================
    # PHASE 1: Load prices (PyArrow)
    # =========================================================================
    timer.start_phase("Phase 1: Load prices (PyArrow)")

    from specparser.amt import prices as prices_module

    start_date = f"{args.start_year - 1}-01-01"
    end_date = f"{args.end_year + 1}-12-31"

    timer.sub_phase("Import PyArrow modules")

    prices_numba = prices_module.load_prices_numba(args.prices, start_date, end_date)

    timer.sub_phase("load_prices_numba() complete")

    print(f"Prices loaded: {len(prices_numba.sorted_keys):,} rows", file=sys.stderr)
    print(f"  Tickers: {len(prices_numba.ticker_to_idx)}", file=sys.stderr)
    print(f"  Fields: {len(prices_numba.field_to_idx)}", file=sys.stderr)

    timer.end_phase()

    # =========================================================================
    # PHASE 2: JIT Warmup
    # =========================================================================
    timer.start_phase("Phase 2: JIT Warmup")

    from specparser.amt.valuation_numba import get_straddle_backtests_numba

    # Small warmup run
    _ = get_straddle_backtests_numba(
        args.pattern, args.start_year, args.start_year,
        args.amt, args.prices,
        valid_only=True, overrides_path=args.overrides,
    )

    timer.end_phase()

    # =========================================================================
    # PHASE 3: Find straddles + expand to days (u8m)
    # =========================================================================
    timer.start_phase("Phase 3: Find straddles + expand to days")

    from specparser.amt import schedules

    timer.sub_phase("Import schedules")

    straddle_days_table = schedules.find_straddle_days_u8m(
        args.amt, args.start_year, args.end_year, args.pattern, live_only=True
    )

    timer.sub_phase("find_straddle_days_u8m() complete")

    asset_u8m = straddle_days_table["rows"][0]
    straddle_u8m = straddle_days_table["rows"][1]
    dates = straddle_days_table["rows"][2]
    straddle_id = straddle_days_table["rows"][3]  # maps each day to source straddle

    n_days = len(dates)
    print(f"Expanded: {n_days:,} days", file=sys.stderr)

    timer.sub_phase("Extract arrays")

    # Compute straddle_starts and straddle_lengths
    from specparser.amt.valuation_numba import _compute_starts_lengths_from_parent_idx
    straddle_starts, straddle_lengths = _compute_starts_lengths_from_parent_idx(straddle_id)
    n_straddles = len(straddle_starts)

    timer.sub_phase("Compute starts/lengths")

    # Convert u8m to strings
    from specparser.amt import strings as strings_module

    unique_indices = straddle_starts
    unique_asset_u8m = asset_u8m[unique_indices]
    unique_straddle_u8m = straddle_u8m[unique_indices]

    timer.sub_phase("Extract unique u8m rows")

    unique_assets = strings_module.u8m2s(unique_asset_u8m)
    unique_straddles = strings_module.u8m2s(unique_straddle_u8m)

    timer.sub_phase("u8m2s conversion")

    straddle_list = [
        (asset.strip(), straddle)
        for asset, straddle in zip(unique_assets.tolist(), unique_straddles.tolist())
    ]

    timer.sub_phase("Build straddle_list")

    print(f"Straddles: {n_straddles:,}", file=sys.stderr)

    timer.end_phase()

    # =========================================================================
    # PHASE 4: Parse straddle strings (xpry, xprm, ntrc, etc.)
    # =========================================================================
    timer.start_phase("Phase 4: Parse straddle strings")

    stryms = []
    ntrcs = []
    assets_for_tickers = []

    for asset, straddle in straddle_list:
        xpry = schedules.xpry(straddle)
        xprm = schedules.xprm(straddle)
        ntrc = schedules.ntrc(straddle)
        stryms.append(f"{xpry}-{xprm:02d}")
        ntrcs.append(ntrc)
        assets_for_tickers.append(asset)

    timer.sub_phase("Parse xpry/xprm/ntrc")

    timer.end_phase()

    # =========================================================================
    # PHASE 5: Resolve tickers (batch) - DETAILED BREAKDOWN
    # =========================================================================
    timer.start_phase("Phase 5: Resolve tickers (detailed)")

    from specparser.amt import loader
    from specparser.amt import asset_straddle_tickers

    # Sub-phase 5a: Get unique (asset, strym, ntrc) combinations
    unique_combos = set(zip(assets_for_tickers, stryms, ntrcs))
    timer.sub_phase(f"5a: Find unique combos ({len(unique_combos):,})")

    # Sub-phase 5b: Pre-load asset data for unique assets
    unique_asset_names = set(assets_for_tickers)
    asset_to_data: dict[str, dict | None] = {}
    for asset_name in unique_asset_names:
        asset_to_data[asset_name] = loader.get_asset(args.amt, asset_name)
    timer.sub_phase(f"5b: Pre-load asset data ({len(unique_asset_names)} assets)")

    # Sub-phase 5c: Resolve tickers using pre-loaded data
    ticker_map: dict[str, dict[str, tuple[str, str]]] = {}
    seen_keys: set[str] = set()
    n_cache_hits = 0
    n_get_tickers_calls = 0

    for asset, strym, ntrc in zip(assets_for_tickers, stryms, ntrcs):
        asset_data = asset_to_data[asset]
        if asset_data is None:
            continue

        vol = asset_data.get("Vol")
        hedge = asset_data.get("Hedge")
        if vol is None or hedge is None:
            continue

        cache_key = asset_straddle_tickers.asset_straddle_ticker_key(
            asset, strym, ntrc, vol, hedge
        )

        if cache_key in seen_keys:
            n_cache_hits += 1
            continue
        seen_keys.add(cache_key)

        n_get_tickers_calls += 1
        ticker_table = asset_straddle_tickers.get_asset_straddle_tickers(
            asset, strym, ntrc, args.amt
        )

        param_map: dict[str, tuple[str, str]] = {}
        for row in ticker_table["rows"]:
            name, ticker, field = row
            param_map[name] = (ticker, field)

        ticker_map[cache_key] = param_map

    timer.sub_phase(f"5c: Build ticker_map ({n_get_tickers_calls} calls, {n_cache_hits:,} cache hits)")

    print(f"Ticker map entries: {len(ticker_map):,}", file=sys.stderr)
    print(f"  get_asset_straddle_tickers calls: {n_get_tickers_calls}", file=sys.stderr)
    print(f"  Cache hits (skipped): {n_cache_hits:,}", file=sys.stderr)

    timer.end_phase()

    # =========================================================================
    # PHASE 6: Prepare backtest arrays - DETAILED BREAKDOWN
    # =========================================================================
    timer.start_phase("Phase 6: Prepare backtest arrays (detailed)")

    from specparser.amt.valuation import _anchor_day
    from specparser.amt import valuation_numba
    import calendar

    # Initialize arrays
    vol_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    vol_field_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge_field_idx = np.full(n_straddles, -1, dtype=np.int32)
    ntry_anchor_date32 = np.zeros(n_straddles, dtype=np.int32)
    xpry_anchor_date32 = np.zeros(n_straddles, dtype=np.int32)
    ntrv_offsets = np.zeros(n_straddles, dtype=np.int32)
    ntry_month_end = np.zeros(n_straddles, dtype=np.int32)
    xpry_month_end = np.zeros(n_straddles, dtype=np.int32)

    timer.sub_phase("6a: Initialize arrays")

    # Track sub-phase timings
    t_parse = 0.0
    t_ticker_lookup = 0.0
    t_anchor = 0.0

    for s, (asset, straddle) in enumerate(straddle_list):
        strym = stryms[s]
        ntrc = ntrcs[s]

        # Parse straddle string
        t0 = time.perf_counter()
        ntry_y = schedules.ntry(straddle)
        ntry_m = schedules.ntrm(straddle)
        xpry_y = schedules.xpry(straddle)
        xpry_m = schedules.xprm(straddle)
        xprc = schedules.xprc(straddle).strip()
        xprv = schedules.xprv(straddle).strip()
        ntrv_str = schedules.ntrv(straddle).strip()
        t_parse += time.perf_counter() - t0

        # Ticker lookup
        t0 = time.perf_counter()
        asset_data = asset_to_data[asset]
        if asset_data is not None:
            vol = asset_data.get("Vol")
            hedge = asset_data.get("Hedge")
            if vol is not None and hedge is not None:
                cache_key = asset_straddle_tickers.asset_straddle_ticker_key(
                    asset, strym, ntrc, vol, hedge
                )
                if cache_key in ticker_map:
                    param_map = ticker_map[cache_key]
                    if "vol" in param_map:
                        vol_ticker, vol_field = param_map["vol"]
                        if vol_ticker in prices_numba.ticker_to_idx:
                            vol_ticker_idx[s] = prices_numba.ticker_to_idx[vol_ticker]
                        if vol_field in prices_numba.field_to_idx:
                            vol_field_idx[s] = prices_numba.field_to_idx[vol_field]
                    if "hedge" in param_map:
                        hedge_ticker, hedge_field = param_map["hedge"]
                        if hedge_ticker in prices_numba.ticker_to_idx:
                            hedge_ticker_idx[s] = prices_numba.ticker_to_idx[hedge_ticker]
                        if hedge_field in prices_numba.field_to_idx:
                            hedge_field_idx[s] = prices_numba.field_to_idx[hedge_field]
        t_ticker_lookup += time.perf_counter() - t0

        # Anchor date computation
        t0 = time.perf_counter()
        _, ntry_num_days = calendar.monthrange(ntry_y, ntry_m)
        ntry_month_end[s] = valuation_numba.ymd_to_date32(ntry_y, ntry_m, ntry_num_days)

        _, xpry_num_days = calendar.monthrange(xpry_y, xpry_m)
        xpry_month_end[s] = valuation_numba.ymd_to_date32(xpry_y, xpry_m, xpry_num_days)

        try:
            ntrv_offsets[s] = int(ntrv_str) if ntrv_str else 0
        except (ValueError, TypeError):
            ntrv_offsets[s] = 0

        entry_anchor = _anchor_day(xprc, xprv, ntry_y, ntry_m, asset, args.overrides)
        if entry_anchor is not None:
            y, m, d = map(int, entry_anchor.split("-"))
            ntry_anchor_date32[s] = valuation_numba.ymd_to_date32(y, m, d)
        else:
            ntry_anchor_date32[s] = np.iinfo(np.int32).max

        expiry_anchor = _anchor_day(xprc, xprv, xpry_y, xpry_m, asset, args.overrides)
        if expiry_anchor is not None:
            y, m, d = map(int, expiry_anchor.split("-"))
            xpry_anchor_date32[s] = valuation_numba.ymd_to_date32(y, m, d)
        else:
            xpry_anchor_date32[s] = np.iinfo(np.int32).max
        t_anchor += time.perf_counter() - t0

    timer.sub_phase(f"6b: Loop over straddles (parse={t_parse:.2f}s, ticker={t_ticker_lookup:.2f}s, anchor={t_anchor:.2f}s)")

    # Also run the actual function for comparison
    from specparser.amt.valuation_numba import _prepare_backtest_arrays_sorted as actual_prepare

    t0 = time.perf_counter()
    backtest_arrays = actual_prepare(
        straddle_list,
        ticker_map,
        prices_numba,
        stryms,
        ntrcs,
        args.amt,
        args.overrides,
    )
    t_actual = time.perf_counter() - t0

    timer.sub_phase(f"6c: Actual _prepare_backtest_arrays_sorted() = {t_actual:.2f}s")

    timer.end_phase()

    # =========================================================================
    # PHASE 7: Run Numba kernel
    # =========================================================================
    timer.start_phase("Phase 7: Run Numba kernel")

    from specparser.amt import valuation_numba

    timer.sub_phase("Import valuation_numba")

    (vol_array, hedge_array, hedge1_array, hedge2_array, hedge3_array,
     ntry_offsets, xpry_offsets,
     strike_array, strike1_array, strike2_array, strike3_array,
     days_to_expiry, mv, delta, opnl, hpnl, pnl, action) = \
        valuation_numba.full_backtest_kernel_sorted(
            prices_numba.sorted_keys,
            prices_numba.sorted_values,
            prices_numba.n_fields,
            prices_numba.n_dates,
            prices_numba.min_date32,
            backtest_arrays.vol_ticker_idx,
            backtest_arrays.vol_field_idx,
            backtest_arrays.hedge_ticker_idx,
            backtest_arrays.hedge_field_idx,
            backtest_arrays.hedge1_ticker_idx,
            backtest_arrays.hedge1_field_idx,
            backtest_arrays.hedge2_ticker_idx,
            backtest_arrays.hedge2_field_idx,
            backtest_arrays.hedge3_ticker_idx,
            backtest_arrays.hedge3_field_idx,
            backtest_arrays.n_hedges,
            straddle_starts,
            straddle_lengths,
            backtest_arrays.ntry_anchor_date32,
            backtest_arrays.xpry_anchor_date32,
            backtest_arrays.ntrv_offsets,
            backtest_arrays.ntry_month_end,
            backtest_arrays.xpry_month_end,
            dates,
        )

    timer.sub_phase("full_backtest_kernel_sorted()")

    # Count valid rows
    valid_count = np.sum(~np.isnan(mv))
    print(f"Valid mv rows: {valid_count:,} / {n_days:,}", file=sys.stderr)

    timer.end_phase()

    # =========================================================================
    # PHASE 8: Build Arrow output
    # =========================================================================
    timer.start_phase("Phase 8: Build Arrow output")

    from specparser.amt.valuation_numba import _build_arrow_output_sorted

    # Build parent_idx for output
    out_parent_idx = np.zeros(n_days, dtype=np.int32)
    for s in range(n_straddles):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        out_parent_idx[start:start+length] = s

    timer.sub_phase("Build parent_idx")

    result = _build_arrow_output_sorted(
        dates=dates,
        vol=vol_array,
        hedge=hedge_array,
        hedge1=hedge1_array,
        hedge2=hedge2_array,
        hedge3=hedge3_array,
        strike=strike_array,
        strike1=strike1_array,
        strike2=strike2_array,
        strike3=strike3_array,
        mv=mv,
        delta=delta,
        opnl=opnl,
        hpnl=hpnl,
        pnl=pnl,
        action=action,
        parent_idx=out_parent_idx,
        backtest_arrays=backtest_arrays,
        straddle_starts=straddle_starts,
        straddle_lengths=straddle_lengths,
        ntry_offsets=ntry_offsets,
        xpry_offsets=xpry_offsets,
        valid_only=True,
    )

    timer.sub_phase("_build_arrow_output_sorted()")

    print(f"Output rows: {result.num_rows:,}", file=sys.stderr)

    timer.end_phase()

    # =========================================================================
    # Report
    # =========================================================================
    timer.report()

    # Summary
    print("\n" + "=" * 80, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"  Pattern:     {args.pattern}", file=sys.stderr)
    print(f"  Year range:  {args.start_year}-{args.end_year}", file=sys.stderr)
    print(f"  Straddles:   {n_straddles:,}", file=sys.stderr)
    print(f"  Days:        {n_days:,}", file=sys.stderr)
    print(f"  Output rows: {result.num_rows:,}", file=sys.stderr)

    total_time = timer.phases[-1][2] - timer.start_time if timer.phases else 0
    rate = n_straddles / total_time if total_time > 0 else 0
    print(f"  Rate:        {rate:,.0f} straddles/sec", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # Clean up
    prices_module.clear_prices_numba()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
