#!/usr/bin/env python3
"""
Benchmark fast string processing for straddle expansion.

Compares:
1. Standard Python approach (schedules.find_straddle_yrs)
2. Fast uint8 matrix approach (strings.py operations)

The fast approach uses Numba-accelerated byte matrix operations to avoid
Python string object creation in inner loops.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path so we can import specparser
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from specparser.amt import loader, schedules
from specparser.amt.strings import (
    strs2u8mat,
    make_ym_matrix,
    cartesian_product,
    cartesian_product_par,
    add_months2specs_inplace,
    u8m2s,
    sep,
    ASCII_PIPE,
    ASCII_SPACE,
    make_u8mat,
)
from numba import njit


# -------------------------------------
# Standard Python approach
# -------------------------------------

def expand_standard(amt_path, pattern, start_year, end_year):
    """Standard Python string approach using schedules.find_straddle_yrs."""
    return schedules.find_straddle_yrs(amt_path, start_year, end_year, pattern, True)


# -------------------------------------
# Fast uint8 matrix approach
# -------------------------------------

def precompute_templates(schedules_table):
    """
    Convert schedules table to uint8 templates.

    Each row in schedules_table has: [schcnt, schid, asset, ntrc, ntrv, xprc, xprv, wgt]

    We need to create a template for the fixed parts of the straddle string:
    |NTRY-MM|XPRY-MM|NTRC|NTRV|XPRC|XPRV|WGT|
                    ^--- fixed parts ---^

    Returns:
        asset_mat: uint8 matrix of asset names, shape (N, asset_width)
        ntrc_vec: uint8 vector of entry codes (N or F), shape (N,)
        fixed_mat: uint8 matrix of |NTRC|NTRV|XPRC|XPRV|WGT|, shape (N, fixed_width)
    """
    rows = schedules_table["rows"]
    cols = schedules_table["columns"]

    if not rows:
        return None, None, None

    asset_idx = cols.index("asset")
    ntrc_idx = cols.index("ntrc")
    ntrv_idx = cols.index("ntrv")
    xprc_idx = cols.index("xprc")
    xprv_idx = cols.index("xprv")
    wgt_idx = cols.index("wgt")

    # Extract columns as lists
    assets = [row[asset_idx] for row in rows]
    ntrcs = [row[ntrc_idx] for row in rows]
    ntrvs = [row[ntrv_idx] for row in rows]
    xprcs = [row[xprc_idx] for row in rows]
    xprvs = [row[xprv_idx] for row in rows]
    wgts = [row[wgt_idx] for row in rows]

    # Build fixed part strings: |NTRC|NTRV|XPRC|XPRV|WGT|
    fixed_parts = [
        f"|{ntrc}|{ntrv}|{xprc}|{xprv}|{wgt}|"
        for ntrc, ntrv, xprc, xprv, wgt in zip(ntrcs, ntrvs, xprcs, xprvs, wgts)
    ]

    # Convert to uint8 matrices
    asset_mat = strs2u8mat(assets)
    fixed_mat = strs2u8mat(fixed_parts)

    # Extract ntrc as uint8 vector (just first char: N or F)
    ntrc_vec = np.array([ord(c[0]) if c else ord(' ') for c in ntrcs], dtype=np.uint8)

    return asset_mat, ntrc_vec, fixed_mat


def expand_fast(amt_path, pattern, start_year, end_year, use_parallel=False, return_timing=False, skip_string_convert=False):
    """
    Fast uint8 matrix approach for straddle expansion.

    Steps:
    1. Get schedules and precompute templates
    2. Generate expiry year-months as uint8 matrix
    3. Cartesian product: templates × year-months
    4. Compute entry dates from expiry dates
    5. Assemble final straddle strings

    Args:
        skip_string_convert: If True, skip converting to Python strings (for benchmarking)
    """
    timings = {}

    # 1. Get schedules
    t0 = time.perf_counter()
    found = schedules.find_schedules(amt_path, pattern=pattern, live_only=True)
    timings["find_schedules"] = time.perf_counter() - t0

    if not found["rows"]:
        result = {"orientation": "row", "columns": ["asset", "straddle"], "rows": []}
        return (result, timings) if return_timing else result

    # 2. Precompute templates
    t0 = time.perf_counter()
    asset_mat, ntrc_vec, fixed_mat = precompute_templates(found)
    timings["precompute_templates"] = time.perf_counter() - t0

    if asset_mat is None:
        result = {"orientation": "row", "columns": ["asset", "straddle"], "rows": []}
        return (result, timings) if return_timing else result

    n_templates = asset_mat.shape[0]

    # 3. Generate expiry year-months
    # make_ym_matrix returns (N, 7) matrix with "YYYY-MM" format
    t0 = time.perf_counter()
    n_months = (end_year - start_year + 1) * 12
    xpr_ym = make_ym_matrix((start_year, 1, end_year, 12))  # shape (n_months, 7)
    timings["make_ym_matrix"] = time.perf_counter() - t0

    # 4. Cartesian product: templates × year-months
    # We want: asset | |ntr_ym| |xpr_ym| fixed_part
    # But we need to compute ntr_ym from xpr_ym + ntrc offset

    t0 = time.perf_counter()

    # First, expand templates by year-months
    # Result shape: (n_templates * n_months, asset_width + 7)
    cart_fn = cartesian_product_par if use_parallel else cartesian_product

    # Create template indices to track which template each row came from
    template_indices = np.repeat(np.arange(n_templates, dtype=np.int64), n_months)

    # Expand asset matrix
    asset_expanded = asset_mat[template_indices]  # (N_total, asset_width)

    # Expand fixed matrix
    fixed_expanded = fixed_mat[template_indices]  # (N_total, fixed_width)

    # Expand ntrc vector
    ntrc_expanded = ntrc_vec[template_indices]  # (N_total,)

    # Tile xpr_ym for each template
    xpr_expanded = np.tile(xpr_ym, (n_templates, 1))  # (N_total, 7)

    timings["expand_arrays"] = time.perf_counter() - t0

    # 5. Compute entry dates from expiry dates using N/F offset
    # N = 1 month before (offset -1), F = 2 months before (offset -2)
    t0 = time.perf_counter()

    # Convert ntrc codes to month offsets
    month_offsets = np.where(
        ntrc_expanded == ord('N'), np.int64(-1),
        np.where(ntrc_expanded == ord('F'), np.int64(-2), np.int64(0))
    )

    ntr_expanded = np.empty_like(xpr_expanded)
    add_months2specs_inplace(ntr_expanded, xpr_expanded, month_offsets)

    timings["compute_entry_dates"] = time.perf_counter() - t0

    # 6. Assemble final straddle strings
    # Format: |NTR_YM|XPR_YM|NTRC|NTRV|XPRC|XPRV|WGT|
    t0 = time.perf_counter()

    n_total = asset_expanded.shape[0]
    pipe = np.full((n_total, 1), ASCII_PIPE, dtype=np.uint8)

    # Concatenate: | + ntr_ym + | + xpr_ym + fixed_part
    straddle_mat = np.hstack([
        pipe,
        ntr_expanded,
        pipe,
        xpr_expanded,
        fixed_expanded,
    ])

    timings["assemble_straddles"] = time.perf_counter() - t0

    # 7. Convert back to Python strings for output
    t0 = time.perf_counter()

    if skip_string_convert:
        # Return uint8 matrices directly (for benchmarking matrix ops only)
        rows = []  # Empty - we're just measuring matrix operations
        timings["convert_to_strings"] = 0.0
    else:
        assets_out = u8m2s(asset_expanded).tolist()
        straddles_out = u8m2s(straddle_mat).tolist()

        # Build output table
        rows = [[a.strip(), s.strip()] for a, s in zip(assets_out, straddles_out)]
        timings["convert_to_strings"] = time.perf_counter() - t0

    result = {"orientation": "row", "columns": ["asset", "straddle"], "rows": rows, "_n_rows": n_total}
    return (result, timings) if return_timing else result


def expand_fast_parsed(amt_path, pattern, start_year, end_year, use_parallel=False, return_timing=False):
    """
    Fast expansion returning pre-parsed columns instead of straddle strings.

    Returns table with columns:
        ["asset", "ntry", "ntrm", "xpry", "xprm", "ntrc", "ntrv", "xprc", "xprv", "wgt"]

    This skips both:
    1. Straddle string assembly (no need to build "|YYYY-MM|YYYY-MM|..." format)
    2. String conversion overhead (numeric fields stay as integers)

    Benefits:
    - Downstream code can use pre-parsed values directly
    - No need to call schedules.xpry(), xprm(), ntrc(), etc.
    """
    timings = {}

    # 1. Get schedules
    t0 = time.perf_counter()
    found = schedules.find_schedules(amt_path, pattern=pattern, live_only=True)
    timings["find_schedules"] = time.perf_counter() - t0

    if not found["rows"]:
        result = {
            "orientation": "row",
            "columns": ["asset", "ntry", "ntrm", "xpry", "xprm", "ntrc", "ntrv", "xprc", "xprv", "wgt"],
            "rows": []
        }
        return (result, timings) if return_timing else result

    # 2. Extract schedule data
    t0 = time.perf_counter()
    cols = found["columns"]
    asset_idx = cols.index("asset")
    ntrc_idx = cols.index("ntrc")
    ntrv_idx = cols.index("ntrv")
    xprc_idx = cols.index("xprc")
    xprv_idx = cols.index("xprv")
    wgt_idx = cols.index("wgt")

    # Extract columns as numpy arrays for vectorized operations
    n_templates = len(found["rows"])
    assets = np.array([row[asset_idx] for row in found["rows"]], dtype=object)
    ntrcs = np.array([row[ntrc_idx] for row in found["rows"]], dtype=object)
    ntrvs = np.array([row[ntrv_idx] for row in found["rows"]], dtype=object)
    xprcs = np.array([row[xprc_idx] for row in found["rows"]], dtype=object)
    xprvs = np.array([row[xprv_idx] for row in found["rows"]], dtype=object)
    wgts = np.array([row[wgt_idx] for row in found["rows"]], dtype=object)

    timings["extract_schedules"] = time.perf_counter() - t0

    # 3. Generate expiry year-months as numpy arrays
    t0 = time.perf_counter()
    n_months = (end_year - start_year + 1) * 12

    # Generate all (year, month) pairs using numpy
    years_range = np.arange(start_year, end_year + 1)
    months_range = np.arange(1, 13)
    xpr_years = np.repeat(years_range, 12)
    xpr_months = np.tile(months_range, end_year - start_year + 1)

    timings["generate_ym"] = time.perf_counter() - t0

    # 4. Expand arrays (cartesian product of templates × year-months)
    t0 = time.perf_counter()

    # Create template indices to track which template each row came from
    template_indices = np.repeat(np.arange(n_templates, dtype=np.int64), n_months)

    # Expand schedule data using numpy indexing
    assets_expanded = assets[template_indices]
    ntrcs_expanded = ntrcs[template_indices]
    ntrvs_expanded = ntrvs[template_indices]
    xprcs_expanded = xprcs[template_indices]
    xprvs_expanded = xprvs[template_indices]
    wgts_expanded = wgts[template_indices]

    # Tile year-months for each template
    xpry_expanded = np.tile(xpr_years, n_templates)
    xprm_expanded = np.tile(xpr_months, n_templates)

    timings["expand_arrays"] = time.perf_counter() - t0

    # 5. Compute entry dates from expiry dates using N/F offset (vectorized)
    t0 = time.perf_counter()

    # Convert ntrc codes to month offsets: N=1, F=2, else=0
    offsets = np.where(ntrcs_expanded == "N", 1,
                       np.where(ntrcs_expanded == "F", 2, 0))

    # Compute entry year-month using vectorized math
    total_months = xpry_expanded * 12 + xprm_expanded - 1 - offsets
    ntry_expanded = total_months // 12
    ntrm_expanded = (total_months % 12) + 1

    timings["compute_entry_dates"] = time.perf_counter() - t0

    # 6. Build output table
    t0 = time.perf_counter()

    n_total = len(assets_expanded)

    # Use zip and list comprehension for faster row building
    rows = [
        [asset, ntry, ntrm, xpry, xprm, ntrc, ntrv, xprc, xprv, wgt]
        for asset, ntry, ntrm, xpry, xprm, ntrc, ntrv, xprc, xprv, wgt
        in zip(
            assets_expanded.tolist(),
            ntry_expanded.tolist(),
            ntrm_expanded.tolist(),
            xpry_expanded.tolist(),
            xprm_expanded.tolist(),
            ntrcs_expanded.tolist(),
            ntrvs_expanded.tolist(),
            xprcs_expanded.tolist(),
            xprvs_expanded.tolist(),
            wgts_expanded.tolist(),
        )
    ]

    timings["build_rows"] = time.perf_counter() - t0

    result = {
        "orientation": "row",
        "columns": ["asset", "ntry", "ntrm", "xpry", "xprm", "ntrc", "ntrv", "xprc", "xprv", "wgt"],
        "rows": rows,
        "_n_rows": n_total
    }
    return (result, timings) if return_timing else result


# -------------------------------------
# Benchmark utilities
# -------------------------------------

def verify_outputs(result_std, result_fast):
    """Verify that both approaches produce the same output."""
    rows_std = sorted([tuple(r) for r in result_std["rows"]])
    rows_fast = sorted([tuple(r) for r in result_fast["rows"]])

    if len(rows_std) != len(rows_fast):
        print(f"MISMATCH: Standard has {len(rows_std)} rows, Fast has {len(rows_fast)} rows")
        return False

    mismatches = 0
    for i, (std, fast) in enumerate(zip(rows_std, rows_fast)):
        if std != fast:
            mismatches += 1
            if mismatches <= 5:
                print(f"MISMATCH at row {i}:")
                print(f"  Standard: {std}")
                print(f"  Fast:     {fast}")

    if mismatches > 0:
        print(f"Total mismatches: {mismatches}")
        return False

    print(f"VERIFIED: {len(rows_std)} rows match")
    return True


def benchmark(amt_path, pattern, start_year, end_year, iterations=5, use_parallel=False, show_breakdown=False, matrix_only=False, include_parsed=False):
    """Run approaches and compare. Returns (result_std, result_fast, min_fast_time)."""

    print(f"Benchmarking: pattern='{pattern}', years={start_year}-{end_year}", file=sys.stderr)
    print(f"Iterations: {iterations}, Parallel: {use_parallel}, Matrix-only: {matrix_only}", file=sys.stderr)
    print(file=sys.stderr)

    # Warmup (important for Numba JIT compilation)
    print("Warming up...", file=sys.stderr)
    _ = expand_standard(amt_path, pattern, start_year, end_year)
    _ = expand_fast(amt_path, pattern, start_year, end_year, use_parallel, skip_string_convert=matrix_only)
    if include_parsed:
        _ = expand_fast_parsed(amt_path, pattern, start_year, end_year, use_parallel)
    print(file=sys.stderr)

    # Standard approach
    print("Running standard approach...", file=sys.stderr)
    times_std = []
    result_std = None
    for i in range(iterations):
        t0 = time.perf_counter()
        result_std = expand_standard(amt_path, pattern, start_year, end_year)
        elapsed = time.perf_counter() - t0
        times_std.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.2f}ms", file=sys.stderr)

    # Fast approach (straddle strings)
    print("\nRunning fast approach (strings)...", file=sys.stderr)
    times_fast = []
    result_fast = None
    all_timings_fast = []
    for i in range(iterations):
        t0 = time.perf_counter()
        result_fast, timings = expand_fast(amt_path, pattern, start_year, end_year, use_parallel, return_timing=True, skip_string_convert=matrix_only)
        elapsed = time.perf_counter() - t0
        times_fast.append(elapsed)
        all_timings_fast.append(timings)
        print(f"  Run {i+1}: {elapsed*1000:.2f}ms", file=sys.stderr)

    # Fast parsed approach (pre-parsed columns)
    times_parsed = []
    result_parsed = None
    all_timings_parsed = []
    if include_parsed:
        print("\nRunning fast approach (parsed)...", file=sys.stderr)
        for i in range(iterations):
            t0 = time.perf_counter()
            result_parsed, timings = expand_fast_parsed(amt_path, pattern, start_year, end_year, use_parallel, return_timing=True)
            elapsed = time.perf_counter() - t0
            times_parsed.append(elapsed)
            all_timings_parsed.append(timings)
            print(f"  Run {i+1}: {elapsed*1000:.2f}ms", file=sys.stderr)

    # Report
    min_std = min(times_std)
    min_fast = min(times_fast)
    min_parsed = min(times_parsed) if times_parsed else 0
    speedup_fast = min_std / min_fast if min_fast > 0 else 0
    speedup_parsed = min_std / min_parsed if min_parsed > 0 else 0

    # Standardized benchmark output
    n_rows = result_fast.get("_n_rows", len(result_fast["rows"])) if matrix_only else len(result_std["rows"])
    rate = n_rows / min_fast if min_fast > 0 else 0
    rate_parsed = n_rows / min_parsed if min_parsed > 0 else 0

    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("BENCHMARK RESULTS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Straddles:       {n_rows:,}", file=sys.stderr)
    print(f"  Total time:      {min_fast*1000:.2f}ms", file=sys.stderr)
    print(f"  Rate:            {rate:,.1f} straddles/sec", file=sys.stderr)
    print(f"  Output rows:     {n_rows:,}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Show timing breakdown if requested
    if show_breakdown:
        print(file=sys.stderr)
        print("COMPARISON:", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        print(f"  Standard (min):      {min_std*1000:8.2f}ms  (baseline)", file=sys.stderr)
        print(f"  Fast strings (min):  {min_fast*1000:8.2f}ms  ({speedup_fast:.1f}x faster)", file=sys.stderr)
        if include_parsed:
            print(f"  Fast parsed (min):   {min_parsed*1000:8.2f}ms  ({speedup_parsed:.1f}x faster)", file=sys.stderr)
        print("-" * 60, file=sys.stderr)

        if all_timings_fast:
            print(file=sys.stderr)
            print("FAST STRINGS BREAKDOWN (avg):", file=sys.stderr)
            print("-" * 60, file=sys.stderr)

            avg_timings = {}
            for key in all_timings_fast[0].keys():
                avg_timings[key] = sum(t[key] for t in all_timings_fast) / len(all_timings_fast)

            total = sum(avg_timings.values())
            for key, val in sorted(avg_timings.items(), key=lambda x: -x[1]):
                pct = (val / total * 100) if total > 0 else 0
                print(f"  {key:25s} {val*1000:8.2f}ms  ({pct:5.1f}%)", file=sys.stderr)
            print("-" * 60, file=sys.stderr)
            print(f"  {'TOTAL':25s} {total*1000:8.2f}ms", file=sys.stderr)

        if include_parsed and all_timings_parsed:
            print(file=sys.stderr)
            print("FAST PARSED BREAKDOWN (avg):", file=sys.stderr)
            print("-" * 60, file=sys.stderr)

            avg_timings = {}
            for key in all_timings_parsed[0].keys():
                avg_timings[key] = sum(t[key] for t in all_timings_parsed) / len(all_timings_parsed)

            total = sum(avg_timings.values())
            for key, val in sorted(avg_timings.items(), key=lambda x: -x[1]):
                pct = (val / total * 100) if total > 0 else 0
                print(f"  {key:25s} {val*1000:8.2f}ms  ({pct:5.1f}%)", file=sys.stderr)
            print("-" * 60, file=sys.stderr)
            print(f"  {'TOTAL':25s} {total*1000:8.2f}ms", file=sys.stderr)

        print(file=sys.stderr)

    return result_std, result_fast, min_fast


def main():
    parser = argparse.ArgumentParser(
        description="Fast string processing for straddle expansion"
    )
    parser.add_argument("pattern", help="Regex pattern to match assets (e.g., '^LA Comdty')")
    parser.add_argument("start_year", type=int, help="Start year")
    parser.add_argument("end_year", type=int, help="End year")

    parser.add_argument("--amt", default="data/amt.yml",
                        help="Path to AMT YAML file (default: data/amt.yml)")
    parser.add_argument("--benchmark", "-b", nargs="?", const=5, type=int, metavar="N",
                        help="Benchmark mode: run N iterations and report stats (default: 5)")
    parser.add_argument("--parallel", "-p", action="store_true",
                        help="Use parallel cartesian product")
    parser.add_argument("--verify", "-v", action="store_true",
                        help="Verify that outputs match (requires --benchmark)")
    parser.add_argument("--breakdown", "-t", action="store_true",
                        help="Show timing breakdown (requires --benchmark)")
    parser.add_argument("--matrix-only", action="store_true",
                        help="Skip string conversion in fast approach (measures matrix ops only)")
    parser.add_argument("--parsed", action="store_true",
                        help="Use pre-parsed output format (columns instead of straddle strings)")

    args = parser.parse_args()

    # Benchmark mode: run comparisons and show stats only
    if args.benchmark:
        result_std, result_fast, _ = benchmark(
            args.amt, args.pattern, args.start_year, args.end_year,
            args.benchmark, args.parallel, args.breakdown, args.matrix_only,
            include_parsed=args.parsed
        )

        if args.verify:
            print(file=sys.stderr)
            verify_outputs(result_std, result_fast)

        return 0

    # Normal mode: run expansion and output the table
    if args.parsed:
        # Use pre-parsed output format
        # Warmup
        _ = expand_fast_parsed(args.amt, args.pattern, args.start_year, args.end_year, args.parallel)
        # Run expansion
        result = expand_fast_parsed(args.amt, args.pattern, args.start_year, args.end_year, args.parallel)
    else:
        # Use straddle string format
        # Warmup (important for Numba JIT)
        _ = expand_fast(args.amt, args.pattern, args.start_year, args.end_year, args.parallel)
        # Run expansion
        result = expand_fast(args.amt, args.pattern, args.start_year, args.end_year, args.parallel)

    # Output the table
    if result["rows"]:
        from specparser.amt import loader
        loader.print_table(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
