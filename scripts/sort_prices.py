#!/usr/bin/env python
"""Sort prices.parquet by (ticker, field, date) for faster loading.

When prices.parquet is pre-sorted, load_prices_numba() can skip the
expensive np.argsort() step, saving ~0.13s (~29% of load time).

Usage:
    uv run python scripts/sort_prices.py data/prices.parquet data/prices_sorted.parquet
"""
import argparse
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc


def main():
    parser = argparse.ArgumentParser(
        description="Sort prices.parquet by (ticker, field, date) for faster loading"
    )
    parser.add_argument("input", help="Input parquet file")
    parser.add_argument("output", help="Output sorted parquet file")
    parser.add_argument(
        "--verify", "-v", action="store_true", help="Verify output is sorted"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Loading {input_path}...")
    start = time.perf_counter()
    table = pq.read_table(str(input_path))
    load_time = time.perf_counter() - start
    print(f"  Rows: {table.num_rows:,}")
    print(f"  Load time: {load_time:.2f}s")

    print("Sorting by (ticker, field, date)...")
    start = time.perf_counter()
    sort_indices = pc.sort_indices(
        table,
        sort_keys=[
            ("ticker", "ascending"),
            ("field", "ascending"),
            ("date", "ascending"),
        ],
    )
    sorted_table = table.take(sort_indices)
    sort_time = time.perf_counter() - start
    print(f"  Sort time: {sort_time:.2f}s")

    print(f"Writing {output_path}...")
    start = time.perf_counter()
    pq.write_table(sorted_table, str(output_path))
    write_time = time.perf_counter() - start
    print(f"  Write time: {write_time:.2f}s")

    if args.verify:
        print("Verifying sort order...")
        # Re-read and check
        verify_table = pq.read_table(str(output_path))

        # Dictionary encode to get indices
        # Note: dictionary_encode assigns indices by order of first appearance,
        # so for sorted data, the indices will be monotonically non-decreasing
        ticker_dict = pc.dictionary_encode(verify_table["ticker"]).combine_chunks()
        field_dict = pc.dictionary_encode(verify_table["field"]).combine_chunks()

        ticker_idx = ticker_dict.indices.to_numpy()
        field_idx = field_dict.indices.to_numpy()

        import numpy as np

        epoch = np.datetime64("1970-01-01", "D")
        date_arr = verify_table["date"].to_numpy()
        date_int32 = (date_arr - epoch).astype(np.int32)
        min_date32 = int(date_int32.min())
        max_date32 = int(date_int32.max())
        n_dates = max_date32 - min_date32 + 1
        n_fields = len(field_dict.dictionary)

        date_offset = (date_int32 - min_date32).astype(np.int64)
        composite_key = (
            ticker_idx.astype(np.int64) * n_fields + field_idx
        ) * n_dates + date_offset

        is_sorted = np.all(composite_key[:-1] <= composite_key[1:])
        if is_sorted:
            print("  ✓ Output composite key is monotonic (load_prices_numba will skip sort)")
        else:
            # This is expected if dictionary encoding assigns indices differently
            # Check if actual string sort order is correct instead
            print("  Note: Composite key not monotonic (dictionary encoding order differs)")
            print("  Checking actual string sort order...")

            # Check ticker is non-decreasing
            tickers = verify_table["ticker"].to_pylist()
            ticker_sorted = all(tickers[i] <= tickers[i+1] for i in range(len(tickers)-1))

            if not ticker_sorted:
                print("  ✗ Tickers are NOT sorted!", file=sys.stderr)
                return 1

            # For each ticker group, check field is non-decreasing
            # For each ticker+field group, check date is non-decreasing
            # (This is expensive but thorough)
            fields = verify_table["field"].to_pylist()
            dates = verify_table["date"].to_pylist()

            prev_ticker, prev_field, prev_date = None, None, None
            for i, (t, f, d) in enumerate(zip(tickers, fields, dates)):
                if prev_ticker is not None:
                    if t < prev_ticker:
                        print(f"  ✗ Ticker out of order at row {i}", file=sys.stderr)
                        return 1
                    if t == prev_ticker:
                        if f < prev_field:
                            print(f"  ✗ Field out of order at row {i}", file=sys.stderr)
                            return 1
                        if f == prev_field and d < prev_date:
                            print(f"  ✗ Date out of order at row {i}", file=sys.stderr)
                            return 1
                prev_ticker, prev_field, prev_date = t, f, d

            print("  ✓ Output is correctly sorted by (ticker, field, date)")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
