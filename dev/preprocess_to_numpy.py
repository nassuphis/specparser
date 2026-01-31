"""One-time conversion: Parquet → NumPy format for faster loading.

Run this once to create .npy/.npz files from parquet data.
This avoids Arrow/Parquet overhead at runtime.
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    print("Converting Parquet → NumPy format...")

    # -------------------------------------------------------------------------
    # Convert ticker/field dictionaries
    # -------------------------------------------------------------------------
    print("  Loading ticker dict...")
    ticker = pq.read_table(DATA_DIR / "prices_ticker_dict.parquet")["ticker"].to_pylist()
    print("  Loading field dict...")
    field = pq.read_table(DATA_DIR / "prices_field_dict.parquet")["field"].to_pylist()

    print(f"  Saving ticker dict ({len(ticker)} entries)...")
    np.save(DATA_DIR / "prices_ticker_dict.npy", np.array(ticker, dtype="U100"))
    print(f"  Saving field dict ({len(field)} entries)...")
    np.save(DATA_DIR / "prices_field_dict.npy", np.array(field, dtype="U100"))

    # -------------------------------------------------------------------------
    # Convert prices with pre-built block metadata
    # -------------------------------------------------------------------------
    print("  Loading prices...")
    t = pq.read_table(DATA_DIR / "prices_keyed_sorted.parquet")
    px_key = t["key"].to_numpy().astype(np.int32)
    px_date = t["date"].to_numpy().astype("datetime64[D]").astype(np.int32)
    px_value = t["value"].to_numpy().astype(np.float64)

    print(f"  Building block metadata ({len(px_key):,} prices)...")
    chg = np.flatnonzero(px_key[1:] != px_key[:-1]) + 1
    starts = np.r_[0, chg].astype(np.int32)
    ends = np.r_[chg, len(px_key)].astype(np.int32)
    keys = px_key[starts]
    max_key = int(px_key.max())
    block_of = np.full(max_key + 1, -1, dtype=np.int32)
    block_of[keys] = np.arange(len(keys), dtype=np.int32)

    print(f"  Saving prices npz ({len(starts)} blocks)...")
    np.savez(
        DATA_DIR / "prices_keyed_sorted_np.npz",
        date=px_date,
        value=px_value,
        starts=starts,
        ends=ends,
        block_of=block_of,
    )

    print()
    print("Created:")
    print(f"  {DATA_DIR / 'prices_ticker_dict.npy'}")
    print(f"  {DATA_DIR / 'prices_field_dict.npy'}")
    print(f"  {DATA_DIR / 'prices_keyed_sorted_np.npz'}")


if __name__ == "__main__":
    main()
