"""Test first-call overhead for Numba parallel functions."""
import numpy as np
from numba import njit, prange
import time

@njit(parallel=True, cache=True)
def _merge_per_key(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                   px_block_of, px_starts, px_ends, px_date, px_value, out):
    n_groups = len(g_keys)
    n_blocks = len(px_block_of)
    for gi in prange(n_groups):
        key = g_keys[gi]
        if key < 0 or key >= n_blocks:
            continue
        b = px_block_of[key]
        if b < 0:
            continue
        ps = px_starts[b]
        pe = px_ends[b]
        pi0 = ps
        for si in range(g_starts[gi], g_ends[gi]):
            start = m_start_s[si]
            length = m_len_s[si]
            if length <= 0:
                continue
            out0 = m_out0_s[si]
            end = start + length
            while pi0 < pe and px_date[pi0] < start:
                pi0 += 1
            pi = pi0
            while pi < pe:
                d = px_date[pi]
                if d >= end:
                    break
                out[out0 + (d - start)] = px_value[pi]
                pi += 1
    return out

# Load real data
print("Loading data...")
pz = np.load('data/prices_keyed_sorted_np.npz', allow_pickle=False)
px_date = np.ascontiguousarray(pz['date'])
px_value = np.ascontiguousarray(pz['value'])
px_starts = np.ascontiguousarray(pz['starts'])
px_ends = np.ascontiguousarray(pz['ends'])
px_block_of = np.ascontiguousarray(pz['block_of'])

# Create small safe test data
n_straddles = 1000
n_groups = 100
total_days = 100000

g_keys = np.arange(n_groups, dtype=np.int32) % len(px_block_of)
g_starts = np.arange(n_groups, dtype=np.int32) * (n_straddles // n_groups)
g_ends = np.minimum(g_starts + n_straddles // n_groups, n_straddles).astype(np.int32)
m_start_s = np.repeat(np.arange(11000, 11000 + n_groups, dtype=np.int32), n_straddles // n_groups)
m_len_s = np.full(n_straddles, 60, dtype=np.int32)
m_out0_s = (np.arange(n_straddles, dtype=np.int32) * 60).clip(0, total_days - 60)

print("Running benchmark...")
for run in range(5):
    out = np.full(total_days, np.nan, dtype=np.float64)
    t0 = time.perf_counter()
    out = _merge_per_key(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                         px_block_of, px_starts, px_ends, px_date, px_value, out)
    t1 = time.perf_counter()
    print(f"  Run {run+1}: {(t1-t0)*1000:.1f}ms")
