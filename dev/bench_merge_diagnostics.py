"""Diagnostics for sweep merge: understand group characteristics and overlap patterns."""
import numpy as np
from numba import njit
import time
from pathlib import Path
from typing import Any
import yaml

# ============================================================================
# Run the pipeline up to the merge prep stage
# ============================================================================

# String dtype constants
TICKER_U = "U100"
FIELD_U = "U100"
FUT_MONTH_MAP_LEN = 12

def map_to_id_searchsorted(query_u, sorted_u, order_arr):
    pos = np.searchsorted(sorted_u, query_u)
    pos_clipped = np.minimum(pos, len(sorted_u) - 1)
    valid = (pos < len(sorted_u)) & (sorted_u[pos_clipped] == query_u)
    return np.where(valid, order_arr[pos_clipped], -1).astype(np.int32)

print("Loading data and running pipeline to merge prep stage...")
print("=" * 60)

# Load YAML
amt_resolved = str(Path("data/amt.yml").resolve())
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

with open(amt_resolved, "r") as f:
    run_options = yaml.load(f, Loader=Loader)

# Processing amt
amt = run_options.get("amt", {})
expiry_schedules = run_options.get("expiry_schedules")

amap: dict[str, dict[str, Any]] = {}
for asset_data in amt.values():
    if isinstance(asset_data, dict):
        underlying = asset_data.get("Underlying")
        if underlying and asset_data.get("WeightCap") > 0:
            amap[underlying] = asset_data

anames = np.array(list(amap.keys()), dtype=np.dtypes.StringDType())
idx_map = dict(zip(list(amap.keys()), range(len(anames))))
nps = np.dtypes.StringDType()

hedge_source = np.array(list(map(lambda a: amap[a]["Hedge"].get("Source", ""), anames)), dtype=nps)
hedge_ticker = np.array(list(map(lambda a: amap[a]["Hedge"].get("Ticker", ""), anames)), dtype=nps)
hedge_field = np.array(list(map(lambda a: amap[a]["Hedge"].get("Field", ""), anames)), dtype=nps)
hedge_hedge = np.array(list(map(lambda a: amap[a]["Hedge"].get("hedge", ""), anames)), dtype=nps)
hedge_hedge1 = np.array(list(map(lambda a: amap[a]["Hedge"].get("hedge1", ""), anames)), dtype=nps)
hedge_ccy = np.array(list(map(lambda a: amap[a]["Hedge"].get("ccy", ""), anames)), dtype=nps)
hedge_tenor = np.array(list(map(lambda a: amap[a]["Hedge"].get("tenor", ""), anames)), dtype=nps)
hedge_fut_month_map = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_month_map", " " * FUT_MONTH_MAP_LEN), anames)), dtype=nps)
hedge_min_year_offset = np.array(list(map(lambda a: amap[a]["Hedge"].get("min_year_offset", "0"), anames)), dtype=nps)
hedge_fut_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_code", ""), anames)), dtype=nps)
hedge_market_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("market_code", ""), anames)), dtype=nps)

hedge_sources, hedge_source_id = np.unique(hedge_source, return_inverse=True)
hs2id_map = dict(zip(hedge_sources, range(len(hedge_sources))))
HEDGE_FUT = hs2id_map["fut"]
hedge_source_id_fut = hedge_source_id == HEDGE_FUT
HEDGE_NONFUT = hs2id_map["nonfut"]
hedge_source_id_nonfut = hedge_source_id == HEDGE_NONFUT
HEDGE_CDS = hs2id_map["cds"]
hedge_source_id_cds = hedge_source_id == HEDGE_CDS
HEDGE_CALC = hs2id_map["calc"]
hedge_source_id_calc = hedge_source_id == HEDGE_CALC

calc_hedge1 = hedge_ccy + "_fsw0m_" + hedge_tenor

hedge_fut_month_mtrx = hedge_fut_month_map.astype('S12').view('S1').reshape(-1, FUT_MONTH_MAP_LEN).astype('U1')
hedge_min_year_offset_int = hedge_min_year_offset.astype(np.int64)

achk = np.array([np.sum(np.frombuffer(x.encode('ascii'), dtype=np.uint8)) for x in anames], dtype=np.int64)
aschcnt = np.array(list(map(lambda a: len(expiry_schedules[amap[a]["Options"]]), anames)), dtype=np.int64)

easchcnt = np.repeat(aschcnt, aschcnt)
eastmp = np.concatenate(list(map(
    lambda a: np.array(expiry_schedules[amap[a]["Options"]], dtype="|U20"),
    anames
)), dtype="|U20")

easchj = np.arange(np.sum(aschcnt)) - np.repeat(np.cumsum(aschcnt) - aschcnt, aschcnt)
easchi = np.repeat(np.arange(len(anames)), aschcnt)

easntrcv, _, rest = np.strings.partition(eastmp, '_')
ntrc_flat = np.strings.slice(easntrcv, 1)
ntrv_flat = np.strings.slice(easntrcv, 1, 20)

easxprcv, _, rest = np.strings.partition(rest, '_')
conds = [
    easxprcv == "OVERRIDE",
    (np.strings.slice(easxprcv, 2) == "BD") & np.isin(np.strings.slice(easxprcv, 2, 20), ["a", "b", "c", "d"]),
    np.strings.slice(easxprcv, 2) == "BD"
]
choices_xprc = [easxprcv, "BD", "BD"]
choices_xprv = [
    "",
    (easchj * (20 // easchcnt + 1) + achk[easchi] % 5 + 1).astype("U"),
    np.strings.slice(easxprcv, 2, 20)
]
xprc_flat = np.select(conds, choices_xprc, default=np.strings.slice(easxprcv, 1))

ntrc_uniq, ntrc_ids_flat = np.unique(ntrc_flat, return_inverse=True)
ntrc2id = dict(zip(ntrc_uniq, range(len(ntrc_uniq))))
STR_F = ntrc2id.get("F", -1)

max_schedules = int(np.max(easchcnt))
schedule_id_matrix = np.full((len(amap), max_schedules, 5), -1, dtype=np.int32)
schedule_id_matrix[easchi, easchj, 0] = ntrc_ids_flat

# Load ID dicts
ticker_arr = np.load("data/prices_ticker_dict.npy", allow_pickle=False)
field_arr = np.load("data/prices_field_dict.npy", allow_pickle=False)
n_fields = len(field_arr)

ticker_to_id = {s: i for i, s in enumerate(ticker_arr)}
field_to_id = {s: i for i, s in enumerate(field_arr)}
ticker_order = np.argsort(ticker_arr)
ticker_sorted = ticker_arr[ticker_order]

hedge_ticker_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_ticker], dtype=np.int32)
hedge_hedge_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_hedge], dtype=np.int32)
calc_hedge1_tid = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge1], dtype=np.int32)
hedge_field_fid = np.array([field_to_id.get(str(s), -1) for s in hedge_field], dtype=np.int32)

PX_LAST_FID = field_to_id.get("PX_LAST", -1)
EMPTY_FID = field_to_id.get("", -1)

# Compute straddles
ym = np.arange(2001*12+1-1, 2026*12+1-1, dtype=np.int64)
ym_len = len(ym)

inas = anames
inidx = np.array([idx_map[a] for a in inas], dtype=np.uint64)
inasc = aschcnt[inidx]
sidx = np.repeat(inidx, inasc)
sc = np.repeat(inasc, inasc)
si1 = np.arange(np.sum(inasc))
si2 = np.repeat((np.cumsum(inasc)-inasc), inasc)
si = si1 - si2

smidx = np.repeat(sidx, ym_len)
smym = np.tile(ym, len(sidx))
smlen = len(smidx)

year_vec = smym // 12
month_vec = smym % 12 + 1

dpm = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int8)
leap_feb = np.full(len(smidx), 29, dtype=np.int8)

year0, month0 = year_vec, month_vec
leap0 = (((year0 % 4 == 0) & ((year0 % 100 != 0) | (year0 % 400 == 0)))) & (month0 == 2)
days0_vec = np.where(leap0, leap_feb, dpm[month0 - 1])

year1, month1 = (year_vec*12+(month_vec-1)-1) // 12, (year_vec*12+(month_vec-1)-1) % 12 + 1
leap1 = (((year1 % 4 == 0) & ((year1 % 100 != 0) | (year1 % 400 == 0)))) & (month1 == 2)
days1_vec = np.where(leap1, leap_feb, dpm[month1 - 1])

year2, month2 = (year_vec*12+(month_vec-1)-2) // 12, (year_vec*12+(month_vec-1)-2) % 12 + 1
leap2 = (((year2 % 4 == 0) & ((year2 % 100 != 0) | (year2 % 400 == 0)))) & (month2 == 2)
days2_vec = np.where(leap2, leap_feb, dpm[month2 - 1])

schcnt_vec = np.repeat(sc, ym_len)
schid_vec = np.repeat(si, ym_len)

schedule_id_matrix_smidx = schedule_id_matrix[smidx, schid_vec, :]
ntrc_id_vec = schedule_id_matrix_smidx[:, 0]

day_count_vec = days0_vec + days1_vec + np.where(ntrc_id_vec == STR_F, days2_vec, 0)

# Hedge tickers
hedge_source_id_nonfut_smidx = hedge_source_id_nonfut[smidx]
hedge_source_id_fut_smidx = hedge_source_id_fut[smidx]
hedge_source_id_cds_smidx = hedge_source_id_cds[smidx]
hedge_source_id_calc_smidx = hedge_source_id_calc[smidx]

cond_hedge = [
    hedge_source_id_nonfut_smidx,
    hedge_source_id_fut_smidx,
    hedge_source_id_cds_smidx,
    hedge_source_id_calc_smidx
]

fut_idx = np.flatnonzero(hedge_source_id_fut_smidx)

hedge_fut_code_m = hedge_fut_code[smidx[fut_idx]]
hedge_fut_month_code_m = hedge_fut_month_mtrx[smidx[fut_idx], month_vec[fut_idx]-1]
month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
hedge_opt_month_code_m = month_code[month_vec[fut_idx]-1]

myo_m = hedge_min_year_offset_int[smidx[fut_idx]]
yo_m = np.maximum(np.where(hedge_fut_month_code_m < hedge_opt_month_code_m, 1, 0), myo_m)

hedge_fut_yeartxt_m = (year_vec[fut_idx] + yo_m).astype("U")
hedge_fut_tail_m = hedge_fut_month_code_m + hedge_fut_yeartxt_m + " " + hedge_market_code[smidx[fut_idx]]
hedge_fut_ticker_m = hedge_fut_code_m + hedge_fut_tail_m
hedge_fut_ticker_m_u = hedge_fut_ticker_m.astype(TICKER_U)

hedge_fut_ticker = np.full(smlen, "", dtype=nps)
hedge_fut_ticker[fut_idx] = hedge_fut_ticker_m

hedge_hedge_smidx = hedge_hedge[smidx]
calc_hedge1_smidx = calc_hedge1[smidx]

# Precompute days
_ym_base = 2000 * 12
_ym_range = np.arange(2000*12, 2027*12)
_ym_dates = (
    (_ym_range // 12).astype('U') + '-' +
    np.char.zfill((_ym_range % 12 + 1).astype('U'), 2) + '-01'
).astype('datetime64[D]')
_ym_epoch = _ym_dates.astype(np.int64)

d_start_ym = np.where(ntrc_id_vec == STR_F, year2 * 12 + month2 - 1, year1 * 12 + month1 - 1)
total_days = int(np.sum(day_count_vec))

# Load prices
pz = np.load("data/prices_keyed_sorted_np.npz", allow_pickle=False)
px_date = np.ascontiguousarray(pz["date"])
px_value = np.ascontiguousarray(pz["value"])
px_starts = np.ascontiguousarray(pz["starts"])
px_ends = np.ascontiguousarray(pz["ends"])
px_block_of = np.ascontiguousarray(pz["block_of"])

# Map monthly IDs
hedge_fut_ticker_tid_m = map_to_id_searchsorted(hedge_fut_ticker_m_u, ticker_sorted, ticker_order)

hedge_ticker_tid_smidx = hedge_ticker_tid[smidx]
hedge_hedge_tid_smidx = hedge_hedge_tid[smidx]
calc_hedge1_tid_smidx = calc_hedge1_tid[smidx]

hedge_fut_ticker_tid = np.full(smlen, -1, dtype=np.int32)
hedge_fut_ticker_tid[fut_idx] = hedge_fut_ticker_tid_m

hedge_field_fid_smidx = hedge_field_fid[smidx]

choices_tid = [hedge_ticker_tid_smidx, hedge_fut_ticker_tid, hedge_hedge_tid_smidx, calc_hedge1_tid_smidx]
choices_fid = [hedge_field_fid_smidx, PX_LAST_FID, PX_LAST_FID, EMPTY_FID]

month_tid = np.select(cond_hedge, choices_tid, default=-1).astype(np.int32)
month_fid = np.select(cond_hedge, choices_fid, default=-1).astype(np.int32)

# Sweep merge prep
month_key = (month_tid * np.int32(n_fields) + month_fid).astype(np.int32)
month_start_epoch = _ym_epoch[d_start_ym - _ym_base].astype(np.int32)
month_out0 = (np.cumsum(day_count_vec) - day_count_vec).astype(np.int32)

order = np.lexsort((month_start_epoch, month_key))
m_key_s = month_key[order]
m_start_s = month_start_epoch[order]
m_len_s = day_count_vec[order].astype(np.int32)
m_out0_s = month_out0[order]

max_valid_key = len(px_block_of) - 1
valid_mask = (m_key_s >= 0) & (m_key_s <= max_valid_key)
m_key_valid = m_key_s[valid_mask].astype(np.int32)
m_start_valid = m_start_s[valid_mask].astype(np.int32)
m_len_valid = m_len_s[valid_mask].astype(np.int32)
m_out0_valid = m_out0_s[valid_mask].astype(np.int32)

chg = np.flatnonzero(m_key_valid[1:] != m_key_valid[:-1]) + 1
g_starts = np.r_[0, chg].astype(np.int32)
g_ends = np.r_[chg, len(m_key_valid)].astype(np.int32)
g_keys = m_key_valid[g_starts]

print()
print("=" * 60)
print("SWEEP MERGE DIAGNOSTICS")
print("=" * 60)

# ============================================================================
# Group-level statistics
# ============================================================================
print("\n1. GROUP STATISTICS")
print("-" * 40)

n_groups = len(g_keys)
group_sizes = g_ends - g_starts

print(f"Total groups (unique keys):     {n_groups:,}")
print(f"Total straddles (valid):        {len(m_key_valid):,}")
print(f"Total straddles (all):          {smlen:,}")
print(f"Filtered out (invalid key):     {smlen - len(m_key_valid):,}")

print(f"\nStraddles per group:")
print(f"  min:    {group_sizes.min()}")
print(f"  max:    {group_sizes.max()}")
print(f"  mean:   {group_sizes.mean():.1f}")
print(f"  median: {np.median(group_sizes):.1f}")
print(f"  p95:    {np.percentile(group_sizes, 95):.0f}")
print(f"  p99:    {np.percentile(group_sizes, 99):.0f}")

# ============================================================================
# Interval length statistics
# ============================================================================
print("\n2. INTERVAL LENGTH STATISTICS")
print("-" * 40)

print(f"m_len (days per straddle):")
print(f"  min:    {m_len_valid.min()}")
print(f"  max:    {m_len_valid.max()}")
print(f"  mean:   {m_len_valid.mean():.1f}")
print(f"  median: {np.median(m_len_valid):.0f}")

# Distribution of lengths
len_counts = np.bincount(m_len_valid, minlength=100)
print(f"\nLength distribution (top 10):")
for length in np.argsort(-len_counts)[:10]:
    if len_counts[length] > 0:
        print(f"  {length:3d} days: {len_counts[length]:,} ({100*len_counts[length]/len(m_len_valid):.1f}%)")

# ============================================================================
# Overlap analysis per group
# ============================================================================
print("\n3. OVERLAP ANALYSIS (per group)")
print("-" * 40)

@njit
def analyze_overlaps(g_starts, g_ends, m_start, m_len):
    """Compute overlap statistics per group."""
    n_groups = len(g_starts)

    # For each group, count how many runs there are and total overlap factor
    run_counts = np.zeros(n_groups, dtype=np.int32)
    max_overlaps = np.zeros(n_groups, dtype=np.int32)
    total_work = np.zeros(n_groups, dtype=np.int64)  # sum of all interval lengths
    union_work = np.zeros(n_groups, dtype=np.int64)  # union span

    for gi in range(n_groups):
        s0 = g_starts[gi]
        s1 = g_ends[gi]
        if s0 >= s1:
            continue

        # Count runs and overlaps
        n_runs = 0
        max_overlap = 1

        si = s0
        while si < s1:
            run_start = m_start[si]
            run_end = run_start + m_len[si]
            total_work[gi] += m_len[si]
            overlap_count = 1

            sj = si + 1
            while sj < s1:
                st = m_start[sj]
                en = st + m_len[sj]
                if st > run_end:
                    break
                if en > run_end:
                    run_end = en
                overlap_count += 1
                total_work[gi] += m_len[sj]
                sj += 1

            union_work[gi] += run_end - run_start
            max_overlap = max(max_overlap, overlap_count)
            n_runs += 1
            si = sj

        run_counts[gi] = n_runs
        max_overlaps[gi] = max_overlap

    return run_counts, max_overlaps, total_work, union_work

run_counts, max_overlaps, total_work, union_work = analyze_overlaps(
    g_starts, g_ends, m_start_valid, m_len_valid
)

print(f"Runs per group:")
print(f"  min:    {run_counts.min()}")
print(f"  max:    {run_counts.max()}")
print(f"  mean:   {run_counts.mean():.1f}")
print(f"  total:  {run_counts.sum():,}")

print(f"\nMax overlapping intervals per group:")
print(f"  min:    {max_overlaps.min()}")
print(f"  max:    {max_overlaps.max()}")
print(f"  mean:   {max_overlaps.mean():.1f}")
print(f"  p50:    {np.median(max_overlaps):.0f}")
print(f"  p95:    {np.percentile(max_overlaps, 95):.0f}")

# Overlap factor = total_work / union_work (how much redundant scanning)
valid_union = union_work > 0
overlap_factor = total_work[valid_union] / union_work[valid_union]
print(f"\nOverlap factor (total_work / union_work):")
print(f"  min:    {overlap_factor.min():.2f}")
print(f"  max:    {overlap_factor.max():.2f}")
print(f"  mean:   {overlap_factor.mean():.2f}")
print(f"  median: {np.median(overlap_factor):.2f}")

# ============================================================================
# Run length analysis
# ============================================================================
print("\n4. RUN LENGTH ANALYSIS")
print("-" * 40)

@njit
def analyze_run_lengths(g_starts, g_ends, m_start, m_len):
    """Compute run length statistics."""
    n_groups = len(g_starts)

    # Collect all run lengths
    max_runs = 0
    for gi in range(n_groups):
        max_runs += g_ends[gi] - g_starts[gi]

    run_lengths = np.zeros(max_runs, dtype=np.int32)
    run_idx = 0

    for gi in range(n_groups):
        s0 = g_starts[gi]
        s1 = g_ends[gi]
        if s0 >= s1:
            continue

        si = s0
        while si < s1:
            run_start = m_start[si]
            run_end = run_start + m_len[si]

            sj = si + 1
            while sj < s1:
                st = m_start[sj]
                en = st + m_len[sj]
                if st > run_end:
                    break
                if en > run_end:
                    run_end = en
                sj += 1

            run_lengths[run_idx] = run_end - run_start
            run_idx += 1
            si = sj

    return run_lengths[:run_idx]

run_lengths = analyze_run_lengths(g_starts, g_ends, m_start_valid, m_len_valid)

print(f"Run lengths (buffer sizes needed):")
print(f"  min:    {run_lengths.min()} days")
print(f"  max:    {run_lengths.max()} days")
print(f"  mean:   {run_lengths.mean():.1f} days")
print(f"  median: {np.median(run_lengths):.0f} days")
print(f"  p95:    {np.percentile(run_lengths, 95):.0f} days")
print(f"  p99:    {np.percentile(run_lengths, 99):.0f} days")
print(f"  total:  {len(run_lengths):,} runs")

# Distribution of run lengths
print(f"\nRun length distribution:")
bins = [0, 30, 60, 90, 120, 150, 180, 365, 1000, 10000]
for i in range(len(bins)-1):
    count = np.sum((run_lengths >= bins[i]) & (run_lengths < bins[i+1]))
    pct = 100 * count / len(run_lengths)
    print(f"  {bins[i]:4d}-{bins[i+1]:4d} days: {count:,} runs ({pct:.1f}%)")

# ============================================================================
# Price data per key
# ============================================================================
print("\n5. PRICE DATA PER KEY")
print("-" * 40)

price_counts = np.zeros(n_groups, dtype=np.int32)
for gi in range(n_groups):
    key = g_keys[gi]
    if key >= 0 and key < len(px_block_of):
        b = px_block_of[key]
        if b >= 0:
            price_counts[gi] = px_ends[b] - px_starts[b]

valid_prices = price_counts[price_counts > 0]
print(f"Prices per key (where available):")
print(f"  min:    {valid_prices.min()}")
print(f"  max:    {valid_prices.max()}")
print(f"  mean:   {valid_prices.mean():.1f}")
print(f"  median: {np.median(valid_prices):.0f}")
print(f"  total:  {len(valid_prices):,} keys with prices")

# ============================================================================
# Work estimate: current vs run-based
# ============================================================================
print("\n6. WORK ESTIMATE")
print("-" * 40)

# Current approach: scans each straddle's dates
current_work = np.sum(m_len_valid.astype(np.int64))

# Run-based approach: scans union once + copies
run_based_work = np.sum(run_lengths.astype(np.int64))

print(f"Current approach (per-straddle scan):")
print(f"  Total date comparisons: ~{current_work:,}")

print(f"\nRun-based approach (scan once + copy):")
print(f"  Total date comparisons: ~{run_based_work:,}")
print(f"  Plus: ~{current_work:,} buffer copies (memcpy-like)")

print(f"\nPotential scan reduction: {current_work / run_based_work:.1f}x")
print(f"  (but adds allocation/copy overhead)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Groups:           {n_groups:,}
Straddles:        {len(m_key_valid):,}
Total runs:       {len(run_lengths):,}
Avg overlap:      {overlap_factor.mean():.1f}x
Avg run length:   {run_lengths.mean():.0f} days
Max run length:   {run_lengths.max()} days

Run-buffering viability:
- Overlap factor ~{overlap_factor.mean():.1f}x suggests {overlap_factor.mean():.1f}x redundant scanning
- Run lengths mostly {int(np.percentile(run_lengths, 5))}-{int(np.percentile(run_lengths, 95))} days (p5-p95)
- {100 * np.sum(run_lengths <= 365) / len(run_lengths):.1f}% of runs <= 365 days (small buffers)

Recommendation:
""")

if overlap_factor.mean() > 2.0 and np.percentile(run_lengths, 99) < 1000:
    print("  -> Run-buffering likely beneficial")
    print(f"     ({overlap_factor.mean():.1f}x overlap, runs mostly small)")
elif overlap_factor.mean() < 1.5:
    print("  -> Run-buffering unlikely to help (low overlap)")
else:
    print("  -> Run-buffering may help but test needed")
    print(f"     (some large runs: p99={int(np.percentile(run_lengths, 99))} days)")
