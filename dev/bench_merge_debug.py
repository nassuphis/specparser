"""Debug: Compare benchmark data vs main pipeline data."""
import numpy as np
from numba import njit, prange
import time
from pathlib import Path
from typing import Any
import yaml

# ============================================================================
# Numba kernel (same as backtest_standalone.py)
# ============================================================================
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

# ============================================================================
# Load the exact same data as backtest_standalone.py
# ============================================================================
TICKER_U = "U100"
FUT_MONTH_MAP_LEN = 12

def map_to_id_searchsorted(query_u, sorted_u, order_arr):
    pos = np.searchsorted(sorted_u, query_u)
    pos_clipped = np.minimum(pos, len(sorted_u) - 1)
    valid = (pos < len(sorted_u)) & (sorted_u[pos_clipped] == query_u)
    return np.where(valid, order_arr[pos_clipped], -1).astype(np.int32)

print("Loading data (same as backtest_standalone.py)...")

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
    lambda a: np.array(expiry_schedules[amap[a]["Options"]], dtype="|U20"), anames
)), dtype="|U20")

easchj = np.arange(np.sum(aschcnt)) - np.repeat(np.cumsum(aschcnt) - aschcnt, aschcnt)
easchi = np.repeat(np.arange(len(anames)), aschcnt)

easntrcv, _, rest = np.strings.partition(eastmp, '_')
ntrc_flat = np.strings.slice(easntrcv, 1)

ntrc_uniq, ntrc_ids_flat = np.unique(ntrc_flat, return_inverse=True)
ntrc2id = dict(zip(ntrc_uniq, range(len(ntrc_uniq))))
STR_F = ntrc2id.get("F", -1)

max_schedules = int(np.max(easchcnt))
schedule_id_matrix = np.full((len(amap), max_schedules, 5), -1, dtype=np.int32)
schedule_id_matrix[easchi, easchj, 0] = ntrc_ids_flat

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

_ym_base = 2000 * 12
_ym_range = np.arange(2000*12, 2027*12)
_ym_dates = (
    (_ym_range // 12).astype('U') + '-' +
    np.char.zfill((_ym_range % 12 + 1).astype('U'), 2) + '-01'
).astype('datetime64[D]')
_ym_epoch = _ym_dates.astype(np.int64)

d_start_ym = np.where(ntrc_id_vec == STR_F, year2 * 12 + month2 - 1, year1 * 12 + month1 - 1)
total_days = int(np.sum(day_count_vec))

pz = np.load("data/prices_keyed_sorted_np.npz", allow_pickle=False)
px_date = np.ascontiguousarray(pz["date"])
px_value = np.ascontiguousarray(pz["value"])
px_starts = np.ascontiguousarray(pz["starts"])
px_ends = np.ascontiguousarray(pz["ends"])
px_block_of = np.ascontiguousarray(pz["block_of"])

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
m_key_valid = np.ascontiguousarray(m_key_s[valid_mask].astype(np.int32))
m_start_valid = np.ascontiguousarray(m_start_s[valid_mask].astype(np.int32))
m_len_valid = np.ascontiguousarray(m_len_s[valid_mask].astype(np.int32))
m_out0_valid = np.ascontiguousarray(m_out0_s[valid_mask].astype(np.int32))

chg = np.flatnonzero(m_key_valid[1:] != m_key_valid[:-1]) + 1
g_starts = np.ascontiguousarray(np.r_[0, chg].astype(np.int32))
g_ends = np.ascontiguousarray(np.r_[chg, len(m_key_valid)].astype(np.int32))
g_keys = np.ascontiguousarray(m_key_valid[g_starts])

print(f"Data loaded.")
print()
print("Data characteristics:")
print(f"  g_keys.shape:      {g_keys.shape}, dtype={g_keys.dtype}")
print(f"  g_starts.shape:    {g_starts.shape}, dtype={g_starts.dtype}")
print(f"  g_ends.shape:      {g_ends.shape}, dtype={g_ends.dtype}")
print(f"  m_start_valid:     {m_start_valid.shape}, dtype={m_start_valid.dtype}")
print(f"  m_len_valid:       {m_len_valid.shape}, dtype={m_len_valid.dtype}")
print(f"  m_out0_valid:      {m_out0_valid.shape}, dtype={m_out0_valid.dtype}")
print(f"  px_block_of:       {px_block_of.shape}, dtype={px_block_of.dtype}")
print(f"  px_starts:         {px_starts.shape}, dtype={px_starts.dtype}")
print(f"  px_ends:           {px_ends.shape}, dtype={px_ends.dtype}")
print(f"  px_date:           {px_date.shape}, dtype={px_date.dtype}")
print(f"  px_value:          {px_value.shape}, dtype={px_value.dtype}")
print(f"  total_days:        {total_days:,}")
print()

# Check contiguity
print("Contiguity check:")
for name, arr in [("g_keys", g_keys), ("g_starts", g_starts), ("g_ends", g_ends),
                   ("m_start_valid", m_start_valid), ("m_len_valid", m_len_valid),
                   ("m_out0_valid", m_out0_valid), ("px_block_of", px_block_of),
                   ("px_starts", px_starts), ("px_ends", px_ends),
                   ("px_date", px_date), ("px_value", px_value)]:
    print(f"  {name}: contiguous={arr.flags['C_CONTIGUOUS']}")

print()

# ============================================================================
# Warmup
# ============================================================================
print("Warming up JIT...")
out = np.full(total_days, np.nan, dtype=np.float64)
_merge_per_key(g_keys[:10], g_starts[:10], g_ends[:10],
               m_start_valid, m_len_valid, m_out0_valid,
               px_block_of, px_starts, px_ends, px_date, px_value, out)
print("Warmup done.")
print()

# ============================================================================
# Benchmark
# ============================================================================
print("Running benchmark (5 runs)...")
times = []
for i in range(5):
    out = np.full(total_days, np.nan, dtype=np.float64)
    t0 = time.perf_counter()
    out = _merge_per_key(g_keys, g_starts, g_ends,
                         m_start_valid, m_len_valid, m_out0_valid,
                         px_block_of, px_starts, px_ends, px_date, px_value, out)
    t1 = time.perf_counter()
    times.append(t1 - t0)
    print(f"  Run {i+1}: {(t1-t0)*1000:.1f}ms, found={np.sum(~np.isnan(out)):,}")

print()
print(f"Median: {np.median(times)*1000:.1f}ms")
print(f"Min:    {min(times)*1000:.1f}ms")
print(f"Max:    {max(times)*1000:.1f}ms")
