"""
Optimized backtest using Numba typed dicts and pre-computed arrays.

This version eliminates nested dict lookups from the inner loop by:
1. Pre-parsing schedule strings into arrays
2. Pre-computing asset properties into arrays
3. Using array indexing instead of dict lookups

Compared to backtest_standalone.py which does ~227k dict lookups.
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict
import time
from pathlib import Path
from typing import Any
import yaml

try:
    from yaml import CSafeLoader as yaml_loader
except ImportError:
    from yaml import SafeLoader as yaml_loader

# Import loader for real data experiments
from specparser.amt import loader, table


import pyarrow.parquet as pq
import pyarrow.compute as pc

# ============================================================================
# PHASE 0: Load configuration (same as original)
# ============================================================================

_DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int8)
_OPT_MONTH_CODES = "FGHJKMNQUVXZ"

print("loading yaml........", end="")
start_time = time.perf_counter()

amt_resolved = (Path(__file__).resolve().parent.parent / "data" / "amt.yml")
with open(amt_resolved, "r") as f:
    run_options = yaml.load(f,Loader=yaml_loader)
amt = run_options.get("amt", {})
expiry_schedules = run_options.get("expiry_schedules")

# Build underlying_map (filter to active assets)
underlying_map: dict[str, dict[str, Any]] = {}

for asset_data in amt.values():
    if isinstance(asset_data, dict):
        underlying = asset_data.get("Underlying")
        if underlying and asset_data.get("WeightCap", 0) > 0:
            underlying_map[underlying] = asset_data

schedule_matrix = np.full((len(underlying_map),4,5),"",dtype=np.dtypes.StringDType())
schedule_length = np.full(len(underlying_map),0,dtype=np.uint8)

for idx, asset in enumerate(underlying_map.values()):
    underlying = asset.get("Underlying")
    assid = np.sum(np.frombuffer(underlying.encode('ascii'),dtype=np.uint8))
    schedule_name = asset.get("Options")
    underlying_schedules = expiry_schedules[schedule_name]
    schedule_length[idx]=np.uint8(len(underlying_schedules))
    for i, underlying_schedule in enumerate(underlying_schedules):
        parts = underlying_schedule.split("_")
        schedule_matrix[idx,i,0] = parts[0][0]
        schedule_matrix[idx,i,1] = parts[0][1:]
        if parts[1] == "OVERRIDE":
            schedule_matrix[idx,i,2] = "OVERRIDE"
            schedule_matrix[idx,i,3] = ""
        elif parts[1][:2] == "BD":
            schedule_matrix[idx,i,2] = "BD"
            if parts[1][2] in ["a","b","c","d"]:
                schedule_matrix[idx,i,3] = str( i * (20 // (len(underlying_schedules) + 1)) + assid % 5 + 1 )
            else:
                schedule_matrix[idx,i,3] = parts[1][2:]
        elif parts[1][0] in ("F", "R", "W"):
            schedule_matrix[idx,i,2] = parts[1][0]
            schedule_matrix[idx,i,3] = parts[1][1:]
        else:
            schedule_matrix[idx,i,2] = parts[1]
            schedule_matrix[idx,i,3] = ""
        schedule_matrix[idx,i,4] = parts[2]

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({len(list(underlying_map.keys()))} assets)")
print(schedule_matrix[0,:,:])
print("load prices.........", end="")
start_time = time.perf_counter()

table = pq.read_table("data/prices_sorted.parquet")

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({table.num_rows:,} rows read)")

# ============================================================================
# PHASE 1: Create typed dicts for index lookups
# ============================================================================
print("creating dicts......", end="")
start_time = time.perf_counter()

n_assets = len(underlying_map)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
# ============================================================================
# PHASE 2: Pre-parse schedules into structured arrays
# ============================================================================
print("parsing schedule....", end="")
start_time = time.perf_counter()

# ============================================================================
# PHASE 3: Pre-compute asset properties into arrays
# ============================================================================

# Asset -> schedule mapping
asset_schedule_idx = np.full(n_assets, -1, dtype=np.int64)

# Hedge source: 0=none, 1=fut, 2=nonfut, 3=cds, 4=calc
HEDGE_NONE, HEDGE_FUT, HEDGE_NONFUT, HEDGE_CDS, HEDGE_CALC = 0, 1, 2, 3, 4
asset_hedge_source = np.zeros(n_assets, dtype=np.int8)

# Vol source: 0=none, 1=BBG, 2=CV, 3=BBG_LMEVOL
VOL_NONE, VOL_BBG, VOL_CV, VOL_BBG_LMEVOL = 0, 1, 2, 3
asset_vol_source = np.zeros(n_assets, dtype=np.int8)

# Futures config (12 months of month codes as ASCII)
asset_fut_month_map = np.zeros((n_assets, 12), dtype=np.uint8)
asset_fut_code = np.full(n_assets, "", dtype=np.dtypes.StringDType())
asset_market_code = np.full(n_assets, "", dtype=np.dtypes.StringDType())
asset_min_year_offset = np.zeros(n_assets, dtype=np.int8)

# Non-fut/static hedge config
asset_hedge_ticker = np.full(n_assets, "", dtype=np.dtypes.StringDType())
asset_hedge_field = np.full(n_assets, "", dtype=np.dtypes.StringDType())

# CDS config
asset_cds_hedge = np.full(n_assets, "", dtype=np.dtypes.StringDType())
asset_cds_hedge1 = np.full(n_assets, "", dtype=np.dtypes.StringDType())

# Calc config
asset_calc_ccy = np.full(n_assets, "", dtype=np.dtypes.StringDType())
asset_calc_tenor = np.zeros(n_assets, dtype=np.int32)

# Vol config
asset_vol_ticker = np.full(n_assets, "", dtype=np.dtypes.StringDType())
asset_vol_near = np.full(n_assets, "", dtype=np.dtypes.StringDType())
asset_vol_far = np.full(n_assets, "", dtype=np.dtypes.StringDType())

# Has valid vol for Far (for F schedules)
asset_has_valid_far = np.zeros(n_assets, dtype=np.bool_)

for asset_idx, asset in enumerate(assets):
    asset_data = underlying_map[asset]

    # Schedule mapping
    schedule_name = asset_data.get("Options")
    if schedule_name and schedule_name in schedule_to_idx:
        asset_schedule_idx[asset_idx] = schedule_to_idx[schedule_name]

    # Hedge config
    hedge = asset_data.get("Hedge")
    if hedge:
        source = hedge.get("Source", "")
        if source == "fut":
            asset_hedge_source[asset_idx] = HEDGE_FUT
            asset_fut_code[asset_idx] = hedge.get("fut_code", "")
            asset_market_code[asset_idx] = hedge.get("market_code", "")
            asset_min_year_offset[asset_idx] = hedge.get("min_year_offset", 0)
            fut_month_map = hedge.get("fut_month_map", "")
            for m, code in enumerate(fut_month_map[:12]):
                asset_fut_month_map[asset_idx, m] = ord(code)
        elif source == "nonfut":
            asset_hedge_source[asset_idx] = HEDGE_NONFUT
            asset_hedge_ticker[asset_idx] = hedge.get("Ticker", "")
            asset_hedge_field[asset_idx] = hedge.get("Field", "")
        elif source == "cds":
            asset_hedge_source[asset_idx] = HEDGE_CDS
            asset_cds_hedge[asset_idx] = hedge.get("hedge", "")
            asset_cds_hedge1[asset_idx] = hedge.get("hedge1", "")
        elif source == "calc":
            asset_hedge_source[asset_idx] = HEDGE_CALC
            asset_calc_ccy[asset_idx] = hedge.get("ccy", "")
            asset_calc_tenor[asset_idx] = hedge.get("tenor", 0)

    # Vol config
    vol = asset_data.get("Vol")
    if vol:
        source = vol.get("Source", "")
        if source == "BBG":
            asset_vol_source[asset_idx] = VOL_BBG
            asset_vol_ticker[asset_idx] = vol.get("Ticker", "")
            asset_vol_near[asset_idx] = vol.get("Near", "")
            asset_vol_far[asset_idx] = vol.get("Far", "")
            # Check if Far is valid
            far = vol.get("Far", "")
            near = vol.get("Near", "")
            if far and far != "" and far != "NONE" and far != near:
                asset_has_valid_far[asset_idx] = True
        elif source == "CV":
            asset_vol_source[asset_idx] = VOL_CV
            asset_vol_near[asset_idx] = vol.get("Near", "")
        elif source == "BBG_LMEVOL":
            asset_vol_source[asset_idx] = VOL_BBG_LMEVOL
            asset_has_valid_far[asset_idx] = True  # BBG_LMEVOL always has valid far
            # For BBG_LMEVOL, use hedge config for fut_code, market_code, etc.
            if hedge:
                asset_fut_code[asset_idx] = hedge.get("fut_code", "")
                asset_market_code[asset_idx] = hedge.get("market_code", "")
                asset_min_year_offset[asset_idx] = hedge.get("min_year_offset", 0)
                fut_month_map = hedge.get("fut_month_map", "")
                for m, code in enumerate(fut_month_map[:12]):
                    asset_fut_month_map[asset_idx, m] = ord(code)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ============================================================================
# PHASE 4: Numba-accelerated numeric computation
# ============================================================================
print("numba setup.........", end="")
start_time = time.perf_counter()

@njit
def compute_days_in_month(year: int, month: int) -> int:
    """Compute days in month accounting for leap years."""
    days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        return 29
    return days[month - 1]

@njit
def fill_numeric_arrays(
    year_month_start: int,
    year_month_end: int,
    n_assets: int,
    asset_schedule_idx,      # int64[n_assets]
    schedule_offsets,        # int64[n_schedules]
    schedule_counts,         # int8[n_schedules]
    # Output arrays
    out_year,
    out_month,
    out_asset_idx,
    out_schedule_offset,
    out_schcnt,
    out_schid,
    out_days0,
    out_days1,
    out_days2,
):
    """Fill numeric output arrays. Returns number of straddles written."""
    j = 0
    for year_month in range(year_month_start, year_month_end + 1):
        year = year_month // 12
        month = year_month % 12 + 1

        # Pre-compute days for this month and previous 2 months
        days0 = compute_days_in_month(year, month)
        ym1 = year_month - 1
        days1 = compute_days_in_month(ym1 // 12, ym1 % 12 + 1)
        ym2 = year_month - 2
        days2 = compute_days_in_month(ym2 // 12, ym2 % 12 + 1)

        for asset_idx in range(n_assets):
            schedule_idx = asset_schedule_idx[asset_idx]
            if schedule_idx < 0:
                continue

            start = schedule_offsets[schedule_idx]
            count = schedule_counts[schedule_idx]

            for i in range(count):
                out_year[j] = year
                out_month[j] = month
                out_asset_idx[j] = asset_idx
                out_schedule_offset[j] = start + i
                out_schcnt[j] = count
                out_schid[j] = i
                out_days0[j] = days0
                out_days1[j] = days1
                out_days2[j] = days2
                j += 1

    return j

# ============================================================================
# MAIN EXECUTION
# ============================================================================

year_start = 2001
month_start = 1
year_month_start = year_start * 12 + month_start - 1
year_end = 2026
month_end = 1
year_month_end = year_end * 12 + month_end - 1
month_count = year_month_end - year_month_start + 1

# Calculate total straddle count
monthly_straddle_count = 0
for asset_idx in range(n_assets):
    schedule_idx = asset_schedule_idx[asset_idx]
    if schedule_idx >= 0:
        monthly_straddle_count += int(schedule_counts[schedule_idx])
straddle_count = monthly_straddle_count * month_count

# Allocate output arrays
year_vec = np.zeros(straddle_count, dtype=np.int16)
month_vec = np.zeros(straddle_count, dtype=np.int8)
asset_idx_vec = np.zeros(straddle_count, dtype=np.int32)  # intermediate
schedule_offset_vec = np.zeros(straddle_count, dtype=np.int32)  # intermediate
schcnt_vec = np.zeros(straddle_count, dtype=np.int8)
schid_vec = np.zeros(straddle_count, dtype=np.int8)
days0_vec = np.zeros(straddle_count, dtype=np.int8)
days1_vec = np.zeros(straddle_count, dtype=np.int8)
days2_vec = np.zeros(straddle_count, dtype=np.int8)

# String output arrays
asset_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
ntrc_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
ntrv_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
xprc_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
xprv_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
wgt_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge1t_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge1f_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge2t_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge2f_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge3t_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge3f_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge4t_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge4f_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
volt_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
volf_vec = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
day_count_vec = np.zeros(straddle_count, dtype=np.int8)

# Warm up JIT
_ = fill_numeric_arrays(
    year_month_start, year_month_start,  # just one month
    n_assets, asset_schedule_idx, schedule_offsets, schedule_counts,
    np.zeros(monthly_straddle_count, dtype=np.int16),
    np.zeros(monthly_straddle_count, dtype=np.int8),
    np.zeros(monthly_straddle_count, dtype=np.int32),
    np.zeros(monthly_straddle_count, dtype=np.int32),
    np.zeros(monthly_straddle_count, dtype=np.int8),
    np.zeros(monthly_straddle_count, dtype=np.int8),
    np.zeros(monthly_straddle_count, dtype=np.int8),
    np.zeros(monthly_straddle_count, dtype=np.int8),
    np.zeros(monthly_straddle_count, dtype=np.int8),
)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
# ============================================================================
# TIMED SECTION: Main computation
# ============================================================================
print("vectorized calc.....", end="")
start_time = time.perf_counter()

# Step 1: Fill numeric arrays with Numba
n_written = fill_numeric_arrays(
    year_month_start, year_month_end,
    n_assets, asset_schedule_idx, schedule_offsets, schedule_counts,
    year_vec, month_vec, asset_idx_vec, schedule_offset_vec,
    schcnt_vec, schid_vec, days0_vec, days1_vec, days2_vec,
)

# Step 2: Fill string arrays using vectorized numpy indexing where possible
# Vectorized: simple index-based copies
asset_vec[:] = asset_names[asset_idx_vec]
ntrc_vec[:] = sch_ntrc[schedule_offset_vec]
ntrv_vec[:] = sch_ntrv[schedule_offset_vec]
xprc_vec[:] = sch_xprc[schedule_offset_vec]
wgt_vec[:] = sch_wgt[schedule_offset_vec]
xprv_vec[:] = sch_xprv_base[schedule_offset_vec]  # default, overwritten for BD letter cases

# Vectorized day_count calculation
is_N = ntrc_vec == "N"
is_F = ntrc_vec == "F"
day_count_vec[is_N] = (days0_vec[is_N].astype(np.int16) + days1_vec[is_N].astype(np.int16)).astype(np.int8)
day_count_vec[is_F] = (days0_vec[is_F].astype(np.int16) + days1_vec[is_F].astype(np.int16) + days2_vec[is_F].astype(np.int16)).astype(np.int8)

# Vectorized vol/hedge for static sources (BBG, CV, nonfut, cds)
# Get asset properties indexed by output position
asset_hedge_source_vec = asset_hedge_source[asset_idx_vec]
asset_vol_source_vec = asset_vol_source[asset_idx_vec]
asset_has_valid_far_vec = asset_has_valid_far[asset_idx_vec]

# BBG vol
bbg_mask = asset_vol_source_vec == VOL_BBG
volt_vec[bbg_mask] = asset_vol_ticker[asset_idx_vec[bbg_mask]]
bbg_N_mask = bbg_mask & is_N
bbg_F_mask = bbg_mask & is_F
volf_vec[bbg_N_mask] = asset_vol_near[asset_idx_vec[bbg_N_mask]]
volf_vec[bbg_F_mask] = asset_vol_far[asset_idx_vec[bbg_F_mask]]

# CV vol
cv_mask = asset_vol_source_vec == VOL_CV
volt_vec[cv_mask] = asset_vol_near[asset_idx_vec[cv_mask]]
volf_vec[cv_mask] = "none"

# Nonfut hedge
nonfut_mask = asset_hedge_source_vec == HEDGE_NONFUT
hedge1t_vec[nonfut_mask] = asset_hedge_ticker[asset_idx_vec[nonfut_mask]]
hedge1f_vec[nonfut_mask] = asset_hedge_field[asset_idx_vec[nonfut_mask]]

# CDS hedge
cds_mask = asset_hedge_source_vec == HEDGE_CDS
hedge1t_vec[cds_mask] = asset_cds_hedge[asset_idx_vec[cds_mask]]
hedge1f_vec[cds_mask] = "PX_LAST"
hedge2t_vec[cds_mask] = asset_cds_hedge1[asset_idx_vec[cds_mask]]
hedge2f_vec[cds_mask] = "PX_LAST"

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
print("conditional calc....", end="")
start_time = time.perf_counter()

# Step 3: Fill remaining arrays that need conditional logic (fut, BBG_LMEVOL, calc, BD letter)
# These require string concatenation or complex logic
for j in range(straddle_count):
    asset_idx = asset_idx_vec[j]
    sch_offset = schedule_offset_vec[j]
    year = year_vec[j]
    month = month_vec[j]
    schcnt = schcnt_vec[j]
    schid = schid_vec[j]
    ntrc = ntrc_vec[j]

    # xprv - handle BD letter case (default already copied vectorized)
    if sch_xprv_is_bd_letter[sch_offset]:
        assid = asset_assid[asset_idx]
        xprv_vec[j] = str(schid * (20 // (schcnt + 1)) + assid % 5 + 1)

    # Skip if no hedge/vol config
    hedge_source = asset_hedge_source[asset_idx]
    vol_source = asset_vol_source[asset_idx]
    if hedge_source == HEDGE_NONE or vol_source == VOL_NONE:
        continue

    # BBG_LMEVOL vol (requires string concat) - BBG and CV are vectorized above
    if vol_source == VOL_BBG_LMEVOL:
        fut_month_code = chr(asset_fut_month_map[asset_idx, month - 1])
        opt_month_code = _OPT_MONTH_CODES[month - 1]
        year_offset = max(1 if fut_month_code < opt_month_code else 0, asset_min_year_offset[asset_idx])
        volt_vec[j] = asset_fut_code[asset_idx] + "R" + fut_month_code + str(year + year_offset) + " " + asset_market_code[asset_idx]
        volf_vec[j] = "PX_LAST"

    # Hedge ticker - NONFUT and CDS are vectorized above
    if hedge_source == HEDGE_FUT:
        fut_month_code = chr(asset_fut_month_map[asset_idx, month - 1])
        opt_month_code = _OPT_MONTH_CODES[month - 1]
        year_offset = max(1 if fut_month_code < opt_month_code else 0, asset_min_year_offset[asset_idx])
        hedge1t_vec[j] = asset_fut_code[asset_idx] + fut_month_code + str(year + year_offset) + " " + asset_market_code[asset_idx]
        hedge1f_vec[j] = "PX_LAST"
    # CDS is vectorized above
    elif hedge_source == HEDGE_CALC:
        ccy = asset_calc_ccy[asset_idx]
        tenor = asset_calc_tenor[asset_idx]
        hedge1t_vec[j] = ccy + "_fsw0m_" + str(tenor)
        hedge2t_vec[j] = ccy + "_fsw6m_" + str(tenor)
        hedge3t_vec[j] = ccy + "_pva0m_" + str(tenor)
        hedge4t_vec[j] = ccy + "_pva6m_" + str(tenor)

    # Day count is now computed vectorized above

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({year_vec.shape[0]:,} straddles)")

# ============================================================================
# OUTPUT
# ============================================================================

result = {
    "orientation": "numpy",
    "columns": [
        "year",
        "month",
        "asset",
        "schcnt", "schid",
        "ntrc", "ntrv", "xprc", "xprv", "wgt",
        "hedge1t", "hedge1f",
        "hedge2t", "hedge2f",
        "hedge3t", "hedge3f",
        "hedge4t", "hedge4f",
        "volt", "volf",
        "days0", "days1", "days2",
        "day_count",
    ],
    "rows": [
        year_vec,
        month_vec,
        asset_vec,
        schcnt_vec,
        schid_vec,
        ntrc_vec,
        ntrv_vec,
        xprc_vec,
        xprv_vec,
        wgt_vec,
        hedge1t_vec,
        hedge1f_vec,
        hedge2t_vec,
        hedge2f_vec,
        hedge3t_vec,
        hedge3f_vec,
        hedge4t_vec,
        hedge4f_vec,
        volt_vec,
        volf_vec,
        days0_vec, days1_vec, days2_vec,
        day_count_vec,
    ]
}
