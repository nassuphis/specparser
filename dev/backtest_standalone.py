import numpy as np
from numba import njit, types
import time
from pathlib import Path
from typing import Any
import yaml
import pyarrow.parquet as pq
import pyarrow.compute as pc

script_start_time = time.perf_counter()
print("loading yaml".ljust(20, "."), end="")
start_time = time.perf_counter()
amt_resolved = str(Path("data/amt.yml").resolve())

try: 
    from yaml import CSafeLoader as Loader
except ImportError: 
    from yaml import SafeLoader as Loader


with open(amt_resolved, "r") as f:
    run_options = yaml.load(f, Loader=Loader)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
print("processing amt".ljust(20, "."), end="")
start_time = time.perf_counter()

amt = run_options.get("amt", {})

expiry_schedules = run_options.get("expiry_schedules")

#with open("data/amt.yml","r") as f: amt=yaml.safe_load(f)["amt"]
amap: dict[str, dict[str, Any]] = {}

for asset_data in amt.values():
    if isinstance(asset_data, dict):
        underlying = asset_data.get("Underlying")
        if underlying and asset_data.get("WeightCap")>0:
            amap[underlying] = asset_data

anames = np.array(list(amap.keys()),dtype=np.dtypes.StringDType())
idx_map = dict(zip(list(amap.keys()),range(len(anames))))

# cache all straddle-related data into numpy arrays for vectorized access
nps = np.dtypes.StringDType()

# some info is converted into integers for speed
hedge_source = np.array(list(map(lambda a:amap[a]["Hedge"].get("Source",""),anames)),dtype=nps)
hedge_sources, hedge_source_id = np.unique(hedge_source,return_inverse=True)
hs2id_map = dict(zip(hedge_sources,range(len(hedge_sources)))) # integer codes for speed
HEDGE_FUT = hs2id_map["fut"]
hedge_source_id_fut = hedge_source_id == HEDGE_FUT
HEDGE_NONFUT = hs2id_map["nonfut"]
hedge_source_id_nonfut = hedge_source_id == HEDGE_NONFUT
HEDGE_CDS = hs2id_map["cds"]
hedge_source_id_cds = hedge_source_id == HEDGE_CDS
HEDGE_CALC = hs2id_map["calc"]
hedge_source_id_calc = hedge_source_id == HEDGE_CALC

hedge_ticker = np.array(list(map(lambda a:amap[a]["Hedge"].get("Ticker",""),anames)),dtype=nps)
hedge_field = np.array(list(map(lambda a:amap[a]["Hedge"].get("Field",""),anames)),dtype=nps)
hedge_hedge = np.array(list(map(lambda a:amap[a]["Hedge"].get("hedge",""),anames)),dtype=nps)
hedge_hedge1 = np.array(list(map(lambda a:amap[a]["Hedge"].get("hedge1",""),anames)),dtype=nps)

# calc-type hedges are fixed concats, do here
hedge_ccy = np.array(list(map(lambda a:amap[a]["Hedge"].get("ccy",""),anames)),dtype=nps)
hedge_tenor = np.array(list(map(lambda a:amap[a]["Hedge"].get("tenor",""),anames)),dtype=nps)
calc_hedge1 = hedge_ccy + "_fsw0m_"+ hedge_tenor
calc_hedge2 = hedge_ccy + "_fsw6m_"+ hedge_tenor
calc_hedge3 = hedge_ccy + "_pva0m_"+ hedge_tenor
calc_hedge4 = hedge_ccy + "_pva6m_"+ hedge_tenor

# matrix of assets x months -> month_codes
hedge_fut_month_map = np.array(list(map(lambda a:amap[a]["Hedge"].get("fut_month_map"," "*12),anames)),dtype=nps)
hedge_fut_month_mtrx = hedge_fut_month_map.astype('S12').view('S1').reshape(-1,12).astype('U1')

hedge_min_year_offset = np.array(list(map(lambda a:amap[a]["Hedge"].get("min_year_offset","0"),anames)),dtype=nps)
hedge_min_year_offset_int = hedge_min_year_offset.astype(np.int64) # this is used as a number
hedge_fut_code = np.array(list(map(lambda a:amap[a]["Hedge"].get("fut_code",""),anames)),dtype=nps)
hedge_market_code = np.array(list(map(lambda a:amap[a]["Hedge"].get("market_code",""),anames)),dtype=nps)

vol_source = np.array(list(map(lambda a:amap[a]["Vol"].get("Source",""),anames)),dtype=nps)
vol_sources, vol_source_id = np.unique(vol_source,return_inverse=True)
vs2id_map = dict(zip(vol_sources,range(len(vol_sources))))
VOL_BBG_LMEVOL = vs2id_map["BBG_LMEVOL"]
vol_source_id_bbg_lmevol = vol_source_id == VOL_BBG_LMEVOL
VOL_BBG = vs2id_map["BBG"]
vol_source_id_bbg = vol_source_id == VOL_BBG
VOL_CV = vs2id_map["CV"]
vol_source_id_cv = vol_source_id == VOL_CV

vol_ticker = np.array(list(map(lambda a:amap[a]["Vol"].get("Ticker",""),anames)),dtype=nps)
vol_near = np.array(list(map(lambda a:amap[a]["Vol"].get("Near",""),anames)),dtype=nps)
vol_far = np.array(list(map(lambda a:amap[a]["Vol"].get("Far",""),anames)),dtype=nps)

# checksum of asset names
achk = np.array([np.sum(np.frombuffer(x.encode('ascii'),dtype=np.uint8)) for x in anames],dtype=np.int64)

# schedule count
aschcnt = np.array(list(map(
    lambda a:len(expiry_schedules[amap[a]["Options"]]),
    anames
)),dtype=np.int64)

# expanded count
easchcnt = np.repeat(aschcnt,aschcnt)
# expanded schedules
eastmp = np.concatenate(list(map(
    lambda a:np.array(expiry_schedules[amap[a]["Options"]],dtype="|U20"),
    anames
)),dtype="|U20")

schedule_matrix = np.full((len(amap),np.max(easchcnt),5),"",dtype=np.dtypes.StringDType())
easchj = np.arange(np.sum(aschcnt))-np.repeat(np.cumsum(aschcnt)-aschcnt,aschcnt)
easchi = np.repeat(np.arange(len(anames)),aschcnt)

#ea_ntrcv,_,rest = np.strings.partition(eastmp,'_')
easntrcv, _, rest = np.strings.partition(eastmp,'_')
schedule_matrix[easchi,easchj,0] = np.strings.slice(easntrcv,1)
schedule_matrix[easchi,easchj,1] = np.strings.slice(easntrcv,1,20)
easxprcv, _, rest = np.strings.partition(rest,'_')
conds=[
    easxprcv=="OVERRIDE",
    (np.strings.slice(easxprcv,2)=="BD") & np.isin(np.strings.slice(easxprcv,2,20),["a","b","c","d"]),
    np.strings.slice(easxprcv,2)=="BD"
]
choices_xprc=[
    easxprcv,
    "BD",
    "BD"
]
choices_xprv=[
    "",
    (easchj  * (20 // easchcnt + 1) + achk[easchi] % 5 + 1).astype("U"),
    np.strings.slice(easxprcv,2,20)
]
schedule_matrix[easchi,easchj,2]=np.select(conds,choices_xprc,default=np.strings.slice(easxprcv,1))
schedule_matrix[easchi,easchj,3]=np.select(conds,choices_xprv,default=np.strings.slice(easxprcv,1,20))
schedule_matrix[easchi,easchj,4], _, rest = np.strings.partition(rest,'_')
uniq_schedules, schedule_id_flat = np.unique(schedule_matrix, return_inverse=True)
sch2id_map = dict(zip(uniq_schedules,range(len(uniq_schedules))))
schedule_id_matrix = schedule_id_flat.reshape(schedule_matrix.shape)
STR_OVERRIDE = sch2id_map["OVERRIDE"]
STR_N = sch2id_map["N"]
STR_F = sch2id_map["F"]
STR_R = sch2id_map["R"]
STR_W = sch2id_map["W"]
STR_BD = sch2id_map["BD"]

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
# ============================================================================
# compute straddles
# ============================================================================
print("compute straddles".ljust(20, "."), end="")
start_time = time.perf_counter()

# months
ym = np.arange(2001*12+1-1,2026*12+1-1,dtype=np.int64)
ym_len = len(ym)


# inas: input asset
inas = anames
inalen = len(inas)
# inidx : asset-count-length numpy array of index into numpy matrices with info
inidx = np.array([idx_map[a] for a in inas],dtype=np.uint64)
# inasc : input asset straddle count. # straddles by asset
inasc = aschcnt[inidx] # lengths
sidx = np.repeat(inidx,inasc)  # by straddle asset index
sc  = np.repeat(inasc,inasc)   # by straddle strad count for this index
si1 = np.arange(np.sum(inasc)) 
si2 = np.repeat((np.cumsum(inasc)-inasc),inasc)
si  = si1 - si2                # straddle id within this asset


# fancy indexing for the whole loop, the straddle-month-loop
smidx  = np.repeat(sidx,ym_len)
smym   = np.tile(ym,len(sidx))
smlen  = len(smidx)

# asset, year, month
asset_vec  =  inas[smidx]
year_vec   =  smym //12
month_vec  =  smym % 12 + 1

# straddle days
dpm = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],dtype=np.int8)
leap_feb = np.full(len(smidx),29,dtype=np.int8)

year0, month0 = year_vec, month_vec
leap0 = ( ( (year0 % 4 == 0) & ( (year0 % 100 != 0) | (year0 % 400 == 0)))) & (month0==2)
days0_vec = np.where(leap0,leap_feb,dpm[month0 - 1])

year1, month1 = (year_vec*12+(month_vec-1)-1) // 12, (year_vec*12+(month_vec-1)-1) % 12 + 1
leap1 = (( (year1 % 4 == 0) & ( (year1 % 100 != 0) | (year1 % 400 == 0)))) & (month1==2)
days1_vec = np.where(leap1,leap_feb,dpm[month1 - 1])

year2, month2 = (year_vec*12+(month_vec-1)-2) // 12, (year_vec*12+(month_vec-1)-2) % 12 + 1
leap2 = (( (year2 % 4 == 0) & ((year2 % 100 != 0 ) | ( year2 % 400 == 0)))) & (month2==2)
days2_vec = np.where(leap2,leap_feb,dpm[month2 - 1])

# straddle 
schcnt_vec = np.repeat(sc,ym_len)
schid_vec  = np.repeat(si,ym_len)
schedule_matrix_smidx = schedule_matrix[smidx,schid_vec,:]
ntrc_vec   = schedule_matrix_smidx[:,0]
ntrv_vec   = schedule_matrix_smidx[:,1]
xprc_vec   = schedule_matrix_smidx[:,2]
xprv_vec   = schedule_matrix_smidx[:,3]
wgt_vec    = schedule_matrix_smidx[:,4]

schedule_id_matrix_smidx = schedule_id_matrix[smidx,schid_vec,:]
ntrc_id_vec   = schedule_id_matrix_smidx[:,0]
ntrv_id_vec   = schedule_id_matrix_smidx[:,1]
xprc_id_vec   = schedule_id_matrix_smidx[:,2]
xprv_id_vec   = schedule_id_matrix_smidx[:,3]

# total day-count
day_count_vec = days0_vec + days1_vec + np.where(ntrc_id_vec==STR_F,days2_vec,0)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
# ============================================================================
# straddle hedge tickers
# ============================================================================
print("hedge tickers".ljust(20, "."), end="")
start_time = time.perf_counter()

hedge_source_id_nonfut_smidx = hedge_source_id_nonfut[smidx]
hedge_source_id_fut_smidx    = hedge_source_id_fut[smidx]
hedge_source_id_cds_smidx    = hedge_source_id_cds[smidx]
hedge_source_id_calc_smidx   = hedge_source_id_calc[smidx]

cond_hedge = [
    hedge_source_id_nonfut_smidx, 
    hedge_source_id_fut_smidx, 
    hedge_source_id_cds_smidx, 
    hedge_source_id_calc_smidx
]

# hedge ticker for nonfut
hedge_ticker_smidx = hedge_ticker[smidx]

# future calc for fut
#
# TODO: this calculation is expensive and only applied to futures
# use mask np.flatnozero(hedge_source_id_smidx == HEDGE_FUT)
#

hedge_fut_code_smidx = hedge_fut_code[smidx]
hedge_fut_month_code_smidx = hedge_fut_month_mtrx[smidx,month_vec-1]
month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
hedge_opt_month_code_smidx = month_code[month_vec-1]

myo_smidx = hedge_min_year_offset_int[smidx]
yo_smidx = np.maximum(np.where(hedge_fut_month_code_smidx < hedge_opt_month_code_smidx,1,0),myo_smidx)

hedge_fut_yeartxt_smidx = (year_vec+yo_smidx).astype("U")
hedge_fut_tail_smidx = hedge_fut_month_code_smidx + hedge_fut_yeartxt_smidx + " " + hedge_market_code[smidx]


hedge_fut_ticker = hedge_fut_code_smidx+hedge_fut_tail_smidx

# hedge from cds, hedge1 from calc
hedge_hedge_smidx = hedge_hedge[smidx]
calc_hedge1_smidx = calc_hedge1[smidx]

choices_hedge1t = [ hedge_ticker_smidx, hedge_fut_ticker, hedge_hedge_smidx, calc_hedge1_smidx ]
choices_hedge1f = [ hedge_field[smidx], "PX_LAST", "PX_LAST", "" ]
hedge1t_vec = np.select(cond_hedge,choices_hedge1t,default="")
hedge1f_vec = np.select(cond_hedge,choices_hedge1f,default="")

choices_hedge2t = ["","",hedge_hedge1[smidx],calc_hedge2[smidx]]
choices_hedge2f = ["","","PX_LAST",""]
hedge2t_vec = np.select(cond_hedge,choices_hedge1f,default="")
hedge2f_vec = np.select(cond_hedge,choices_hedge2f,default="")

choices_hedge3t = ["","","",calc_hedge3[smidx]]
#choices_hedge3f = ["","","",""]
hedge3t_vec = np.select(cond_hedge,choices_hedge3t,default="")
hedge3f_vec = np.full(len(hedge3t_vec),"",dtype="U")

#choices_hedge4t = ["","","",calc_hedge4[smidx]]
#choices_hedge4f = ["","","",""]
hedge4t_vec = np.where(cond_hedge[3],calc_hedge4[smidx],"")
hedge4f_vec = hedge3f_vec

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
# ============================================================================
# straddle vol tickers (select)
# ============================================================================
print("vol tickers (select)".ljust(20, "."), end="")
start_time = time.perf_counter()

vol_source_id_smidx = vol_source_id[smidx]
vol_source_id_bbg_smidx = vol_source_id_bbg[smidx]
vol_source_id_bbg_lmevol_smidx = vol_source_id_bbg_lmevol[smidx]
vol_source_id_bbg_cv_smidx = vol_source_id_cv[smidx]

cond_vol = [
    ( vol_source_id_bbg_smidx ) & ( ntrc_id_vec==STR_N ),
    ( vol_source_id_bbg_smidx ) & ( ntrc_id_vec==STR_F ),
    ( vol_source_id_bbg_lmevol_smidx ),
    ( vol_source_id_bbg_cv_smidx )
]

vol_ticker_smidx = vol_ticker[smidx]
vol_near_smidx   = vol_near[smidx]
vol_far_smidx    = vol_far[smidx]
vol_tkrz_smidx   = hedge_fut_code_smidx + "R" + hedge_fut_tail_smidx

choices_volt = [vol_ticker_smidx,vol_ticker_smidx,vol_tkrz_smidx,vol_near_smidx]
volt_vec      = np.select(cond_vol,choices_volt,default="")

choices_volf = [vol_near_smidx,vol_far_smidx,"PX_LAST","none"]
volf_vec      = np.select(cond_vol,choices_volf,default="")


print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
# ============================================================================
# straddle vol tickers (conditional)
# ============================================================================
print("vol tickers (cond)".ljust(20, "."), end="")
start_time = time.perf_counter()
# was slower then select versions
"""
idx1  = np.flatnonzero( ( vol_source_id_bbg_smidx ) & ( ntrc_id_vec==STR_N ) )
idx2  = np.flatnonzero( ( vol_source_id_bbg_smidx ) & ( ntrc_id_vec==STR_F ) )
idx3  = np.flatnonzero( vol_source_id_bbg_lmevol_smidx ) 
idx4  = np.flatnonzero( vol_source_id_bbg_cv_smidx  )

vol_ticker_smidx = vol_ticker[smidx]
vol_near_smidx   = vol_near[smidx]
vol_far_smidx    = vol_far[smidx]

volt_vec = np.full(len(smidx),"",dtype=nps)
volf_vec = np.full(len(smidx),"",dtype=nps)

volt_vec[idx1] = vol_ticker_smidx[idx1]
volf_vec[idx1] = vol_near_smidx[idx1]
volt_vec[idx2] = vol_ticker_smidx[idx2]
volf_vec[idx2] = vol_far_smidx[idx2]

volt_vec[idx3] = hedge_fut_code_smidx[idx3] + "R" + hedge_fut_tail_smidx[idx3]
volf_vec[idx3] = "PX_LAST"

volt_vec[idx4] = vol_near_smidx[idx4]
volf_vec[idx4] = "none"
"""

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
# ============================================================================
# precumpute days
# ============================================================================
print("precompute days".ljust(20, "."), end="")
start_time = time.perf_counter()

# Precompute days-since-epoch lookup for all year-months (2000-01 to 2026-12)
# This avoids slow datetime64 string parsing
_ym_base = 2000 * 12  # base year-month
_ym_range = np.arange(2000*12, 2027*12)  # year-months as integers
_ym_dates = (
    (_ym_range // 12).astype('U') + '-' +
    np.char.zfill((_ym_range % 12 + 1).astype('U'), 2) + '-01'
).astype('datetime64[D]')
_ym_epoch = _ym_dates.astype(np.int64)  # days since 1970-01-01

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
# ============================================================================
# expand to daily calendar
# ============================================================================
print("expand days".ljust(20, "."), end="")
start_time = time.perf_counter()


# Start year-month for each straddle (as integer year*12 + month-1)
d_start_ym = np.where(ntrc_vec == "F", year2 * 12 + month2 - 1, year1 * 12 + month1 - 1)

# Day index within each straddle (0, 1, 2, ..., day_count-1)
total_days = np.sum(day_count_vec)
di = np.arange(total_days) - np.repeat(np.cumsum(day_count_vec) - day_count_vec, day_count_vec)

# Straddle index for each day (index into the 222K monthly straddle vectors)
d_stridx = np.repeat(np.arange(len(day_count_vec)), day_count_vec)

# Asset index for each day (index into asset-level arrays like anames, hedge_ticker, etc)
d_smidx = np.repeat(smidx, day_count_vec)

# Integer arrays - fast
d_schid = np.repeat(schid_vec, day_count_vec)

# String arrays: DON'T copy - use d_stridx to index into monthly vectors when needed
# e.g., asset_vec[d_stridx], hedge1t_vec[d_stridx], volt_vec[d_stridx]

# Compute days-since-epoch using lookup table (fast integer indexing)
d_epoch = _ym_epoch[d_start_ym[d_stridx] - _ym_base] + di

# Skip Y/M/D extraction - use d_epoch directly for joining with prices
# The prices file has date as datetime64[D], convert to epoch days for O(1) lookup

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({len(list(amap.keys()))} assets)")
# ============================================================================
# load price data
# ============================================================================
print("load prices".ljust(20, "."), end="")
start_time = time.perf_counter()
px = pq.read_table("data/prices_sorted.parquet")

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({px.num_rows:,} rows read)")
# ============================================================================
# convert date to epoch days for fast joining
# ============================================================================
print("extract epoch".ljust(20, "."), end="")
start_time = time.perf_counter()
px_epoch = px.column('date').to_numpy().astype('datetime64[D]').astype(np.int64)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms (total day-straddles: {total_days:,})")
print("-"*40)
print(f"total: {1e3*(time.perf_counter()-script_start_time):0.3f}ms")
# ============================================================================
# show results
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
        "days0","days1","days2",
        "day_count",
    ],
    "rows": [
        year_vec,
        month_vec,
        asset_vec,
        schcnt_vec,
        schid_vec ,
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
        days0_vec,days1_vec,days2_vec,
        day_count_vec,
    ]

}

