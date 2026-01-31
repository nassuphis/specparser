import numpy as np
from numba import njit, types
import time
from pathlib import Path
from typing import Any
import yaml
import pyarrow.parquet as pq
import pyarrow.compute as pc


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
nps = np.dtypes.StringDType()
hedge_source = np.array(list(map(lambda a:amap[a]["Hedge"].get("Source",""),anames)),dtype=nps)
hedge_ticker = np.array(list(map(lambda a:amap[a]["Hedge"].get("Ticker",""),anames)),dtype=nps)
hedge_field = np.array(list(map(lambda a:amap[a]["Hedge"].get("Field",""),anames)),dtype=nps)
hedge_hedge = np.array(list(map(lambda a:amap[a]["Hedge"].get("hedge",""),anames)),dtype=nps)
hedge_hedge1 = np.array(list(map(lambda a:amap[a]["Hedge"].get("hedge1",""),anames)),dtype=nps)
hedge_hedge2 = np.array(list(map(lambda a:amap[a]["Hedge"].get("hedge2",""),anames)),dtype=nps)
hedge_hedge3 = np.array(list(map(lambda a:amap[a]["Hedge"].get("hedge3",""),anames)),dtype=nps)
hedge_hedge4 = np.array(list(map(lambda a:amap[a]["Hedge"].get("hedge4",""),anames)),dtype=nps)
hedge_ccy = np.array(list(map(lambda a:amap[a]["Hedge"].get("ccy",""),anames)),dtype=nps)
hedge_tenor = np.array(list(map(lambda a:amap[a]["Hedge"].get("tenor",""),anames)),dtype=nps)
hedge_fut_month_map = np.array(list(map(lambda a:amap[a]["Hedge"].get("fut_month_map"," "*12),anames)),dtype=nps)
hedge_min_year_offset = np.array(list(map(lambda a:amap[a]["Hedge"].get("min_year_offset","0"),anames)),dtype=nps)
hedge_fut_code = np.array(list(map(lambda a:amap[a]["Hedge"].get("fut_code",""),anames)),dtype=nps)
hedge_market_code = np.array(list(map(lambda a:amap[a]["Hedge"].get("market_code",""),anames)),dtype=nps)
vol_source = np.array(list(map(lambda a:amap[a]["Vol"].get("Source",""),anames)),dtype=nps)
vol_ticker = np.array(list(map(lambda a:amap[a]["Vol"].get("Ticker",""),anames)),dtype=nps)
vol_near = np.array(list(map(lambda a:amap[a]["Vol"].get("Near",""),anames)),dtype=nps)
vol_far = np.array(list(map(lambda a:amap[a]["Vol"].get("Far",""),anames)),dtype=nps)
achk = np.array([np.sum(np.frombuffer(x.encode('ascii'),dtype=np.uint8)) for x in anames],dtype=np.int64)
aschnam = np.array(list(map(lambda a:amap[a]["Options"],anames)),dtype=nps)
aschlen = np.array(list(map(
    lambda a:len(expiry_schedules[amap[a]["Options"]]),
    anames
)),dtype=np.int64)
easchcnt = np.repeat(aschlen,aschlen)
eastmp = np.concatenate(list(map(
    lambda a:np.array(expiry_schedules[amap[a]["Options"]],dtype="|U20"),
    anames
)),dtype="|U20")

schedule_matrix = np.full((len(amap),np.max(easchcnt),5),"",dtype=np.dtypes.StringDType())
easchj = np.arange(np.sum(aschlen))-np.repeat(np.cumsum(aschlen)-aschlen,aschlen)
easchi = np.repeat(np.arange(len(anames)),aschlen)

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

# encode strings as integers for speed
hedge_sources = list(set(hedge_source))
hs2id_map = dict(zip(hedge_sources,range(len(hedge_sources))))
id2hs_map = dict(zip(range(len(hedge_sources)),hedge_sources))
vol_sources = list(set(vol_source))
vs2id_map = dict(zip(vol_sources,range(len(vol_sources))))
id2vs_map = dict(zip(range(len(vol_sources)),vol_sources))
VOL_BBG_LMEVOL = vs2id_map["BBG_LMEVOL"]
VOL_BBG        = vs2id_map["BBG"]
VOL_CV         = vs2id_map["CV"]
HEDGE_FUT      = hs2id_map["fut"]
HEDGE_NONFUT   = hs2id_map["nonfut"]
HEDGE_FUT      = hs2id_map["fut"]
HEDGE_CDS      = hs2id_map["cds"]
HEDGE_CALC     = hs2id_map["calc"]
# dim0: asset, dim1: 0=hedge,1=vol, values as above
asset_sources = np.full((len(amap),2),0,dtype=np.uint32) 

asset_hedge_tickers = np.full((len(amap),8),"",dtype=np.dtypes.StringDType()) 
asset_vol_tickers = np.full((len(amap),3),"",dtype=np.dtypes.StringDType()) 
asset_ids = np.array([np.sum(np.frombuffer(x.encode('ascii'),dtype=np.uint8)) for x in anames],dtype=np.uint64)

for idx, asset in enumerate(amap.values()):
    underlying = asset.get("Underlying")
    # assets_ids
    # asset_sources
    asset_sources[idx,0] = hs2id_map[hedge_source[idx]] # hedge source
    asset_sources[idx,1] = vs2id_map[vol_source[idx]] # vol source
    # asset hedge tickers, fields
    if hedge_source[idx]=="nonfut":
        asset_hedge_tickers[idx,0] = hedge_ticker[idx]
        asset_hedge_tickers[idx,1] = hedge_field[idx]
    if hedge_source[idx]=="cds":
        asset_hedge_tickers[idx,2] = hedge_hedge[idx]
        asset_hedge_tickers[idx,3] = hedge_hedge1[idx]
    if hedge_source[idx]=="calc":
        asset_hedge_tickers[idx,4] = hedge_ccy[idx]+"_fsw0m_"+hedge_tenor[idx]
        asset_hedge_tickers[idx,5] = hedge_ccy[idx]+"_fsw6m_"+hedge_tenor[idx]
        asset_hedge_tickers[idx,6] = hedge_ccy[idx]+"_pva0m_"+hedge_tenor[idx]
        asset_hedge_tickers[idx,7] = hedge_ccy[idx]+"_pva6m_"+hedge_tenor[idx]   
    # asset vol tickers, fields
    asset_vol_tickers[idx,0] = vol_ticker[idx]
    asset_vol_tickers[idx,1] = vol_near[idx]
    asset_vol_tickers[idx,2] = vol_far[idx]
    # fill in schedule matrix, length
 

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({len(list(amap.keys()))} assets)")
#print(hedge_sources)
#print(vol_sources)
#print(schedule_matrix[idx_map["EURUSD Curncy"],:,:])
#print(hedge_ticker[idx_map["EURUSD Curncy"]])
#print(vol_ticker[idx_map["EURUSD Curncy"]])
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
print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ============================================================================
# allocate straddles
# ============================================================================
print("compute straddles".ljust(20, "."), end="")
start_time = time.perf_counter()

# months
ym = np.arange(2001*12+1-1,2026*12+1-1,dtype=np.int64)
ym_len = len(ym)


# inas: input asset
inas = anames
inalen = len(inas)
# asset_idxs : asset-count-length numpy array of index into numpy matrices with info
inidx = np.array([idx_map[a] for a in inas],dtype=np.uint64)
# inasc : input asset straddle count. # straddles by asset
inasc = aschlen[inidx] # lengths
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
year_vec =  smym //12
month_vec = smym % 12 + 1

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
ntrc_vec   = schedule_matrix[smidx,schid_vec,0]
ntrv_vec   = schedule_matrix[smidx,schid_vec,1]
xprc_vec   = schedule_matrix[smidx,schid_vec,2]
xprv_vec   = schedule_matrix[smidx,schid_vec,3]
wgt_vec    = schedule_matrix[smidx,schid_vec,4]

# total day-count
day_count_vec = days0_vec + days1_vec + np.where(ntrc_vec=="F",days2_vec,0)

cond_hedge = [
    asset_sources[smidx,0]==HEDGE_NONFUT,
    asset_sources[smidx,0]==HEDGE_FUT,
    asset_sources[smidx,0]==HEDGE_CDS,
    asset_sources[smidx,0]==HEDGE_CALC
]

fut_month_code = np.strings.slice(hedge_fut_month_map[smidx],month_vec - 1,month_vec)
opt_month_code = np.strings.slice("FGHJKMNQUVXZ",month_vec - 1,month_vec)
myo = (hedge_min_year_offset[smidx]).astype(np.int64)
yo = np.maximum(np.where(fut_month_code < opt_month_code,1,0),myo)

choices_hedge1t = [
    hedge_ticker[smidx],
    hedge_fut_code[smidx]+fut_month_code[smidx]+(year_vec+yo).astype("U")+" " + hedge_market_code[smidx],
    hedge_hedge[smidx],
    hedge_ccy[smidx]+"_fsw0m_"+hedge_tenor[smidx]
]
hedge1t_vec = np.select(cond_hedge,choices_hedge1t,default="")

choices_hedge1f = [
    hedge_field[smidx],
    "PX_LAST",
    "PX_LAST",
    ""
]
hedge1f_vec = np.select(cond_hedge,choices_hedge1f,default="")

choices_hedge2t = ["","",hedge_hedge1[smidx],hedge_ccy[smidx]+"_fsw6m_"+hedge_tenor[smidx]]
hedge2t_vec = np.select(cond_hedge,choices_hedge1f,default="")

choices_hedge2f = ["","","PX_LAST",""]
hedge2f_vec = np.select(cond_hedge,choices_hedge2f,default="")

choices_hedge3t = ["","","",hedge_ccy[smidx]+"_pva0m_"+hedge_tenor[smidx]]
hedge3t_vec = np.select(cond_hedge,choices_hedge3t,default="")

choices_hedge3f = ["","","",""]
hedge3f_vec = np.select(cond_hedge,choices_hedge3f,default="")

choices_hedge4t = ["","","",hedge_ccy[smidx]+"_pva6m_"+hedge_tenor[smidx]]
hedge4t_vec = np.select(cond_hedge,choices_hedge4t,default="")

choices_hedge4f = ["","","",""]
hedge4f_vec = np.select(cond_hedge,choices_hedge4f,default="")

cond_vol = [
    ( asset_sources[smidx,1]==VOL_BBG ) & ( ntrc_vec=="N" ),
    ( asset_sources[smidx,1]==VOL_BBG ) & ( ntrc_vec=="F" ),
    ( asset_sources[smidx,1]==VOL_BBG_LMEVOL ),
    ( asset_sources[smidx,1]==VOL_CV )
]

choices_volt = [
    vol_ticker[smidx],
    vol_ticker[smidx],
    hedge_fut_code[smidx]+"R"+fut_month_code[smidx]+(year_vec+yo).astype("U")+" " + hedge_market_code[smidx],
    vol_near[smidx]
]
choices_volf = [
    vol_near[smidx],
    vol_far[smidx],
    "PX_LAST",
    "none"
]

volt_vec      = np.select(cond_vol,choices_volt,default="")
volf_vec      = np.select(cond_vol,choices_volf,default="")


print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
#print(f"{asset_vec.shape[0]:,} assets")
#print(f"{year_vec.shape[0]:,} years")
#print(f"{month_vec.shape[0]:,} years")
#print(f" monthly straddle_count: {np.sum(aschlen[inidx])}")
#print(f"   total straddle_count: {len(smidx):,}")

# ============================================================================
# expand to daily calendar
# ============================================================================
print("expand to days".ljust(20, "."), end="")
start_time = time.perf_counter()

# Precompute days-since-epoch lookup for all year-months (2000-01 to 2026-12)
# This avoids slow datetime64 string parsing
_ym_base = 2000 * 12  # base year-month
_ym_range = np.arange(2000*12, 2027*12)  # year-months as integers
_ym_dates = ((_ym_range // 12).astype('U') + '-' +
             np.char.zfill((_ym_range % 12 + 1).astype('U'), 2) + '-01').astype('datetime64[D]')
_ym_epoch = _ym_dates.astype(np.int64)  # days since 1970-01-01

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

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms (total day-straddles: {total_days:,})")

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

