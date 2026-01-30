import numpy as np
from numba import njit, types
from numba.typed import Dict as ndict
import time
from pathlib import Path
from typing import Any
import yaml

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader


import pyarrow.parquet as pq
import pyarrow.compute as pc

# Import loader for real data experiments
from specparser.amt import loader, table


print("loading yaml........", end="")
start_time = time.perf_counter()

# Get asset data for a specific asset
_DAYS_PER_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
amt_resolved = (Path(__file__).resolve().parent.parent / "data" / "amt.yml")
with open(amt_resolved, "r") as f: run_options = yaml.load(f, Loader=Loader)
amt = run_options.get("amt", {})
expiry_schedules = run_options.get("expiry_schedules")
underlying_map: dict[str, dict[str, Any]] = {}

for asset_data in amt.values():
    if isinstance(asset_data, dict):
        underlying = asset_data.get("Underlying")
        if underlying and asset_data.get("WeightCap")>0:
            underlying_map[underlying] = asset_data

hedge_sources = list(set([ x["Hedge"]["Source"]  for x in underlying_map.values()]))
hs2id_map = dict(zip(hedge_sources,range(len(hedge_sources))))
id2hs_map = dict(zip(range(len(hedge_sources)),hedge_sources))
vol_sources = list(set([ x["Vol"]["Source"]  for x in underlying_map.values()]))
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
asset_sources = np.full((len(underlying_map),2),0,dtype=np.uint32) 

asset_hedge_tickers = np.full((len(underlying_map),8),"",dtype=np.dtypes.StringDType()) 
asset_vol_tickers = np.full((len(underlying_map),3),"",dtype=np.dtypes.StringDType()) 

asset_name = np.array(list(underlying_map.keys()),dtype=np.dtypes.StringDType())
idx_map = dict(zip(asset_name.tolist(),range(len(asset_name))))

schedule_matrix = np.full((len(underlying_map),4,5),"",dtype=np.dtypes.StringDType())
schedule_length = np.full(len(underlying_map),0,dtype=np.uint8)
asset_ids = np.full(len(underlying_map),0,dtype=np.uint32)

for idx, asset in enumerate(underlying_map.values()):
    underlying = asset.get("Underlying")
    # assets_ids
    asset_ids[idx] = np.sum(np.frombuffer(underlying.encode('ascii'),dtype=np.uint8))
    # asset_sources
    asset_sources[idx,0] = hs2id_map[asset["Hedge"]["Source"]] # hedge source
    asset_sources[idx,1] = vs2id_map[asset["Vol"]["Source"]] # vol source
    # asset hedge tickers, fields
    if asset["Hedge"]["Source"]=="nonfut":
        asset_hedge_tickers[idx,0] = asset["Hedge"].get("Ticker","")
        asset_hedge_tickers[idx,1] = asset["Hedge"].get("Field","")
    if asset["Hedge"]["Source"]=="cds":
        asset_hedge_tickers[idx,2] = asset["Hedge"].get("hedge","")
        asset_hedge_tickers[idx,3] = asset["Hedge"].get("hedge1","")
    if asset["Hedge"]["Source"]=="calc":
        asset_hedge_tickers[idx,4] = asset["Hedge"]['ccy']+"_fsw0m_"+str(asset["Hedge"]['tenor'])
        asset_hedge_tickers[idx,5] = asset["Hedge"]['ccy']+"_fsw6m_"+str(asset["Hedge"]['tenor'])
        asset_hedge_tickers[idx,6] = asset["Hedge"]['ccy']+"_pva0m_"+str(asset["Hedge"]['tenor'])
        asset_hedge_tickers[idx,7] = asset["Hedge"]['ccy']+"_pva6m_"+str(asset["Hedge"]['tenor'])   
    # asset vol tickers, fields
    asset_vol_tickers[idx,0] = asset["Vol"].get("Ticker","")
    asset_vol_tickers[idx,1] = asset["Vol"].get("Near","")
    asset_vol_tickers[idx,2] = asset["Vol"].get("Far","")
    # fill in schedule matrix, length
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
                schedule_matrix[idx,i,3] = str( i * (20 // (len(underlying_schedules) + 1)) + asset_ids[idx] % 5 + 1 )
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
print(hedge_sources)
print(vol_sources)
print(schedule_matrix[idx_map["EURUSD Curncy"],:,:])
print(asset_hedge_tickers[idx_map["EURUSD Curncy"],:])
print(asset_vol_tickers[idx_map["EURUSD Curncy"],:])
print("load prices.........", end="")
start_time = time.perf_counter()

table = pq.read_table("data/prices_sorted.parquet")

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({table.num_rows:,} rows read)")
# ============================================================================
# allocate straddles
# ============================================================================
print("allocate straddles..", end="")
start_time = time.perf_counter()

# months
year_months = np.arange(2001*12+1-1,2026*12+1-1,dtype=np.int64)
month_count = len(year_months)


assets = np.array(list(underlying_map.keys()),dtype=np.dtypes.StringDType()) # arbitrary asset list
asset_idxs = np.array([idx_map[asset] for asset in assets],dtype=np.int64)
straddle_asset_idxs = np.repeat(asset_idxs,schedule_length[asset_idxs])

# fancy indexing for the whole loop
month_asset_straddle_idxs = np.repeat(straddle_asset_idxs,month_count)
straddle_count = len(month_asset_straddle_idxs)

#
asset_vec  = asset_name[month_asset_straddle_idxs]
year_vec = np.tile(year_months // 12,len(straddle_asset_idxs)).astype(np.int16)
month_vec = np.tile(year_months % 12 + 1,len(straddle_asset_idxs)).astype(np.int8)

# 
schcnt_vec = schedule_length[month_asset_straddle_idxs]
schid_vec  = np.full(straddle_count, np.uint8(0), dtype=np.int8)
ntrc_vec   = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
ntrv_vec   = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
xprc_vec   = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
xprv_vec   = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
wgt_vec    = np.full(straddle_count, "", dtype=np.dtypes.StringDType())

hedge1t_vec    = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge2t_vec   = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge3t_vec   = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge4t_vec   = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
volt_vec      = np.full(straddle_count, "", dtype=np.dtypes.StringDType())

hedge1f_vec    = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge2f_vec   = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge3f_vec   = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
hedge4f_vec   = np.full(straddle_count, "", dtype=np.dtypes.StringDType())
volf_vec      = np.full(straddle_count, "", dtype=np.dtypes.StringDType())


days0_vec = np.full(straddle_count, np.uint8(0), dtype=np.int8)
days1_vec = np.full(straddle_count, np.uint8(0), dtype=np.int8)
days2_vec = np.full(straddle_count, np.uint8(0), dtype=np.int8)
day_count_vec = np.full(straddle_count, np.uint8(0), dtype=np.int8)


print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
print(f"{asset_vec.shape[0]:,} assets")
print(f"{year_vec.shape[0]:,} years")
print(f"{month_vec.shape[0]:,} years")
print(f" monthly straddle_count: {np.sum(schedule_length[asset_idxs])}")
print(f"   total straddle_count: {straddle_count:,}")
# ============================================================================
# compute straddles
# ============================================================================
print("compute straddles...", end="")
start_time = time.perf_counter()

j=-1
for year_month in year_months:

    year, month =  year_month // 12, year_month % 12 + 1

    for asset in assets:

        asset_data = underlying_map[asset]
        idx = idx_map[asset]
        assid = asset_ids[idx]
        schcnt = schedule_length[idx]
        
        for i in range(schcnt):
            j=j+1
            schid_vec[j]  = i
            ntrc_vec[j] = schedule_matrix[idx,i,0]
            ntrv_vec[j] = schedule_matrix[idx,i,1]
            xprc_vec[j] = schedule_matrix[idx,i,2]
            xprv_vec[j] = schedule_matrix[idx,i,3]            
            wgt_vec[j]  = schedule_matrix[idx,i,4]

            hedge = asset_data["Hedge"]
            if hedge is None: continue
            vol = asset_data["Vol"]
            if vol is None: continue

            if ntrc_vec[i]=="F" and  asset_sources[idx,1]!=VOL_BBG_LMEVOL: 
                if asset_vol_tickers[idx,2]=="": continue
                if asset_vol_tickers[idx,1]==asset_vol_tickers[idx,2]: continue
                if asset_vol_tickers[idx,2]=="NONE": continue

            if asset_sources[idx,1]==VOL_BBG and ntrc_vec[j]=="N":
                volt_vec[j] = asset_vol_tickers[idx,0] #vol["Ticker"]
                volf_vec[j] = asset_vol_tickers[idx,1] #vol["Near"]

            if asset_sources[idx,1]==VOL_BBG and ntrc_vec[j]=="F":
                volt_vec[j] = asset_vol_tickers[idx,0] # vol["Ticker"]
                volf_vec[j] = asset_vol_tickers[idx,2] #vol["Far"]

            if asset_sources[idx,1]==VOL_CV :
                volt_vec[j] = asset_vol_tickers[idx,1] #vol["Near"]
                volf_vec[j] = "none"

            if asset_sources[idx,1]==VOL_BBG_LMEVOL:
                fut_month_code = hedge["fut_month_map"][month - 1]
                opt_month_code = "FGHJKMNQUVXZ"[month - 1]
                year_offset = max(1 if fut_month_code < opt_month_code else 0, hedge["min_year_offset"])
                volt_vec[j] = hedge["fut_code"]+"R"+fut_month_code+str(year + year_offset)+" " + hedge["market_code"] 
                volf_vec[j] = "PX_LAST"
            
            if asset_sources[idx,0]==HEDGE_NONFUT:
                hedge1t_vec[j] = asset_hedge_tickers[idx,0]
                hedge1f_vec[j] = asset_hedge_tickers[idx,1]

            elif asset_sources[idx,0]==HEDGE_FUT:
                fut_month_code = hedge["fut_month_map"][month - 1]
                opt_month_code = "FGHJKMNQUVXZ"[month - 1]
                year_offset = max(1 if fut_month_code < opt_month_code else 0, hedge["min_year_offset"])
                hedge1t_vec[j] = hedge["fut_code"]+fut_month_code+str(year + year_offset)+" " + hedge["market_code"] 
                hedge1f_vec[j] = "PX_LAST"
    
            elif asset_sources[idx,0]==HEDGE_CDS:
                hedge1t_vec[j] =  asset_hedge_tickers[idx,2]
                hedge2t_vec[j] =  asset_hedge_tickers[idx,3]
                hedge1f_vec[j] = "PX_LAST"
                hedge2f_vec[j] = "PX_LAST"

            elif asset_sources[idx,0]==HEDGE_CALC:
                hedge1t_vec[j] = asset_hedge_tickers[idx,4]
                hedge2t_vec[j] = asset_hedge_tickers[idx,5]
                hedge3t_vec[j] = asset_hedge_tickers[idx,6]
                hedge4t_vec[j] = asset_hedge_tickers[idx,7]
                hedge1f_vec[j] = ""
                hedge2f_vec[j] = ""
                hedge3f_vec[j] = ""
                hedge4f_vec[j] = ""

            year0, month0 = year, month
            year1, month1 = (year*12+(month-1)-1) // 12, (year*12+(month-1)-1) % 12 + 1
            year2, month2 = (year*12+(month-1)-2) // 12, (year*12+(month-1)-2) % 12 + 1

            days0_vec[j] = 29 if (year0 % 4 == 0 and (year0 % 100 != 0 or year0 % 400 == 0)) and (month0==2) else _DAYS_PER_MONTH[month0 - 1]
            days1_vec[j] = 29 if (year1 % 4 == 0 and (year1 % 100 != 0 or year1 % 400 == 0)) and (month1==2) else _DAYS_PER_MONTH[month1 - 1]
            days2_vec[j] = 29 if (year2 % 4 == 0 and (year2 % 100 != 0 or year2 % 400 == 0)) and (month2==2) else _DAYS_PER_MONTH[month2 - 1]

            if ntrc_vec[j]=="N":
                day_count_vec[j] = days0_vec[j] + days1_vec[j]
            elif ntrc_vec[j]=="F":
                day_count_vec[j] = days0_vec[j] + days1_vec[j] + days2_vec[j]
            else:
                day_count_vec[j]=0

   
print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({year_vec.shape[0]:,} straddles)")

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

