
from typing import Any
from . import loader

_MEMOIZE_ENABLED = True

_ASSET_STRADDLE_TICKER_CACHE = {}
_ASSET_STRADDLE_TICKER_COLUMNS=["name","ticker","field"]

def set_memoize_enabled(enabled: bool) -> None:
    """Enable or disable memoization for ticker functions."""
    global _MEMOIZE_ENABLED
    _MEMOIZE_ENABLED = enabled


def clear_ticker_caches() -> None:
    """Clear all ticker-related caches."""
    _ASSET_STRADDLE_TICKER_CACHE.clear()


def split_ticker(ticker,year,month):
        if ":" in ticker:
            ticker_pre,ym,ticker_post =  ticker.split(":",2)
            if ym<f"{year}-{month:02d}":
                return ticker_pre
            else:
                return ticker_post
        else:
            return ticker

def make_fut_ticker(fut_code,fut_month_map,min_year_offset,market_code,qualifier,year,month):
    month_map = "FGHJKMNQUVXZ"
    fut_month_code = fut_month_map[month - 1]
    opt_month_code = month_map[month - 1]
    year_offset = max(1 if fut_month_code < opt_month_code else 0, min_year_offset)
    fut_year = year + year_offset
    return  f"{fut_code}{qualifier}{fut_month_code}{fut_year} {market_code}"

def asset_straddle_ticker_key(
    asset: str, 
    strym: str, 
    ntrc:str,
    vol:dict,
    hedge:dict
):
    if vol["Source"]=="BBG_LMEVOL":
        cache_key = asset+"|"+strym+"|"+ntrc
    elif hedge["Source"]=="fut":
        cache_key = asset+"|"+strym+"|"+ntrc
    elif hedge["Source"]=="nonfut" and ":" in hedge["Ticker"]:
        cache_key = asset+"|"+strym+"|"+ntrc
    elif hedge["Source"]=="cds" and ":" in hedge["hedge"]:
        cache_key = asset+"|"+strym+"|"+ntrc
    elif hedge["Source"]=="cds" and ":" in hedge["hedge1"]:
        cache_key = asset+"|"+strym+"|"+ntrc
    else: 
        cache_key = asset
    return cache_key

def get_asset_straddle_tickers(asset: str, strym: str, ntrc:str, amt_path: str) -> dict[str,Any]:

    ticker_dict = dict(
        orientation="row",
        columns=_ASSET_STRADDLE_TICKER_COLUMNS,
        rows = []
    )

    asset_data = loader.get_asset(amt_path,asset)
    if asset_data is None : return ticker_dict
    hedge = asset_data["Hedge"]
    if hedge is None: return ticker_dict
    vol = asset_data["Vol"]
    if vol is None: return ticker_dict

    if ntrc=="F":
        if vol.get("Far","")=="": return ticker_dict
        if vol.get("Near","")==vol.get("Far",""): return ticker_dict
        if vol.get("Far","")=="NONE": return ticker_dict

    cache_key = asset_straddle_ticker_key(asset,strym,ntrc,vol,hedge)

    if _MEMOIZE_ENABLED and cache_key in _ASSET_STRADDLE_TICKER_CACHE:
        return _ASSET_STRADDLE_TICKER_CACHE[cache_key]

    year, month = int(strym[0:4]), int(strym[5:7])

    if vol["Source"]=="BBG" and ntrc=="N":
       ticker_dict["rows"].append(["vol",vol["Ticker"],vol["Near"]])

    if vol["Source"]=="BBG" and ntrc=="F":
        ticker_dict["rows"].append(["vol",vol["Ticker"],vol["Far"]])

    if vol["Source"]=="CV":
        ticker_dict["rows"].append(["vol",vol["Near"],"none"])

    if vol["Source"]=="BBG_LMEVOL":
        vol_ticker =  make_fut_ticker(
            fut_code=hedge["fut_code"],
            fut_month_map=hedge["fut_month_map"],
            min_year_offset=hedge["min_year_offset"],
            market_code=hedge["market_code"],
            qualifier="R",
            year=year,
            month=month,
        )
        ticker_dict["rows"].append(["vol",vol_ticker,"PX_LAST"])

    if hedge["Source"]=="fut":
        fut_ticker = make_fut_ticker(
            fut_code=hedge["fut_code"],
            fut_month_map=hedge["fut_month_map"],
            min_year_offset=hedge["min_year_offset"],
            market_code=hedge["market_code"],
            qualifier="",
            year=year,
            month=month,
        )
        ticker_dict["rows"].append(["hedge",fut_ticker,"PX_LAST"])

    if hedge["Source"]=="nonfut":
        ticker_dict["rows"].append(["hedge",split_ticker(hedge["Ticker"],year,month),hedge["Field"]])

    if hedge["Source"]=="cds":
        ticker_dict["rows"].append(["hedge",split_ticker(hedge["hedge"],year,month),"PX_LAST"])
        ticker_dict["rows"].append(["hedge1",split_ticker(hedge["hedge1"],year,month),"PX_LAST"])

    if hedge["Source"]=="calc":
        ticker_dict["rows"].append(["hedge1",f"{hedge['ccy']}_fsw0m_{hedge['tenor']}",""])
        ticker_dict["rows"].append(["hedge2",f"{hedge['ccy']}_fsw6m_{hedge['tenor']}",""])
        ticker_dict["rows"].append(["hedge3",f"{hedge['ccy']}_pva0m_{hedge['tenor']}",""])
        ticker_dict["rows"].append(["hedge4",f"{hedge['ccy']}_pva6m_{hedge['tenor']}",""])
    
    if _MEMOIZE_ENABLED:
        _ASSET_STRADDLE_TICKER_CACHE[cache_key] = ticker_dict

    return ticker_dict

def _ym_range(ntry: str, xpry: str) -> list[str]:
    sy, sm = int(ntry[:4]), int(ntry[5:7])
    ey, em = int(xpry[:4]), int(xpry[5:7])
    start = sy * 12 + (sm - 1)
    end   = ey * 12 + (em - 1)
    # months: start .. end inclusive  => range(start, end+1)
    return [f"{m//12:04d}-{(m%12)+1:02d}" for m in range(start, end + 1)]

def find_assets_straddles_tickers(pattern: str, ntry: str, xpry:str, amt_path: str):

    assets = [asset for _, asset in loader._iter_assets(amt_path, live_only=True, pattern=pattern)]
    months = _ym_range(ntry, xpry)
    ntrcs  = ("N", "F")

    # find column indices once
    cols = _ASSET_STRADDLE_TICKER_COLUMNS
    i_ticker = cols.index("ticker")
    i_field  = cols.index("field")

    rows_out: list[list[str]] = []
    seen: set[tuple[str, str]] = set()

    for asset in assets:
        for ym in months:
            for ntrc in ntrcs:
                straddles = get_asset_straddle_tickers(asset, ym, ntrc, amt_path)
                for row in straddles["rows"]:
                    key = (row[i_ticker],row[i_field])
                    if key in seen: 
                        #print(f"dupe: {row[i_ticker]},{row[i_field]}")
                        continue
                    seen.add(key)
                    rows_out.append(row)

    return {
        "columns": _ASSET_STRADDLE_TICKER_COLUMNS,
        "orientation": "row",
        "rows": rows_out,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    from .table import print_table

    parser = argparse.ArgumentParser(
        description="Asset straddle ticker utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get tickers for a single asset/straddle
  uv run python -m specparser.amt.asset_straddle_tickers data/amt.yml --get "CL Comdty" --ym 2024-06 --ntrc N

  # Find all unique tickers for assets matching a pattern over a date range
  uv run python -m specparser.amt.asset_straddle_tickers data/amt.yml --find "Comdty$" --start 2024-01 --end 2024-12

  # Find tickers for all live assets
  uv run python -m specparser.amt.asset_straddle_tickers data/amt.yml --find "." --start 2024-01 --end 2024-06
""",
    )
    parser.add_argument("amt_path", help="Path to AMT YAML file")
    parser.add_argument("--get", metavar="ASSET", help="Get tickers for a single asset")
    parser.add_argument("--ym", metavar="YYYY-MM", help="Year-month for --get (e.g., 2024-06)")
    parser.add_argument("--ntrc", choices=["N", "F"], default="N", help="Entry code: N=near, F=far (default: N)")
    parser.add_argument("--find", metavar="PATTERN", help="Find tickers for assets matching regex pattern")
    parser.add_argument("--start", metavar="YYYY-MM", help="Start year-month for --find")
    parser.add_argument("--end", metavar="YYYY-MM", help="End year-month for --find")

    args = parser.parse_args()

    if args.get:
        if not args.ym:
            parser.error("--get requires --ym")
        result = get_asset_straddle_tickers(args.get, args.ym, args.ntrc, args.amt_path)
        print_table(result)

    elif args.find:
        if not args.start or not args.end:
            parser.error("--find requires --start and --end")
        result = find_assets_straddles_tickers(args.find, args.start, args.end, args.amt_path)
        print_table(result)
        print(f"\nTotal unique tickers: {len(result['rows'])}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
