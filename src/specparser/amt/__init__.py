# -------------------------------------
# AMT (Asset Management Table) utilities
# -------------------------------------
"""
Module for processing AMT YAML files.

AMT files are YAML files with a specific structure containing asset definitions
under the 'amt' key, where each asset has an 'Underlying' identifier.

This package provides utilities for:
- Loading and querying AMT files (loader)
- Extracting tickers from assets (tickers)
- Expanding schedules and building straddles (schedules)
"""

# Re-export all public functions for backward compatibility
# Users can continue to use: from specparser.amt import load_amt, expand_schedules, etc.

from .loader import (
    load_amt,
    clear_cache,
    get_value,
    get_aum,
    get_leverage,
    get_asset,
    find_assets,
    cached_assets,
    get_table,
    table_column,
    format_table,
    print_table,
    bind_rows,
    table_unique_rows,
    _iter_assets,
    assets,
    asset_class,
    _compile_rules,
    _match_rules,
    asset_table,
    asset_group,
)

from .tickers import (
    _split_ticker,
    get_tschemas,
    find_tschemas,
    fut_spec2ticker,
    fut_norm2act,
    clear_normalized_cache,
    _tschma_dict_expand_bbgfc,
    _tschma_dict_expand_split,
    asset_straddle_tickers,
    straddle_days,
    find_tickers,
)

from .schedules import (
    get_schedule,
    find_schedules,
    _split_code_value,
    expand,
    expand_ym,
    get_expand,
    get_expand_ym,
    # straddle parsing
    ntr,
    ntry,
    ntrm,
    xpr,
    xpry,
    xprm,
    ntrc,
    ntrv,
    xprc,
    xprv,
    wgt,
)

__all__ = [
    # loader
    "load_amt",
    "clear_cache",
    "get_value",
    "get_aum",
    "get_leverage",
    "get_asset",
    "find_assets",
    "cached_assets",
    "get_table",
    "table_column",
    "format_table",
    "print_table",
    "bind_rows",
    "table_unique_rows",
    "_iter_assets",
    "assets",
    "asset_class",
    "_compile_rules",
    "_match_rules",
    "asset_table",
    "asset_group",
    # tickers
    "_split_ticker",
    "get_tschemas",
    "find_tschemas",
    "fut_spec2ticker",
    "fut_norm2act",
    "clear_normalized_cache",
    "_tschma_dict_expand_bbgfc",
    "_tschma_dict_expand_split",
    "asset_straddle",
    "straddle_days",
    "find_tickers",
    # schedules
    "get_schedule",
    "find_schedules",
    "_split_code_value",
    "expand",
    "expand_ym",
    "get_expand",
    "get_expand_ym",
    # straddle parsing
    "ntr",
    "ntry",
    "ntrm",
    "xpr",
    "xpry",
    "xprm",
    "ntrc",
    "ntrv",
    "xprc",
    "xprv",
    "wgt",
]
