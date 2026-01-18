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
    find_underlyings,
    list_assets,
    get_table,
    format_table,
    print_table,
    _iter_assets,
    assets,
    live_assets,
    live_class,
    _compile_rules,
    _match_rules,
    live_table,
    live_group,
)

from .tickers import (
    _split_ticker,
    asset_tickers,
    fut_ticker,
    normalized2actual,
    clear_normalized_cache,
    _expand_bbgfc_row,
    _expand_split_ticker_row,
    asset_straddle,
    straddle_days,
    live_tickers,
)

from .schedules import (
    get_schedule,
    find_schedules,
    _split_code_value,
    live_schedules,
    expand,
    find_expand,
)

__all__ = [
    # loader
    "load_amt",
    "clear_cache",
    "get_value",
    "get_aum",
    "get_leverage",
    "get_asset",
    "find_underlyings",
    "list_assets",
    "get_table",
    "format_table",
    "print_table",
    "_iter_assets",
    "assets",
    "live_assets",
    "live_class",
    "_compile_rules",
    "_match_rules",
    "live_table",
    "live_group",
    # tickers
    "_split_ticker",
    "asset_tickers",
    "fut_ticker",
    "normalized2actual",
    "clear_normalized_cache",
    "_expand_bbgfc_row",
    "_expand_split_ticker_row",
    "asset_straddle",
    "straddle_days",
    "live_tickers",
    # schedules
    "get_schedule",
    "find_schedules",
    "_split_code_value",
    "live_schedules",
    "expand",
    "find_expand",
]
