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

Imports are lazy to avoid RuntimeWarning when running submodules as scripts.
Use: from specparser.amt import load_amt, expand, etc.
"""

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
    "_iter_assets",
    "assets",
    "asset_class",
    "_compile_rules",
    "_match_rules",
    "asset_table",
    "asset_group",
    # table utilities (from table.py)
    "table_column",
    "table_select_columns",
    "table_add_column",
    "table_drop_columns",
    "table_replace_value",
    "table_bind_rows",
    "table_unique_rows",
    "table_join",
    "table_unchop",
    "table_chop",
    "format_table",
    "print_table",
    # backward compatibility alias
    "bind_rows",
    # tickers
    "_split_ticker",
    "get_tschemas",
    "find_tschemas",
    "fut_spec2ticker",
    "fut_norm2act",
    "fut_act2norm",
    "clear_normalized_cache",
    "_tschma_dict_expand_bbgfc",
    "_tschma_dict_expand_split",
    "asset_straddle_tickers",
    "get_straddle_days",
    "find_tickers",
    "find_tickers_ym",
    # schedules
    "get_schedule",
    "find_schedules",
    "_split_code_value",
    "expand",
    "expand_ym",
    "get_expand",
    "get_expand_ym",
    "get_days_ym",
    "clear_days_cache",
    "year_month_days",
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

# Lazy import mapping: attribute -> (module, name)
_LAZY_IMPORTS = {
    # loader
    "load_amt": (".loader", "load_amt"),
    "clear_cache": (".loader", "clear_cache"),
    "get_value": (".loader", "get_value"),
    "get_aum": (".loader", "get_aum"),
    "get_leverage": (".loader", "get_leverage"),
    "get_asset": (".loader", "get_asset"),
    "find_assets": (".loader", "find_assets"),
    "cached_assets": (".loader", "cached_assets"),
    "get_table": (".loader", "get_table"),
    "_iter_assets": (".loader", "_iter_assets"),
    "assets": (".loader", "assets"),
    "asset_class": (".loader", "asset_class"),
    "_compile_rules": (".loader", "_compile_rules"),
    "_match_rules": (".loader", "_match_rules"),
    "asset_table": (".loader", "asset_table"),
    "asset_group": (".loader", "asset_group"),
    # table utilities (from table.py)
    "table_column": (".table", "table_column"),
    "table_select_columns": (".table", "table_select_columns"),
    "table_add_column": (".table", "table_add_column"),
    "table_drop_columns": (".table", "table_drop_columns"),
    "table_replace_value": (".table", "table_replace_value"),
    "table_bind_rows": (".table", "table_bind_rows"),
    "table_unique_rows": (".table", "table_unique_rows"),
    "table_join": (".table", "table_join"),
    "table_unchop": (".table", "table_unchop"),
    "table_chop": (".table", "table_chop"),
    "format_table": (".table", "format_table"),
    "print_table": (".table", "print_table"),
    # backward compatibility alias
    "bind_rows": (".table", "table_bind_rows"),
    # tickers
    "_split_ticker": (".tickers", "_split_ticker"),
    "get_tschemas": (".tickers", "get_tschemas"),
    "find_tschemas": (".tickers", "find_tschemas"),
    "fut_spec2ticker": (".tickers", "fut_spec2ticker"),
    "fut_norm2act": (".tickers", "fut_norm2act"),
    "fut_act2norm": (".tickers", "fut_act2norm"),
    "clear_normalized_cache": (".tickers", "clear_normalized_cache"),
    "_tschma_dict_expand_bbgfc": (".tickers", "_tschma_dict_expand_bbgfc"),
    "_tschma_dict_expand_split": (".tickers", "_tschma_dict_expand_split"),
    "asset_straddle_tickers": (".tickers", "asset_straddle_tickers"),
    "get_straddle_days": (".tickers", "get_straddle_days"),
    "find_tickers": (".tickers", "find_tickers"),
    "find_tickers_ym": (".tickers", "find_tickers_ym"),
    # schedules
    "get_schedule": (".schedules", "get_schedule"),
    "find_schedules": (".schedules", "find_schedules"),
    "_split_code_value": (".schedules", "_split_code_value"),
    "expand": (".schedules", "expand"),
    "expand_ym": (".schedules", "expand_ym"),
    "get_expand": (".schedules", "get_expand"),
    "get_expand_ym": (".schedules", "get_expand_ym"),
    "year_month_days": (".schedules", "year_month_days"),
    "get_days_ym": (".schedules", "get_days_ym"),
    "clear_days_cache": (".schedules", "clear_days_cache"),
    # straddle parsing
    "ntr": (".schedules", "ntr"),
    "ntry": (".schedules", "ntry"),
    "ntrm": (".schedules", "ntrm"),
    "xpr": (".schedules", "xpr"),
    "xpry": (".schedules", "xpry"),
    "xprm": (".schedules", "xprm"),
    "ntrc": (".schedules", "ntrc"),
    "ntrv": (".schedules", "ntrv"),
    "xprc": (".schedules", "xprc"),
    "xprv": (".schedules", "xprv"),
    "wgt": (".schedules", "wgt"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_name, __package__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
