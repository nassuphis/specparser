# -------------------------------------
# Normalized vs Actual 
# Commodity Futures tickers
# -------------------------------------
"""
Ticker extraction and transformation utilities.

Handles extracting tickers from assets, expanding futures specs,
split ticker handling, and straddle info building.
"""
import csv
import re
from pathlib import Path

# -------------------------------------
#  map cache
# -------------------------------------

_MEMOIZE_ENABLED: bool = True

# Cache for normalized to actual futures mapping
_NORMALIZED_CACHE: dict[str, dict[str, str]] = {}
# Cache for actual to normalized futures mapping (reverse)
_ACTUAL_CACHE: dict[str, dict[str, str]] = {}


def set_memoize_enabled(enabled: bool) -> None:
    """Enable or disable memoization for ticker functions."""
    global _MEMOIZE_ENABLED
    _MEMOIZE_ENABLED = enabled


def clear_chain_caches() -> None:
    """Clear all chain-related caches."""
    _NORMALIZED_CACHE.clear()
    _ACTUAL_CACHE.clear()


def fut_norm2act(csv_path: str | Path, ticker: str) -> str | None:
    """
    Convert a normalized BBG futures ticker to the actual BBG ticker.

    Uses the CSV lookup table to map normalized tickers (e.g., "LAF2025 Comdty")
    to actual BBG tickers (e.g., "LA F25 Comdty").

    The CSV is loaded once and cached for subsequent lookups.

    Args:
        csv_path: Path to the CSV file with normalized_future,actual_future columns
        ticker: The normalized futures ticker to look up

    Returns:
        The actual BBG ticker if found, or None if not found

    Example:
        >>> actual = normalized2actual("data/current_bbg_chain_data.csv", "LAF2025 Comdty")
        >>> actual  # Returns actual ticker or None
    """
    csv_path = str(Path(csv_path).resolve())

    # Load and cache the CSV if not already cached
    if csv_path not in _NORMALIZED_CACHE:
        mapping = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = row.get("normalized_future", "")
                actual = row.get("actual_future", "")
                if normalized and actual: mapping[normalized] = actual
        _NORMALIZED_CACHE[csv_path] = mapping

    return _NORMALIZED_CACHE[csv_path].get(ticker)


def fut_act2norm(csv_path: str | Path, ticker: str) -> str | None:
    """
    Convert an actual BBG futures ticker to the normalized ticker.

    This is the inverse of fut_norm2act. Uses the same CSV lookup table
    to map actual BBG tickers (e.g., "LA F25 Comdty") back to normalized
    tickers (e.g., "LAF2025 Comdty").

    The CSV is loaded once and cached for subsequent lookups.

    Args:
        csv_path: Path to the CSV file with normalized_future,actual_future columns
        ticker: The actual BBG futures ticker to look up

    Returns:
        The normalized ticker if found, or None if not found
    """
    csv_path = str(Path(csv_path).resolve())

    # Load and cache the reverse mapping if not already cached
    if csv_path not in _ACTUAL_CACHE:
        mapping = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = row.get("normalized_future", "")
                actual = row.get("actual_future", "")
                if normalized and actual:
                    mapping[actual] = normalized
        _ACTUAL_CACHE[csv_path] = mapping

    return _ACTUAL_CACHE[csv_path].get(ticker)





