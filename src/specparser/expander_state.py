# -------------------------------------
# expander shared state
# -------------------------------------
"""
Shared state for the expander DSL:
- DICT: user-populated registry for @{...} lookups
- NAMES: constants available in expressions
- RNG: single source of randomness
"""
import math
import random
import secrets

# ============================================================
# The term dictionary
# ============================================================

DICT: dict[str, str] = {
    "aa": "X",
    "ab": "Y",
    "ba": "Z"
}


def set_dict(d: dict[str, str]) -> None:
    """Replace DICT contents with provided dictionary."""
    DICT.clear()
    DICT.update(d)


def get_dict() -> dict[str, str]:
    """Return the current DICT."""
    return DICT


def clear_dict() -> None:
    """Clear the DICT."""
    DICT.clear()


# ============================================================
# Constants/names for expressions
# ============================================================

NAMES: dict[str, object] = {
    "pi": complex(math.pi),
}


def set_name(name: str, value: object) -> None:
    """Set a constant in NAMES."""
    NAMES[name] = value


def get_name(name: str) -> object:
    """Get a constant from NAMES."""
    return NAMES.get(name)


def get_names() -> dict[str, object]:
    """Return the current NAMES dict."""
    return NAMES


# ============================================================
# Random number generator
# ============================================================

_DEFAULT_SEED = secrets.randbits(128)
RNG = random.Random(_DEFAULT_SEED)


def seed(x: int | str | None = None) -> int:
    """
    Set the random seed for the expander RNG.

    Args:
        x: Seed value. If None or "auto"/"rand"/"random"/"entropy",
           reseeds from OS entropy. Otherwise uses int(x).

    Returns:
        The seed that was used.
    """
    if x is None or str(x).lower() in ("auto", "rand", "random", "entropy"):
        s = secrets.randbits(128)
    else:
        s = int(x)
    RNG.seed(s)
    return s


def get_rng() -> random.Random:
    """Return the RNG instance."""
    return RNG
