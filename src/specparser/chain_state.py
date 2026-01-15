# -------------------------------------
# chain shared state
# -------------------------------------
"""
Shared state for the chain parser:
- NAMES: constants available in scalar expressions
- FUNCS: functions available in scalar expressions
- ALLOWED_OPS: whitelisted operators for simpleeval
"""
import ast
import operator as op
import math
import cmath
import random

import numpy as np

from . import slots as slotfuns


# ============================================================
# Operators whitelist for simpleeval
# ============================================================

ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
}


# ============================================================
# Constants for expressions
# ============================================================

NAMES: dict[str, complex | float] = {
    "pi": math.pi,
    "tau": math.tau,
    "e": math.e,
    "inf": float("inf"),
    "nan": float("nan"),
    "j": 1j,
    "i": 1j,  # allow i for imaginary
    "zero": 0 + 0j,
    "one": 1 + 1j,
}


def set_const(name: str, value: complex | float) -> None:
    """Set a constant in NAMES."""
    NAMES[name.strip().lower()] = complex(value)


def get_const(name: str) -> complex:
    """Get a constant from NAMES."""
    return complex(NAMES[name.strip().lower()])


# ============================================================
# Helper functions
# ============================================================

def real_max(x, y):
    """Max comparing only real parts."""
    return max(x.real, y.real)


def real_min(x, y):
    """Min comparing only real parts."""
    return min(x.real, y.real)


def lerp(start, end, i, n):
    """Linear interpolation: return i-th value (1-based) of n samples from start to end."""
    x = np.linspace(start, end, n)
    return x[i - 1]


def rint(N):
    """Random integer from 1 to N."""
    return random.randint(1, N)


def rfloat(a, b):
    """Random float between a and b."""
    return random.uniform(a, b)


# ============================================================
# Functions registry
# ============================================================

FUNCS: dict[str, object] = {
    # trig (complex-aware)
    "sin": cmath.sin,
    "cos": cmath.cos,
    "tan": cmath.tan,
    # math (complex-aware)
    "sqrt": cmath.sqrt,
    "log": cmath.log,
    "exp": cmath.exp,
    "abs": abs,
    # comparison (real-part only)
    "min": real_min,
    "max": real_max,
    # interpolation
    "lerp": lerp,
    # random
    "rint": rint,
    "rfloat": rfloat,
    # slots
    "slotmax": slotfuns.max_slot,
    "slotmin": slotfuns.first_free_slot,
    "slots": slotfuns.free_slots,
}
