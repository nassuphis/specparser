# -------------------------------------
# render-time functions (#{...})
# -------------------------------------
"""
Functions available in #{...} expressions.
These run at render time (once per output row) and return scalar values.

Also contains the render context machinery used by expander.py.
"""
import contextvars
import re
from typing import Any

from . import expander_state as state
from . import slots as slotfuns
from . import files as filefuns
from . import dates as datefuns


# ============================================================
# Render context
# ============================================================

_RENDER_CTX: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "RENDER_CTX", default=None
)


def get_render_ctx() -> dict[str, Any]:
    """Get the current render context. Raises if not in render context."""
    ctx = _RENDER_CTX.get()
    if ctx is None:
        raise RuntimeError("render-time function called outside render context")
    return ctx


def set_render_ctx(ctx: dict[str, Any] | None):
    """Set the render context. Returns token for reset."""
    return _RENDER_CTX.set(ctx)


def reset_render_ctx(token):
    """Reset render context using token from set_render_ctx."""
    _RENDER_CTX.reset(token)


def with_render_ctx(fn):
    """Decorator: inject current render context as first arg (hidden from spec)."""
    def wrapper(*args, **kwargs):
        return fn(get_render_ctx(), *args, **kwargs)
    return wrapper


# ============================================================
# Render-time functions
# ============================================================

def render_choose(*args):
    """Randomly choose one of the arguments."""
    return f"{state.RNG.choice(args)}"


def render_search_keys(pat, exclude=None):
    """Return a random key from DICT matching pattern, optionally excluding one."""
    rx = re.compile(pat)
    exc = None if exclude is None else str(exclude)
    matches = [
        k for k in state.DICT.keys()
        if rx.fullmatch(k) and (exc is None or k != exc)
    ]
    return f"{state.RNG.choice(matches) if matches else None}"


def render_search_values(pat, exclude=None):
    """Return a random value from DICT where key matches pattern."""
    rx = re.compile(pat)
    exc = None if exclude is None else str(exclude)
    matches = [
        state.DICT[k] for k in state.DICT.keys()
        if rx.fullmatch(k) and (exc is None or state.DICT[k] != exc)
    ]
    return f"{state.RNG.choice(matches) if matches else None}"


def render_rvalues():
    """Return a random value from DICT."""
    vals = [state.DICT[k] for k in state.DICT.keys()]
    return f"{state.RNG.choice(vals)}"


def render_rint(N):
    """Return a random integer from 1 to N."""
    return f"{state.RNG.randint(1, N)}"


def render_rfloat(a, b):
    """Return a random float between a and b."""
    return f"{state.RNG.uniform(a, b)}"


def render_rfloat3(a, b):
    """Return a random float between a and b, rounded to 3 decimals."""
    return f"{round(state.RNG.uniform(a, b), 3)}"


def render_num(x):
    """Convert to float."""
    return float(x)


def render_i(x):
    """Convert to integer."""
    return int(float(x))


def render_zfill(x, w):
    """Zero-pad x to width w."""
    return str(x).zfill(int(w))


def render_fmt(s, *args):
    """Format string s with args."""
    return str(s).format(*args)


def render_at(seq, idx):
    """Return element at index idx from seq."""
    return seq[int(idx)]


def render_wat(seq, idx):
    """Return element at index idx from seq, with wrap-around."""
    i = int(idx)
    n = len(seq)
    if n == 0:
        return f"{None}"
    return f"{seq[i % n]}"


# -------------------------------------
# render from file
# -------------------------------------

def render_rline2(fn: str, delim=":") -> str:
    """Return two random lines from file, joined by delimiter."""
    l1 = filefuns.get_random_line(fn)
    l2 = filefuns.get_random_line(fn)
    return f"{l1}{delim}{l2}"


def render_r2line(fn1: str, fn2: str, delim=":") -> str:
    """Return random lines from two files, joined by delimiter."""
    l1 = filefuns.get_random_line(fn1)
    l2 = filefuns.get_random_line(fn2)
    return f"{l1}{delim}{l2}"


# -------------------------------------
# convenience
# -------------------------------------

@with_render_ctx
def render_lerp(ctx, start, end):
    """Linear interpolation based on row position."""
    row = int(ctx["row"])
    nrows = int(ctx["nrows"])
    t = 0.0 if nrows <= 1 else (row - 1) / (nrows - 1)
    value = float(start) + (float(end) - float(start)) * t
    return f"{value}"


@with_render_ctx
def render_square(ctx, start, end):
    """Generate square coordinates interpolated by row position."""
    row = int(ctx["row"])
    nrows = int(ctx["nrows"])
    t = 0.0 if nrows <= 1 else (row - 1) / (nrows - 1)
    size = float(start) + (float(end) - float(start)) * t
    return f"-{size}:-{size}:{size}:{size}"


# ============================================================
# Function registry
# ============================================================

REF_FUNCS: dict[str, object] = {
    # randomized selections
    "choose": render_choose,
    "rval": render_rvalues,
    "rint": render_rint,
    "rfloat": render_rfloat,
    "rfloat3": render_rfloat3,
    # search DICT
    "key": render_search_keys,
    "value": render_search_values,
    # type conversions
    "num": render_num,
    "i": render_i,
    "str": str,
    # formatting
    "zfill": render_zfill,
    "fmt": render_fmt,
    # access dimensions
    "at": render_at,
    "wat": render_wat,
    # file access
    "line": filefuns.get_line,
    "rline": filefuns.get_random_line,
    "rline2": render_rline2,
    "r2line": render_r2line,
    # convenience functions
    "square": render_square,
    "lerp": render_lerp,
    # slot utilities
    "first": slotfuns.first_free_slot,
    # date utilities
    "expiry": datefuns.expiry,
    "entry": datefuns.entry,
}
