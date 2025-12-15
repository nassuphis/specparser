#!/usr/bin/env python
"""
expander.py

Scanner + choice expansion core.

Scanner output (structure, no expansion):
  - Lit(text)
  - Dim("LIST",  raw)   where raw is list-inner text OR a single-item sugar "{...}" / "${...}"
  - Dim("PLIST", raw)   where raw is progressive-list inner text (same syntax as LIST)
  - Ref(key)            render-time reference, key is "row" or a digit string ("1","2",...)

Choice expansion (union semantics inside a LIST/PLIST inner):
  - {...} numeric ranges:
      {a:b}        step mode with step = +1/-1 (always)
      {a:b_step}   step mode with explicit step magnitude
      {a:b|N}      linspace mode with N samples inclusive
    padding preserved for integer endpoints with leading zeros.

  - ${...} simpleeval:
      scalar -> single choice
      iterable -> multiple choices
    always returns list[str] (never iterates over a string’s characters).

  - nested [...] inside list items:
      a[b,c]d -> abd, acd
"""

from __future__ import annotations

import re
import math
import ast
import operator as op
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
from collections.abc import Iterable
from itertools import product
import random
from pathlib import Path
from simpleeval import EvalWithCompoundTypes


# ============================================================
# Segments (scanner output)
# ============================================================

DimKind = Literal["LIST", "PLIST"]

@dataclass(frozen=True)
class Lit:
    text: str

@dataclass(frozen=True)
class Dim:
    dim_kind: DimKind
    raw: str  # LIST/PLIST inner text; for implicit sugar, raw includes "{...}" or "${...}" or "@{...}"

@dataclass(frozen=True)
class Ref:
    key: str  # "row" or "1","2",...

@dataclass(frozen=True)
class Init:
    expr: str

Segment = Union[Lit, Dim, Ref, Init]


# ============================================================
# The term dictionary
# ============================================================

# User-populated registry for @{...}
DICT: dict[str, str] = {
    "aa": "X",
    "ab": "Y",
    "ba": "Z"
}

def set_dict(d: dict[str, str]) -> None:
    DICT.clear()
    DICT.update(d)

# ============================================================
# Simpleeval config for ${...}, #{...}
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

NAMES = {
    "pi": complex(math.pi),
}

# --- choice time functions

def search_keys_expand(pat):
    rx = re.compile(pat)
    matches = [ k for k in DICT.keys() if rx.fullmatch(k) ]
    return matches

def search_values_expand(pat):
    rx = re.compile(pat)   
    matches = [ DICT[k] for k in DICT.keys() if rx.fullmatch(k) ]
    return matches

FUNCS: dict[str, object] = {
    "key": search_keys_expand,
    "value": search_values_expand,
}

# --- render time functions

def choose(*args):
    return random.choice(args)

def search_keys_ref(pat, exclude=None):
    rx = re.compile(pat)
    exc = None if exclude is None else str(exclude)
    matches = [
        k for k in DICT.keys()
        if rx.fullmatch(k) and (exc is None or k != exc)
    ]
    return random.choice(matches) if matches else None

def search_values_ref(pat, exclude=None):
    rx = re.compile(pat)
    exc = None if exclude is None else str(exclude)
    matches = [
        DICT[k] for k in DICT.keys()
        if rx.fullmatch(k) and (exc is None or DICT[k] != exc)
    ]
    return random.choice(matches) if matches else None

def rint(N):
    return random.randint(1,N)

def rfloat(a,b):
    return random.uniform(a,b)

def num(x): return float(x)
def i(x): return int(float(x))
def zfill(x, w): return str(x).zfill(int(w))
def fmt(s, *args): return str(s).format(*args)
def at(seq, idx): return seq[int(idx)]
def wat(seq, idx):
    i = int(idx)
    n = len(seq)
    if n == 0:
        return None  # or raise
    return seq[i % n]     # wrap-around

REF_FUNCS: dict[str, object] = {
    "choose": choose,
    "key": search_keys_ref,
    "value": search_values_ref,
    "rint": rint,
    "rfloat": rfloat,
    "num": num,
    "i": i,
    "zfill": zfill,
    "fmt":  fmt,
    "at": at,
    "wat": wat,

    # add functions later if you want (seq, etc.)
}

# --- init time functions

RNG = random.Random()

def seed_init(x):
    RNG.seed(int(x))
    return int(x)

def set_const_init(name, value):
    NAMES[str(name)] = value
    return value

def set_dict_init(**kwargs):
    """Replace DICT with provided key/value pairs."""
    DICT.clear()
    for k, v in kwargs.items():
        DICT[str(k)] = str(v)
    return len(DICT)

def add_dict_init(**kwargs):
    """Update DICT with provided key/value pairs."""
    for k, v in kwargs.items():
        DICT[str(k)] = str(v)
    return len(kwargs)

def load_init(path, *, mode="new", start=1, strip=True, skip_empty=False, encoding="utf-8"):
    """
    Load a text file into DICT:
      key = str(line_number)
      value = line (optionally stripped)

    mode:
      - "new": replace DICT
      - "add": update DICT
    """
    p = Path(str(path))
    lines = p.read_text(encoding=encoding).splitlines()

    d = {}
    i = int(start)
    for line in lines:
        s = line.strip() if strip else line
        if skip_empty and s == "":
            i += 1
            continue
        d[str(i)] = s
        i += 1

    if mode == "new":
        DICT.clear()
    elif mode != "add":
        raise ValueError("mode must be 'new' or 'add'")

    DICT.update(d)
    return len(d)


INIT_FUNCS = {
    "seed": seed_init, 
    "const": set_const_init, 
    "new": set_dict_init,
    "add": add_dict_init,
    "load": load_init
}

# ============================================================
# regexes
# ============================================================

_INT_RE = re.compile(r"^[+-]?\d+$")
_DICTSEL_RE = re.compile(r"^@\{(.+)\}$")

# ============================================================
# Low-level scanners
# ============================================================

def _is_int_like(s: str) -> bool:
    return bool(_INT_RE.match(s.strip()))


def _needs_pad(a_s: str, b_s: str) -> int:
    """
    Padding is purely lexical:
      - strip sign
      - if either endpoint has a leading zero (and length > 1), pad to max width
    """
    a2 = a_s.strip().lstrip("+-")
    b2 = b_s.strip().lstrip("+-")
    pad_on = (len(a2) > 1 and a2.startswith("0")) or (len(b2) > 1 and b2.startswith("0"))
    return max(len(a2), len(b2)) if pad_on else 0


def _scan_balanced(s: str, start: int, open_ch: str, close_ch: str) -> int | None:
    """
    Given s[start] == open_ch, return index one-past matching close_ch.
    Supports nesting of the same delimiter type.
    """
    depth = 1
    i = start + 1
    n = len(s)
    while i < n and depth > 0:
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
        i += 1
    return i if depth == 0 else None


def _scan_list_body(s: str, lbrack_index: int) -> int | None:
    """
    Find matching ']' for '[' at lbrack_index, skipping protected regions inside:
      @{...}, ${...}, {...}, and nested [...]
    """
    i = lbrack_index + 1
    n = len(s)
    while i < n:
        ch = s[i]

        if ch == "@" and i + 1 < n and s[i + 1] == "{":
            end = _scan_balanced(s, i + 1, "{", "}")
            if end is None:
                return None
            i = end
            continue

        if ch == "$" and i + 1 < n and s[i + 1] == "{":
            end = _scan_balanced(s, i + 1, "{", "}")
            if end is None:
                return None
            i = end
            continue

        if ch == "{":
            end = _scan_balanced(s, i, "{", "}")
            if end is None:
                return None
            i = end
            continue

        if ch == "[":
            end = _scan_balanced(s, i, "[", "]")
            if end is None:
                return None
            i = end
            continue

        if ch == "]":
            return i + 1

        i += 1

    return None


# ============================================================
# Stage 1: scanner that outputs Lit/Dim/Ref segments (incl. sugar)
# ============================================================

def scan_segments(spec: str) -> List[Segment]:
    """
    One-pass scanner that outputs:
      - Lit(text)
      - Ref(key)             for #{...}
      - Dim("PLIST", inner)  for >[ ... ]
      - Dim("LIST", inner)   for [ ... ]
      - Dim("LIST", "${...}") for implicit ${ ... } in text
      - Dim("LIST", "@{...}") for implicit @{ ... } in text
      - Dim("LIST", "{...}")  for implicit { ... } in text
    """
    segs: List[Segment] = []
    buf: List[str] = []

    def flush_lit() -> None:
        if buf:
            segs.append(Lit("".join(buf)))
            buf.clear()

    i = 0
    n = len(spec)
    while i < n:
        ch = spec[i]

        # --- REF: #{...} ---
        # REF is render-time only (does not create a dimension).
        # Must be recognized before "{...}" sugar, otherwise "#{...}" would be mis-scanned as "{...}".
        if ch == "#" and i + 1 < n and spec[i + 1] == "{":
            flush_lit()
            end = _scan_balanced(spec, i + 1, "{", "}")
            if end is None:
                segs.append(Lit(spec[i:]))
                break
            inner = spec[i+2:end-1].strip()
            segs.append(Ref(inner))
            i = end
            continue

        # --- INIT: !{...} ---
        elif ch == "!" and i + 1 < n and spec[i + 1] == "{":
            flush_lit()
            end = _scan_balanced(spec, i + 1, "{", "}")
            if end is None:
                segs.append(Lit(spec[i:]))
                break
            expr = spec[i + 2 : end - 1].strip()
            segs.append(Init(expr))
            i = end
            continue

        # --- PLIST: >[ ... ] ---
        elif ch == ">" and i + 1 < n and spec[i + 1] == "[":
            flush_lit()
            end = _scan_list_body(spec, i + 1)
            if end is None:
                raise ValueError("Unterminated progressive list")
            inner = spec[i + 2 : end - 1]
            segs.append(Dim("PLIST", inner))
            i = end
            continue

        # --- LIST: [ ... ] ---
        elif ch == "[":
            flush_lit()
            end = _scan_list_body(spec, i)
            if end is None:
                raise ValueError("Unterminated list")
            inner = spec[i + 1 : end - 1]
            segs.append(Dim("LIST", inner))
            i = end
            continue

        # --- implicit ${...} sugar => LIST dim ---
        elif ch == "$" and i + 1 < n and spec[i + 1] == "{":
            flush_lit()
            end = _scan_balanced(spec, i + 1, "{", "}")
            if end is None:
                segs.append(Lit(spec[i:]))
                break
            segs.append(Dim("LIST", spec[i:end]))
            i = end
            continue

        # --- implicit @{...} sugar => LIST dim ---
        elif ch == "@" and i + 1 < n and spec[i + 1] == "{":
            flush_lit()
            end = _scan_balanced(spec, i + 1, "{", "}")
            if end is None:
                segs.append(Lit(spec[i:]))
                break
            segs.append(Dim("LIST", spec[i:end]))
            i = end
            continue

        # --- implicit {...} sugar => LIST dim ---
        elif ch == "{":
            flush_lit()
            end = _scan_balanced(spec, i, "{", "}")
            if end is None:
                segs.append(Lit(spec[i:]))
                break
            segs.append(Dim("LIST", spec[i:end]))
            i = end
            continue

        else:
            buf.append(ch)
            i += 1

    flush_lit()
    return segs


# ============================================================
# Stage 2: split list inner on top-level commas
# ============================================================

def split_list_items(inner: str) -> List[str]:
    """
    Split LIST/PLIST inner text on commas at depth 0 only.

    Protected regions:
      - brace_depth: {...}, ${...}, @{...} (all use braces)
      - paren_depth: (...)
      - brack_depth: [...]
    """
    out: List[str] = []
    buf: List[str] = []

    brace_depth = 0
    paren_depth = 0
    brack_depth = 0

    i = 0
    n = len(inner)
    while i < n:
        ch = inner[i]

        if ch == "{":
            brace_depth += 1
            buf.append(ch)
        elif ch == "}":
            if brace_depth > 0:
                brace_depth -= 1
            buf.append(ch)

        elif ch == "(":
            paren_depth += 1
            buf.append(ch)
        elif ch == ")":
            if paren_depth > 0:
                paren_depth -= 1
            buf.append(ch)

        elif ch == "[":
            brack_depth += 1
            buf.append(ch)
        elif ch == "]":
            if brack_depth > 0:
                brack_depth -= 1
            buf.append(ch)

        elif ch == "," and brace_depth == 0 and paren_depth == 0 and brack_depth == 0:
            out.append("".join(buf).strip())
            buf.clear()
        else:
            buf.append(ch)

        i += 1

    out.append("".join(buf).strip())
    return [x for x in out if x != ""]


# ============================================================
# Stage 3: union choice expansion for LIST inners
# ============================================================

# - range {....}
def _parse_range_body(body: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse body in one of these forms:
      a:b
      a:b_step
      a:b|N

    Returns (a_s, b_s, mode, third_s) where:
      mode in {"step", "lin"}
      third_s is "" for implicit step=1, else step string or N string.
    """
    body = body.strip()
    if ":" not in body:
        return None

    a_s, rest = body.split(":", 1)
    a_s = a_s.strip()
    rest = rest.strip()

    if "|" in rest:
        b_s, n_s = rest.split("|", 1)
        b_s = b_s.strip()
        n_s = n_s.strip()
        if not n_s:
            return None
        return (a_s, b_s, "lin", n_s)

    if "_" in rest:
        b_s, step_s = rest.split("_", 1)
        b_s = b_s.strip()
        step_s = step_s.strip()
        if not step_s:
            return None
        return (a_s, b_s, "step", step_s)

    b_s = rest.strip()
    return (a_s, b_s, "step", "")


def _expand_range_body(body: str) -> List[str]:
    parsed = _parse_range_body(body)
    if parsed is None:
        return ["{" + body + "}"]

    a_s, b_s, mode, third_s = parsed

    try:
        a_val = int(a_s) if _is_int_like(a_s) else float(a_s)
        b_val = int(b_s) if _is_int_like(b_s) else float(b_s)
    except Exception:
        return ["{" + body + "}"]

    a_is_int = _is_int_like(a_s)
    b_is_int = _is_int_like(b_s)
    pad_w = _needs_pad(a_s, b_s) if (a_is_int and b_is_int) else 0

    def fmt_int(xi: int) -> str:
        if pad_w <= 1:
            return str(xi)
        return ("-" if xi < 0 else "") + f"{abs(xi):0{pad_w}d}"

    def fmt_float(x: float) -> str:
        return f"{float(x):.15g}"

    out: List[str] = []

    if mode == "lin":
        if not _is_int_like(third_s):
            return ["{" + body + "}"]
        N = int(third_s)
        if N <= 0:
            return ["{" + body + "}"]

        if N == 1:
            vals = [float(a_val)]
        else:
            a_f = float(a_val)
            b_f = float(b_val)
            step = (b_f - a_f) / (N - 1)
            vals = [a_f + i * step for i in range(N)]

        use_int = False
        if a_is_int and b_is_int:
            rng = int(b_val) - int(a_val)
            if rng == 0 and N == 1:
                use_int = True
            elif rng != 0 and N == abs(rng) + 1:
                use_int = True

        if use_int:
            for v in vals:
                out.append(fmt_int(int(round(v))))
        else:
            for v in vals:
                out.append(fmt_float(v))
        return out

    # step mode (implicit or explicit)
    if third_s == "":
        step_mag = 1.0
    else:
        try:
            step_mag = float(third_s)
        except Exception:
            return ["{" + body + "}"]
        if not math.isfinite(step_mag) or step_mag == 0.0:
            return ["{" + body + "}"]

    direction = 1.0 if float(b_val) >= float(a_val) else -1.0
    step = direction * abs(step_mag)

    use_int = bool(a_is_int and b_is_int and float(step).is_integer())

    a_f = float(a_val)
    b_f = float(b_val)

    max_iters = 1_000_000
    x = a_f
    it = 0

    if step > 0:
        cond = lambda v: v <= b_f or math.isclose(v, b_f, rel_tol=0.0, abs_tol=1e-15)
    else:
        cond = lambda v: v >= b_f or math.isclose(v, b_f, rel_tol=0.0, abs_tol=1e-15)

    while cond(x):
        out.append(fmt_int(int(round(x))) if use_int else fmt_float(x))
        x += step
        it += 1
        if it > max_iters:
            return ["{" + body + "}"]

    return out


def _expand_range_token(token: str) -> Optional[List[str]]:
    brace_depth = 0
    paren_depth = 0
    brack_depth = 0

    i = 0
    n = len(token)
    while i < n:
        ch = token[i]

        if ch == "(":
            paren_depth += 1
        elif ch == ")" and paren_depth > 0:
            paren_depth -= 1

        elif ch == "[":
            brack_depth += 1
        elif ch == "]" and brack_depth > 0:
            brack_depth -= 1

        elif ch == "{":
            if paren_depth == 0 and brack_depth == 0 and brace_depth == 0:
                end = _scan_balanced(token, i, "{", "}")
                if end is None:
                    return [token]
                pre = token[:i]
                body = token[i + 1 : end - 1]
                suf = token[end:]
                vals = _expand_range_body(body)
                return [f"{pre}{v}{suf}" for v in vals]
            brace_depth += 1
        elif ch == "}" and brace_depth > 0:
            brace_depth -= 1

        i += 1

    return None


# - dict @{....}

def _expand_dict_token(token: str) -> Optional[List[str]]:
    """
    If token is exactly '@{regex}', expand to matching entries from DICT (by key).
    Values, fullmatch on keys.
    """
    m = _DICTSEL_RE.match(token.strip())
    if not m:
        return None

    if not DICT:
        return [token]

    pat = m.group(1)
    try:
        rx = re.compile(pat)
    except re.error:
        return [token]

    matches = [str(DICT[k]) for k in DICT.keys() if rx.fullmatch(k)]
    return matches if matches else []


# - fcall ${....}

def _to_str_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (str, bytes)):
        return [v.decode() if isinstance(v, bytes) else v]
    if isinstance(v, (int, float, complex, bool)):
        return [str(v)]
    if isinstance(v, dict):
        return [str(v)]
    if isinstance(v, Iterable):
        return [str(x) for x in v]
    return [str(v)]


def _eval_fcall_body(expr: str) -> Optional[List[str]]:
    s = expr.strip()
    try:
        se = EvalWithCompoundTypes(names=NAMES, functions=FUNCS, operators=ALLOWED_OPS)
        vals = _to_str_list(se.eval(s))
        return vals if vals else None
    except Exception:
        return None


def _expand_fcall_token(token: str) -> Optional[List[str]]:
    brace_depth = 0
    paren_depth = 0
    brack_depth = 0

    i = 0
    n = len(token)
    while i < n:
        ch = token[i]

        if ch == "{":
            brace_depth += 1
        elif ch == "}" and brace_depth > 0:
            brace_depth -= 1

        elif ch == "(":
            paren_depth += 1
        elif ch == ")" and paren_depth > 0:
            paren_depth -= 1

        elif ch == "[":
            brack_depth += 1
        elif ch == "]" and brack_depth > 0:
            brack_depth -= 1

        if (
            ch == "$"
            and i + 1 < n
            and token[i + 1] == "{"
            and brace_depth == 0
            and paren_depth == 0
            and brack_depth == 0
        ):
            end = _scan_balanced(token, i + 1, "{", "}")
            if end is None:
                return [token]
            pre = token[:i]
            body = token[i + 2 : end - 1]
            suf = token[end:]
            vals = _eval_fcall_body(body)
            if not vals:
                return [token]
            return [f"{pre}{v}{suf}" for v in vals]

        i += 1

    return None


# - list [....]
def _expand_nested_list_token(token: str) -> Optional[List[str]]:
    brace_depth = 0
    paren_depth = 0
    brack_depth = 0

    i = 0
    n = len(token)
    while i < n:
        ch = token[i]

        if ch == "{":
            brace_depth += 1
        elif ch == "}" and brace_depth > 0:
            brace_depth -= 1

        elif ch == "(":
            paren_depth += 1
        elif ch == ")" and paren_depth > 0:
            paren_depth -= 1

        elif ch == "[":
            if brace_depth == 0 and paren_depth == 0 and brack_depth == 0:
                end = _scan_balanced(token, i, "[", "]")
                if end is None:
                    return [token]
                pre = token[:i]
                inner = token[i + 1 : end - 1]
                suf = token[end:]
                choices = expand_list_choices(inner)
                return [f"{pre}{c}{suf}" for c in choices]
            brack_depth += 1
        elif ch == "]" and brack_depth > 0:
            brack_depth -= 1

        i += 1

    return None


# - expand items in list
def expand_item_union(item: str) -> List[str]:
    pending = [item]
    changed = True

    while changed:
        changed = False
        next_pending: List[str] = []
        # Expansion is sequential, not “cartesian within a single token”.
        # Order matters only for output ordering, not for the set of results.
        # Priority: nested [...] → ${...} → @{...} → {...}
        for tok in pending:

            ex1 = _expand_nested_list_token(tok)
            if ex1 is not None and not (len(ex1) == 1 and ex1[0] == tok):
                next_pending.extend(ex1)
                changed = True
                continue

            exf = _expand_fcall_token(tok)
            if exf is not None and not (len(exf) == 1 and exf[0] == tok):
                next_pending.extend(exf)
                changed = True
                continue

            exd = _expand_dict_token(tok)
            if exd is not None and not (len(exd) == 1 and exd[0] == tok):
                next_pending.extend(exd)
                changed = True
                continue

            ex2 = _expand_range_token(tok)
            if ex2 is not None and not (len(ex2) == 1 and ex2[0] == tok):
                next_pending.extend(ex2)
                changed = True
                continue

            next_pending.append(tok)

        pending = next_pending

    return pending


def expand_list_choices(inner: str) -> List[str]:
    items = split_list_items(inner)
    out: List[str] = []
    for it in items:
        out.extend(expand_item_union(it))
    return out


def progressive_choices(inner: str) -> List[str]:
    items = split_list_items(inner)
    expanded_items: List[str] = []
    for it in items:
        expanded_items.extend(expand_item_union(it))

    out: List[str] = []
    acc: List[str] = []
    for it in expanded_items:
        acc.append(it)
        out.append(",".join(acc))
    return out


# ============================================================
# Cartesian product
# ============================================================

class DimSeries:
    def __init__(self, choices: list[str], stride: int):
        self._choices = choices
        self._stride = int(stride)
        self._n = len(choices)

    def __getitem__(self, row_1based):
        r = int(row_1based)
        if r < 1:
            raise IndexError("row is 1-based (>= 1)")
        idx = ((r - 1) // self._stride) % self._n
        return self._choices[idx]

    def __len__(self):
        return self._n
    

def _realize_dim(d: Dim) -> List[str]:
    return expand_list_choices(d.raw) if d.dim_kind == "LIST" else progressive_choices(d.raw)


def expand(spec: str, *, limit: int = 0) -> List[str]:
    """
    Full expansion:
      - scan_segments(spec) -> [Lit/Dim/Ref]
      - realize each Dim to a list of choices
      - cartesian product across dims
      - render each row by interleaving Lit + selected Dim value + resolving Ref
    """
    segs = scan_segments(spec)

    # init-time code (runs once)
    init_names = dict(NAMES)
    init_se = EvalWithCompoundTypes(names=init_names, functions=INIT_FUNCS, operators=ALLOWED_OPS)
    for s in segs:
        if isinstance(s, Init):
            init_se.eval(s.expr) 

    dims: List[Dim] = [s for s in segs if isinstance(s, Dim)]
    dim_choices: List[List[str]] = [_realize_dim(d) for d in dims]

    if any(len(c) == 0 for c in dim_choices):
        return []

    sizes = [len(c) for c in dim_choices]
    strides = []
    acc = 1
    for sz in reversed(sizes):
        strides.append(acc)
        acc *= sz
    strides.reverse()

    base_render_names = dict(NAMES)

    # expose per-dimension choice lists: choices1, choices2, ...
    for j, choices_j in enumerate(dim_choices, start=1):
        base_render_names[f"choices{j}"] = choices_j

    # expose row-aligned expanded dimension series: dim1, dim2, ...
    for j, (choices_j, stride_j) in enumerate(zip(dim_choices, strides), start=1):
        base_render_names[f"dim{j}"] = DimSeries(choices_j, stride_j)

    # optional convenience
    base_render_names["ndims"] = len(dim_choices)
    base_render_names["nrows"] = acc  # total rows, product of sizes (can be big)

    out: List[str] = []

    if not dim_choices:
        render_names = dict(base_render_names)
        render_names["row"] = 1
        se = EvalWithCompoundTypes(names=render_names, functions=REF_FUNCS, operators=ALLOWED_OPS)

        parts = []
        for s in segs:
            if isinstance(s, Lit):
                parts.append(s.text)
            elif isinstance(s, Ref):
                try:
                    parts.append(str(se.eval(s.key)))
                except Exception:
                    parts.append(f"#{{{s.key}}}")
        out.append("".join(parts))
        return out

    # Render-time eval for #{...}:
    # Create a per-row evaluator with row + selected dimension values in `render_names`.
    # Do NOT mutate global NAMES; keep render context isolated.
    for row_i, picks in enumerate(product(*dim_choices), start=1):
        render_names = dict(base_render_names)
        render_names["row"] = row_i
        # Selected dimension values are exposed as d1, d2, ... for #{...} expressions.
        # (Values are strings; add helper funcs in REF_FUNCS if you want numeric ops.)
        for j, v in enumerate(picks, start=1):
            render_names[f"d{j}"] = v

        se = EvalWithCompoundTypes(names=render_names, functions=REF_FUNCS, operators=ALLOWED_OPS)

        parts = []
        dim_i = 0
        for s in segs:
            if isinstance(s, Lit):
                parts.append(s.text)
            elif isinstance(s, Dim):
                parts.append(picks[dim_i])
                dim_i += 1
            elif isinstance(s, Ref):
                try:
                    parts.append(str(se.eval(s.key)))
                except Exception:
                    parts.append(f"#{{{s.key}}}")  # leave literal on eval failure
            elif isinstance(s, Init):
                continue
            else:
                pass #should be error?
        
        # Important: append ONCE per row (after all segments are rendered),
        # not once per segment.
        out.append("".join(parts))
        if limit and len(out) >= limit:
            break

    return out


# ============================================================
# Selftest
# ============================================================

def _selftest() -> None:
    def st(spec: str):
        segs = scan_segments(spec)
        out = []
        for s in segs:
            if isinstance(s, Lit):
                out.append(("LIT", s.text))
            elif isinstance(s, Dim):
                out.append(("DIM", s.dim_kind, s.raw))
            else:
                out.append(("REF", s.key))
        return out

    # --- scanner tests ---
    assert st("abc") == [("LIT", "abc")]
    assert st("x[a,b]y") == [("LIT", "x"), ("DIM", "LIST", "a,b"), ("LIT", "y")]
    assert st("x>[a,b]y") == [("LIT", "x"), ("DIM", "PLIST", "a,b"), ("LIT", "y")]
    assert st("[${fun(1,2)}]") == [("DIM", "LIST", "${fun(1,2)}")]
    assert st("[@{/^A/}]") == [("DIM", "LIST", "@{/^A/}")]
    assert st("[{1:3}]") == [("DIM", "LIST", "{1:3}")]
    assert st("[a[b],c]") == [("DIM", "LIST", "a[b],c")]
    assert st("[[a,b]]") == [("DIM", "LIST", "[a,b]")]
    assert st("sdsda${asd}{01:03} #{row}") == [
        ("LIT", "sdsda"),
        ("DIM", "LIST", "${asd}"),
        ("DIM", "LIST", "{01:03}"),
        ("LIT", " "),
        ("REF", "row"),
    ]

    # --- list-item splitting tests (NO expansion here) ---
    assert split_list_items("a,b,c") == ["a", "b", "c"]
    assert split_list_items(" a , b , c ") == ["a", "b", "c"]
    assert split_list_items("") == []
    assert split_list_items(",,") == []
    assert split_list_items("a,,b,") == ["a", "b"]
    assert split_list_items("fun(x,y),z") == ["fun(x,y)", "z"]
    assert split_list_items("${[1,2,3]},x") == ["${[1,2,3]}", "x"]
    assert split_list_items("@{/a,b/},x") == ["@{/a,b/}", "x"]
    assert split_list_items("{0:1_0.25},x") == ["{0:1_0.25}", "x"]
    assert split_list_items("a[b,c],d") == ["a[b,c]", "d"]

    # --- range parsing / expansion tests ---
    assert _expand_range_body("1:3") == ["1", "2", "3"]
    assert _expand_range_body("3:1") == ["3", "2", "1"]
    assert _expand_range_body("0.5:3.5") == ["0.5", "1.5", "2.5", "3.5"]
    assert _expand_range_body("0:1_0.25") == ["0", "0.25", "0.5", "0.75", "1"]
    assert _expand_range_body("1:10_2") == ["1", "3", "5", "7", "9"]
    assert _expand_range_body("0:1|5") == ["0", "0.25", "0.5", "0.75", "1"]

    # padding
    assert _expand_range_body("01:03") == ["01", "02", "03"]
    assert _expand_range_body("-003:003") == ["-003", "-002", "-001", "000", "001", "002", "003"]
    assert _expand_range_body("01:05_2") == ["01", "03", "05"]

    # {a:b} always step=1, even for floats
    assert _expand_range_body("0.1:0.3") == ["0.1"]
    assert _expand_range_body("0:1_0.5") == ["0", "0.5", "1"]
    assert _expand_range_body("0:1|3") == ["0", "0.5", "1"]

    # --- ${...} expansion tests ---
    assert expand_list_choices("1,${2+4},100") == ["1", "6", "100"]
    assert expand_list_choices("1,${[4,5,6]},100") == ["1", "4", "5", "6", "100"]
    assert expand_list_choices("x${[1,2]}y") == ["x1y", "x2y"]
    assert expand_list_choices("1,${pi},100") == ["1", str(complex(math.pi)), "100"]

    # --- nested [] union expander tests ---
    assert expand_list_choices("a[b,c]d,e") == ["abd", "acd", "e"]
    assert expand_list_choices("x[{01:03},y]") == ["x01", "x02", "x03", "xy"]
    assert expand_list_choices("[{01:05_2},a]") == ["01", "03", "05", "a"]

    # multiple expansions in a single item
    assert expand_list_choices("a{1:3}b") == ["a1b", "a2b", "a3b"]
    assert expand_list_choices("a${[1,2]}b") == ["a1b", "a2b"]

    # mix range + fcall in one token (note: A/B must be strings)
    assert expand_list_choices('x{01:03}y${["A","B"]}') == [
        "x01yA", "x02yA", "x03yA",
        "x01yB", "x02yB", "x03yB",
    ]

    # nested [] producing items that contain ranges
    assert expand_list_choices("p[{1:3},q]s") == ["p1s", "p2s", "p3s", "pqs"]

    # nested [] producing items that contain fcalls
    assert expand_list_choices("p[${[1,2]},q]s") == ["p1s", "p2s", "pqs"]

    # progressive choices
    assert progressive_choices("a,b,c") == ["a", "a,b", "a,b,c"]
    assert progressive_choices("${[1,2]},x") == ["1", "1,2", "1,2,x"]

    # dictionary lookup (values, fullmatch)
    DICT.clear()
    DICT.update({"aa": "X", "ab": "Y", "ba": "Z"})
    assert expand_list_choices("@{a.}") == ["X", "Y"]
    assert expand_list_choices("pre@{aa}post") == ["pre@{aa}post"]
    assert expand_list_choices("@{(}") == ["@{(}"]

    # --- full expand tests (cartesian + refs) ---
    assert expand("[1,2] #{row}") == ["1 1", "2 2"]
    assert expand("[a,b]{1:2}::#{1}-#{2}-#{row}") == [
        "a1::a-1-1",
        "a2::a-2-2",
        "b1::b-1-3",
        "b2::b-2-4",
    ]

    print("selftest: OK")


# ============================================================
# CLI
# ============================================================

def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="expander core (scanner + choice expansion).")
    p.add_argument("spec", nargs="?", help="Input spec string")
    p.add_argument("--selftest", action="store_true", help="Run selftest and exit")
    p.add_argument("--choices", action="store_true", help="Spec must be a single top-level [ ... ]; print its expanded choices")
    p.add_argument("--pchoices", action="store_true", help="Spec must be a single top-level >[ ... ]; print its progressive choices")
    p.add_argument("--expand", action="store_true", help="Expand full spec and print lines")
    p.add_argument("--limit", type=int, default=0, help="Limit printed expansion lines (0 = no limit)")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return 0

    if args.spec is None:
        p.error("spec is required unless --selftest is given")

    if args.choices:
        segs = scan_segments(args.spec)
        if len(segs) != 1 or not isinstance(segs[0], Dim) or segs[0].dim_kind != "LIST":
            raise SystemExit("--choices expects spec to be a single top-level [ ... ] list")
        for c in expand_list_choices(segs[0].raw):
            print(c)
        return 0

    if args.pchoices:
        segs = scan_segments(args.spec)
        if len(segs) != 1 or not isinstance(segs[0], Dim) or segs[0].dim_kind != "PLIST":
            raise SystemExit("--pchoices expects spec to be a single top-level >[ ... ] list")
        for c in progressive_choices(segs[0].raw):
            print(c)
        return 0

    if args.expand:
        for line in expand(args.spec, limit=args.limit):
            print(line)
        return 0

    segs = scan_segments(args.spec)
    for s in segs:
        if isinstance(s, Lit):
            print("LIT ", repr(s.text))
        elif isinstance(s, Dim):
            print("DIM ", s.dim_kind, repr(s.raw))
        else:
            print("REF ", repr(s.key))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
