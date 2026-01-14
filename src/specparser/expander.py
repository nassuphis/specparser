#!/usr/bin/env python
"""
expander.py

Scanner + choice expansion core.

Scanner output (structure, no expansion):
  - Lit(text)
  - Dim("LIST",  raw)   where raw is list-inner text OR a single-item sugar "{...}" / "${...}"
  - Dim("PLIST", raw)   where raw is progressive-list inner text (same syntax as LIST) but inside a  ">[ ... ]"
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
from pathlib import Path
import importlib.util
import sys  # <-- add

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent))

def _load_sibling(name):
    mod_path = Path(__file__).resolve().with_name(f"{name}.py")
    mod_spec = importlib.util.spec_from_file_location(f"_{name}_local", mod_path)
    assert mod_spec and mod_spec.loader
    mod = importlib.util.module_from_spec(mod_spec)
    # IMPORTANT: register before exec_module (dataclasses expects this)
    sys.modules[mod_spec.name] = mod
    mod_spec.loader.exec_module(mod)
    return mod

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    # running as a standalone script: python ../specparser/expander.py ...
    sp = _load_sibling("specparser")
else:
    # imported as part of the specparser package: from specparser import expander
    from . import specparser as sp


from rasterizer import image2spec 

import re
import math
import numpy as np
import ast
import operator as op
from dataclasses import dataclass
import contextvars
from typing import List, Literal, Optional, Tuple, Union, Any
from collections.abc import Iterable
from itertools import product
import random
import secrets
from pathlib import Path
from simpleeval import EvalWithCompoundTypes
import subprocess

# ============================================================
# Context handler
# ============================================================

_RENDER_CTX: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "RENDER_CTX", default=None
)

def _render_ctx() -> dict[str, Any]:
    ctx = _RENDER_CTX.get()
    if ctx is None:
        raise RuntimeError("render-time function called outside render context")
    return ctx

def with_render_ctx(fn):
    """Decorator: inject current render_names as first arg (hidden from spec)."""
    def wrapper(*args, **kwargs):
        return fn(_render_ctx(), *args, **kwargs)
    return wrapper

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
# these are vectors, the cross product
# of all is computed

# -------------------------------------
#  search dicr
# -------------------------------------

def search_keys_expand(pat):
    rx = re.compile(pat)
    matches = [ k for k in DICT.keys() if rx.fullmatch(k) ]
    return matches

def search_values_expand(pat):
    rx = re.compile(pat)   
    matches = [ DICT[k] for k in DICT.keys() if rx.fullmatch(k) ]
    return matches

# -------------------------------------
#  random numbers
# -------------------------------------

def rint_expand(N):
    return RNG.randint(1,N)

def rfloat_expand(a,b):
    return RNG.uniform(a,b)

# -------------------------------------
#  0 -> 1 steps
# -------------------------------------

def seq(num):
    vec = np.linspace(0.0,1.0,num)
    out=[f"{x}" for x in vec]
    return out

# -------------------------------------
#  read files
# -------------------------------------

line_dict_expand: dict[str, list[str]] = {}

def lines_expand(fn: str, lno: int):
    global line_dict_expand

    if fn not in line_dict_expand:
        try:
            with open(fn, "r", encoding="utf-8") as f:
                line_dict_expand[fn] = f.read().splitlines()
        except FileNotFoundError:
            print(f"cant read from'{fn}'")
            return None
    if not line_dict_expand[fn]:
        print(f"file is empty: '{fn}'")
        return None
    return RNG.choices(line_dict_expand[fn],k=lno)
  
    
def lines2_expand(fn: str, lno: int,delim= ":"):
    l1=lines_expand(fn,lno)
    l2=lines_expand(fn,lno)
    if l1 is None or l2 is None: return None
    return [f"{a}{delim}{b}" for a, b in zip(l1, l2, strict=True)]


# -------------------------------------
# spec files
# -------------------------------------

# read the whole specfile
def specfile(specfile: str): 
    # normalize spec filename
    p = Path(specfile)
    if p.suffix != ".spec": p = p.with_suffix(".spec")
    try:
        specs = p.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return specs

# read specific slots from specfile
def specfile_slots(specfile: str | Path, slots): # read specific slots
    if isinstance(slots, Iterable) and not isinstance(slots, (str, bytes)):
        want = {str(i) for i in slots}
    else:
        want = {str(slots)}

    p = Path(specfile)
    if p.suffix != ".spec": p = p.with_suffix(".spec")

    try: specs = p.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError: return []

    out = []
    for s in specs:
        d = sp.split_chain(s)
        if "slot" not in d or not d["slot"]: continue
        if d["slot"][0] in want: out.append(s)
    return out



# -------------------------------------
# slot management
# -------------------------------------

def used_files(schema: str) -> list[str]:
    base = Path(schema)
    dirpath = base.parent
    stem = base.name

    pat = re.compile(rf"^{re.escape(stem)}_(\d+)\.jpg$")
    files: list[str] = []

    for p in dirpath.iterdir():
        if not p.is_file():
            continue
        if pat.match(p.name):
            files.append(str(p))

    return files

def slots2jpegs(schema: str, slots: Iterable[int]) -> list[str]:
    base = Path(schema)
    dirpath = base.parent
    stem = base.name
    out: list[str] = []
    for i in sorted(set(slots)):
        p = dirpath / f"{stem}_{i:05d}.jpg"
        if p.is_file():
            out.append(str(p))
    return out

def slots2specs(schema: str, slots: Iterable[int]) -> list[str]:
    base = Path(schema)
    dirpath = base.parent
    stem = base.name
    out: list[str] = []
    for i in sorted(set(slots)):
        p = dirpath / f"{stem}_{i:05d}.spec"
        if p.is_file():
            out.append(str(p))
    return out

def used_slots(schema: str) -> list[int]:
    base = Path(schema)
    stem = base.name

    pat = re.compile(rf"^{re.escape(stem)}_(\d+)\.jpg$")
    used: set[int] = set()

    for fname in used_files(schema):
        p = Path(fname)
        m = pat.match(p.name)
        if m:
            used.add(int(m.group(1)))

    return list(used)

def max_slot(schema: str) -> int | None:
    used = set(used_slots(schema))
    if not used: return None
    return max(used)

def first_free_slot(schema: str) -> int:
    used = set(used_slots(schema))
    if not used: return 1
    universe = set(range(1,max(used)+2))
    return min(universe - used)

def free_slots(schema: str, required: int) -> list[int]:
    used = set(used_slots(schema))
    if not used: return list(range(1,required+1))
    universe = set(range(1,max(used)+required+1)) # required slot count fits
    free = universe - used
    return sorted(free)[:required]

def slots(required: int) -> list[int]:
    return free_slots(DICT["outschema"], required)


# -------------------------------------
#  spec modifiers
# -------------------------------------

def spec_rot(spec,rot):
    d = sp.split_chain(spec)
    d["rot"]=[str(rot)]
    spec=sp.concat_chain(d)
    return spec

def specs_rot(specs: List[str],rot):
    new_specs = []
    for spec in specs: new_specs.append(spec_rot(spec,rot))
    return new_specs


def spec_replace(spec: str, key: str, old: str, new: str) -> str:
    d = sp.split_chain(spec)
    if key in d:
        d[key] = [v.replace(old, new) for v in d[key]]
    return sp.concat_chain(d)

def specs_replace(specs: str | list[str], key: str, old: str, new: str) -> list[str]:
    if isinstance(specs, str): specs = [specs]
    return [spec_replace(s, key, old, new) for s in specs]

# -------------------------------------
# spec management
# -------------------------------------

def spec2slot(spec: str, slot: int) -> str:
    d = sp.split_chain(spec)
    d["slot"] = [str(slot)]
    return sp.concat_chain(d)

def spec2free(spec: str):
    return spec2slot(spec,first_free_slot(DICT["outschema"]))

def specs2free(specs: List[str]):
    if isinstance(specs, str): specs = [specs]
    new_specs = []
    for spec, slot in zip(specs, free_slots(DICT["outschema"], len(specs))): 
        new_specs.append(spec2slot(spec,slot))
    return new_specs

# -------------------------------------
#  spec files
# -------------------------------------

def specfile2free(specf: str):
    specs = specfile(specf)
    if not specs: return None  # or [] if you prefer
    return specs2free(specs)

def specfile_slots2free(specf: str, slots):
    specs = specfile_slots(specf, slots)
    if not specs: return None  # or [] if you prefer
    return specs2free(specs)

# -------------------------------------
# images
# -------------------------------------

def image(imgfile: str):
    return image2spec.read_spec_exiftool(imgfile)

def image2free(imgfile: str):
    return spec2free(image(imgfile))

def images(
    schema: str,              # "filedir/filestem"
    suffices: Iterable[int],  # e.g. [1,3,4] or range(1,10)
) -> list[str]:
    specs: list[str] = []
    for fn in slots2jpegs(schema,suffices): specs.append(image(str(fn)))
    return specs

def images2free(
    schema: str,              # "filedir/filestem"
    suffices: Iterable[int],  # e.g. [1,3,4] or range(1,10)
) -> list[str]:
    print(f"images2free: {len(suffices)}")
    return specs2free(images(schema,suffices))


# -------------------------------------
#  ocr
# -------------------------------------

SCRIPT = Path(__file__).resolve().parent.parent / "lyapunov" / "extract_spec.sh"

def ocr(imagefile):
    print(f"OCR file:{imagefile}")
    spec = subprocess.check_output(["bash", str(SCRIPT), imagefile],text=True)
    print(f"OCR:{spec}")
    d = sp.split_chain(spec)
    d["source"]=[imagefile]
    spec=sp.concat_chain(d)
    return spec

def ocr2free(imagefile):
    return spec2free(ocr(imagefile))

def ocrs2free(schema,slots):
    specs=[]
    for jpeg in slots2jpegs(schema,slots): specs.append(ocr(jpeg))
    return specs2free(specs)

FUNCS: dict[str, object] = {
    "range":        range,
    "rint":         rint_expand,
    "rfloat":       rfloat_expand,
    "key":          search_keys_expand,
    "value":        search_values_expand,
    # sequence
    "seq":          seq,
    # files
    "lines":        lines_expand,
    "lines2":       lines2_expand,
    # specs in files
    "specf":        specfile_slots2free,
    # specs in images
    "img":          images2free,
    # ocr files
    "ocr":          ocr2free,
    "ocrs":         ocrs2free,
    # individual specs
    "spec":         spec2free,
    "rot":          specs_rot,
    "swap":         specs_replace,
    "free":         specs2free,
    # utilities
    "slots":        slots,
    "used_slots":   used_slots,
    "free_slots":   free_slots,
    "slots2jpegs":  slots2jpegs,
    "slots2specs":  slots2specs,
    "first":        first_free_slot,
}

#######################################
#
# render time functions
#
#######################################

# these need to be valies not vectors

def render_choose(*args):
    return f"{RNG.choice(args)}"

def render_search_keys(pat, exclude=None):
    rx = re.compile(pat)
    exc = None if exclude is None else str(exclude)
    matches = [
        k for k in DICT.keys()
        if rx.fullmatch(k) and (exc is None or k != exc)
    ]
    return f"{RNG.choice(matches) if matches else None}"

def render_search_values(pat, exclude=None):
    rx = re.compile(pat)
    exc = None if exclude is None else str(exclude)
    matches = [
        DICT[k] for k in DICT.keys()
        if rx.fullmatch(k) and (exc is None or DICT[k] != exc)
    ]
    return f"{RNG.choice(matches) if matches else None}"

def render_rvalues():
    vals = [DICT[k] for k in DICT.keys()]
    return f"{RNG.choice(vals)}"

def render_rint(N):
    return f"{RNG.randint(1,N)}"

def render_rfloat(a,b):
    return f"{RNG.uniform(a,b)}"

def render_rfloat3(a,b):
    return f"{round(RNG.uniform(a,b),3)}"

def render_num(x): return float(x)
def render_i(x): return int(float(x))
def render_zfill(x, w): return str(x).zfill(int(w))
def render_fmt(s, *args): return str(s).format(*args)
def render_at(seq, idx): return seq[int(idx)]
def render_wat(seq, idx):
    i = int(idx)
    n = len(seq)
    if n == 0:
        return f"{None}"  # or raise
    return f"{seq[i % n]}"     # wrap-around

# -------------------------------------
#  render from file
# -------------------------------------

# global cache
line_dict_ref: dict[str, list[str]] = {}

def render_file_line(fn: str, lno: int) -> str:
    """
    Return line number lno (0-based) from file fn.
    Caches file contents across calls.
    """
    global line_dict_ref
    if fn not in line_dict_ref:
        with open(fn, "r", encoding="utf-8") as f:
            # keep lines without trailing newline
            line_dict_ref[fn] = f.read().splitlines()
    try:
        return f"{line_dict_ref[fn][lno]}"
    except IndexError:
        raise IndexError(f"line number {lno} out of range for file '{fn}'")

def render_random_file_line(fn: str) -> str:
    """
    Return a random line from file fn.
    Caches file contents across calls.
    """
    global line_dict_ref

    if fn not in line_dict_ref:
        with open(fn, "r", encoding="utf-8") as f:
            line_dict_ref[fn] = f.read().splitlines()

    if not line_dict_ref[fn]:
        raise ValueError(f"file '{fn}' is empty")

    return f"{RNG.choice(line_dict_ref[fn])}"

def render_rline2(fn: str, delim=":") -> str:
    l1 = render_random_file_line(fn)
    l2 = render_random_file_line(fn)
    return f"{l1}{delim}{l2}"

def render_r2line(fn1: str, fn2: str, delim=":") -> str:
    l1 = render_random_file_line(fn1)
    l2 = render_random_file_line(fn2)
    return f"{l1}{delim}{l2}"


# -------------------------------------
#  convenience
# -------------------------------------

@with_render_ctx
def render_lerp(ctx,start,end):
    row = int(ctx["row"])
    nrows = int(ctx["nrows"])
    t = 0.0 if nrows <= 1 else (row - 1) / (nrows - 1)
    value = float(start) + (float(end) - float(start)) * t
    return f"{value}"

@with_render_ctx
def render_square(ctx,start,end):
    row = int(ctx["row"])
    nrows = int(ctx["nrows"])
    t = 0.0 if nrows <= 1 else (row - 1) / (nrows - 1)
    size = float(start) + (float(end) - float(start)) * t
    return f"-{size}:-{size}:{size}:{size}"

# -------------------------------------
#  function registry
# -------------------------------------

REF_FUNCS: dict[str, object] = {
    # randomized selections
    "choose":   render_choose,
    "rval":     render_rvalues,
    "rint":     render_rint,
    "rfloat":   render_rfloat,
    "rfloat3":  render_rfloat3,
    # search DICT
    "key":      render_search_keys,
    "value":    render_search_values,
    # type conversions
    "num":      render_num, # numeric
    "i":        render_i, # integer
    "str":      str,
    # formatting
    "zfill":    render_zfill, #
    "fmt":      render_fmt,
    # access dimensions
    "at":       render_at,
    "wat":      render_wat,
    # file access
    "line":     render_file_line,
    "rline":    render_random_file_line,
    "rline2":   render_rline2,
    "r2line":   render_r2line,
    # convenience functions
    "square":   render_square,
    "lerp":     render_lerp,
    # add functions later if you want (seq, etc.)
    "first":     first_free_slot,
}

# --- init time functions

#  RNG (single source of randomness) ---
_DEFAULT_SEED = secrets.randbits(128)  # different each process run
RNG = random.Random(_DEFAULT_SEED)

def seed_init(x=None):
    """
    !{seed(123)}      -> deterministic seed
    !{seed()}         -> reseed from OS entropy
    !{seed("auto")}   -> same as seed()
    !{seed(256)}      -> if you pass a large int, it's fine
    """
    if x is None or str(x).lower() in ("auto", "rand", "random", "entropy"):
        s = secrets.randbits(128)
    else:
        s = int(x)
    RNG.seed(s)
    return s

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
# Bozo Macros
# ============================================================

MACROS: dict[str, str] = {}
_MACRO_KEY_RE = re.compile(r"^@[A-Z0-9@]+$")

def macro(s: str) -> str:
    """
    Simple macro expansion:
    replace all occurrences of keys like @FOO with their values.
    """
    out = s
    for k, v in MACROS.items():
        if _MACRO_KEY_RE.fullmatch(k):
            out = out.replace(k, v)
    return out

def macro_init(fn: str) -> bool:
    """
    Initialize global MACROS from a file with lines like:
    @MACRO=value

    Returns True on success, False on failure.
    """
    global MACROS

    try:
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue

                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()

                if _MACRO_KEY_RE.fullmatch(key):
                    MACROS[key] = val

        return True

    except OSError as e:
        print(f"failed to load macro file '{fn}': {e}")
        return False

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
        #NAMES.update(base_render_names)
        tok = _RENDER_CTX.set(render_names)
        try:
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
        finally:
            _RENDER_CTX.reset(tok)
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

        #NAMES.update(render_names)
        tok = _RENDER_CTX.set(render_names)
        try:
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
        finally:
            _RENDER_CTX.reset(tok)

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
    assert expand("[a,b]{1:2}::#{d1}-#{d2}-#{row}") == [
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
    p.add_argument("--macro", action="store_true", help="Just expand macro")
    p.add_argument("--choices", action="store_true", help="Spec must be a single top-level [ ... ]; print its expanded choices")
    p.add_argument("--pchoices", action="store_true", help="Spec must be a single top-level >[ ... ]; print its progressive choices")
    p.add_argument("--expand", action="store_true", help="Expand full spec and print lines")
    p.add_argument("--limit", type=int, default=0, help="Limit printed expansion lines (0 = no limit)")
    args = p.parse_args()

    macro_init("macros.txt")

    if args.selftest:
        _selftest()
        return 0

    if args.spec is None:
        p.error("spec is required unless --selftest is given")

    if args.macro:
        print(macro(args.spec))
        return 0

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
        for line in expand(macro(args.spec), limit=args.limit):
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
