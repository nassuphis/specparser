#!/usr/bin/env python
"""
expandspec.py
--------------

A tiny, human-readable DSL that expands specs into lists of strings.

Grammar (strict)
================
Only these constructs are supported:

- **Cartesian list**: `[a,b,c]`
  Produces a union of items. Items may contain a single numeric range `{...}` or a dict selector `@{...}`.
  Nested `[...]` is not supported (a literal `[` inside an item is treated as a normal character).

- **Progressive list**: `>[a,b,c]`
  Expands to the **progressive prefixes** of the list, joined by commas:
  `>[a,b,c]` → `a`, `a,b`, `a,b,c`
   Items follow the same rules as cartesian list items (each item may contain one `{...}` or `@{...}`).

- **Numeric range**: `{a:b[:c]}`
  Expands to numbers on a line from `a` to `b` (inclusive).
  Rules:
  • If two fields and both are integers → integer step of +1 or −1.
  • If three fields:
      - If the third field contains a '.' or 'e'/'E' → step size (float or int). Direction auto-corrected if needed.
      - Else, if `a` and `b` are integers:
          ▸ If the third field (N) divides the integer range exactly (or N==1), treat it as an integer step size.
          ▸ Otherwise, treat it as a COUNT (N evenly spaced samples, inclusive).
      - Otherwise → COUNT (N evenly spaced samples, inclusive).
  Formatting:
  • If both endpoints are integers and step is integer (or COUNT lands on integers), results are integers.
  • Zero-padding is preserved if either endpoint has leading zeros (e.g., `{01:03}` → `01,02,03`).
  • Floats are emitted with up to 15 significant digits.

- **Dict selector**: `@{<regex>}` or `@{/regex/flags}`
  Expands to names matched from a provided set (`--names`).
  Flags (if using `/regex/flags`):
  • `i` = ignore case
  • `m` = multiline
  If no `--names` are supplied, the selector is kept literal (no expansion).

- **Reference**: `#{n}`
  Inserts the value chosen for the **n-th list-like unit** (1-based, left-to-right) in the current cartesian combination.
  List-like units are:
  1) Numeric producers created from `{...}` embedded in text
  2) Top-level cartesian lists `[ ... ]`
  3) Whole-token dict selectors `@{...}`

Semantics
=========
- **All lists are cartesian.** There is **no implicit zip**
- references are **by number only** (`#{1}`, `#{2}`, …).
- **Zipping happens only via references**: `#{n}` reproduces the selection from the n-th list-like dimension within the current cartesian product.

Composition & parsing
=====================
- The spec is parsed left-to-right into segments:
  literals, producers (`{...}` inside text), cartesian lists (`[...]`), whole-token selectors (`@{...}`), and references (`#{n}`).
- Each producer/list/selector is assigned a 1-based ordinal (its **list-like id**) for `#{n}` to reference.
- The full cartesian product of all list-like units is enumerated; references are resolved **after** a combination is chosen.

Examples
========
1) Cartesian list:
   `[a,b,c]` → `a`, `b`, `c`

2) Integer ranges with padding and sign:
   `{01:03}` → `01`, `02`, `03`
   `{-003:003:3}` → `-003`, `000`, `003`

3) Float step vs count:
   `{0:1:0.5}`   → `0`, `0.5`, `1`
   `{0:1:5}`     → `0`, `0.25`, `0.5`, `0.75`, `1`

4) Regex selector (with names):
   `@{/^A/}` with names `[AAPL, MSFT, NVDA, AMZN]` → `AAPL, AMZN`

5) Cartesian product & references:
   Spec: `a{1:3}[p,o,r]-->#{2}`
   List-like units:
     (1) `{1:3}`   → picks 1,2,3
     (2) `[p,o,r]` → picks p,o,r
   Output (cartesian + ref to (2)):
     `a1p-->p, a1o-->o, a1r-->r, a2p-->p, a2o-->o, a2r-->r, a3p-->p, a3o-->o, a3r-->r`

6) References anywhere:
   `#{2}:: a{1:3}[p,o,r]`
   → `p:: a1p, o:: a1o, r:: a1r, p:: a2p, o:: a2o, r:: a2r, p:: a3p, o:: a3o, r:: a3r`

7) Progressive list:
`>[a,b,c,d]` → `a`, `a,b`, `a,b,c`, `a,b,c,d`

CLI
===
  python expandspec.py --in "pre[foo,bar]{1:2}"
  python expandspec.py --in "[{-003:003:1}]"
  python expandspec.py --in "[@{/^A/}]" --names names.json
Options:
  --unique, --sort, --limit, --count, --sep, --json, --names, --exit-empty,
  --selftest, --selftest-verbose

Error behavior
==============
- Invalid `{...}` body or malformed tokens are kept as literal text (no hard failure).
- Unknown references (e.g., `#{99}`) are left as literal.
"""

import re
import math
import json
import sys
from itertools import product
from typing import List, Tuple, Union, Iterable, Optional, Dict
import operator as op
import ast
from simpleeval import SimpleEval
import numpy as np

# =========================
# Tokenization regexes
# =========================

_fcall_simple_re = re.compile(r"^(?P<prefix>.*)\$\{(?P<body>[^{}]+)\}(?P<suffix>.*)$")
_brace_re = re.compile(r"^(?P<prefix>.*)\{(?P<body>[^{}]+)\}(?P<suffix>.*)$")
_int_re = re.compile(r"^[+-]?\d+$")
_dictsel_re = re.compile(r"^@\{(.+)\}$")              # whole-token @{...}
_js_regex_re = re.compile(r"^/(.*?)/([im]*)$")        # /re/flags
_ref_re = re.compile(r"#\{(\d+)\}")      # #{n} where n is digits only

# =========================
# Helpers
# =========================

def _is_int_like(s: str) -> bool:
    return bool(_int_re.match(s.strip()))

def _parse_num(s: str) -> Tuple[Union[float, int], bool]:
    s = s.strip()
    if _is_int_like(s):
        return int(s), True
    return float(s), False

def _needs_pad(a: str, b: str) -> int:
    a2, b2 = a.lstrip("+-"), b.lstrip("+-")
    pad_on = (len(a2) > 1 and a2[0] == "0") or (len(b2) > 1 and b2[0] == "0")
    return max(len(a2), len(b2)) if pad_on else 0

def _compile_regex(expr: str) -> re.Pattern:
    m = _js_regex_re.match(expr.strip())
    if m:
        pat, flags_s = m.group(1), m.group(2)
        flags = 0
        if 'i' in flags_s: flags |= re.IGNORECASE
        if 'm' in flags_s: flags |= re.MULTILINE
        return re.compile(pat, flags)
    return re.compile(expr)

def _coerce_names(names: Union[None, Iterable[str], dict]) -> List[str]:
    if names is None:
        return []
    if isinstance(names, dict):
        return list(names.keys())
    return list(names)

# =========================
# Numeric range expander
# =========================

def _expand_single_brace(token: str, *, trim_for_items: bool = False) -> List[str]:
    """
    Numeric range expander for a single {...} inside token; if none, return [token] unchanged.

    Supports:
      - {a:b:step}   step is float/int step size (direction auto-corrected)
      - {a:b:N}      N is integer COUNT (inclusive linspace), unless N is a valid integer STEP that
                     divides the integer range cleanly (then treat as STEP).
    """
    tok_input = token.strip() if trim_for_items else token
    m = _brace_re.match(tok_input)
    if not m:
        return [tok_input]

    pre, suf = m.group("prefix"), m.group("suffix")
    body = m.group("body").strip()
    parts = [p.strip() for p in body.split(":")]
    if len(parts) not in (2, 3):
        return [tok_input]

    a_s, b_s = parts[0], parts[1]
    try:
        a_val, a_is_int = _parse_num(a_s)
        b_val, b_is_int = _parse_num(b_s)
    except Exception:
        return [tok_input]

    int_pad_width = _needs_pad(a_s, b_s) if (a_is_int and b_is_int) else 0

    # Determine mode
    if len(parts) == 2:
        if a_is_int and b_is_int:
            step_mode = ("step", 1 if b_val >= a_val else -1)
        else:
            return [tok_input]
    else:
        third = parts[2]
        third_is_intlike = _is_int_like(third)
        looks_floaty = any(c in third for c in ".eE")

        try:
            if looks_floaty and not third_is_intlike:
                # explicit float STEP
                step_val = float(third)
                if step_val == 0.0 or not math.isfinite(step_val):
                    return [tok_input]
                if (b_val - a_val) * step_val < 0:
                    step_val = -step_val
                step_mode = ("step", step_val)
            elif third_is_intlike:
                N = int(third)
                if N == 0:
                    return [tok_input]
                if a_is_int and b_is_int:
                    # treat as STEP only if it divides the integer range cleanly; else COUNT
                    rng = int(b_val) - int(a_val)
                    if N == 1 or (rng != 0 and rng % N == 0):
                        step_val = N if (b_val >= a_val) else -N
                        step_mode = ("step", step_val)
                    else:
                        step_mode = ("count", N)
                else:
                    step_mode = ("count", N)
            else:
                return [tok_input]
        except Exception:
            return [tok_input]

    # Formatting decision
    def _should_use_int_format() -> bool:
        if not (a_is_int and b_is_int):
            return False
        if step_mode[0] == "step":
            return float(step_mode[1]).is_integer()
        else:
            N = step_mode[1]
            return (b_val - a_val) == int(b_val - a_val) and N == abs(int(b_val - a_val)) + 1

    use_int_format = _should_use_int_format()

    def _fmt_num(x: Union[float, int]) -> str:
        if use_int_format:
            xi = int(round(x))
            if int_pad_width <= 1:
                return str(xi)
            return ("-" if xi < 0 else "") + f"{abs(xi):0{int_pad_width}d}"
        return f"{float(x):.15g}"

    # Generate
    out: List[str] = []
    if step_mode[0] == "count":
        N = step_mode[1]
        if N == 1:
            vals = [a_val]
        else:
            delta = (b_val - a_val) / (N - 1)
            vals = [a_val + i * delta for i in range(N)]
        for v in vals:
            out.append(f"{pre}{_fmt_num(v)}{suf}")
        return out

    step = step_mode[1]
    max_iters = 1_000_000
    if step > 0:
        cond = lambda v: v <= b_val or math.isclose(v, b_val, rel_tol=0, abs_tol=1e-15)
    else:
        cond = lambda v: v >= b_val or math.isclose(v, b_val, rel_tol=0, abs_tol=1e-15)

    x = a_val
    i = 0
    while cond(x):
        out.append(f"{pre}{_fmt_num(x)}{suf}")
        x = x + step
        i += 1
        if i > max_iters:
            return [tok_input]
    return out

# =========================
# Dict selector (@{...})
# =========================

def _expand_dict_selector(token: str, names: List[str], *, trim_for_items: bool = False) -> List[str] | None:
    tok_input = token.strip() if trim_for_items else token
    m = _dictsel_re.match(tok_input)
    if not m:
        return None
    if not names:
        return [tok_input]
    expr = m.group(1).strip()
    try:
        rx = _compile_regex(expr)
    except re.error:
        return [tok_input]
    out = [n for n in names if rx.search(n)]
    return out

# =========================
# Tokenizer (safe)
#   Produces parts: ("text", chunk), ("list", inner), ("zip", inner)
# =========================

def _tokenize_parts(spec: str):
    """
    Produce parts: ("text", chunk), ("list", inner)
    - Keeps @{...} and {...} as single text chunks (parsed later).
    - Recognizes [ ... ] as 'list'.
    - EXPLICIT [[...]] IS NOT SUPPORTED: raises ValueError.
    """
    parts = []
    i, n = 0, len(spec)
    buf = []

    def flush_text():
        if buf:
            parts.append(("text", "".join(buf)))
            buf.clear()

    while i < n:
        ch = spec[i]
        # >[ ... ] progressive list (prefix-accumulating)
        if ch == '>' and i + 1 < n and spec[i + 1] == '[':
            flush_text()
            j = i + 2
            depth = 1
            while j < n and depth > 0:
                cj = spec[j]
                # skip @{...} inside list
                if cj == '@' and j + 1 < n and spec[j + 1] == '{':
                    k = j + 2
                    d = 1
                    while k < n and d > 0:
                        if spec[k] == '{': d += 1
                        elif spec[k] == '}': d -= 1
                        k += 1
                    j = k
                    continue
                # skip ${...} inside list  <-- ADD THIS BLOCK
                if cj == '$' and j + 1 < n and spec[j + 1] == '{':
                    k = j + 2
                    d = 1
                    while k < n and d > 0:
                        if spec[k] == '{': d += 1
                        elif spec[k] == '}': d -= 1
                        k += 1
                    j = k
                    continue
                # skip { ... } inside list
                if cj == '{':
                    k = j + 1
                    d = 1
                    while k < n and d > 0:
                        if spec[k] == '{': d += 1
                        elif spec[k] == '}': d -= 1
                        k += 1
                    j = k
                    continue
                if cj == '[':
                    # nested '[' char inside item → keep going as char
                    j += 1
                    continue
                if cj == ']':
                    depth -= 1
                    j += 1
                    break
                j += 1
            inner = spec[i + 2 : j - 1] if j - 1 >= i + 2 else ""
            parts.append(("plist", inner))
            i = j
            continue
        # @{ ... } selector → single text chunk
        if ch == '@' and i + 1 < n and spec[i + 1] == '{':
            flush_text()
            j = i + 2
            depth = 1
            while j < n and depth > 0:
                if spec[j] == '{': depth += 1
                elif spec[j] == '}': depth -= 1
                j += 1
            parts.append(("text", spec[i:j]))
            i = j
            continue
        # ${ ... } simpleeval block → single text chunk (balanced)
        if ch == '$' and i + 1 < n and spec[i + 1] == '{':
            flush_text()
            j = i + 2
            depth = 1
            while j < n and depth > 0:
                cj = spec[j]
                # skip nested { } properly
                if cj == '{':
                    depth += 1
                elif cj == '}':
                    depth -= 1
                j += 1
            parts.append(("text", spec[i:j]))  # include entire ${...}
            i = j
            continue
        # { ... } numeric range → single text chunk (NO post-name support)
        if ch == '{':
            flush_text()
            j = i + 1
            depth = 1
            while j < n and depth > 0:
                if spec[j] == '{': depth += 1
                elif spec[j] == '}': depth -= 1
                j += 1
            parts.append(("text", spec[i:j]))
            i = j
            continue

        # [[ ... ]] -> ERROR (explicit zip not supported)
        if ch == '[' and i + 1 < n and spec[i + 1] == '[':
            raise ValueError("Explicit zip [[...]] is not supported. Use [...]+#{n} instead.")

        # [ ... ] cartesian list
        if ch == '[':
            flush_text()
            j = i + 1
            depth = 1
            while j < n and depth > 0:
                cj = spec[j]
                # skip @{...} inside list
                if cj == '@' and j + 1 < n and spec[j + 1] == '{':
                    k = j + 2
                    d = 1
                    while k < n and d > 0:
                        if spec[k] == '{': d += 1
                        elif spec[k] == '}': d -= 1
                        k += 1
                    j = k
                    continue
                # skip ${...} inside list  <-- ADD THIS
                if cj == '$' and j + 1 < n and spec[j + 1] == '{':
                    k = j + 2
                    d = 1
                    while k < n and d > 0:
                        if spec[k] == '{': d += 1
                        elif spec[k] == '}': d -= 1
                        k += 1
                    j = k
                    continue
                # skip { ... } inside list
                if cj == '{':
                    k = j + 1
                    d = 1
                    while k < n and d > 0:
                        if spec[k] == '{': d += 1
                        elif spec[k] == '}': d -= 1
                        k += 1
                    j = k
                    continue
                if cj == '[':
                    # nested normal '[' inside list is just a char; keep going
                    j += 1
                    continue
                if cj == ']':
                    depth -= 1
                    j += 1
                    break
                j += 1
            inner = spec[i + 1 : j - 1] if j - 1 >= i + 1 else ""
            parts.append(("list", inner))
            i = j
            continue

        # default literal
        buf.append(ch)
        i += 1

    flush_text()
    return parts

def _split_into_parts(spec: str):
    return _tokenize_parts(spec)


# =========================
# Segments
# =========================

class _Seg: ...
class _Lit(_Seg):
    def __init__(self, text: str): self.text = text
class _ListChoices(_Seg):
    def __init__(self, choices: List[str], pid: Optional[str] = None):
        self.choices = choices        # cartesian list choices
        self.pid = pid                # list-like ordinal id as string (for #{id})

class _ZipList(_Seg):
    def __init__(self, choices: List[str], zip_pid: Optional[str], pid: Optional[str] = None):
        self.choices = choices      # items of the list
        self.zip_pid = zip_pid      # producer pid that this list zips to
        self.pid = pid              # list-like ordinal id (string), used by #{...} refs

class _Producer(_Seg):
    def __init__(self, pid: str, values: List[str]): self.pid, self.values = pid, values
class _Ref(_Seg):
    def __init__(self, pid: str): self.pid = pid

# =========================
# Split text into [Lit/Ref] segments
# =========================

def _split_list_items(s: str) -> List[str]:
    """
    Split the inner of [ ... ] on top-level commas only.
    Ignores commas inside {...}, ${...}, @{...}.
    """
    out, buf = [], []
    depth = 0  # braces depth for {...}, @{...}, ${...}
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == '{':
            depth += 1
            buf.append(ch)
        elif ch == '}':
            if depth > 0:
                depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            item = "".join(buf).strip()
            if item:
                out.append(item)
            buf = []
        else:
            buf.append(ch)
        i += 1
    item = "".join(buf).strip()
    if item:
        out.append(item)
    return out

def _split_fcall(text: str):
    """
    Return (pre, body, suf) for the FIRST balanced ${...} in text, or None.
    Scans braces so inner lists/dicts don't confuse it.
    """
    i = text.find("${")
    if i < 0:
        return None
    j = i + 2
    n = len(text)
    depth = 1
    while j < n and depth > 0:
        ch = text[j]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        j += 1
    if depth != 0:
        return None
    pre = text[:i]
    body = text[i+2:j-1]
    suf = text[j:]
    return pre, body, suf

def _split_text_with_refs(text: str) -> List[_Seg]:
    segs: List[_Seg] = []
    idx = 0
    for mref in _ref_re.finditer(text):
        if mref.start() > idx:
            segs.append(_Lit(text[idx:mref.start()]))
        segs.append(_Ref(mref.group(1)))
        idx = mref.end()
    if idx < len(text):
        segs.append(_Lit(text[idx:]))
    if not segs:
        segs.append(_Lit(""))
    return segs



ALLOWED_OPS = {
    ast.Add: op.add, 
    ast.Sub: op.sub, 
    ast.Mult: op.mul, 
    ast.Div: op.truediv,
    ast.Pow: op.pow, 
    ast.USub: op.neg, 
    ast.UAdd: op.pos,
    ast.FloorDiv: op.floordiv, 
    ast.Mod: op.mod
}

NAMES = {}

def set_const(name: str, value: complex | float) -> None:
    NAMES[name.strip().lower()] = complex(value)

def seq(start,end,num):
    vec = np.linspace(start,end,num)
    out=[f"{x}" for x in vec]
    return out

def ctoi(z):
    return int(z.real)

def rv(z):
    return z.real

FUNCS = {
    # pick only what you truly need
    "seq": seq, 
    "ctoi": ctoi,
    "rv": rv,
}

# in expandspec.py, near top
DEBUG_EVAL = True  # flip to True temporarily

# replace _eval_fcall_body with same body + debug prints
def _eval_fcall_body(expr: str) -> Optional[list[str]]:
    s = expr.strip()
    try:
        se = SimpleEval(names=NAMES, functions=FUNCS, operators=ALLOWED_OPS)
        v = se.eval(s)
        if isinstance(v, str):
            return [v]
        try:
            return [str(x) for x in v]
        except TypeError:
            return str(v)
    except Exception as e:
        if DEBUG_EVAL:
            print(f"[expandspec] simpleeval failed for '{s}': {type(e).__name__}: {e}", file=sys.stderr)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, str):
            return [v]
        try:
            return [str(x) for x in v]
        except TypeError:
            return str(v)
    except Exception as e2:
        if DEBUG_EVAL:
            print(f"[expandspec] literal_eval failed for '{s}': {type(e2).__name__}: {e2}", file=sys.stderr)
        return None

def _expand_single_fcall(token: str, *, trim_for_items: bool = False) -> list[str]:
    """
    Expand one ${...} (with optional prefix/suffix) using the balanced splitter.
    If not a valid ${...} or evaluation fails, return [token] unchanged.
    """
    tok_input = token.strip() if trim_for_items else token

    # Use the same balanced finder used at top level
    found = _split_fcall(tok_input)
    if not found:
        return [tok_input]
    pre, body, suf = found

    vals = _eval_fcall_body(body)
    if not vals:
        return [tok_input]

    return [f"{pre}{v}{suf}" for v in vals]
    
# =========================
# Expand list blocks (cartesian or zip) into strings
# =========================

def _expand_list_block(block_text: str, names: List[str]) -> List[str]:
    """
    Expand [a,b,c{1:3},@{regex}] into a flat list (union).
    References inside list items are treated literally by design.
    """
    items = _split_list_items(block_text)
    out: List[str] = []
    for it in items:

        sel = _expand_dict_selector(it, names, trim_for_items=True)
        if sel is not None:
            out.extend(sel)
            continue

         # NEW: ${...} simpleeval in list items
        fvals = _expand_single_fcall(it, trim_for_items=True)
        if not (len(fvals) == 1 and fvals[0] == it):
            out.extend(fvals)
            continue

        out.extend(_expand_single_brace(it, trim_for_items=True))

    return out

def _expand_progressive_list(block_text: str, names: List[str]) -> List[str]:
    """
    Expand >[a,b,c] into progressive comma-joined prefixes:
      ['a', 'a,b', 'a,b,c']
    Items use the same item-expansion rules as normal lists.
    """
    items = _expand_list_block(block_text, names)
    out: List[str] = []
    acc: List[str] = []
    for it in items:
        acc.append(it)
        out.append(",".join(acc))
    return out

# =========================
# Parse plain text into segments (Lit/Ref/Producer or ListChoices from dictsel)
# =========================

def _parse_plain_to_segments(text: str, names: List[str], ordinal_start: int) -> Tuple[List[_Seg], int]:
    """
    Parse a plain text chunk into segments:
      - Whole-token @{...} -> _ListChoices (cartesian)
      - Embedded {...}     -> _Producer (list-like unit), no naming, ordinal only
      - Otherwise          -> split into _Lit / _Ref
    """
    # whole-segment dict selector?
    sel = _expand_dict_selector(text, names, trim_for_items=False)
    if sel is not None:
        return ([_ListChoices(sel)], ordinal_start)
    
    # --- ${ ... } via simpleeval (balanced; works embedded or standalone) ---
    fsplit = _split_fcall(text)
    if fsplit is not None:
        pre_text, body_expr, suf_text = fsplit
        core_vals = _eval_fcall_body(body_expr)
        if not core_vals:
            return (_split_text_with_refs(text), ordinal_start)

        pid = str(ordinal_start)
        next_ord = ordinal_start + 1

        segs: List[_Seg] = []
        segs.extend(_split_text_with_refs(pre_text))
        segs.append(_Producer(pid, core_vals))   # evaluates to list-like unit
        segs.extend(_split_text_with_refs(suf_text))
        return (segs, next_ord)


    # embedded { ... } numeric producer (no @name support)
    m = _brace_re.match(text)
    if m:
        pre_text, suf_text = m.group("prefix"), m.group("suffix")
        body = m.group("body").strip()

        # expand only the body to get core values
        core_vals = _expand_single_brace("{" + body + "}")
        # if not a valid {...} expansion, fall back to literal+refs parsing
        if not core_vals or (len(core_vals) == 1 and core_vals[0] == "{" + body + "}"):
            return (_split_text_with_refs(text), ordinal_start)

        # producer id is the current ordinal; advance for the next one
        pid = str(ordinal_start)
        next_ord = ordinal_start + 1

        segs: List[_Seg] = []
        segs.extend(_split_text_with_refs(pre_text))
        segs.append(_Producer(pid, core_vals))   # producer emits only the core values
        segs.extend(_split_text_with_refs(suf_text))
        return (segs, next_ord)

    # no braces → split by refs (or literal)
    return (_split_text_with_refs(text), ordinal_start)

# =========================
# Build segments from parts, assigning zip lists
# =========================

def _segments_from_parts(parts: List[Tuple[str, str]], names: List[str]) -> List[_Seg]:
    """
    Build the flat segment list with these rules:
      • All lists are cartesian (no implicit adjacency zip, no explicit zip).
      • Whole-token @{...} selectors become cartesian lists.
      • Numeric {...} inside text becomes a producer unit.
      • Each list-like unit (producer or list) gets an ordinal (1-based) so #{n} can reference it.
    """
    segs: List[_Seg] = []
    ordinal_for_producers = 1   # local producer counter (not used for refs)
    listlike_ordinal = 1        # the ONLY ordinal exposed to #{n}

    def _append_producer(p: _Producer):
        nonlocal listlike_ordinal
        # Store the exposed ordinal for this producer
        p.oid = str(listlike_ordinal)  # type: ignore[attr-defined]
        listlike_ordinal += 1
        segs.append(p)

    for kind, content in parts:
        if kind == "list":
            choices = _expand_list_block(content, names)
            if not choices:
                return []
            lc = _ListChoices(choices, pid=str(listlike_ordinal))
            listlike_ordinal += 1
            segs.append(lc)
        elif kind == "plist":
            choices = _expand_progressive_list(content, names)
            if not choices:
                return []
            lc = _ListChoices(choices, pid=str(listlike_ordinal))
            listlike_ordinal += 1
            segs.append(lc)

        else:
            sgs, ordinal_for_producers = _parse_plain_to_segments(content, names, ordinal_for_producers)
            for s in sgs:
                if isinstance(s, _Producer):
                    _append_producer(s)
                else:
                    segs.append(s)

    return segs

# =========================
# Public API
# =========================

def expand_cartesian_lists(spec: str, *, names: Union[None, Iterable[str], dict] = None) -> List[str]:
    """
    Expand with:
      • Numeric ranges {a:b[:c]} create producer list-like units.
      • [ ... ] are cartesian lists.
      • @{...} as whole-token expands to a cartesian list.
      • #{n} references the n-th list-like unit (1-based), which can be either a producer
        or a list; it yields the currently selected value from that unit.
      • No explicit zip syntax; no naming; no promotion.
    """
    name_list = _coerce_names(names)
    parts = _split_into_parts(spec)
    segs = _segments_from_parts(parts, name_list)
    if not segs:
        return []

    # Build dimensions in left-to-right order; track list-like ordinals
    dims: List[List[Union[int, str, None]]] = []
    # Map exposed ordinal string -> (kind, seg_ref, dim_index)
    ll_map: Dict[str, Tuple[str, object, int]] = {}

    current_ll_ord = 1
    for s in segs:
        if isinstance(s, _Producer):
            vals = s.values
            dim_index = len(dims)
            dims.append(list(range(len(vals))) or [0])  # index selection
            # record this list-like ordinal for refs
            ll_map[str(current_ll_ord)] = ("producer", s, dim_index)
            current_ll_ord += 1

        elif isinstance(s, _ListChoices):
            dim_index = len(dims)
            dims.append(s.choices if s.choices else [""])
            ll_map[str(current_ll_ord)] = ("list", s, dim_index)
            current_ll_ord += 1

        else:  # _Lit, _Ref
            dims.append([None])

    results: List[str] = []
    for combo in product(*dims):
        # Render immediate pieces (leave #{n} placeholders for a final pass)
        out_parts: List[str] = []
        for s, choice in zip(segs, combo):
            if isinstance(s, _Lit):
                out_parts.append(s.text)
            elif isinstance(s, _ListChoices):
                out_parts.append(str(choice))
            elif isinstance(s, _Producer):
                out_parts.append(s.values[int(choice)])
            elif isinstance(s, _Ref):
                out_parts.append(f"#{{{s.pid}}}")
            else:
                out_parts.append("")

        # Final pass: resolve #{n} using ll_map and this combo
        def _ref_replace(m: re.Match) -> str:
            pid = m.group(1)  # numeric ordinal string
            if pid not in ll_map:
                return m.group(0)
            kind, seg_ref, dim_index = ll_map[pid]
            sel = combo[dim_index]
            if kind == "producer":
                sprod: _Producer = seg_ref  # type: ignore[assignment]
                try:
                    return sprod.values[int(sel)]
                except Exception:
                    return m.group(0)
            else:  # "list"
                # selection is already the chosen string
                try:
                    return str(sel)
                except Exception:
                    return m.group(0)

        rendered = _ref_re.sub(_ref_replace, "".join(out_parts))
        results.append(rendered)

    return results

def _selftest(verbose: bool = False) -> int:
    """
    Tests for the 'all cartesian; refs by number only; no [[...]]' rules.
    """
    def eq(got, exp, msg=""):
        nonlocal fails, passed
        if got == exp:
            passed += 1
            if verbose:
                print(f"✅ {msg or 'ok'}")
        else:
            fails += 1
            print("❌", msg or "mismatch")
            print("   expected:", exp)
            print("   got     :", got)

    def run(spec, expected, names=None, msg=""):
        got = expand_cartesian_lists(spec, names=names)
        eq(got, expected, msg or spec)

    def run_count(spec, expected_count, names=None, msg=""):
        got = expand_cartesian_lists(spec, names=names)
        ok = (len(got) == expected_count)
        nonlocal fails, passed
        if ok:
            passed += 1
            if verbose:
                print(f"✅ {msg or spec} (count={expected_count})")
        else:
            fails += 1
            print("❌", msg or spec)
            print("   expected count:", expected_count)
            print("   got count     :", len(got))
            if verbose and got:
                print("   sample        :", got[:min(8, len(got))])

    def run_raises(spec, msg=""):
        nonlocal fails, passed
        try:
            expand_cartesian_lists(spec)
        except ValueError:
            passed += 1
            if verbose:
                print(f"✅ {msg or spec} (raised ValueError as expected)")
        except Exception as e:
            fails += 1
            print("❌", msg or spec)
            print("   expected ValueError, got:", type(e).__name__, str(e))
        else:
            fails += 1
            print("❌", msg or spec)
            print("   expected ValueError, but no exception was raised")

    fails = 0
    passed = 0

    # ---------- Basics ----------
    run("[a,b,c]", ["a","b","c"], msg="cartesian list")
    run("pre{1:3}suf", ["pre1suf","pre2suf","pre3suf"], msg="int range")
    run("{-003:003:1}", ["-003","-002","-001","000","001","002","003"], msg="signed padded ints")
    run("[{-003:003:1}]",
        ["-003","-002","-001","000","001","002","003"], msg="padded ints in list")

    # ---------- Floats & count ----------
    run("[{1:2:1e-1}]",
        ["1","1.1","1.2","1.3","1.4","1.5","1.6","1.7","1.8","1.9","2"],
        msg="float step as list")

    run("x{0:1:5}y",
        ["x0y","x0.25y","x0.5y","x0.75y","x1y"],
        msg="count mode (inclusive 5 samples)")

    # ---------- Cartesian products ----------
    run("[x,y]{01:02}", ["x01","x02","y01","y02"], msg="cartesian multiply")
    run_count("a{1:3}[p,o,r]", 9, msg="producer × list (3*3)")
    run_count("a{1:3}[p,o,r][P,O,R]", 27, msg="three cartesian dims (3*3*3)")

    # ---------- Dict selector ----------
    sample_names = ["AAPL","MSFT","NVDA","AMZN","META","ORCL","IBM","alpha","Beta"]
    run("[@{/^A/}]", ["AAPL","AMZN"], names=sample_names, msg="regex selector ^A (case-sensitive)")
    run("@{/^a/i}", ["AAPL","AMZN","alpha"], names=sample_names, msg="regex selector ^a with /i flag")
    run("@{/A/}", ["AAPL","NVDA","AMZN","META"], names=sample_names, msg="regex selector anywhere A")

    # ---------- Refs by ordinal ONLY ----------
    run("a{1:3}b#{1}", ["a1b1","a2b2","a3b3"], msg="#{1} refers to first list-like unit (producer)")
    run("[x,y]#{1}", ["xx","yy"], msg="#{1} refers to first list-like unit (list)")

    # Refs can be anywhere in the text; they resolve after selection
    run("#{2}:: a{1:3}[p,o,r]",
        ["p:: a1p","o:: a1o","r:: a1r",
         "p:: a2p","o:: a2o","r:: a2r",
         "p:: a3p","o:: a3o","r:: a3r"],
        msg="ref before the list resolves")

    run("a{1:2}[p,o]/#{2}",
        ["a1p/p","a1o/o","a2p/p","a2o/o"],
        msg="#{2} mirrors the 2nd list-like unit across cartesian")

    # ---------- NEW: Unknown reference left literal ----------
    run("a{1:2}X#{9}", ["a1X#{9}","a2X#{9}"], msg="unknown ref remains literal")

    # ---------- NEW: Multiple refs to same dim ----------
    run("A{1:2}[x,y]-#{1}-#{2}-#{2}",
        ["A1x-1-x-x","A1y-1-y-y","A2x-2-x-x","A2y-2-y-y"],
        msg="reusing refs from producer and list")

    # ---------- NEW: Count vs step disambiguation on integers ----------
    # range 0..10 with third=5 divides exactly => STEP of 5
    run("{0:10:5}", ["0","5","10"], msg="int third that divides range -> step")
    # range 0..10 with third=4 doesn't divide => COUNT of 4, inclusive
    run("{0:10:4}", ["0","3.33333333333333","6.66666666666667","10"], msg="int third that does not divide -> count")

    # ---------- NEW: Negative float step and direction ----------
    run("{1:0:-0.25}", ["1","0.75","0.5","0.25","0"], msg="negative float step descending")

    # ---------- NEW: Empty list yields empty expansion ----------
    run_count("[]", 0, msg="empty list -> empty result")

    # ---------- NEW: Dict selector inside a list item ----------
    run("[pre,@{/^A/},post]{1:2}--#{2}",
        ["pre1--1","pre2--2",
         "AAPL1--1","AAPL2--2",
         "AMZN1--1","AMZN2--2",
         "post1--1","post2--2"],
        names=["AAPL","MSFT","AMZN"],
        msg="selector as list item expands inline")

    # ---------- NEW: Mixed producers + lists + refs to later dim ----------
    run("X{1:2}Y{01:02}[a,b]::#{3}-#{1}-#{2}-#{3}",
        ["X1Y01a::a-1-01-a","X1Y01b::b-1-01-b",
         "X1Y02a::a-1-02-a","X1Y02b::b-1-02-b",
         "X2Y01a::a-2-01-a","X2Y01b::b-2-01-b",
         "X2Y02a::a-2-02-a","X2Y02b::b-2-02-b"],
        msg="refs to third list-like dim")
    
    # ---------- NEW: Empty list yields empty expansion ----------
    run_count("[]", 0, msg="empty list -> empty result")
 
    # ---------- NEW: Progressive list ----------
    run(">[a,b,c,d]", ["a","a,b","a,b,c","a,b,c,d"], msg="progressive list")
    run("X>[p,o]Y", ["XpY","Xp,oY"], msg="progressive list inside text, 2 items")

    # ---------- NEW: ${...} simpleeval calls ----------

    # Top-level literal list
    run('${["A","B"]}', ["A", "B"], msg="${...} top-level literal list")

    # Embedded in text
    run('n:1e5,${["A","B"]},w:20',
        ["n:1e5,A,w:20", "n:1e5,B,w:20"],
        msg="${...} embedded in text")

    # Inside cartesian list
    run('[${["A","B"]}]', ["A", "B"], msg="${...} inside [ ... ] list")

    # Progressive list containing ${...}
    run('>[${["a","b"]},x]', ["a", "a,b", "a,b,x"], msg="progressive list with ${...}")

    # Prefix/suffix around ${...}
    run('pre${["u","v"]}suf', ["preusuf", "prevsuf"], msg="prefix/suffix with ${...}")

    # Cartesian with another producer
    run('A${["x","y"]}B{1:2}',
        ["AxB1", "AxB2", "AyB1", "AyB2"],
        msg="${...} cartesian with numeric {..}")

    # Reference to ${...} (it is a list-like unit)
    run('${["p","q"]}::#{1}', ["p::p", "q::q"], msg="reference mirrors ${...} selection")

    # Comma inside elements must not split list items
    run('[${["A,B","C"]}]', ["A,B", "C"], msg="commas inside ${...} items are preserved")

    # Call a registered function via FUNCS (seq is defined above)
    run('${seq(0,1,3)}', ["0.0", "0.5", "1.0"], msg="${seq(...)} via FUNCS")

    # Invalid expression -> left literal
    run('${not_defined}', ['${not_defined}'], msg="invalid ${...} remains literal")

    # ---- Inject a deterministic jsample and test single- and two-digit bounds ----
    _FUNCS_SAVE = dict(FUNCS)
    try:
        def _jsample(a, b):
            # inclusive numeric strings like your julia spec expects
            return [f"{k}" for k in range(int(a), int(b) + 1)]
        FUNCS["jsample"] = _jsample

        run('${jsample(9,9)}',  ["9"], msg="jsample single value")
        run('${jsample(9,10)}', ["9", "10"], msg="jsample two-digit upper bound")

        run('n:5e6,eqn:10,c:${jsample(9,10)},w:2',
            ["n:5e6,eqn:10,c:9,w:2", "n:5e6,eqn:10,c:10,w:2"],
            msg="embedded jsample(9,10) in full spec")
    finally:
        FUNCS.clear(); FUNCS.update(_FUNCS_SAVE)


    if verbose:
        print(f"\nPassed: {passed}, Failed: {fails}")
    return fails

# =========================
# CLI
# =========================

def _read_input_arg_or_stdin(arg: Union[str, None]) -> str:
    if arg is not None:
        return arg
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""

def _load_names_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    try:
        obj = json.loads(data)
        if isinstance(obj, dict):
            return list(obj.keys())
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return [line.strip() for line in data.splitlines() if line.strip()]

def _cli():
    import argparse

    p = argparse.ArgumentParser(
        prog="expandspec",
        description="Expand specs: lists [..], ranges {..}, selectors @{..}, refs #{n} (n is the list-like ordinal).",
    )
    p.add_argument("--in", dest="spec", help="Spec string to expand. If omitted, read from stdin.")
    p.add_argument("--names", help="Path to names file (json or newline list) for @{...} selectors.")
    p.add_argument("--sep", default="\n", help="Separator used when printing results (default: newline).")
    p.add_argument("--json", action="store_true", help="Emit JSON array instead of joined text.")
    p.add_argument("--unique", action="store_true", help="Deduplicate results (preserve order).")
    p.add_argument("--sort", action="store_true", help="Sort results (after dedup if enabled).")
    p.add_argument("--limit", type=int, default=0, help="Limit number of printed items (0 = no limit).")
    p.add_argument("--count", action="store_true", help="Only print the count of results.")
    p.add_argument("--exit-empty", action="store_true", help="Exit with code 2 if expansion is empty.")
    # --- add these:
    p.add_argument("--selftest", action="store_true", help="Run built-in unit tests and exit.")
    p.add_argument("--selftest-verbose", action="store_true", help="Verbose self-test output.")

    args = p.parse_args()

    # --- run tests and exit
    if args.selftest or args.selftest_verbose:
        failures = _selftest(verbose=args.selftest_verbose)
        sys.exit(0 if failures == 0 else 2)

    spec = _read_input_arg_or_stdin(args.spec)
    names = _load_names_from_file(args.names) if args.names else None

    try:
        out = expand_cartesian_lists(spec, names=names)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    if args.unique:
        seen = set()
        tmp = []
        for x in out:
            if x not in seen:
                seen.add(x)
                tmp.append(x)
        out = tmp
    if args.sort:
        out = sorted(out)
    if args.limit and args.limit > 0:
        out = out[:args.limit]

    if args.count:
        print(len(out))
        print()
        return

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print()
        return

    print(args.sep.join(out))
    print()

    if args.exit_empty and not out:
        sys.exit(2)

if __name__ == "__main__":
    _cli()
