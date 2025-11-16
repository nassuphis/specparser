#!/usr/bin/env python
# specparser.py — minimal CLI-safe parser (names kept as strings)

import re
import ast
from simpleeval import SimpleEval, NameNotDefined, InvalidExpression, DEFAULT_OPERATORS
import operator as op
import math
import cmath
import json
import argparse
import sys
import numpy as np

__all__ = [
    "SpecParseError",
    "set_const",
    "extract_used_names",
    "parse_chain",
    "parse_names_and_args",
    "parse_args_only",
]

# ---------- errors ----------

class SpecParseError(ValueError):
    pass

# ---------- scalar parsing ----------

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

NAMES = {
    "pi": math.pi, 
    "tau": math.tau, 
    "e": math.e,
    "inf": float("inf"), 
    "nan": float("nan"),
    "j": 1j, 
    "i": 1j,  # allow i for imaginary
    "zero": 0 + 0j,
    "one":  1 + 1j,
}

def set_const(name: str, value: complex | float) -> None:
    NAMES[name.strip().lower()] = complex(value)

def get_const(name: str):
    return complex(NAMES[name.strip().lower()])

def real_max(x,y):
    return max(x.real,y.real)

def real_min(x,y):
    return min(x.real,y.real)

def lerp(start,end,i,n):
    x=np.linspace(start,end,n)
    return x[i-1]



FUNCS = {
    # pick only what you truly need
    "sin": cmath.sin, 
    "cos": cmath.cos, 
    "tan": cmath.tan,
    "sqrt": cmath.sqrt, 
    "log": cmath.log, 
    "exp": cmath.exp,
    "abs": abs,
    "min": real_min, 
    "max": real_max,
    "lerp": lerp,
}

def simple_eval_number(expr: str) -> complex:
    se = SimpleEval(names=NAMES, functions=FUNCS, operators=ALLOWED_OPS)
    v = se.eval(expr)
    return complex(v)

_frac_exp_re = re.compile(
    r"""
    ^\s*
    ([+-]?\d*\.?\d+)       # base
    (?:e([+-]?\d*\.?\d+))? # optional fractional exponent
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

def _parse_scalar(tok: str) -> complex:
    t = tok.strip().lower()
    if not t:  return 0.0 + 0.0j
    try:
        v = simple_eval_number(t)
        return complex(v)
    except Exception:
        pass
    m = _frac_exp_re.match(t)
    if m:
        base = float(m.group(1))
        exp_str = m.group(2)
        if exp_str is None:
            return complex(base)
        exp = float(exp_str)
        val = base * math.exp(exp * math.log(10))
        return complex(val)

    raise SpecParseError(f"Invalid scalar literal: {tok!r}")

# ---------- chain parsing (CLI-safe) ----------

SKIP_PREFIXES = {"!"}
def _is_skipped_name(raw: str) -> bool:
    s = raw.lstrip()
    return bool(s) and s[0] in SKIP_PREFIXES

def _strip_nonfunctional_prefix(raw: str) -> str:
    s = raw.lstrip()
    return s[1:] if s.startswith("_") else raw

def _split_top_level(s: str, sep: str) -> list[str]:
    out, buf, depth = [], [], 0
    opens = "([{"; closes = ")]}"
    pairs = dict(zip(closes, opens))
    for ch in s:
        if ch in opens:
            depth += 1
        elif ch in closes:
            # don't let depth go negative if malformed; we stay robust
            if depth > 0:
                depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [x for x in out if x != ""]

def extract_used_names(chain: str) -> set[str]:
    if not chain.strip():
        return set()
    items = _split_top_level(chain, ",")
    return {item.split(":", 1)[0].lower() for item in items if item}


def split_chain(chain: str):
    out = {}
    if not chain.strip():
        return out
    items = _split_top_level(chain, ",")
    for item in items:
        parts = _split_top_level(item, ":")  # colon only at top level
        if not parts:
            continue
        name_raw = parts[0].strip()
        if _is_skipped_name(name_raw):
            continue
        name = _strip_nonfunctional_prefix(name_raw)
        name = name.lower()
        out[name] = parts[1:]
    return out

def parse_chain(chain: str, MAXA: int = 12):
    out = []
    if not chain.strip():
        return out
    items = _split_top_level(chain, ",")
    for item in items:
        parts = _split_top_level(item, ":")
        if not parts:
            continue
        name_raw = parts[0].strip()
        if _is_skipped_name(name_raw):
            continue
        name = _strip_nonfunctional_prefix(name_raw)
        name = name.lower()
        arg_tokens = parts[1:MAXA+1]
        args = tuple(_parse_scalar(tok) for tok in arg_tokens)
        out.append((name, args))
    return out

def parse_names_and_args(chain: str, MAXA: int = 12):
    specs = parse_chain(chain, MAXA=MAXA)
    K = len(specs)
    names = [None] * K
    args = np.zeros((K, MAXA), np.complex128)
    for i, (name, argv) in enumerate(specs):
        names[i] = name
        if argv:
            args[i, :len(argv)] = np.asarray(argv, dtype=np.complex128)
    return names, args

def parse_args_only(chain: str, MAXA: int = 12):
    _, A = parse_names_and_args(chain, MAXA=MAXA)
    return A

# ---------- CLI ----------

def _parse_const_kv(text: str):
    # name=value ; value parsed with same scalar rules
    if "=" not in text:
        raise argparse.ArgumentTypeError("const must be NAME=VALUE")
    k, v = text.split("=", 1)
    k = k.strip().lower()
    if not k:
        raise argparse.ArgumentTypeError("const name is empty")
    try:
        val = _parse_scalar(v)
    except SpecParseError as e:
        raise argparse.ArgumentTypeError(str(e))
    return k, val

def _complex_to_repr(z: complex) -> str:
    # stable CLI-friendly representation
    return f"{z.real:+.12g}{z.imag:+.12g}j"

def _approx_eq(z: complex, w: complex, tol=1e-12) -> bool:
    return abs(z.real - w.real) <= tol and abs(z.imag - w.imag) <= tol

def _check(label: str, got: complex, expected: complex, verbose: bool) -> tuple[bool, str]:
    ok = _approx_eq(got, expected)
    if verbose:
        return ok, f"{'✅' if ok else '❌'} {label}\n   got: {got!r}\n   exp: {expected!r}"
    else:
        return ok, f"{'✅' if ok else '❌'} {label}"

def _run_selftests(verbose: bool=False) -> int:
    """
    Runs a battery of self-tests for scalar parsing, function calls, constants,
    fractional exponents, and top-level splitting in chains.
    Returns 0 on success, non-zero on failure.
    """
    passed = 0
    failed = 0
    logs: list[str] = []

    # snapshot global names (so tests don't leak)
    names_snapshot = dict(NAMES)

    try:
        # ---- scalar + constants ----
        ok, msg = _check("pi constant", _parse_scalar("pi"), complex(math.pi), verbose)
        logs.append(msg); passed += ok; failed += (not ok)

        ok, msg = _check("2**8", _parse_scalar("2**8"), complex(256.0), verbose)
        logs.append(msg); passed += ok; failed += (not ok)

        # Python doesn't accept 1e2.1; your fallback treats it as 10**2.1
        val_1e = 10.0 ** 2.1
        ok, msg = _check("1e2.1 fallback", _parse_scalar("1e2.1"), complex(val_1e), verbose)
        logs.append(msg); passed += ok; failed += (not ok)

        # complex literal and arithmetic
        ok, msg = _check("3+4j", _parse_scalar("3+4j"), complex(3, 4), verbose)
        logs.append(msg); passed += ok; failed += (not ok)

        # ---- functions (cmath) ----
        # sin(pi) ~ 0
        ok, msg = _check("sin(pi)", _parse_scalar("sin(pi)"), complex(cmath.sin(math.pi)), verbose)
        logs.append(msg); passed += ok; failed += (not ok)

        # set const a=3 and test cos(a)
        set_const("a", _parse_scalar("3"))
        ok, msg = _check("cos(a) with a=3", _parse_scalar("cos(a)"), complex(cmath.cos(3)), verbose)
        logs.append(msg); passed += ok; failed += (not ok)

        # custom real_max/real_min via max/min mapping
        # real_max(3+4j, 1+100j) -> max(3,1) = 3 (returned as real 3+0j)
        ok, msg = _check("max(a, 1) real part", _parse_scalar("max(a,1)"), complex(max(3.0, 1.0), 0), verbose)
        logs.append(msg); passed += ok; failed += (not ok)

        # nested functions with internal commas
        # max(sin(pi), cos(0)) -> max(0,1) = 1
        ok, msg = _check("nested max(sin(pi),cos(0))",
                         _parse_scalar("max(sin(pi), cos(0))"), complex(1.0, 0.0), verbose)
        logs.append(msg); passed += ok; failed += (not ok)

        # ---- top-level splitting in chains ----
        chain = "op:1:max(2,3):min(4,5),foo:sin(pi):cos(a)"
        names, A = parse_names_and_args(chain, MAXA=12)
        # names should be ['op', 'foo']
        ok = (names == ['op', 'foo'])
        msg = f"{'✅' if ok else '❌'} split chain names -> {names}"
        logs.append(msg); passed += ok; failed += (not ok)

        # args shape & some values:
        # op args: [1, max(2,3)=3, min(4,5)=4]
        # foo args: [sin(pi) ~ 0, cos(a) ~ cos(3)]
        exp0 = [complex(1,0), complex(3,0), complex(4,0)]
        got0 = [A[0,0], A[0,1], A[0,2]]
        ok = all(_approx_eq(g, e) for g, e in zip(got0, exp0))
        msg = f"{'✅' if ok else '❌'} chain args[0] head -> {got0}"
        logs.append(msg); passed += ok; failed += (not ok)

        exp1 = [complex(cmath.sin(math.pi)), complex(cmath.cos(3))]
        got1 = [A[1,0], A[1,1]]
        ok = all(_approx_eq(g, e) for g, e in zip(got1, exp1))
        msg = f"{'✅' if ok else '❌'} chain args[1] head -> {got1}"
        logs.append(msg); passed += ok; failed += (not ok)

        # extract_used_names should ignore internals
        used = extract_used_names(chain)
        ok = (used == {'op', 'foo'})
        msg = f"{'✅' if ok else '❌'} extract_used_names -> {used}"
        logs.append(msg); passed += ok; failed += (not ok)

    finally:
        # restore original names
        NAMES.clear()
        NAMES.update(names_snapshot)

    # print log
    for line in logs:
        print(line)

    total = passed + failed
    print(f"\nPassed: {passed}, Failed: {failed}, Total: {total}")
    return 0 if failed == 0 else 1

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse pipeline spec 'op:arg1:arg2,op2:arg1,...' into names and complex args."
    )
    ap.add_argument("--spec", required=False, help="Spec string (no quoting/escaping).")
    ap.add_argument("--maxa", type=int, default=12, help="Max args per op (default: 12).")
    ap.add_argument(
        "--const", action="append", default=[],
        help="Add/override constant as NAME=VALUE (VALUE parsed like args). Repeatable."
    )
    ap.add_argument(
        "--format", choices=["pretty", "json", "args"], default="pretty",
        help="Output format: pretty (names + matrix), json, or args (matrix only)."
    )
    ap.add_argument("--selftest", action="store_true", help="Run built-in self tests and exit.")
    ap.add_argument("--selftest-verbose", action="store_true",
                    help="Run self tests with verbose output.")
    args = ap.parse_args(argv)

    if args.selftest or args.selftest_verbose:
        rc = _run_selftests(verbose=args.selftest_verbose)
        return rc
    
    # enforce --spec only if not selftesting
    if not args.spec:
        ap.error("--spec is required unless --selftest or --selftest-verbose is used.")

    # apply constants
    try:
        for kv in args.const:
            k, v = _parse_const_kv(kv)
            set_const(k, v)
    except argparse.ArgumentTypeError as e:
        ap.error(str(e))

    try:
        names, A = parse_names_and_args(args.spec, MAXA=args.maxa)
    except SpecParseError as e:
        print(f"specparser error: {e}", file=sys.stderr)
        return 2

    if args.format == "pretty":
        print("names:", names)
        # print compact args (trim zero tail columns)
        if A.size == 0:
            print("args: []")
            return 0
        # find last nonzero column
        nz_cols = np.where(np.any(A != 0+0j, axis=0))[0]
        last = nz_cols.max() + 1 if nz_cols.size else 0
        Ashow = A[:, :last] if last > 0 else np.zeros((A.shape[0], 0), A.dtype)
        # convert to string grid
        for i, row in enumerate(Ashow):
            row_str = ", ".join(_complex_to_repr(z) for z in row) if row.size else ""
            print(f"args[{i}]: [{row_str}]")
        return 0

    if args.format == "json":
        # JSON friendly: complex as {"re":..,"im":..}
        def cjson(z: complex): return {"re": float(z.real), "im": float(z.imag)}
        obj = {
            "names": names,
            "args": [[cjson(z) for z in A[i, :]] for i in range(A.shape[0])],
        }
        print(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
        return 0

    if args.format == "args":
        # raw matrix lines
        for i in range(A.shape[0]):
            row = " ".join(_complex_to_repr(z) for z in A[i, :])
            print(row)
        return 0

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

