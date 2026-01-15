#!/usr/bin/env python
# chain.py — minimal CLI-safe parser for key:value chains

"""
Parse pipeline specifications of the form "op:arg1:arg2,op2:arg1,...".

Operations are kept as plain strings; scalar arguments are evaluated
into complex numbers using a restricted expression language.
"""

from __future__ import annotations

import re
import math
import cmath
import json
import argparse
import sys

import numpy as np
from simpleeval import SimpleEval

from . import chain_state as state

__all__ = [
    "SpecParseError",
    "set_const",
    "get_const",
    "extract_used_names",
    "split_chain",
    "concat_chain",
    "parse_chain",
    "parse_names_and_args",
    "parse_args_only",
    "get_required_arg",
]

# Re-export for backwards compatibility
NAMES = state.NAMES
FUNCS = state.FUNCS
ALLOWED_OPS = state.ALLOWED_OPS
set_const = state.set_const
get_const = state.get_const


# ============================================================
# Errors
# ============================================================

class SpecParseError(ValueError):
    pass


# ============================================================
# Scalar parsing
# ============================================================

def simple_eval_number(expr: str) -> complex:
    """Evaluate an expression string to a complex number."""
    se = SimpleEval(names=state.NAMES, functions=state.FUNCS, operators=state.ALLOWED_OPS)
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
    """
    Parse a scalar token into a complex number.

    Supports:
    - SimpleEval expressions with NAMES and FUNCS
    - Fractional exponents like 1e2.1 (interpreted as 10^2.1)
    """
    t = tok.strip().lower()
    if not t:
        return 0.0 + 0.0j
    try:
        v = simple_eval_number(t)
        return complex(v)
    except Exception:
        pass

    # Fallback: fractional exponent notation
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


# ============================================================
# Chain parsing
# ============================================================

SKIP_PREFIXES = {"!"}


def _is_skipped_name(raw: str) -> bool:
    """Check if name should be skipped (starts with !)."""
    s = raw.lstrip()
    return bool(s) and s[0] in SKIP_PREFIXES


def _strip_nonfunctional_prefix(raw: str) -> str:
    """Remove leading underscore (variant marker)."""
    s = raw.lstrip()
    return s[1:] if s.startswith("_") else raw


def split_top_level(s: str, sep: str) -> list[str]:
    """
    Split string on separator, but only at top level (outside brackets).

    Handles (), [], {} nesting.
    """
    out, buf, depth = [], [], 0
    opens = "([{"
    closes = ")]}"
    for ch in s:
        if ch in opens:
            depth += 1
        elif ch in closes:
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
    """Extract the set of operation names from a chain (without parsing args)."""
    if not chain.strip():
        return set()
    items = split_top_level(chain, ",")
    return {item.split(":", 1)[0].lower() for item in items if item}


def split_chain(chain: str) -> dict[str, list[str]]:
    """
    Split a chain into a dict mapping operation names to raw argument lists.

    Args:
        chain: Chain string like "op1:arg1:arg2,op2:arg1"

    Returns:
        Dict like {"op1": ["arg1", "arg2"], "op2": ["arg1"]}
    """
    out = {}
    if not chain.strip():
        return out
    items = split_top_level(chain, ",")
    for item in items:
        parts = split_top_level(item, ":")
        if not parts:
            continue
        name_raw = parts[0].strip()
        if _is_skipped_name(name_raw):
            continue
        name = _strip_nonfunctional_prefix(name_raw)
        name = name.lower()
        out[name] = parts[1:]
    return out


def concat_chain(d: dict[str, list[str]]) -> str:
    """Rebuild a chain string from a dict."""
    return ",".join(f"{k}:" + ":".join(v) for k, v in d.items())


def parse_chain(chain: str, MAXA: int = 12) -> list[tuple[str, tuple[complex, ...]]]:
    """
    Parse a chain into a list of (name, args) tuples.

    Args:
        chain: Chain string
        MAXA: Maximum number of arguments to parse per operation

    Returns:
        List of (name, args) where args is a tuple of complex numbers
    """
    out = []
    if not chain.strip():
        return out
    items = split_top_level(chain, ",")
    for item in items:
        parts = split_top_level(item, ":")
        if not parts:
            continue
        name_raw = parts[0].strip()
        if _is_skipped_name(name_raw):
            continue
        name = _strip_nonfunctional_prefix(name_raw)
        name = name.lower()
        arg_tokens = parts[1 : MAXA + 1]
        args = tuple(_parse_scalar(tok) for tok in arg_tokens)
        out.append((name, args))
    return out


def parse_names_and_args(
    chain: str, MAXA: int = 12
) -> tuple[list[str], np.ndarray]:
    """
    Parse a chain into names list and numpy argument matrix.

    Args:
        chain: Chain string
        MAXA: Maximum arguments per operation (matrix width)

    Returns:
        Tuple of (names, args) where:
        - names: list of operation names
        - args: numpy array of shape (K, MAXA) with dtype complex128
    """
    specs = parse_chain(chain, MAXA=MAXA)
    K = len(specs)
    names = [None] * K
    args = np.zeros((K, MAXA), np.complex128)
    for i, (name, argv) in enumerate(specs):
        names[i] = name
        if argv:
            args[i, : len(argv)] = np.asarray(argv, dtype=np.complex128)
    return names, args


def parse_args_only(chain: str, MAXA: int = 12) -> np.ndarray:
    """Parse a chain and return only the argument matrix."""
    _, A = parse_names_and_args(chain, MAXA=MAXA)
    return A


# ============================================================
# Utilities
# ============================================================

def get_required_arg(spec: str, key: str) -> str:
    """Get a required argument from a spec, raising if missing."""
    d = split_chain(spec)
    vals = d.get(key)
    if not vals or vals[0] is None or str(vals[0]).strip() == "":
        raise ValueError(f"spec missing required '{key}': {spec}")
    return str(vals[0]).strip()


def add_slot(spec: str) -> str:
    """Add a slot number to a spec if not present."""
    from . import slots as slotfuns

    d = split_chain(spec)
    if "slot" in d:
        return spec
    slot = str(slotfuns.first_free_slot(state.NAMES["outschema"]))
    d["slot"] = [slot]
    return concat_chain(d)


def slot_suffix(spec: str, width: int = 5) -> str:
    """Extract or generate a zero-padded slot suffix from a spec."""
    from . import slots as slotfuns

    d = split_chain(spec)
    if "slot" in d:
        vals = d["slot"]
        slot = str(vals[0].strip())
        if slot.isdigit():
            return slot.zfill(width)
        return slot
    slot = str(slotfuns.first_free_slot(state.NAMES["outschema"])).zfill(width)
    return slot


# ============================================================
# CLI helpers
# ============================================================

def _parse_const_kv(text: str) -> tuple[str, complex]:
    """Parse a NAME=VALUE constant definition."""
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
    """Format complex number for CLI output."""
    return f"{z.real:+.12g}{z.imag:+.12g}j"


def _approx_eq(z: complex, w: complex, tol: float = 1e-12) -> bool:
    """Check if two complex numbers are approximately equal."""
    return abs(z.real - w.real) <= tol and abs(z.imag - w.imag) <= tol


def _check(
    label: str, got: complex, expected: complex, verbose: bool
) -> tuple[bool, str]:
    """Check a test result and format output."""
    ok = _approx_eq(got, expected)
    if verbose:
        return ok, f"{'✅' if ok else '❌'} {label}\n   got: {got!r}\n   exp: {expected!r}"
    else:
        return ok, f"{'✅' if ok else '❌'} {label}"


def _run_selftests(verbose: bool = False) -> int:
    """
    Run self-tests for scalar parsing, function calls, constants,
    fractional exponents, and top-level splitting in chains.
    """
    passed = 0
    failed = 0
    logs: list[str] = []

    # snapshot global names (so tests don't leak)
    names_snapshot = dict(state.NAMES)

    try:
        # ---- scalar + constants ----
        ok, msg = _check("pi constant", _parse_scalar("pi"), complex(math.pi), verbose)
        logs.append(msg)
        passed += ok
        failed += not ok

        ok, msg = _check("2**8", _parse_scalar("2**8"), complex(256.0), verbose)
        logs.append(msg)
        passed += ok
        failed += not ok

        # Python doesn't accept 1e2.1; fallback treats it as 10**2.1
        val_1e = 10.0**2.1
        ok, msg = _check("1e2.1 fallback", _parse_scalar("1e2.1"), complex(val_1e), verbose)
        logs.append(msg)
        passed += ok
        failed += not ok

        # complex literal and arithmetic
        ok, msg = _check("3+4j", _parse_scalar("3+4j"), complex(3, 4), verbose)
        logs.append(msg)
        passed += ok
        failed += not ok

        # ---- functions (cmath) ----
        ok, msg = _check(
            "sin(pi)", _parse_scalar("sin(pi)"), complex(cmath.sin(math.pi)), verbose
        )
        logs.append(msg)
        passed += ok
        failed += not ok

        # set const a=3 and test cos(a)
        state.set_const("a", _parse_scalar("3"))
        ok, msg = _check(
            "cos(a) with a=3", _parse_scalar("cos(a)"), complex(cmath.cos(3)), verbose
        )
        logs.append(msg)
        passed += ok
        failed += not ok

        # real_max/real_min via max/min
        ok, msg = _check(
            "max(a, 1) real part",
            _parse_scalar("max(a,1)"),
            complex(max(3.0, 1.0), 0),
            verbose,
        )
        logs.append(msg)
        passed += ok
        failed += not ok

        # nested functions with internal commas
        ok, msg = _check(
            "nested max(sin(pi),cos(0))",
            _parse_scalar("max(sin(pi), cos(0))"),
            complex(1.0, 0.0),
            verbose,
        )
        logs.append(msg)
        passed += ok
        failed += not ok

        # ---- top-level splitting in chains ----
        chain = "op:1:max(2,3):min(4,5),foo:sin(pi):cos(a)"
        names, A = parse_names_and_args(chain, MAXA=12)
        ok = names == ["op", "foo"]
        msg = f"{'✅' if ok else '❌'} split chain names -> {names}"
        logs.append(msg)
        passed += ok
        failed += not ok

        # args shape & values
        exp0 = [complex(1, 0), complex(3, 0), complex(4, 0)]
        got0 = [A[0, 0], A[0, 1], A[0, 2]]
        ok = all(_approx_eq(g, e) for g, e in zip(got0, exp0))
        msg = f"{'✅' if ok else '❌'} chain args[0] head -> {got0}"
        logs.append(msg)
        passed += ok
        failed += not ok

        exp1 = [complex(cmath.sin(math.pi)), complex(cmath.cos(3))]
        got1 = [A[1, 0], A[1, 1]]
        ok = all(_approx_eq(g, e) for g, e in zip(got1, exp1))
        msg = f"{'✅' if ok else '❌'} chain args[1] head -> {got1}"
        logs.append(msg)
        passed += ok
        failed += not ok

        # extract_used_names
        used = extract_used_names(chain)
        ok = used == {"op", "foo"}
        msg = f"{'✅' if ok else '❌'} extract_used_names -> {used}"
        logs.append(msg)
        passed += ok
        failed += not ok

    finally:
        # restore original names
        state.NAMES.clear()
        state.NAMES.update(names_snapshot)

    for line in logs:
        print(line)

    total = passed + failed
    print(f"\nPassed: {passed}, Failed: {failed}, Total: {total}")
    return 0 if failed == 0 else 1


# ============================================================
# CLI
# ============================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse pipeline spec 'op:arg1:arg2,op2:arg1,...' into names and complex args."
    )
    ap.add_argument("--spec", required=False, help="Spec string (no quoting/escaping).")
    ap.add_argument("--split", required=False, help="Split only (no scalar parsing)")
    ap.add_argument("--maxa", type=int, default=12, help="Max args per op (default: 12).")
    ap.add_argument(
        "--const",
        action="append",
        default=[],
        help="Add/override constant as NAME=VALUE. Repeatable.",
    )
    ap.add_argument(
        "--format",
        choices=["pretty", "json", "args"],
        default="pretty",
        help="Output format: pretty (names + matrix), json, or args (matrix only).",
    )
    ap.add_argument("--selftest", action="store_true", help="Run built-in self tests.")
    ap.add_argument(
        "--selftest-verbose", action="store_true", help="Run self tests with verbose output."
    )
    args = ap.parse_args(argv)

    if args.selftest or args.selftest_verbose:
        return _run_selftests(verbose=args.selftest_verbose)

    if args.split:
        ss = split_chain(args.split)
        for i, (n, v) in enumerate(ss.items()):
            print(f"{i} {n}:{v}")
        return 0

    if not args.spec:
        ap.error("--spec is required unless --selftest or --selftest-verbose is used.")

    # apply constants
    try:
        for kv in args.const:
            k, v = _parse_const_kv(kv)
            state.set_const(k, v)
    except argparse.ArgumentTypeError as e:
        ap.error(str(e))

    try:
        names, A = parse_names_and_args(args.spec, MAXA=args.maxa)
    except SpecParseError as e:
        print(f"specparser error: {e}", file=sys.stderr)
        return 2

    if args.format == "pretty":
        print("names:", names)
        if A.size == 0:
            print("args: []")
            return 0
        # find last nonzero column
        nz_cols = np.where(np.any(A != 0 + 0j, axis=0))[0]
        last = nz_cols.max() + 1 if nz_cols.size else 0
        Ashow = A[:, :last] if last > 0 else np.zeros((A.shape[0], 0), A.dtype)
        for i, row in enumerate(Ashow):
            row_str = ", ".join(_complex_to_repr(z) for z in row) if row.size else ""
            print(f"args[{i}]: [{row_str}]")
        return 0

    if args.format == "json":

        def cjson(z: complex):
            return {"re": float(z.real), "im": float(z.imag)}

        obj = {
            "names": names,
            "args": [[cjson(z) for z in A[i, :]] for i in range(A.shape[0])],
        }
        print(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
        return 0

    if args.format == "args":
        for i in range(A.shape[0]):
            row = " ".join(_complex_to_repr(z) for z in A[i, :])
            print(row)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
