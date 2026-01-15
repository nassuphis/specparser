"""Tests for specparser.chain module."""

import cmath
import math
import pytest

from specparser.chain import (
    split_chain,
    concat_chain,
    parse_chain,
    parse_names_and_args,
    extract_used_names,
)
from specparser import chain_state as state


def _approx_eq(a: complex, b: complex, tol: float = 1e-9) -> bool:
    """Check if two complex numbers are approximately equal."""
    return abs(a - b) < tol


class TestSplitChain:
    """Tests for split_chain parser."""

    def test_simple_chain(self):
        result = split_chain("key1:value1,key2:value2")
        assert result == {"key1": ["value1"], "key2": ["value2"]}

    def test_multiple_args(self):
        result = split_chain("op:1:2:3")
        assert result == {"op": ["1", "2", "3"]}

    def test_empty_string(self):
        result = split_chain("")
        assert result == {}

    def test_whitespace_handling(self):
        result = split_chain(" key : value ")
        assert result == {"key": ["value"]}


class TestConcatChain:
    """Tests for concat_chain rebuilder."""

    def test_simple_concat(self):
        d = {"key1": ["value1"], "key2": ["value2"]}
        result = concat_chain(d)
        # Order may vary, check both parts present
        assert "key1:value1" in result
        assert "key2:value2" in result

    def test_multiple_args_concat(self):
        d = {"op": ["1", "2", "3"]}
        result = concat_chain(d)
        assert result == "op:1:2:3"


class TestParseScalar:
    """Tests for scalar expression parsing."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Save and restore NAMES state around each test."""
        snapshot = dict(state.NAMES)
        yield
        state.NAMES.clear()
        state.NAMES.update(snapshot)

    def test_pi_constant(self):
        names, A = parse_names_and_args("x:pi", MAXA=4)
        assert names == ["x"]
        assert _approx_eq(A[0, 0], complex(math.pi))

    def test_power_expression(self):
        names, A = parse_names_and_args("x:2**8", MAXA=4)
        assert _approx_eq(A[0, 0], complex(256.0))

    def test_fractional_exponent(self):
        # 1e2.1 is not valid Python, but chain parser handles as 10**2.1
        val_expected = 10.0 ** 2.1
        names, A = parse_names_and_args("x:1e2.1", MAXA=4)
        assert _approx_eq(A[0, 0], complex(val_expected))

    def test_complex_literal(self):
        names, A = parse_names_and_args("x:3+4j", MAXA=4)
        assert _approx_eq(A[0, 0], complex(3, 4))

    def test_sin_function(self):
        names, A = parse_names_and_args("x:sin(pi)", MAXA=4)
        assert _approx_eq(A[0, 0], complex(cmath.sin(math.pi)))

    def test_cos_with_constant(self):
        state.set_const("a", complex(3))
        names, A = parse_names_and_args("x:cos(a)", MAXA=4)
        assert _approx_eq(A[0, 0], complex(cmath.cos(3)))

    def test_max_function(self):
        state.set_const("a", complex(3))
        names, A = parse_names_and_args("x:max(a,1)", MAXA=4)
        assert _approx_eq(A[0, 0], complex(max(3.0, 1.0), 0))

    def test_nested_functions(self):
        names, A = parse_names_and_args("x:max(sin(pi),cos(0))", MAXA=4)
        assert _approx_eq(A[0, 0], complex(1.0, 0.0))


class TestParseChain:
    """Tests for full chain parsing."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Save and restore NAMES state around each test."""
        snapshot = dict(state.NAMES)
        yield
        state.NAMES.clear()
        state.NAMES.update(snapshot)

    def test_multi_op_chain(self):
        chain = "op:1:max(2,3):min(4,5),foo:sin(pi):cos(0)"
        names, A = parse_names_and_args(chain, MAXA=12)
        assert names == ["op", "foo"]

    def test_chain_args_first_op(self):
        state.set_const("a", complex(3))
        chain = "op:1:max(2,3):min(4,5),foo:sin(pi):cos(a)"
        names, A = parse_names_and_args(chain, MAXA=12)

        # First op: 1, max(2,3)=3, min(4,5)=4
        assert _approx_eq(A[0, 0], complex(1, 0))
        assert _approx_eq(A[0, 1], complex(3, 0))
        assert _approx_eq(A[0, 2], complex(4, 0))

    def test_chain_args_second_op(self):
        state.set_const("a", complex(3))
        chain = "op:1:max(2,3):min(4,5),foo:sin(pi):cos(a)"
        names, A = parse_names_and_args(chain, MAXA=12)

        # Second op: sin(pi)≈0, cos(3)
        assert _approx_eq(A[1, 0], complex(cmath.sin(math.pi)))
        assert _approx_eq(A[1, 1], complex(cmath.cos(3)))


class TestExtractUsedNames:
    """Tests for extract_used_names utility."""

    def test_simple_extraction(self):
        used = extract_used_names("op:1:2,foo:3")
        assert used == {"op", "foo"}

    def test_complex_chain(self):
        used = extract_used_names("op:1:max(2,3):min(4,5),foo:sin(pi):cos(0)")
        assert used == {"op", "foo"}
