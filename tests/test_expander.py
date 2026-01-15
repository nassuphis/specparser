"""Tests for specparser.expander module."""

import math
import os
import tempfile
import pytest

from specparser.expander import (
    scan_segments,
    split_list_items,
    expand_list_choices,
    progressive_choices,
    expand,
    macro,
    macro_init,
    Lit,
    Dim,
    Ref,
    Init,
    DICT,
    MACROS,
)
from specparser.expander import _expand_range_body


class TestScanner:
    """Tests for scan_segments tokenizer."""

    def _scan(self, spec: str):
        """Helper to convert segments to tuples for easier comparison."""
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

    def test_literal_only(self):
        assert self._scan("abc") == [("LIT", "abc")]

    def test_list_dimension(self):
        assert self._scan("x[a,b]y") == [
            ("LIT", "x"),
            ("DIM", "LIST", "a,b"),
            ("LIT", "y"),
        ]

    def test_progressive_list(self):
        assert self._scan("x>[a,b]y") == [
            ("LIT", "x"),
            ("DIM", "PLIST", "a,b"),
            ("LIT", "y"),
        ]

    def test_function_call_in_list(self):
        assert self._scan("[${fun(1,2)}]") == [("DIM", "LIST", "${fun(1,2)}")]

    def test_regex_in_list(self):
        assert self._scan("[@{/^A/}]") == [("DIM", "LIST", "@{/^A/}")]

    def test_range_in_list(self):
        assert self._scan("[{1:3}]") == [("DIM", "LIST", "{1:3}")]

    def test_nested_brackets(self):
        assert self._scan("[a[b],c]") == [("DIM", "LIST", "a[b],c")]

    def test_double_nested_brackets(self):
        assert self._scan("[[a,b]]") == [("DIM", "LIST", "[a,b]")]

    def test_complex_spec(self):
        assert self._scan("sdsda${asd}{01:03} #{row}") == [
            ("LIT", "sdsda"),
            ("DIM", "LIST", "${asd}"),
            ("DIM", "LIST", "{01:03}"),
            ("LIT", " "),
            ("REF", "row"),
        ]


class TestSplitListItems:
    """Tests for split_list_items parser."""

    def test_simple_split(self):
        assert split_list_items("a,b,c") == ["a", "b", "c"]

    def test_whitespace_trim(self):
        assert split_list_items(" a , b , c ") == ["a", "b", "c"]

    def test_empty_string(self):
        assert split_list_items("") == []

    def test_only_commas(self):
        assert split_list_items(",,") == []

    def test_empty_items_filtered(self):
        assert split_list_items("a,,b,") == ["a", "b"]

    def test_function_call_preserved(self):
        assert split_list_items("fun(x,y),z") == ["fun(x,y)", "z"]

    def test_nested_list_preserved(self):
        assert split_list_items("${[1,2,3]},x") == ["${[1,2,3]}", "x"]

    def test_regex_preserved(self):
        assert split_list_items("@{/a,b/},x") == ["@{/a,b/}", "x"]

    def test_range_preserved(self):
        assert split_list_items("{0:1_0.25},x") == ["{0:1_0.25}", "x"]

    def test_brackets_preserved(self):
        assert split_list_items("a[b,c],d") == ["a[b,c]", "d"]


class TestExpandRangeBody:
    """Tests for _expand_range_body range expansion."""

    def test_simple_range(self):
        assert _expand_range_body("1:3") == ["1", "2", "3"]

    def test_reverse_range(self):
        assert _expand_range_body("3:1") == ["3", "2", "1"]

    def test_float_range(self):
        assert _expand_range_body("0.5:3.5") == ["0.5", "1.5", "2.5", "3.5"]

    def test_range_with_step(self):
        assert _expand_range_body("0:1_0.25") == ["0", "0.25", "0.5", "0.75", "1"]

    def test_range_with_integer_step(self):
        assert _expand_range_body("1:10_2") == ["1", "3", "5", "7", "9"]

    def test_linspace(self):
        assert _expand_range_body("0:1|5") == ["0", "0.25", "0.5", "0.75", "1"]

    def test_zero_padding(self):
        assert _expand_range_body("01:03") == ["01", "02", "03"]

    def test_negative_padding(self):
        assert _expand_range_body("-003:003") == [
            "-003", "-002", "-001", "000", "001", "002", "003"
        ]

    def test_padded_with_step(self):
        assert _expand_range_body("01:05_2") == ["01", "03", "05"]

    def test_float_step_one(self):
        # {a:b} always step=1, even for floats
        assert _expand_range_body("0.1:0.3") == ["0.1"]

    def test_float_explicit_step(self):
        assert _expand_range_body("0:1_0.5") == ["0", "0.5", "1"]

    def test_linspace_three(self):
        assert _expand_range_body("0:1|3") == ["0", "0.5", "1"]


class TestExpandListChoices:
    """Tests for expand_list_choices expansion."""

    def test_expression_expansion(self):
        assert expand_list_choices("1,${2+4},100") == ["1", "6", "100"]

    def test_list_expression(self):
        assert expand_list_choices("1,${[4,5,6]},100") == ["1", "4", "5", "6", "100"]

    def test_cartesian_expression(self):
        assert expand_list_choices("x${[1,2]}y") == ["x1y", "x2y"]

    def test_pi_constant(self):
        result = expand_list_choices("1,${pi},100")
        assert result == ["1", str(complex(math.pi)), "100"]

    def test_nested_union(self):
        assert expand_list_choices("a[b,c]d,e") == ["abd", "acd", "e"]

    def test_nested_with_range(self):
        assert expand_list_choices("x[{01:03},y]") == ["x01", "x02", "x03", "xy"]

    def test_nested_with_step_range(self):
        assert expand_list_choices("[{01:05_2},a]") == ["01", "03", "05", "a"]

    def test_inline_range(self):
        assert expand_list_choices("a{1:3}b") == ["a1b", "a2b", "a3b"]

    def test_inline_expression(self):
        assert expand_list_choices("a${[1,2]}b") == ["a1b", "a2b"]

    def test_range_and_expression_cartesian(self):
        assert expand_list_choices('x{01:03}y${["A","B"]}') == [
            "x01yA", "x02yA", "x03yA",
            "x01yB", "x02yB", "x03yB",
        ]

    def test_nested_with_range_items(self):
        assert expand_list_choices("p[{1:3},q]s") == ["p1s", "p2s", "p3s", "pqs"]

    def test_nested_with_expression_items(self):
        assert expand_list_choices("p[${[1,2]},q]s") == ["p1s", "p2s", "pqs"]


class TestProgressiveChoices:
    """Tests for progressive_choices expansion."""

    def test_simple_progressive(self):
        assert progressive_choices("a,b,c") == ["a", "a,b", "a,b,c"]

    def test_progressive_with_expression(self):
        assert progressive_choices("${[1,2]},x") == ["1", "1,2", "1,2,x"]


class TestDictionaryLookup:
    """Tests for @{...} dictionary lookup."""

    def test_regex_lookup(self):
        DICT.clear()
        DICT.update({"aa": "X", "ab": "Y", "ba": "Z"})
        assert expand_list_choices("@{a.}") == ["X", "Y"]

    def test_fullmatch_lookup(self):
        DICT.clear()
        DICT.update({"aa": "X", "ab": "Y", "ba": "Z"})
        # Exact key not matched by @{} returns literal
        assert expand_list_choices("pre@{aa}post") == ["pre@{aa}post"]

    def test_invalid_regex(self):
        DICT.clear()
        assert expand_list_choices("@{(}") == ["@{(}"]


class TestFullExpand:
    """Tests for full expand() with cartesian products and refs."""

    def test_ref_row(self):
        assert expand("[1,2] #{row}") == ["1 1", "2 2"]

    def test_cartesian_with_refs(self):
        assert expand("[a,b]{1:2}::#{d1}-#{d2}-#{row}") == [
            "a1::a-1-1",
            "a2::a-2-2",
            "b1::b-1-3",
            "b2::b-2-4",
        ]

    def test_limit_parameter(self):
        """Test that limit parameter caps output rows."""
        result = expand("[a,b,c,d,e]", limit=3)
        assert len(result) == 3
        assert result == ["a", "b", "c"]

    def test_limit_with_cartesian(self):
        """Test limit with cartesian product."""
        result = expand("[a,b][1,2,3]", limit=4)
        assert len(result) == 4
        assert result == ["a1", "a2", "a3", "b1"]

    def test_no_dimensions(self):
        """Test spec with no dimensions (literal only)."""
        assert expand("hello world") == ["hello world"]

    def test_no_dimensions_with_ref(self):
        """Test literal with ref but no dimensions."""
        assert expand("row is #{row}") == ["row is 1"]

    def test_empty_expression_literal(self):
        """Test that empty ${} is treated as literal."""
        # Empty ${} doesn't evaluate, treated as literal
        assert expand("[${}]") == ["${}"]

    def test_single_dimension(self):
        """Test single dimension expansion."""
        assert expand("[x,y,z]") == ["x", "y", "z"]

    def test_three_dimensions(self):
        """Test three-way cartesian product."""
        result = expand("[a,b][1,2][X,Y]")
        assert len(result) == 8  # 2 * 2 * 2
        assert result[0] == "a1X"
        assert result[-1] == "b2Y"

    def test_ref_ndims(self):
        """Test #{ndims} reference."""
        assert expand("[a,b][1,2] dims=#{ndims}") == [
            "a1 dims=2",
            "a2 dims=2",
            "b1 dims=2",
            "b2 dims=2",
        ]

    def test_ref_nrows(self):
        """Test #{nrows} reference."""
        result = expand("[a,b,c] total=#{nrows}")
        assert all("total=3" in r for r in result)

    def test_ref_invalid_fallback(self):
        """Test that invalid ref expression falls back to literal."""
        result = expand("[a,b] #{undefined_var}")
        assert result == ["a #{undefined_var}", "b #{undefined_var}"]


class TestInitExpressions:
    """Tests for !{...} init-time expressions."""

    def test_init_seed(self):
        """Test that !{seed(...)} sets random seed."""
        # With same seed, random functions should be reproducible
        r1 = expand("!{seed(42)}${rint(1,100)}")
        r2 = expand("!{seed(42)}${rint(1,100)}")
        assert r1 == r2

    def test_init_runs_once(self):
        """Test that init expression doesn't affect output."""
        result = expand("!{seed(123)}[a,b,c]")
        assert result == ["a", "b", "c"]


class TestMacros:
    """Tests for macro expansion."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Save and restore MACROS state."""
        snapshot = dict(MACROS)
        yield
        MACROS.clear()
        MACROS.update(snapshot)

    def test_simple_macro(self):
        """Test simple macro substitution."""
        MACROS["@TEST"] = "replaced"
        assert macro("prefix @TEST suffix") == "prefix replaced suffix"

    def test_macro_no_match(self):
        """Test that non-macro text is unchanged."""
        assert macro("no macros here") == "no macros here"

    def test_multiple_macros(self):
        """Test multiple macro substitutions."""
        MACROS["@A"] = "1"
        MACROS["@B"] = "2"
        assert macro("@A + @B") == "1 + 2"

    def test_macro_init_from_file(self):
        """Test loading macros from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# comment line\n")
            f.write("@FOO=bar\n")
            f.write("@BAZ=qux\n")
            f.write("\n")  # empty line
            path = f.name

        try:
            result = macro_init(path)
            assert result is True
            assert MACROS.get("@FOO") == "bar"
            assert MACROS.get("@BAZ") == "qux"
        finally:
            os.unlink(path)

    def test_macro_init_missing_file(self):
        """Test macro_init with missing file returns False."""
        result = macro_init("/nonexistent/path/macros.txt")
        assert result is False


class TestScannerInit:
    """Tests for Init segment scanning."""

    def _scan(self, spec: str):
        """Helper to convert segments to tuples."""
        segs = scan_segments(spec)
        out = []
        for s in segs:
            if isinstance(s, Lit):
                out.append(("LIT", s.text))
            elif isinstance(s, Dim):
                out.append(("DIM", s.dim_kind, s.raw))
            elif isinstance(s, Ref):
                out.append(("REF", s.key))
            elif isinstance(s, Init):
                out.append(("INIT", s.expr))
        return out

    def test_init_expression(self):
        """Test scanning !{...} init expression."""
        result = self._scan("!{seed(42)}[a,b]")
        assert ("INIT", "seed(42)") in result

    def test_multiple_init(self):
        """Test multiple init expressions."""
        result = self._scan("!{seed(1)}!{const('x',5)}[a]")
        assert ("INIT", "seed(1)") in result
        assert ("INIT", "const('x',5)") in result


class TestExpandRangeBodyEdgeCases:
    """Additional edge case tests for range expansion."""

    def test_single_value_range(self):
        """Test range where start equals end."""
        assert _expand_range_body("5:5") == ["5"]

    def test_large_range(self):
        """Test larger range."""
        result = _expand_range_body("1:100")
        assert len(result) == 100
        assert result[0] == "1"
        assert result[-1] == "100"

    def test_negative_range(self):
        """Test negative number range."""
        assert _expand_range_body("-5:-3") == ["-5", "-4", "-3"]

    def test_negative_to_positive(self):
        """Test range crossing zero."""
        assert _expand_range_body("-2:2") == ["-2", "-1", "0", "1", "2"]

    def test_linspace_two_points(self):
        """Test linspace with only 2 points."""
        assert _expand_range_body("0:10|2") == ["0", "10"]

    def test_linspace_one_point(self):
        """Test linspace with 1 point."""
        assert _expand_range_body("5:10|1") == ["5"]


class TestExpandListChoicesEdgeCases:
    """Additional edge case tests for list choice expansion."""

    def test_single_item(self):
        """Test single item list."""
        assert expand_list_choices("only") == ["only"]

    def test_whitespace_only_items(self):
        """Test that whitespace-only items are filtered."""
        assert expand_list_choices("a,   ,b") == ["a", "b"]

    def test_deeply_nested_brackets(self):
        """Test deeply nested bracket structures expands inner list."""
        # Inner [x,y] is expanded as a union within the outer list
        assert expand_list_choices("a[[x,y]]b") == ["axb", "ayb"]

    def test_math_expression(self):
        """Test mathematical expression evaluation."""
        result = expand_list_choices("${2**10}")
        assert result == ["1024"]

    def test_negative_numbers(self):
        """Test negative number in expression."""
        result = expand_list_choices("${-5}")
        assert result == ["-5"]
