# Testing Lessons Learned

This document captures lessons learned during test development for the AMT module.

## Test Count Confusion

### What Happened
During the test expansion session, there was confusion about test counts:
- Original `test_amt.py`: **64 tests**
- After additions: **120 tests** (121 test functions, 1 skipped due to collection)
- Total project tests went from ~216 to **273**

**No tests were removed.** The confusion arose from comparing:
1. Tests in a single file vs. all project tests
2. Collected tests vs. test function count
3. Different pytest runs at different stages

### Lesson
Always be explicit about scope when discussing test counts:
- "64 tests in test_amt.py" vs "216 tests total"
- Use `pytest --collect-only` for accurate counts

---

## Bug Found During Testing

### Empty Field Parsing Bug in Straddle Strings

**Problem:** The `_parse_straddle()` function used `s.strip("|").split("|")` to parse straddle strings. This broke when any field was empty (consecutive pipes `||`):

```python
# Straddle with empty xprv field (common in real data)
s = "|2023-12|2024-01|N|0|OVERRIDE||33.3|"

# Using strip("|"):
s.strip("|").split("|")  # Returns 6 parts - the empty xprv is lost!

# Using s[1:-1]:
s[1:-1].split("|")  # Returns 7 parts: ['2023-12', '2024-01', 'N', '0', 'OVERRIDE', '', '33.3']
# Empty xprv preserved correctly!
```

**Fix:** Changed from `strip("|")` to `s[1:-1]` to preserve empty internal fields.

**Lesson:** `strip()` removes **all** matching characters from both ends, not just one. For delimited formats with potentially empty fields, use slice notation `[1:-1]` instead.

### Validating Bug Impact with Real Data

After fixing the bug, we verified which fields are actually empty in production:

```bash
# Count straddles for January 2025
uv run python -m specparser.amt.tickers data/amt.yml --expand-ym '.' True 2025 1 | wc -l
# 742 (741 straddles + header)
```

**Real data analysis (741 straddles for 2025-01):**

| Field | Empty Count | Example Pattern | Source |
|-------|-------------|-----------------|--------|
| `xprv` (expiry value) | **88** | `OVERRIDE||33.3` | Schedules using `OVERRIDE` without day number |
| `wgt` (weight) | **0** | N/A | All real schedules have weights |

The bug would have affected **88 straddles** (12% of monthly output) by corrupting straddles with empty `xprv`. The empty `wgt` case only existed in test fixtures.

**Lesson:** After finding a bug through testing, validate its impact against real data. The test fixture had empty `wgt`, but real data has empty `xprv` - both cases need the same fix, but understanding the actual data helps prioritize and verify.

---

## Test Fixture Issues

### Handler Dispatch Mismatch

**Problem:** Test fixture had `Source: "BBG"` for a hedge, but the code's `_HEDGE_HANDLERS` dispatch table didn't include "BBG" as a key:

```python
_HEDGE_HANDLERS = {
    "nonfut": _hedge_nonfut,  # BBG-style ticker
    "cds": _hedge_cds,
    "fut": _hedge_fut,
    "calc": _hedge_calc,
    # "BBG" is NOT here!
}
```

Assets with `Source: "BBG"` fell through to `_hedge_default()`, which creates a spec string instead of using the ticker directly.

**Fix:** Changed test fixture to use `Source: "nonfut"` which correctly handles BBG-style tickers.

**Lesson:** Test fixtures must match the code's dispatch/handler expectations. When testing handler-based code, verify which handler keys exist.

---

## Test Assumptions vs. Reality

### Placeholder Straddles for Missing Schedules

**Assumption:** Assets without an `Options` field would cause `asset_straddle_tickers()` to raise `ValueError`.

**Reality:** The code returns a "placeholder" straddle with empty values:
```python
# Asset without Options field still returns:
{'columns': [...], 'rows': [['NO_SCHED Test', '|2024-06|2024-06||||||']]}
```

The `_schedule_to_rows()` function creates a placeholder row when schedule is empty/None.

**Fix:** Changed test to verify placeholder behavior instead of expecting an exception.

**Lesson:** Read the actual code path before writing tests. The `get_expand_ym()` → `get_schedule()` → `_schedule_to_rows()` chain handles missing schedules gracefully.

---

## Test Organization Patterns

### What Worked Well

1. **Separate test classes by domain:**
   - `TestLoader` - Core loading functions
   - `TestSchedules` - Schedule expansion
   - `TestTickers` - Ticker extraction
   - `TestStraddleParsing` - Straddle string parsing
   - `TestDateConstraints` - Date constraint operators (X, <, >)
   - `TestTickerExpansion` - Higher-level ticker functions
   - `TestTableUtilities` - Table manipulation (bind_rows, unique_rows)
   - `TestEdgeCases` - Error handling
   - `TestIntegration` - End-to-end workflows

2. **Shared fixtures with cleanup:**
   ```python
   @pytest.fixture
   def test_amt_file():
       # Create temp file
       yield path
       # Cleanup
       os.unlink(path)
       clear_cache()
   ```

3. **Inline temp files for edge cases:**
   ```python
   def test_specific_edge_case(self):
       with tempfile.NamedTemporaryFile(...) as f:
           f.write(specific_data)
           # ... test ...
   ```

### What Could Be Improved

1. **Parameterized tests** for similar cases (e.g., all 12 month codes)
2. **Property-based testing** for straddle parsing (any valid straddle should round-trip)
3. **Snapshot testing** for complex table outputs

---

## Testing Internal vs Output Transformations

### Column Removal from Output

**Scenario:** Removed `cls` and `type` columns from `asset_straddle_tickers()` output.

**What changed:**
- Old output columns: `['asset', 'cls', 'type', 'param', 'source', 'ticker', 'field', 'straddle']`
- New output columns: `['asset', 'param', 'source', 'ticker', 'field', 'straddle']`

**Tests that needed updating:**
1. `test_asset_straddle_tickers_basic` - Checks output column structure
2. `test_asset_straddle_tickers_filters_near` - Used hardcoded index for `type` column
3. `test_asset_straddle_tickers_filters_far` - Used hardcoded index for `type` column

**Tests that did NOT need updating:**
- `TestStraddleTickerFiltering` unit tests - These test `_filter_straddle_tickers()` which still operates on the original 7-column input format internally

**Lesson:** When a function transforms data internally but produces different output:
1. **Unit tests for internal logic** can remain unchanged (they test the internal API)
2. **Integration tests for output format** need updating (they test the external API)

The `_filter_straddle_tickers()` function receives rows with all 7 columns and filters them. The column removal happens *after* filtering in `asset_straddle_tickers()`. This separation meant:
- 8 unit tests for filtering rules: **unchanged**
- 2 integration tests checking final output: **updated**

### Index-Based vs Column-Name-Based Testing

**Problem:** Tests used hardcoded indices like `r[2]` to access the `type` column:

```python
# Before: brittle index-based access
types = [r[2] for r in table["rows"]]
vol_rows = [r for r in table["rows"] if r[2] == "Vol"]
```

**After column removal:** Index 2 now refers to `source`, not `type`.

**Better approach:** For tests checking specific values, use the column name to look up the index, or use semantic checks:

```python
# After: check by param value instead of type
vol_rows = [r for r in table["rows"] if r[1] == "vol"]  # param is at index 1
```

**Lesson:** When testing table output:
- For **structure tests**: Verify exact column list (`assert table["columns"] == [...]`)
- For **value tests**: Either look up column indices dynamically or use semantic identifiers that survive schema changes

---

## Key Takeaways

1. **Test the actual behavior, not assumed behavior** - Read the code path first
2. **Be careful with string methods** - `strip()` vs slice notation have different semantics
3. **Fixture data must match handler expectations** - Verify dispatch keys exist
4. **Clear caches between tests** - Module-level caches cause test interference
5. **Explicit test counts** - Always specify file/project scope when discussing counts
6. **Tests found a real bug** - The empty field parsing bug would have affected 12% of production straddles
7. **Separate internal and output tests** - Internal transformation tests and output format tests have different update requirements
8. **Avoid brittle index-based access** - Column indices change when schema changes; use semantic checks where possible
