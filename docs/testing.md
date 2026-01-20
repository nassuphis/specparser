# Testing Philosophy

This document describes the testing approach for the specparser AMT module.

## Overview

Tests are located in `tests/test_amt.py` and cover the three main submodules:
- `loader.py` - Core file loading, caching, and asset queries
- `schedules.py` - Schedule expansion and straddle string handling
- `tickers.py` - Ticker extraction and transformation

## Testing Principles

### 1. Test Organization

Tests are organized into classes by functional area:

| Class | Purpose |
|-------|---------|
| `TestLoader` | Core loader functions (load, cache, assets, tables) |
| `TestSchedules` | Schedule expansion and straddle packing |
| `TestTickers` | Ticker extraction and transformation |
| `TestTableUtilities` | Table manipulation functions (bind_rows, unique_rows) |
| `TestStraddleParsing` | Straddle string parsing functions |
| `TestDateConstraints` | Date constraint parsing (X, <, >) |
| `TestEdgeCases` | Error handling and boundary conditions |
| `TestIntegration` | End-to-end workflows |

### 2. Fixture Strategy

**Temporary Files**: Tests use `tempfile.NamedTemporaryFile` for creating test data files. This ensures:
- Tests are isolated from actual data files
- Cleanup happens automatically via pytest fixtures
- Tests can run in any environment

**Cache Clearing**: Always call `clear_cache()` at the start of tests that depend on file loading. The AMT module caches loaded files, so stale cache can cause test interference.

```python
def test_example(self, test_amt_file):
    clear_cache()  # Always clear cache first
    data = load_amt(test_amt_file)
    # ... assertions
```

**Fixture Cleanup**: Fixtures should clean up in a `finally` block or use `yield`:

```python
@pytest.fixture
def test_amt_file():
    # Setup
    with tempfile.NamedTemporaryFile(...) as f:
        f.write(test_data)
        path = f.name
    yield path
    # Cleanup
    os.unlink(path)
    clear_cache()
```

### 3. Test Coverage Goals

**Function Coverage**: Every exported function should have at least one test covering:
- Happy path (normal operation)
- Edge cases (empty inputs, missing data)
- Error conditions (invalid inputs)

**Parameter Coverage**: Functions with multiple parameters should test:
- Default parameter values
- Non-default parameter combinations
- Optional parameters (None vs provided value)

**Integration Coverage**: Key workflows should have end-to-end tests that verify multiple functions work together correctly.

### 4. Test Naming Conventions

Test names should clearly describe what is being tested:

```python
def test_<function>_<scenario>(self):
    """Brief description of what is being tested."""
```

Examples:
- `test_load_amt` - basic functionality
- `test_load_amt_caching` - specific behavior
- `test_load_amt_not_found` - error case
- `test_find_assets_live_only` - parameter variation

### 5. Assertion Best Practices

**Be Specific**: Assert exact values when possible, not just truthy/falsy:

```python
# Good
assert len(table["rows"]) == 3
assert table["columns"] == ["asset", "cls", "type"]

# Less Good
assert table["rows"]
assert "asset" in table["columns"]
```

**Test Structure**: For table results, verify both columns and row structure:

```python
table = some_function()
assert table["columns"] == ["expected", "columns"]
assert len(table["rows"]) == expected_count
# Verify specific row content
assert table["rows"][0] == [expected_values]
```

**Error Testing**: Use `pytest.raises` with `match` for error messages:

```python
with pytest.raises(ValueError, match="Column 'x' not found"):
    table_column(table, "x")
```

## Test Data Design

### Minimal Test Fixtures

Test fixtures should be minimal but complete:

```yaml
# Good: Only includes what's needed for the test
amt:
  Asset1:
    Underlying: "TEST Comdty"
    WeightCap: 0.1
    Options: "simple"
expiry_schedules:
  simple:
    - "N1_OVERRIDE15"
```

### Coverage of Variations

The main test fixture (`test_amt_file`) includes multiple asset types to cover:
- Different hedge sources (fut, nonfut, calc, cds)
- Different Vol configurations (Near/Far, NONE fields)
- Split tickers
- Multiple schedule types (monthly, quarterly)

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test class
uv run pytest tests/test_amt.py::TestLoader

# Run specific test
uv run pytest tests/test_amt.py::TestLoader::test_load_amt

# Run with coverage
uv run pytest --cov=specparser.amt
```

## Adding New Tests

When adding a new function:

1. Add basic functionality test
2. Add edge case tests (empty inputs, missing data)
3. Add error case tests if function can raise exceptions
4. Add parameter variation tests for non-trivial parameters
5. Consider if integration test is needed

When fixing a bug:

1. First write a test that reproduces the bug (should fail)
2. Fix the bug
3. Verify the test passes
4. Consider adding related edge case tests

## Test Categories

### Unit Tests

Test individual functions in isolation:

```python
def test_split_ticker_normal(self):
    result = _split_ticker("CL1 Comdty", "hedge")
    assert result == [("CL1 Comdty", "hedge")]
```

### Integration Tests

Test functions working together:

```python
def test_full_schedule_expansion_workflow(self, test_amt_file):
    clear_cache()
    found = find_assets(test_amt_file, "Comdty$", live_only=True)
    asset = found["rows"][0][0]
    expanded = get_expand(test_amt_file, asset, 2024, 2024)
    assert len(expanded["rows"]) > 0
```

### Parameterized Tests

For testing multiple similar cases:

```python
@pytest.mark.parametrize("month,expected_code", [
    (1, "F"), (2, "G"), (3, "H"), (6, "M"), (12, "Z")
])
def test_fut_spec2ticker_months(self, month, expected_code):
    spec = "fut_code:CL,fut_month_map:FGHJKMNQUVXZ,market_code:Comdty"
    result = fut_spec2ticker(spec, 2024, month)
    assert expected_code in result
```

## Common Pitfalls

1. **Forgetting cache clearing**: Tests may pass in isolation but fail when run together
2. **Path type assumptions**: Test both `str` and `Path` objects
3. **Order dependence**: Tests should not depend on execution order
4. **Incomplete cleanup**: Temporary files must be cleaned up to avoid disk clutter
5. **Over-specific assertions**: Don't assert on implementation details that might change
