# Plan: Remove `table_join` Alias and Migrate to `table_stack_cols`

## Problem

The `table_join` function name is misleading:
- It sounds like a SQL-style key-based join
- But it actually stacks columns side-by-side assuming row alignment
- We now have proper key-based joins (`table_left_join`, `table_inner_join`)
- Keeping `table_join` creates confusion about which function to use

## Solution

Remove the `table_join` alias entirely and update all callers to use `table_stack_cols`.

## Impact Analysis

### Files with `table_join` references:

| File | Line | Usage Type |
|------|------|------------|
| `src/specparser/amt/table.py` | 375 | Function definition |
| `src/specparser/amt/__init__.py` | 49, 132 | Export |
| `src/specparser/amt/loader.py` | 211 | Import |
| `src/specparser/amt/loader.py` | 329 | Call in `asset_group()` |
| `src/specparser/amt/loader.py` | 411 | Call in CLI `--merge` |
| `src/specparser/amt/loader.py` | 543-549 | Self-test |
| `tests/test_amt.py` | 32, 734-738 | Import and test |
| `docs/table.md` | 320-331 | Documentation |
| `notebooks/table.ipynb` | Multiple | Documentation/examples |
| `plans/loader-notebook-plan.md` | 249 | Comment reference |

## Implementation

### Phase 1: Update `loader.py` (internal caller)

```python
# Change import (line 211):
# FROM: table_join,
# TO:   table_stack_cols,

# Change asset_group() (line 329):
def asset_group(path: str | Path, live_only: bool = False, pattern: str = ".") -> dict[str, Any]:
    """live with group, subgroup, liquidity, and limit override."""
    return table_stack_cols(  # was: table_join
        asset_table(path, "group_table", default="error", live_only=live_only, pattern=pattern),
        asset_table(path, "subgroup_table", default="", live_only=live_only, pattern=pattern),
        asset_table(path, "liquidity_table", default="1", live_only=live_only, pattern=pattern),
        asset_table(path, "limit_overrides", default="", live_only=live_only, pattern=pattern),
    )

# Change CLI --merge (line 411):
print_table(table_stack_cols(*tables))  # was: table_join

# Update self-test (lines 543-549):
# Test table_stack_cols
t1 = {"columns": ["key", "a"], "rows": [["k1", 1], ["k2", 2]]}
t2 = {"columns": ["key", "b"], "rows": [["k1", 10], ["k2", 20]]}
merged = table_stack_cols(t1, t2)
assert merged["columns"] == ["key", "a", "b"], f"table_stack_cols: wrong columns"
# Note: table_stack_cols returns column-oriented, so check orientation
assert merged["orientation"] == "column", f"table_stack_cols: wrong orientation"
print("  table_stack_cols: OK")
```

### Phase 2: Update `tests/test_amt.py`

```python
# Change import (line 32):
# FROM: table_join,
# TO:   table_stack_cols,

# Rename/update test (lines 734-738):
def test_table_stack_cols_basic(self):
    """Test table_stack_cols returns column-oriented result."""
    t1 = {"orientation": "row", "columns": ["key", "a"], "rows": [["k1", 1]]}
    t2 = {"orientation": "row", "columns": ["key", "b"], "rows": [["k1", 2]]}
    result = table_stack_cols(t1, t2)
    assert result["orientation"] == "column"
    assert result["columns"] == ["key", "a", "b"]
```

### Phase 3: Remove from exports

**`src/specparser/amt/__init__.py`:**
- Remove `"table_join"` from `__all__` (line 49)
- Remove `"table_join": (".table", "table_join")` from `_LAZY_IMPORTS` (line 132)

**`src/specparser/amt/table.py`:**
- Delete the `table_join` function (lines 375-393)

### Phase 4: Update documentation

**`docs/table.md`:**
- Remove or update the `table_join` section (lines 320-331)
- Can mention it was removed in favor of `table_stack_cols`

**`notebooks/table.ipynb`:**
- Remove `table_join` from imports
- Remove `table_join` example cell
- Keep `table_stack_cols` examples

**`plans/loader-notebook-plan.md`:**
- Update comment to reference `table_stack_cols` instead

## Behavior Change Note

`table_join` returned **row-oriented** output (for backward compatibility).
`table_stack_cols` returns **column-oriented** output (for performance).

Callers that need row-oriented output should use:
```python
table_to_rows(table_stack_cols(...))
```

However, examining the actual usages:
- `asset_group()` - Used with `print_table()` which handles both orientations ✓
- CLI `--merge` - Used with `print_table()` which handles both orientations ✓
- Self-test - Just checking columns/rows, needs update for orientation

## Files to Modify

1. **src/specparser/amt/loader.py** - Update import, `asset_group()`, CLI, self-test
2. **src/specparser/amt/table.py** - Delete `table_join` function
3. **src/specparser/amt/__init__.py** - Remove exports
4. **tests/test_amt.py** - Update import and test
5. **docs/table.md** - Update documentation
6. **notebooks/table.ipynb** - Update examples
7. **plans/loader-notebook-plan.md** - Update comment (optional, low priority)

## Verification

1. Run tests: `uv run python -m pytest tests/test_amt.py -v`
2. Run loader self-test: `uv run python -m specparser.amt.loader --selftest`
3. Verify `from specparser.amt import table_join` raises `AttributeError`
4. Regenerate notebook PDF if needed
