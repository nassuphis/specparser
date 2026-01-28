# Plan: Backtest Script Improvements

## Overview

Three improvements requested:
1. **Delay u8m→string conversion** - Push uint8 matrix processing further downstream
2. **Skip output in benchmark mode** - Don't print straddle table when `--benchmark` flag is set
3. **Standardize benchmark output** - All three scripts show same metrics format

---

## Part 1: Delay u8m→string Conversion

### Current Bottleneck

From `backtest_strings.py` profiling:
- **88% of time** spent in `convert_to_strings` (u8m2s + tolist + strip)
- Matrix operations are 13x faster than Python string approach
- But we lose most of that gain when converting back to strings

### Downstream Parsing Analysis

After `find_straddle_yrs()`, straddle strings are parsed in these locations:

| Location | Function | What's Parsed | Frequency |
|----------|----------|---------------|-----------|
| backtest.py:167-168 | `xpry()`, `xprm()` | Expiry year/month | 1x per straddle (grouping) |
| tickers.py:536-537 | `xpry()`, `xprm()`, `ntrc()` | Expiry + entry code | 1x per straddle |
| prices.py:383 | `straddle_days()` | Entry/expiry dates | 1x per straddle |
| valuation.py:558-563 | `ntrc/ntrv/xprc/xprv()` | All fields | 1x per straddle |

### Parsing Functions (all in schedules.py)

```python
# Position-based extraction (fast)
ntry(s) → int(s[1:5])      # Entry year
ntrm(s) → int(s[6:8])      # Entry month
xpry(s) → int(s[9:13])     # Expiry year
xprm(s) → int(s[14:16])    # Expiry month

# Split-based extraction (slower - calls _parse_straddle)
ntrc(s) → _parse_straddle(s)[2]  # Entry code (N/F)
ntrv(s) → _parse_straddle(s)[3]  # Entry value
xprc(s) → _parse_straddle(s)[4]  # Expiry code
xprv(s) → _parse_straddle(s)[5]  # Expiry value
wgt(s)  → _parse_straddle(s)[6]  # Weight
```

### Proposed Solution

**Option A: Pre-parsed Integer Table (Recommended)**

Instead of returning `["asset", "straddle"]` with string straddles, return a pre-parsed table:

```python
{
    "columns": ["asset", "ntry", "ntrm", "xpry", "xprm", "ntrc", "ntrv", "xprc", "xprv", "wgt"],
    "rows": [
        ["LA Comdty", 2023, 12, 2024, 1, "N", "0", "OVERRIDE", "", "33.3"],
        ...
    ]
}
```

**Benefits:**
- No string parsing needed downstream
- All fields pre-extracted as native types
- Can still reconstruct straddle string if needed

**Changes Required:**

1. Add `expand_fast_parsed()` in `backtest_strings.py`:
   - Returns pre-parsed columns instead of straddle strings
   - Skip the straddle string assembly step entirely

2. Update `backtest.py` grouping logic (lines 163-170):
   ```python
   # Current:
   xpry = schedules.xpry(straddle)
   xprm = schedules.xprm(straddle)

   # New:
   xpry = row[xpry_idx]  # Direct column access
   xprm = row[xprm_idx]
   ```

3. Update downstream functions to accept pre-parsed data:
   - `filter_tickers()` - accept (asset, xpry, xprm, ntrc) directly
   - `straddle_days()` - accept (ntry, ntrm, xpry, xprm) directly
   - `_add_action_column()` - accept parsed values directly

**Option B: Keep Straddle Strings, Add Vectorized Parsing**

Add batch parsing functions to `strings.py`:

```python
@njit
def parse_xpry_batch(straddle_mat: np.ndarray) -> np.ndarray:
    """Extract expiry year from all rows at once."""
    # Read positions 9-13 as 4-digit integers
    return read_4digits_batch(straddle_mat, 9)
```

**Deferred to Future:**
- More invasive changes needed
- Requires modifying core schedules/tickers/valuation APIs
- Better to first validate the approach with Option A

---

## Part 2: Benchmark Mode Behavior ✅ COMPLETE

### Desired Behavior

| Mode | Backtest Table Output | Benchmark Stats |
|------|----------------------|-----------------|
| Normal (no flag) | Yes | No |
| `--benchmark` | No | Yes |

### Current Behavior (CORRECT)

| Script | `--benchmark` flag | Outputs Table? | Shows Stats? |
|--------|-------------------|----------------|--------------|
| backtest.py | `--benchmark N` | No ✓ | Yes ✓ |
| backtest_fast.py | `--benchmark` | No ✓ | Yes ✓ |
| backtest_strings.py | `--benchmark` | No ✓ | Yes ✓ |

**Flag combinations for backtest_fast.py:**
- `--benchmark` only → BENCHMARK RESULTS only
- `--timing` only → TIMING BREAKDOWN + table output
- `--benchmark --timing` → TIMING BREAKDOWN + BENCHMARK RESULTS (no table)

**Flag combinations for backtest_strings.py:**
- `--benchmark` only → BENCHMARK RESULTS only
- No flag → expanded straddles table output
- `--benchmark --breakdown` → BENCHMARK RESULTS + comparison/timing details

---

## Part 3: Standardize Benchmark Output

### Current Output Formats

**backtest.py** (lines 217-220):
```
Benchmark (3 runs, 2304 straddles, 16 workers):
  Min: 72.53s  (31.8 straddles/sec)
  Avg: 75.12s
  Max: 78.21s
```

**backtest_fast.py** (lines 221-228):
```
============================================================
BENCHMARK RESULTS
============================================================
  Straddles:     2,304
  Total time:    9.85s
  Rate:          3010.7 straddles/sec
  Output rows:   32,920
============================================================
```

**backtest_strings.py** (lines 258-272):
```
============================================================
RESULTS
============================================================
  Rows generated:    213,408

  Standard (avg):    183.98ms
  ...
```

### Proposed Unified Format

All three scripts should output this when `--benchmark` is used:

```
============================================================
BENCHMARK RESULTS
============================================================
  Straddles:       2,304
  Total time:      9.85s
  Rate:            3,010.7 straddles/sec
  Output rows:     32,920
============================================================
```

### Changes Required

**backtest.py** (lines 217-220):
```python
# Replace:
print(f"\nBenchmark ({args.benchmark} runs, {total_straddles} straddles, {num_workers} workers):", file=sys.stderr)
print(f"  Min: {min(times):.2f}s  ({total_straddles/min(times):.1f} straddles/sec)", file=sys.stderr)
print(f"  Avg: {sum(times)/len(times):.2f}s", file=sys.stderr)
print(f"  Max: {max(times):.2f}s", file=sys.stderr)

# With:
print(f"\n{'='*60}", file=sys.stderr)
print(f"BENCHMARK RESULTS", file=sys.stderr)
print(f"{'='*60}", file=sys.stderr)
print(f"  Straddles:       {total_straddles:,}", file=sys.stderr)
print(f"  Total time:      {min(times):.2f}s", file=sys.stderr)
print(f"  Rate:            {total_straddles/min(times):,.1f} straddles/sec", file=sys.stderr)
print(f"  Output rows:     {len(all_rows):,}", file=sys.stderr)
print(f"{'='*60}\n", file=sys.stderr)
```

**backtest_fast.py** (lines 221-228):
- Already close to target format
- Just align field widths for consistency

**backtest_strings.py** (lines 258-272):
- Update RESULTS section to match format
- Note: This script measures expansion, not valuation, so "Output rows" = straddles generated

---

## Implementation Order

### Phase 1: Benchmark Mode Behavior ✅ COMPLETE

**Status:** COMPLETE

**Completed:**
1. [x] `backtest_fast.py`:
   - `--benchmark` → shows ONLY benchmark stats (no table, no timing breakdown)
   - No flag → shows table output only
   - `--timing` → shows timing breakdown + table output
   - `--benchmark --timing` → shows timing breakdown + benchmark stats (no table)

2. [x] `backtest_strings.py`:
   - `--benchmark` → shows ONLY benchmark stats (no comparison details unless `--breakdown`)
   - No flag → shows expanded straddles table output
   - Added normal output mode (runs fast expansion, outputs table)

3. [x] Standardized benchmark stats format across all scripts:
   ```
   ============================================================
   BENCHMARK RESULTS
   ============================================================
     Straddles:       2,304
     Total time:      9.85s
     Rate:            3,010.7 straddles/sec
     Output rows:     32,920
   ============================================================
   ```

4. [x] Tested all scripts in both modes

**Benchmark Results (LA Comdty 2001-2024):**

| Script | What it Measures | Time | Rate |
|--------|------------------|------|------|
| backtest.py | Full valuation (multiprocessing) | ~8.6s | 11 straddles/sec |
| backtest_fast.py | Full valuation (single-threaded) | ~9.2s | 3,158 straddles/sec |
| backtest_strings.py | String expansion only | ~0.67ms | 3.4M straddles/sec |

**Note:** backtest_strings.py measures only the expansion step (2,304 straddles in → 2,304 rows out), while backtest_fast.py measures full valuation (2,304 straddles → 32,920 daily valuation rows).

### Phase 2: Pre-parsed Data ✅ COMPLETE

**Goal:** Skip straddle string assembly entirely by returning pre-parsed columns.

**Completed:**
1. [x] Added `expand_fast_parsed()` to `backtest_strings.py`:
   - Returns columns: `["asset", "ntry", "ntrm", "xpry", "xprm", "ntrc", "ntrv", "xprc", "xprv", "wgt"]`
   - Skips straddle string assembly step
   - Returns numeric types (ntry, ntrm, xpry, xprm as int)

2. [x] Added `--parsed` flag to benchmark and output modes

3. [x] Benchmark results (LA Comdty 2001-2024, 2,304 straddles):

| Approach | Time | Rate | Speedup vs Standard |
|----------|------|------|---------------------|
| Standard (Python strings) | 1.71ms | 1.3M straddles/sec | 1.0x (baseline) |
| Fast strings (u8m → strings) | 0.71ms | 3.2M straddles/sec | 2.4x |
| **Fast parsed (pre-parsed)** | **0.63ms** | **3.7M straddles/sec** | **2.7x** |

**Key findings:**
- Fast parsed is **12% faster** than fast strings (0.63ms vs 0.71ms)
- Fast strings spends 66% of time in `convert_to_strings`
- Fast parsed spends 73% of time in `build_rows` (converting numpy arrays to Python lists)
- Both approaches are bottlenecked by Python list/string object creation

**Usage:**
```bash
# Normal output (straddle strings)
uv run python scripts/backtest_strings.py '^LA Comdty' 2024 2024

# Pre-parsed output (columns)
uv run python scripts/backtest_strings.py '^LA Comdty' 2024 2024 --parsed

# Benchmark comparing all approaches
uv run python scripts/backtest_strings.py '^LA Comdty' 2001 2024 --benchmark --breakdown --parsed
```

### Phase 3: Downstream Integration (Future)

4. [ ] Update `backtest_fast.py` to use pre-parsed data
5. [ ] Update downstream functions to accept pre-parsed data:
   - `filter_tickers()` - accept (asset, xpry, xprm, ntrc) directly
   - `straddle_days()` - accept (ntry, ntrm, xpry, xprm) directly
   - `_add_action_column()` - accept parsed values directly
6. [ ] Full end-to-end benchmark

---

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/backtest.py` | Standardize benchmark output format |
| `scripts/backtest_fast.py` | Skip output in benchmark mode, standardize format |
| `scripts/backtest_strings.py` | Standardize benchmark output format |

---

## Verification

### Phase 1 Tests (all passing)

```bash
# Test benchmark mode skips table output (stdout should be empty)
uv run python scripts/backtest_fast.py '^LA Comdty' 2024 2024 --benchmark > /tmp/out.txt
wc -l /tmp/out.txt  # Should be 0 (all output to stderr) ✓

# Test normal mode outputs table (no benchmark stats)
uv run python scripts/backtest_fast.py '^LA Comdty' 2024 2024 | head -3  # Shows table ✓

# Test backtest_strings.py normal mode outputs table
uv run python scripts/backtest_strings.py '^LA Comdty' 2024 2024 | head -3  # Shows table ✓

# Compare benchmark output formats (all show same format)
uv run python scripts/backtest.py '^LA Comdty' 2024 2024 --benchmark 1 2>&1 | grep -A6 "BENCHMARK"
uv run python scripts/backtest_fast.py '^LA Comdty' 2024 2024 --benchmark 2>&1 | grep -A6 "BENCHMARK"
uv run python scripts/backtest_strings.py '^LA Comdty' 2024 2024 --benchmark 2>&1 | grep -A6 "BENCHMARK"
```

### Phase 2 Tests (all passing)

```bash
# Compare all expansion approaches
uv run python scripts/backtest_strings.py '^LA Comdty' 2001 2024 --benchmark --breakdown --parsed

# Test pre-parsed output
uv run python scripts/backtest_strings.py '^LA Comdty' 2024 2024 --parsed | head -3
# Output:
# asset	ntry	ntrm	xpry	xprm	ntrc	ntrv	xprc	xprv	wgt
# LA Comdty	2023	12	2024	1	N	0	OVERRIDE		33.3
# LA Comdty	2024	1	2024	2	N	0	OVERRIDE		33.3
```
