# Backtest Results Data Structure Plan

## Inferred Structures from R Code

### 1. Index/Dimension Files

**asset.parquet**
```
Columns: asset_string, asset
- asset_string: asset names (from colnames of pnl cube)
- asset: 1-indexed integer sequence
```

**date.parquet**
```
Columns: date_string, date
- date_string: date as string (from rownames of pnl cube)
- date: 1-indexed integer sequence
```

### 2. Cube Files Pattern

3D arrays are stored as flattened parquet with dimension files:
- `{cube_name}.parquet` - flattened values
- `{cube_name}_{dim1}.parquet` - dimension 1 labels (typically date)
- `{cube_name}_{dim2}.parquet` - dimension 2 labels (typically asset)
- `{cube_name}_{dim3}.parquet` - dimension 3 labels (slice/result/signal)

**Known cubes:**
| Cube | Dim1 | Dim2 | Dim3 |
|------|------|------|------|
| pnl_cube | date | asset | slice |
| entry_cube | date | asset | slice |
| expiry_cube | date | asset | slice |
| decay_cube | date | asset | slice |
| vega_cube | date | asset | slice |
| vol_cube | date | asset | slice |
| entry_weight_cube | date | asset | slice |
| option_pnl_cube | date | asset | slice |
| delta_pnl_cube | date | asset | slice |
| residuals | date | asset | residual |
| signals | date | asset | signal |
| signal_ranks | date | asset | signal |
| reference | date | asset | result |
| strategy | date | asset | result |

### 3. Strategy Cube Contents (strategy_result.parquet)

The `strategy` cube contains these slices:
- asset_class, sub_class
- pnl, cap, liquidity
- winsorized_pnl
- rank_rescale, rank_sum, rank_sum_median, rank_centered
- conviction, is_live
- normalized_signal, risk_weight, risk_signal
- class_weight, class_signal
- beta, norm, redistribute
- hedged, wpnl

### 4. Flat Tables

**backtest.parquet** - Main backtest data with columns:
- date, entry_date, expiry_date
- Underlying, auxUnderlying
- Entry, Expiry (codes)
- pnl, option_pnl, delta_pnl
- vega, decay, vol
- EntryWeight, strike
- straddle, fsw, pva, mv, spot, bidask, delta, RiskAdj, WeightCap

**stats.parquet** - Aggregated statistics:
- group, mean_pnl, mean_pve, mean_nve
- sd_nz_pnl, sharpe_pnl, hit_pnl, dd_pnl, etc.

**signal_hedged_stats.parquet**, **signal_norm_stats.parquet**:
- signal column + all stats from make_stats()

**recent_entries.parquet**:
- asset, hedge_asset, class, straddle
- entry_code, expiry_code
- stagger_weight, entry_date, expiry_date, strike
- strat_weight, straddle_weight, hedge, position

### 5. Config Files

**yaml_config.parquet**: `text`, `i` (line content, line number)
**json_config.parquet**: likely similar structure
**run_options.parquet**: configuration options
**asset_config.parquet**: per-asset config (cap, limit, liquidity, risk, group)

## Verification Tests

### Test 1: Verify dimension file structures
```python
# Check asset.parquet
df = pd.read_parquet('asset.parquet')
assert 'asset_string' in df.columns
assert 'asset' in df.columns
assert df['asset'].is_monotonic_increasing

# Check date.parquet
df = pd.read_parquet('date.parquet')
assert 'date_string' in df.columns
assert 'date' in df.columns
```

### Test 2: Verify cube structure
```python
# For each cube, check:
# 1. Main file exists
# 2. All 3 dimension files exist
# 3. Dimensions are consistent

def verify_cube(base_path, cube_name, dim_names):
    main = pd.read_parquet(f'{base_path}/{cube_name}.parquet')
    dims = []
    for dn in dim_names:
        dim_file = f'{base_path}/{cube_name}_{dn}.parquet'
        dims.append(pd.read_parquet(dim_file))

    # Check that product of dimensions matches main file size
    expected_size = 1
    for d in dims:
        expected_size *= len(d)

    return len(main) == expected_size
```

### Test 3: Verify cube can be reshaped
```python
def load_cube(base_path, cube_name):
    main = pd.read_parquet(f'{base_path}/{cube_name}.parquet')
    # Infer dimensions from companion files
    # Reshape to 3D numpy array
    pass
```

### Test 4: Verify strategy cube slices
```python
result = pd.read_parquet('strategy_result.parquet')
expected_slices = [
    'asset_class', 'sub_class', 'pnl', 'cap', 'liquidity',
    'winsorized_pnl', 'rank_rescale', 'rank_sum', 'rank_sum_median',
    'rank_centered', 'conviction', 'is_live', 'normalized_signal',
    'risk_weight', 'risk_signal', 'class_weight', 'class_signal',
    'beta', 'norm', 'redistribute', 'hedged', 'wpnl'
]
# Check result contains expected slices
```

### Test 5: Cross-reference dimensions
```python
# asset.parquet should match strategy_asset.parquet
# date.parquet should match strategy_date.parquet
```

## Next Steps

1. Run verification tests on actual data
2. Document any discrepancies
3. Build cube loader utility
4. Create visualization of cube contents
