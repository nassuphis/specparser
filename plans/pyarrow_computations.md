# PyArrow Compute Functions Reference

PyArrow provides 305 vectorized compute functions via `pyarrow.compute` (imported as `pc`).
These functions operate on Arrow arrays and are highly optimized for performance.

```python
import pyarrow as pa
import pyarrow.compute as pc

arr = pa.array([1, 2, 3, 4, 5])
result = pc.sum(arr)  # returns 15
```

## Function Categories

Functions are organized by:
1. **Type**: What kind of function (scalar, aggregate, vector)
2. **Data Type**: What data types they operate on
3. **Purpose**: What they do (transform, compare, aggregate, etc.)

### Function Types

| Type | Description | Output |
|------|-------------|--------|
| **ScalarFunction** | Element-wise operations | Same length as input |
| **ScalarAggregateFunction** | Reduce to single value | Scalar |
| **VectorFunction** | Structural operations | Variable length |
| **HashAggregateFunction** | Group-by aggregations | Per-group values |
| **MetaFunction** | Type-dependent dispatch | Varies |

---

## 1. Arithmetic Operations

Element-wise math operations on numeric arrays.

### Basic Arithmetic

| Function | Description | Example |
|----------|-------------|---------|
| `add(x, y)` | x + y (wraps on overflow) | `pc.add([1,2], [3,4])` → `[4,6]` |
| `add_checked(x, y)` | x + y (error on overflow) | |
| `subtract(x, y)` | x - y | `pc.subtract([5,3], [1,2])` → `[4,1]` |
| `subtract_checked(x, y)` | x - y (error on overflow) | |
| `multiply(x, y)` | x * y | `pc.multiply([2,3], [4,5])` → `[8,15]` |
| `multiply_checked(x, y)` | x * y (error on overflow) | |
| `divide(x, y)` | x / y | `pc.divide([10,20], [2,4])` → `[5,5]` |
| `divide_checked(x, y)` | x / y (error on zero) | |
| `negate(x)` | -x | `pc.negate([1,-2,3])` → `[-1,2,-3]` |
| `negate_checked(x)` | -x (error on overflow) | |
| `abs(x)` | Absolute value | `pc.abs([-1,2,-3])` → `[1,2,3]` |
| `abs_checked(x)` | Absolute value (error on overflow) | |
| `sign(x)` | Sign (-1, 0, 1) | `pc.sign([-5,0,3])` → `[-1,0,1]` |

### Power and Roots

| Function | Description | Example |
|----------|-------------|---------|
| `power(x, y)` | x^y | `pc.power([2,3], [3,2])` → `[8,9]` |
| `power_checked(x, y)` | x^y (error on overflow) | |
| `sqrt(x)` | Square root | `pc.sqrt([4,9,16])` → `[2,3,4]` |
| `sqrt_checked(x)` | Square root (error if negative) | |

### Logarithms and Exponentials

| Function | Description | Example |
|----------|-------------|---------|
| `exp(x)` | e^x | `pc.exp([0,1,2])` → `[1, 2.718, 7.389]` |
| `expm1(x)` | e^x - 1 (accurate for small x) | |
| `ln(x)` | Natural log | `pc.ln([1, 2.718, 7.389])` → `[0,1,2]` |
| `ln_checked(x)` | Natural log (error if ≤0) | |
| `log10(x)` | Base-10 log | `pc.log10([1,10,100])` → `[0,1,2]` |
| `log10_checked(x)` | Base-10 log (error if ≤0) | |
| `log2(x)` | Base-2 log | `pc.log2([1,2,4,8])` → `[0,1,2,3]` |
| `log2_checked(x)` | Base-2 log (error if ≤0) | |
| `log1p(x)` | ln(1+x) (accurate for small x) | |
| `log1p_checked(x)` | ln(1+x) (error if ≤-1) | |
| `logb(x, b)` | Log base b | `pc.logb([8,27], [2,3])` → `[3,3]` |
| `logb_checked(x, b)` | Log base b (checked) | |

### Trigonometric

| Function | Description |
|----------|-------------|
| `sin(x)` | Sine |
| `sin_checked(x)` | Sine (checked) |
| `cos(x)` | Cosine |
| `cos_checked(x)` | Cosine (checked) |
| `tan(x)` | Tangent |
| `tan_checked(x)` | Tangent (checked) |
| `asin(x)` | Arc sine |
| `asin_checked(x)` | Arc sine (checked) |
| `acos(x)` | Arc cosine |
| `acos_checked(x)` | Arc cosine (checked) |
| `atan(x)` | Arc tangent |
| `atan2(y, x)` | Arc tangent of y/x |

### Hyperbolic

| Function | Description |
|----------|-------------|
| `sinh(x)` | Hyperbolic sine |
| `cosh(x)` | Hyperbolic cosine |
| `tanh(x)` | Hyperbolic tangent |
| `asinh(x)` | Inverse hyperbolic sine |
| `acosh(x)` | Inverse hyperbolic cosine |
| `acosh_checked(x)` | Inverse hyperbolic cosine (checked) |
| `atanh(x)` | Inverse hyperbolic tangent |
| `atanh_checked(x)` | Inverse hyperbolic tangent (checked) |

### Rounding

| Function | Description | Example |
|----------|-------------|---------|
| `round(x)` | Round to nearest integer | `pc.round([1.4, 1.5, 1.6])` → `[1,2,2]` |
| `round_to_multiple(x)` | Round to multiple | |
| `ceil(x)` | Ceiling (round up) | `pc.ceil([1.1, 1.9])` → `[2,2]` |
| `floor(x)` | Floor (round down) | `pc.floor([1.1, 1.9])` → `[1,1]` |
| `trunc(x)` | Truncate (toward zero) | `pc.trunc([-1.5, 1.5])` → `[-1,1]` |

### Bitwise Operations

| Function | Description | Example |
|----------|-------------|---------|
| `bit_wise_and(x, y)` | Bitwise AND | `pc.bit_wise_and([5,6], [3,3])` → `[1,2]` |
| `bit_wise_or(x, y)` | Bitwise OR | |
| `bit_wise_xor(x, y)` | Bitwise XOR | |
| `bit_wise_not(x)` | Bitwise NOT | |
| `shift_left(x, n)` | Left shift | `pc.shift_left([1,2], [2,2])` → `[4,8]` |
| `shift_left_checked(x, n)` | Left shift (checked) | |
| `shift_right(x, n)` | Right shift | |
| `shift_right_checked(x, n)` | Right shift (checked) | |

---

## 2. Comparison Operations

Element-wise comparisons returning boolean arrays.

### Relational

| Function | Description | Example |
|----------|-------------|---------|
| `equal(x, y)` | x == y | `pc.equal([1,2,3], [1,2,4])` → `[T,T,F]` |
| `not_equal(x, y)` | x != y | |
| `less(x, y)` | x < y | `pc.less([1,2,3], [2,2,2])` → `[T,F,F]` |
| `less_equal(x, y)` | x <= y | |
| `greater(x, y)` | x > y | |
| `greater_equal(x, y)` | x >= y | |

### Null Checks

| Function | Description | Example |
|----------|-------------|---------|
| `is_null(x)` | True if null | `pc.is_null([1,None,3])` → `[F,T,F]` |
| `is_valid(x)` | True if not null | `pc.is_valid([1,None,3])` → `[T,F,T]` |
| `is_nan(x)` | True if NaN | |
| `is_inf(x)` | True if infinite | |
| `is_finite(x)` | True if finite | |
| `true_unless_null(x)` | True unless null | |

### Membership

| Function | Description | Example |
|----------|-------------|---------|
| `is_in(x, value_set)` | x in value_set | `pc.is_in([1,2,3], value_set=[2,4])` → `[F,T,F]` |
| `index_in(x, value_set)` | Index of x in value_set | |

---

## 3. Logical Operations

Boolean operations.

| Function | Description | Example |
|----------|-------------|---------|
| `and(x, y)` | Logical AND | `pc.and_([T,T,F], [T,F,F])` → `[T,F,F]` |
| `and_kleene(x, y)` | AND with null propagation | |
| `and_not(x, y)` | x AND NOT y | |
| `and_not_kleene(x, y)` | x AND NOT y (Kleene) | |
| `or(x, y)` | Logical OR | |
| `or_kleene(x, y)` | OR with null propagation | |
| `xor(x, y)` | Logical XOR | |
| `invert(x)` | Logical NOT | `pc.invert([T,F])` → `[F,T]` |

**Note:** Python `and`/`or` are keywords, so use `pc.and_()` and `pc.or_()`.

---

## 4. Aggregate Functions

Reduce arrays to scalar values.

### Basic Aggregates

| Function | Description | Example |
|----------|-------------|---------|
| `sum(x)` | Sum of values | `pc.sum([1,2,3,4])` → `10` |
| `product(x)` | Product of values | `pc.product([1,2,3,4])` → `24` |
| `mean(x)` | Arithmetic mean | `pc.mean([1,2,3,4])` → `2.5` |
| `min(x)` | Minimum value | `pc.min([3,1,4,1])` → `1` |
| `max(x)` | Maximum value | `pc.max([3,1,4,1])` → `4` |
| `min_max(x)` | Both min and max | `pc.min_max([3,1,4])` → `{min:1, max:4}` |

### Statistical Aggregates

| Function | Description |
|----------|-------------|
| `stddev(x)` | Standard deviation |
| `variance(x)` | Variance |
| `skew(x)` | Skewness |
| `kurtosis(x)` | Kurtosis |
| `quantile(x)` | Quantile (VectorFunction) |
| `approximate_median(x)` | Approximate median (fast) |
| `tdigest(x)` | T-digest for quantiles |

### Counting

| Function | Description | Example |
|----------|-------------|---------|
| `count(x)` | Count non-null values | `pc.count([1,None,3])` → `2` |
| `count_all()` | Count all (including null) | |
| `count_distinct(x)` | Count unique values | `pc.count_distinct([1,2,2,3])` → `3` |

### Position-Based

| Function | Description |
|----------|-------------|
| `first(x)` | First non-null value |
| `last(x)` | Last non-null value |
| `first_last(x)` | Both first and last |
| `index(x)` | Index of first occurrence |
| `any(x)` | Any value is true |
| `all(x)` | All values are true |

---

## 5. Cumulative Functions

Running/cumulative operations (VectorFunctions).

| Function | Description | Example |
|----------|-------------|---------|
| `cumulative_sum(x)` | Running sum | `pc.cumulative_sum([1,2,3])` → `[1,3,6]` |
| `cumulative_sum_checked(x)` | Running sum (checked) | |
| `cumulative_prod(x)` | Running product | `pc.cumulative_prod([1,2,3])` → `[1,2,6]` |
| `cumulative_prod_checked(x)` | Running product (checked) | |
| `cumulative_min(x)` | Running minimum | `pc.cumulative_min([3,1,4])` → `[3,1,1]` |
| `cumulative_max(x)` | Running maximum | `pc.cumulative_max([3,1,4])` → `[3,3,4]` |
| `cumulative_mean(x)` | Running mean | |

---

## 6. Element-Wise Selection

| Function | Description | Example |
|----------|-------------|---------|
| `if_else(cond, x, y)` | x if cond else y | `pc.if_else([T,F], [1,2], [3,4])` → `[1,4]` |
| `case_when(conds, values)` | Multi-way conditional | |
| `choose(indices, *arrays)` | Select from arrays by index | |
| `coalesce(*arrays)` | First non-null value | `pc.coalesce([None,2], [1,3])` → `[1,2]` |
| `max_element_wise(*arrays)` | Element-wise max | `pc.max_element_wise([1,5], [3,2])` → `[3,5]` |
| `min_element_wise(*arrays)` | Element-wise min | |

---

## 7. String Operations

Operations on string arrays.

### Case Transformation

| Function | Description | Example |
|----------|-------------|---------|
| `utf8_upper(s)` | UPPERCASE | `pc.utf8_upper(["hello"])` → `["HELLO"]` |
| `utf8_lower(s)` | lowercase | |
| `utf8_capitalize(s)` | Capitalize first letter | |
| `utf8_title(s)` | Title Case | |
| `utf8_swapcase(s)` | Swap case | |
| `ascii_upper(s)` | ASCII uppercase (faster) | |
| `ascii_lower(s)` | ASCII lowercase (faster) | |
| `ascii_capitalize(s)` | ASCII capitalize | |
| `ascii_title(s)` | ASCII title case | |
| `ascii_swapcase(s)` | ASCII swapcase | |

### Trimming and Padding

| Function | Description |
|----------|-------------|
| `utf8_trim(s)` | Trim characters from both ends |
| `utf8_ltrim(s)` | Trim from left |
| `utf8_rtrim(s)` | Trim from right |
| `utf8_trim_whitespace(s)` | Trim whitespace |
| `utf8_ltrim_whitespace(s)` | Trim whitespace from left |
| `utf8_rtrim_whitespace(s)` | Trim whitespace from right |
| `utf8_lpad(s)` | Left-pad to width |
| `utf8_rpad(s)` | Right-pad to width |
| `utf8_center(s)` | Center in width |
| `ascii_trim(s)` | ASCII trim |
| `ascii_ltrim(s)` | ASCII left trim |
| `ascii_rtrim(s)` | ASCII right trim |
| `ascii_trim_whitespace(s)` | ASCII trim whitespace |
| `ascii_ltrim_whitespace(s)` | ASCII left trim whitespace |
| `ascii_rtrim_whitespace(s)` | ASCII right trim whitespace |
| `ascii_lpad(s)` | ASCII left pad |
| `ascii_rpad(s)` | ASCII right pad |
| `ascii_center(s)` | ASCII center |

### Search and Match

| Function | Description | Example |
|----------|-------------|---------|
| `starts_with(s)` | Check prefix | `pc.starts_with(["hello"], pattern="he")` → `[T]` |
| `ends_with(s)` | Check suffix | |
| `match_substring(s)` | Contains substring | |
| `match_substring_regex(s)` | Regex match | |
| `match_like(s)` | SQL LIKE pattern | |
| `find_substring(s)` | Find substring index | |
| `find_substring_regex(s)` | Find regex match index | |
| `count_substring(s)` | Count substring occurrences | |
| `count_substring_regex(s)` | Count regex matches | |

### Replace

| Function | Description |
|----------|-------------|
| `replace_substring(s)` | Replace substring |
| `replace_substring_regex(s)` | Replace regex match |
| `utf8_replace_slice(s)` | Replace slice |
| `binary_replace_slice(s)` | Replace binary slice |

### Split

| Function | Description |
|----------|-------------|
| `split_pattern(s)` | Split on pattern |
| `split_pattern_regex(s)` | Split on regex |
| `utf8_split_whitespace(s)` | Split on whitespace |
| `ascii_split_whitespace(s)` | ASCII split whitespace |

### String Info

| Function | Description |
|----------|-------------|
| `utf8_length(s)` | String length (characters) |
| `binary_length(s)` | Byte length |
| `string_is_ascii(s)` | Check if ASCII-only |
| `utf8_is_alnum(s)` | All alphanumeric |
| `utf8_is_alpha(s)` | All alphabetic |
| `utf8_is_digit(s)` | All digits |
| `utf8_is_decimal(s)` | All decimal |
| `utf8_is_numeric(s)` | All numeric |
| `utf8_is_lower(s)` | All lowercase |
| `utf8_is_upper(s)` | All uppercase |
| `utf8_is_space(s)` | All whitespace |
| `utf8_is_title(s)` | Title case |
| `utf8_is_printable(s)` | All printable |
| `ascii_is_alnum(s)` | ASCII alphanumeric |
| `ascii_is_alpha(s)` | ASCII alphabetic |
| `ascii_is_decimal(s)` | ASCII decimal |
| `ascii_is_lower(s)` | ASCII lowercase |
| `ascii_is_upper(s)` | ASCII uppercase |
| `ascii_is_space(s)` | ASCII space |
| `ascii_is_title(s)` | ASCII title |
| `ascii_is_printable(s)` | ASCII printable |

### Other String

| Function | Description |
|----------|-------------|
| `utf8_reverse(s)` | Reverse string |
| `ascii_reverse(s)` | ASCII reverse |
| `utf8_slice_codeunits(s)` | Slice by codeunits |
| `utf8_normalize(s)` | Unicode normalization |
| `utf8_zero_fill(s)` | Zero-fill |
| `binary_reverse(s)` | Reverse binary |
| `binary_slice(s)` | Slice binary |
| `binary_repeat(s, n)` | Repeat binary |
| `binary_join(list, sep)` | Join with separator |
| `binary_join_element_wise(...)` | Element-wise join |
| `extract_regex(s)` | Extract regex groups |
| `extract_regex_span(s)` | Extract regex span |

---

## 8. Date/Time Operations

Operations on temporal types (date, time, timestamp, duration).

### Component Extraction

| Function | Description | Example |
|----------|-------------|---------|
| `year(t)` | Extract year | `pc.year(timestamps)` |
| `month(t)` | Extract month (1-12) | |
| `day(t)` | Extract day of month | |
| `hour(t)` | Extract hour | |
| `minute(t)` | Extract minute | |
| `second(t)` | Extract second | |
| `millisecond(t)` | Extract millisecond | |
| `microsecond(t)` | Extract microsecond | |
| `nanosecond(t)` | Extract nanosecond | |
| `subsecond(t)` | Extract subsecond fraction | |
| `day_of_week(t)` | Day of week (0=Monday) | |
| `day_of_year(t)` | Day of year (1-366) | |
| `week(t)` | Week of year | |
| `iso_week(t)` | ISO week | |
| `iso_year(t)` | ISO year | |
| `us_week(t)` | US week (Sunday start) | |
| `us_year(t)` | US year | |
| `quarter(t)` | Quarter (1-4) | |
| `iso_calendar(t)` | ISO calendar struct | |
| `year_month_day(t)` | Year, month, day struct | |

### Differences

| Function | Description |
|----------|-------------|
| `years_between(t1, t2)` | Years between timestamps |
| `quarters_between(t1, t2)` | Quarters between |
| `months_between(t1, t2)` | Months between |
| `weeks_between(t1, t2)` | Weeks between |
| `days_between(t1, t2)` | Days between |
| `hours_between(t1, t2)` | Hours between |
| `minutes_between(t1, t2)` | Minutes between |
| `seconds_between(t1, t2)` | Seconds between |
| `milliseconds_between(t1, t2)` | Milliseconds between |
| `microseconds_between(t1, t2)` | Microseconds between |
| `nanoseconds_between(t1, t2)` | Nanoseconds between |
| `day_time_interval_between(t1, t2)` | Day-time interval |
| `month_interval_between(t1, t2)` | Month interval |
| `month_day_nano_interval_between(t1, t2)` | Full interval |

### Rounding

| Function | Description |
|----------|-------------|
| `ceil_temporal(t)` | Ceiling to unit |
| `floor_temporal(t)` | Floor to unit |
| `round_temporal(t)` | Round to unit |

### Formatting and Parsing

| Function | Description | Example |
|----------|-------------|---------|
| `strftime(t)` | Format timestamp | `pc.strftime(ts, format="%Y-%m-%d")` |
| `strptime(s)` | Parse string to timestamp | |

### Timezone

| Function | Description |
|----------|-------------|
| `assume_timezone(t)` | Assign timezone |
| `local_timestamp(t)` | Convert to local time |
| `is_dst(t)` | Is daylight saving |
| `is_leap_year(t)` | Is leap year |

---

## 9. Array/Structural Operations

Operations that change array structure.

### Filtering and Selection

| Function | Description | Example |
|----------|-------------|---------|
| `filter(x, mask)` | Filter by boolean mask | `pc.filter([1,2,3], [T,F,T])` → `[1,3]` |
| `take(x, indices)` | Select by indices | `pc.take([10,20,30], [2,0])` → `[30,10]` |
| `array_take(x, indices)` | Array-specific take | |
| `array_filter(x, mask)` | Array-specific filter | |
| `drop_null(x)` | Remove nulls | `pc.drop_null([1,None,3])` → `[1,3]` |

### Sorting

| Function | Description | Example |
|----------|-------------|---------|
| `sort_indices(x)` | Indices that would sort | `pc.sort_indices([3,1,2])` → `[1,2,0]` |
| `array_sort_indices(x)` | Array sort indices | |
| `partition_nth_indices(x)` | Partial sort indices | |
| `select_k_unstable(x)` | Select top-k | |
| `rank(x)` | Rank values | |
| `rank_normal(x)` | Normal score ranking | |
| `rank_quantile(x)` | Quantile ranking | |

### Unique and Dedup

| Function | Description | Example |
|----------|-------------|---------|
| `unique(x)` | Unique values | `pc.unique([1,2,2,3])` → `[1,2,3]` |
| `value_counts(x)` | Count occurrences | |

### Fill and Replace

| Function | Description |
|----------|-------------|
| `fill_null_forward(x)` | Forward fill nulls |
| `fill_null_backward(x)` | Backward fill nulls |
| `replace_with_mask(x, mask, value)` | Replace where mask is true |

### Differences

| Function | Description | Example |
|----------|-------------|---------|
| `pairwise_diff(x)` | Differences between elements | `pc.pairwise_diff([1,3,6])` → `[null,2,3]` |
| `pairwise_diff_checked(x)` | Differences (checked) | |

### List Operations

| Function | Description |
|----------|-------------|
| `list_element(x, i)` | Get element at index |
| `list_slice(x)` | Slice lists |
| `list_flatten(x)` | Flatten nested lists |
| `list_value_length(x)` | Length of list elements |
| `list_parent_indices(x)` | Parent indices |

### Mode and Stats

| Function | Description |
|----------|-------------|
| `mode(x)` | Most frequent value(s) |
| `winsorize(x)` | Winsorize outliers |

### Indices

| Function | Description |
|----------|-------------|
| `indices_nonzero(x)` | Indices of non-zero values |
| `inverse_permutation(x)` | Inverse permutation |

---

## 10. Type Casting

| Function | Description | Example |
|----------|-------------|---------|
| `cast(x, target_type)` | Cast to type | `pc.cast([1,2,3], pa.float64())` |
| `dictionary_encode(x)` | Encode as dictionary | |
| `dictionary_decode(x)` | Decode dictionary | |

---

## 11. Hash/Group Aggregates

Used internally for group-by operations. These functions take a grouping key and values.

| Function | Description |
|----------|-------------|
| `hash_sum` | Sum per group |
| `hash_product` | Product per group |
| `hash_mean` | Mean per group |
| `hash_min` | Min per group |
| `hash_max` | Max per group |
| `hash_min_max` | Min/max per group |
| `hash_count` | Count per group |
| `hash_count_all` | Count all per group |
| `hash_count_distinct` | Distinct count per group |
| `hash_stddev` | Stddev per group |
| `hash_variance` | Variance per group |
| `hash_skew` | Skew per group |
| `hash_kurtosis` | Kurtosis per group |
| `hash_any` | Any true per group |
| `hash_all` | All true per group |
| `hash_first` | First per group |
| `hash_last` | Last per group |
| `hash_first_last` | First/last per group |
| `hash_one` | One value per group |
| `hash_list` | Collect to list per group |
| `hash_distinct` | Distinct values per group |
| `hash_approximate_median` | Approx median per group |
| `hash_tdigest` | T-digest per group |
| `hash_pivot_wider` | Pivot wider per group |

---

## 12. Struct Operations

| Function | Description |
|----------|-------------|
| `struct_field(x, name)` | Extract struct field |
| `make_struct(*arrays)` | Create struct from arrays |
| `map_lookup(x, key)` | Lookup in map type |

---

## 13. Encoding/Compression

| Function | Description |
|----------|-------------|
| `run_end_encode(x)` | Run-length encode |
| `run_end_decode(x)` | Run-length decode |

---

## 14. Random

| Function | Description |
|----------|-------------|
| `random()` | Generate random array |

---

## 15. Pivot

| Function | Description |
|----------|-------------|
| `pivot_wider(x, keys)` | Pivot from long to wide format |

---

## Summary by Operation Type

### Scalar Functions (Element-wise)
Transform each element independently. Output has same length as input.
- Arithmetic: `add`, `subtract`, `multiply`, `divide`, `power`, `sqrt`, etc.
- Comparison: `equal`, `less`, `greater`, etc.
- Logical: `and`, `or`, `xor`, `invert`
- String: `utf8_upper`, `utf8_lower`, `starts_with`, etc.
- Temporal: `year`, `month`, `day`, `strftime`, etc.
- Conditional: `if_else`, `coalesce`, `case_when`

### Aggregate Functions (Reduction)
Reduce array to single scalar value.
- Numeric: `sum`, `mean`, `min`, `max`, `stddev`, `variance`
- Counting: `count`, `count_distinct`, `count_all`
- Boolean: `any`, `all`
- Position: `first`, `last`, `index`

### Vector Functions (Structural)
Change array structure. Output may have different length.
- Filtering: `filter`, `drop_null`, `take`
- Sorting: `sort_indices`, `rank`
- Unique: `unique`, `value_counts`
- Cumulative: `cumulative_sum`, `cumulative_max`, etc.
- Fill: `fill_null_forward`, `fill_null_backward`
- Lists: `list_flatten`, `list_element`

### Hash Aggregate Functions (Group-by)
Per-group aggregations for group-by operations.
- All standard aggregates with `hash_` prefix

---

## Usage Notes

1. **Null handling**: Most functions skip nulls by default. Use `skip_nulls=False` to propagate.

2. **Checked vs unchecked**: `_checked` variants raise errors on overflow/invalid input.

3. **ASCII vs UTF-8**: ASCII functions are faster but only work on ASCII strings.

4. **Memory**: Functions return new arrays; use `memory_pool` parameter for custom allocation.

5. **Scalars**: Most functions accept both arrays and scalars for broadcasting.

```python
# Scalar broadcasting
pc.add([1, 2, 3], 10)  # → [11, 12, 13]
pc.multiply([1, 2, 3], 2)  # → [2, 4, 6]
```
