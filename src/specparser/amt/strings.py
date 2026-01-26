# -------------------------------------
# Fast String/Matrix Operations
# -------------------------------------
"""
High-performance string generation using uint8 byte matrices.

Key insight: work directly with raw bytes (uint8 arrays) instead of Python strings.
The output is a matrix of ASCII characters that can be zero-copy converted to
NumPy/Arrow string arrays using view().

See docs/strings_comments.md for detailed documentation.
See notebooks/strings.ipynb for examples and benchmarks.
"""

import numpy as np
from numpy.dtypes import StringDType
from numba import njit, prange, types
from numba.typed import Dict

ASCII_MAP = Dict.empty(key_type=types.unicode_type, value_type=types.uint8)
for i in range(128): ASCII_MAP[chr(i)] = np.uint8(i)

ASCII_0 = np.uint8(ord("0"))
ASCII_1 = np.uint8(ord("1"))
ASCII_DASH = np.uint8(ord("-"))
ASCII_SPACE = np.uint8(ord(" "))
ASCII_N = np.uint8(ord("M"))
ASCII_F = np.uint8(ord("F"))
ASCII_A = np.uint8(ord("A"))
ASCII_PIPE = np.uint8(ord("|"))

@njit
def make_u8mat(length,width,fill=ASCII_SPACE):
    """new uint8mat filled whith char"""
    return np.full((length, width), fill , dtype=np.uint8)

@njit
def sep(chars: bytes, n: int = 1) -> np.ndarray:
    """fill matrix uint8mat with byte-string"""
    res = make_u8mat(n,len(chars))
    for i in range(n):
        for j in range(len(chars)):
            res[i,j]=chars[j]
    return res

def strs2u8mat(strings: list[str], width: int = None) -> np.ndarray:
    """Convert list of strings to fixed-width uint8 matrix."""
    if width is None:
        width = max(len(s) for s in strings) if strings else 0

    out = np.full((len(strings), width), ASCII_SPACE, dtype=np.uint8)

    for i, s in enumerate(strings):
        b = s.encode('ascii')
        n = min(len(b), width)
        out[i, :n] = np.frombuffer(b[:n], dtype=np.uint8)

    return out

def u82S(vec_u8: np.ndarray) -> np.ndarray:
    vec_u8 = np.ascontiguousarray(vec_u8)
    view = vec_u8.view(f"|S{len(vec_u8)}")
    return view

def u8m2S(mat_u8: np.ndarray) -> np.ndarray:
    """Convert uint8 matrix to numpy variable length string array"""
    mat_u8 = np.ascontiguousarray(mat_u8)   # important
    n, w = mat_u8.shape
    view = mat_u8.view(f"|S{w}").reshape(n)
    return view

def u82s(vec_u8: np.ndarray) -> np.ndarray:
    view = u82S(vec_u8)
    return view.copy().astype(np.dtypes.StringDType)

def u8m2s(mat_u8: np.ndarray) -> np.ndarray:
    """Convert uint8 matrix to numpy variable length string array"""
    view = u8m2S(mat_u8)
    view_copy = np.copy(view)
    string_vec = view_copy.astype(np.dtypes.StringDType)
    return string_vec

def s2u8(s: str) -> np.ndarray:
    """Convert string to uint8 array."""
    return np.frombuffer(s.encode('ascii'), dtype=np.uint8)


@njit
def _write_2digits(out_row, pos, x):
    # x in [0,99]
    out_row[pos + 0] = ASCII_0 + (x // 10)
    out_row[pos + 1] = ASCII_0 + (x % 10)

@njit
def _write_4digits(out_row, pos, x):
    # x in [0,9999]
    out_row[pos + 0] = ASCII_0 + ((x // 1000) % 10)
    out_row[pos + 1] = ASCII_0 + ((x // 100)  % 10)
    out_row[pos + 2] = ASCII_0 + ((x // 10)   % 10)
    out_row[pos + 3] = ASCII_0 + (x % 10)

@njit
def _write_yyyymm(out_row,pos,yyyy,mm):
    _write_4digits(out_row, pos+0, yyyy)
    out_row[pos+4] = ASCII_DASH
    _write_2digits(out_row, pos+5, mm)

@njit
def _write_yyyymmdd(out_row,pos,yyyy,mm,dd):
    _write_4digits(out_row, pos+0, yyyy)
    out_row[pos+4] = ASCII_DASH
    _write_2digits(out_row, pos+5, mm)
    out_row[pos+7] = ASCII_DASH
    _write_2digits(out_row, pos+8, dd+1)


@njit
def read_1digit(row, pos):
    # Read 1 ASCII digit as int64
    return np.int64(row[pos] - ASCII_0)

@njit
def read_2digits(row, pos):
    # Read 2 ASCII digits as int64
    return np.int64(row[pos] - ASCII_0) * 10 + np.int64(row[pos + 1] - ASCII_0)


@njit
def read_4digits(row, pos):
    # Read 4 ASCII digits as int64
    return (np.int64(row[pos] - ASCII_0) * 1000 +
            np.int64(row[pos + 1] - ASCII_0) * 100 +
            np.int64(row[pos + 2] - ASCII_0) * 10 +
            np.int64(row[pos + 3] - ASCII_0))


@njit
def get_uint8_ym(row):
    """Parse YYYY-MM from uint8 row, return (year, month) as int64."""
    yyyy = read_4digits(row, 0)
    mm = read_2digits(row, 5)
    return yyyy, mm


@njit
def get_uint8_ymd(row):
    """Parse YYYY-MM-DD from uint8 row, return (year, month, day) as int64."""
    yyyy = read_4digits(row, 0)
    mm = read_2digits(row, 5)
    dd = read_2digits(row, 8)
    return yyyy, mm, dd


@njit
def is_leap_year(year):
    return  ( (year % 4 == 0) and  ( (year % 100 != 0) or (year % 400 == 0) ) )

days_per_month = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    dtype=np.int64
)

@njit
def last_day_of_month(year,month):
   if (month == 2) and (is_leap_year(year)): return 29
   return days_per_month[month - 1]

@njit
def days_between(starty, startm, endy, endm):
    start = starty * 12 + (startm - 1)
    end   = endy   * 12 + (endm   - 1)
    n = 0
    for k in range(start,end+1):
        n=n+last_day_of_month(k//12,k % 12 + 1)
    return n

@njit
def add_months(starty, startm, n):
    start = starty * 12 + (startm - 1)
    end   = start + n
    return end // 12, end % 12 + 1

@njit
def make_ym(year,month):
    out = np.empty(7,dtype=np.uint8)
    _write_yyyymm(out,0,year,month)
    return out

@njit
def add_months_ym(spec,n):
        starty, startm = get_uint8_ym(spec)
        endy, endm = add_months(starty, startm, n)
        return make_ym(*add_months(starty, startm, n))

@njit
def add_months_ym_inplace(row,spec,n):
        _write_yyyymm(row,0,*add_months(*get_uint8_ym(spec), n))
        return

@njit
def add_months2specs_inplace(targets,sources,months):
    for i in range(targets.shape[0]):
        target=targets[i]
        source=sources[i]
        n = months[i]
        add_months_ym_inplace(target,source,n)

@njit
def add_months2specs_inplace_NF(targets,sources,months):
    for i in range(targets.shape[0]):
        target=targets[i]
        source=sources[i]
        if months[i]==b'N'[0]:
            add_months_ym_inplace(target,source,1)
        elif months[i]==b'F'[0]:
            add_months_ym_inplace(target,source,2)

@njit
def make_ym_matrix(vals):
    starty, startm, endy, endm = vals
    """
    Inclusive range: (starty,startm) .. (endy,endm)
    Returns uint8 matrix of shape (n, 7) encoding YYYY-MM.
    """
    start = starty * 12 + (startm - 1)
    end   = endy   * 12 + (endm   - 1)
    n = end-start+1
    # YYYY-MM
    if n <= 0: return np.empty((0, 7), dtype=np.uint8)
    out = np.empty((n, 7), dtype=np.uint8)

    for i in range(n):
        ym = i + start
        y, m =  ym// 12, (ym % 12) + 1
        row = out[i]
        _write_yyyymm(row,0,y,m)

    return out

@njit
def make_ymd_matrix(vals):
    starty, startm, endy, endm = vals
    """
    Inclusive range: (starty,startm) .. (endy,endm)
    Returns uint8 matrix of shape (n, 7) encoding YYYY-MM.
    """
    start = starty * 12 + (startm - 1)
    end   = endy   * 12 + (endm   - 1)
    n = days_between(starty,startm,endy,endm)

    # YYYY-MM-DD
    if n <= 0: return np.empty((0, 10), dtype=np.uint8)
    out = np.empty((n, 10), dtype=np.uint8)

    i=0
    for k in range(start,end+1):
        y, m = k // 12, (k % 12) + 1
        days = last_day_of_month(y,m)
        for d in range(days):
            row = out[i]
            _write_yyyymmdd(row,0,y,m,d)
            i=i+1

    return out

@njit
def make_calendar_from_ranges(src):
    """
    src: int matrix shape (R,4) with columns: starty,startm,endy,endm
    Returns:
      src_idx: int64 vector length N (originating row index)
      cal:     uint8 matrix shape (N,10) encoding YYYY-MM-DD
    """
    R = src.shape[0]

    # pass 1: count
    total = 0
    for r in range(R):
        starty = src[r, 0]
        startm = src[r, 1]
        endy   = src[r, 2]
        endm   = src[r, 3]
        total += days_between(starty, startm, endy, endm)

    src_idx = np.empty(total, dtype=np.int64)
    cal     = np.empty((total, 10), dtype=np.uint8)

    # pass 2: fill
    p = 0
    for r in range(R):
        starty = src[r, 0]
        startm = src[r, 1]
        endy   = src[r, 2]
        endm   = src[r, 3]

        start = starty * 12 + (startm - 1)
        end   = endy   * 12 + (endm   - 1)
        if end < start:
            continue

        for k in range(start, end + 1):
            year  = k // 12
            month = (k % 12) + 1
            days  = last_day_of_month(year, month)

            for d0 in range(days):
                _write_yyyymmdd(cal[p], 0, year, month, d0)
                src_idx[p] = r
                p += 1

    return src_idx, cal


@njit(parallel=True)
def make_calendar_from_ranges_par(src):
    """
    Parallel version of make_calendar_from_ranges.

    src: int matrix shape (R,4) with columns: starty,startm,endy,endm
    Returns:
      src_idx: int64 vector length N (originating row index)
      cal:     uint8 matrix shape (N,10) encoding YYYY-MM-DD
    """
    R = src.shape[0]
    if R == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 10), dtype=np.uint8)

    # Pass 1: compute start positions (sequential, O(R))
    # src_starts[r] = where row r starts writing in output
    # src_starts[R] = total
    src_starts = np.empty(R + 1, dtype=np.int64)
    src_starts[0] = 0
    for r in range(R):
        days = days_between(src[r, 0], src[r, 1], src[r, 2], src[r, 3])
        src_starts[r + 1] = src_starts[r] + days

    total = src_starts[R]

    # Allocate output
    src_idx = np.empty(total, dtype=np.int64)
    cal = np.empty((total, 10), dtype=np.uint8)

    # Pass 2: fill dates (parallel)
    for r in prange(R):
        p = src_starts[r]
        starty, startm = src[r, 0], src[r, 1]
        endy, endm = src[r, 2], src[r, 3]

        ym_start = starty * 12 + (startm - 1)
        ym_end = endy * 12 + (endm - 1)

        for ym in range(ym_start, ym_end + 1):
            year = ym // 12
            month = (ym % 12) + 1
            days = last_day_of_month(year, month)

            for d0 in range(days):
                _write_yyyymmdd(cal[p], 0, year, month, d0)
                src_idx[p] = r
                p += 1

    return src_idx, cal


@njit(parallel=True)
def make_calendar_from_specs_par(spec,ntr,xpr):
    """
    Expand straddle specs to calendar dates.

    specs: uint8 matrix shape (R, W) where W >= 16
           Each row: |YYYY-MM|YYYY-MM|...
           Entry yearmonth at positions 1-7
           Expiry yearmonth at positions 9-15

    Returns:
        src_idx: int64 vector length N (originating row index)
        cal:     uint8 matrix shape (N, 10) encoding YYYY-MM-DD
    """
    R = ntr.shape[0]
    if R == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 10), dtype=np.uint8)

    # Pass 1: parse specs and compute start positions (sequential, O(R))
    src_starts = np.empty(R + 1, dtype=np.int64)
    src_starts[0] = 0

    for r in range(R):
        # Parse entry yearmonth from positions 1-7: |YYYY-MM|
        entry_y = read_4digits(ntr[r],0)
        entry_m = read_2digits(ntr[r],5)
        # Parse expiry yearmonth from positions 9-15: |YYYY-MM|
        expiry_y = read_4digits(xpr[r],0)
        expiry_m = read_2digits(xpr[r],5)

        days = days_between(entry_y, entry_m, expiry_y, expiry_m)
        src_starts[r + 1] = src_starts[r] + days

    total = src_starts[R]

    # Allocate output
    src_idx = np.empty(total, dtype=np.int64)
    cal = np.empty((total, spec.shape[1]+10), dtype=np.uint8)

    # Pass 2: fill dates (parallel)
    for r in prange(R):
        p = src_starts[r]

        # Parse again (cheap, avoids storing intermediate arrays)
        entry_y = read_4digits(ntr[r],0)
        entry_m = read_2digits(ntr[r],5)
        expiry_y = read_4digits(xpr[r],0)
        expiry_m = read_2digits(xpr[r],5)

        ym_start = entry_y * 12 + (entry_m - 1)
        ym_end = expiry_y * 12 + (expiry_m - 1)

        for ym in range(ym_start, ym_end + 1):
            year = ym // 12
            month = (ym % 12) + 1
            days = last_day_of_month(year, month)

            for d0 in range(days):
                cal[p,0:spec.shape[1]]=spec[r,:]
                _write_yyyymmdd(cal[p,:], spec.shape[1], year, month, d0)
                src_idx[p] = r
                p += 1

    return src_idx, cal



# -------------------------------------
# Cartesian Product Operations
# -------------------------------------

@njit
def cartesian_product(matrices: tuple) -> np.ndarray:
    """
    Cartesian product of N fixed-width byte matrices.

    Args:
        matrices: tuple of uint8 matrices

    Returns:
        uint8 matrix with shape (product of row counts, sum of widths)

    Example:
        v1 = [["AA"], ["BB"]]  (2 x 2)
        v2 = [["X"], ["Y"], ["Z"]]  (3 x 1)
        cartesian_product((v1, v2)) -> [["AAX"], ["AAY"], ["AAZ"], ["BBX"], ["BBY"], ["BBZ"]]

    For separators, use a (1, 1) matrix:
        sep = np.array([[ord('|')]], dtype=np.uint8)
        cartesian_product((v1, sep, v2)) -> [["AA|X"], ["AA|Y"], ...]
    """
    n = len(matrices)

    # Compute output dimensions and cache sizes/widths
    total_rows = 1
    total_width = 0
    sizes = np.empty(n, dtype=np.int64)
    widths = np.empty(n, dtype=np.int64)
    for i in range(n):
        sizes[i] = matrices[i].shape[0]
        widths[i] = matrices[i].shape[1]
        total_rows *= sizes[i]
        total_width += widths[i]

    out = np.empty((total_rows, total_width), dtype=np.uint8)
    indices = np.zeros(n, dtype=np.int64)

    for p in range(total_rows):
        # Copy current combination
        col_offset = 0
        for i in range(n):
            mat = matrices[i]
            row_idx = indices[i]
            for k in range(widths[i]):
                out[p, col_offset + k] = mat[row_idx, k]
            col_offset += widths[i]

        # Increment indices (odometer style, from right to left)
        for i in range(n - 1, -1, -1):
            indices[i] += 1
            if indices[i] < sizes[i]: break
            indices[i] = 0

    return out


def cartesian_product_np(matrices: tuple) -> np.ndarray:
    """
    Cartesian product of N fixed-width byte matrices using NumPy repeat/tile.

    Pure NumPy implementation - no Numba. Uses index arrays built with
    np.repeat and np.tile, then gathers from input matrices.

    Args:
        matrices: tuple of uint8 matrices

    Returns:
        uint8 matrix with shape (product of row counts, sum of widths)

    Trade-offs vs Numba odometer:
        + No JIT compilation overhead
        + Simple, readable code
        - Allocates n index arrays of size total_rows (memory overhead)
        - Slower for large outputs due to intermediate allocations
    """
    n = len(matrices)
    sizes = np.array([m.shape[0] for m in matrices], dtype=np.int64)
    widths = np.array([m.shape[1] for m in matrices], dtype=np.int64)
    total_rows = np.prod(sizes)
    total_width = np.sum(widths)

    # Build index array for each dimension
    # For dimension k: repeat block size = prod(sizes[k+1:]), tile count = prod(sizes[:k])
    index_arrays = []
    for k in range(n):
        left = np.prod(sizes[k+1:], dtype=np.int64) if k < n - 1 else 1
        right = np.prod(sizes[:k], dtype=np.int64) if k > 0 else 1
        ik = np.tile(np.repeat(np.arange(sizes[k], dtype=np.int64), left), right)
        index_arrays.append(ik)

    # Gather rows from each matrix and concatenate horizontally
    gathered = [matrices[k][index_arrays[k]] for k in range(n)]
    return np.hstack(gathered)


@njit(parallel=True)
def cartesian_product_par(matrices: tuple) -> np.ndarray:
    """
    Parallel cartesian product of N fixed-width byte matrices.

    Uses stride-based index computation for independent iterations,
    enabling multi-threaded execution via prange.

    Args:
        matrices: tuple of uint8 matrices

    Returns:
        uint8 matrix with shape (product of row counts, sum of widths)

    Trade-offs vs sequential odometer:
        + Multi-threaded: scales with available cores
        - Higher per-row cost (n divs + n mods vs ~2 ops)
        - Wins for large outputs (100k+ rows) on multi-core systems
    """
    n = len(matrices)

    # Compute sizes, widths, strides
    sizes = np.empty(n, dtype=np.int64)
    widths = np.empty(n, dtype=np.int64)
    strides = np.empty(n, dtype=np.int64)

    total_rows = 1
    total_width = 0
    for i in range(n):
        sizes[i] = matrices[i].shape[0]
        widths[i] = matrices[i].shape[1]
        total_rows *= sizes[i]
        total_width += widths[i]

    # Compute strides: strides[i] = prod(sizes[i+1:])
    strides[n - 1] = 1
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]

    # Pre-compute column offsets
    col_offsets = np.empty(n, dtype=np.int64)
    col_offsets[0] = 0
    for i in range(1, n):
        col_offsets[i] = col_offsets[i - 1] + widths[i - 1]

    out = np.empty((total_rows, total_width), dtype=np.uint8)

    # Parallel loop - each iteration is independent
    for p in prange(total_rows):
        for i in range(n):
            row_idx = (p // strides[i]) % sizes[i]
            mat = matrices[i]
            offset = col_offsets[i]
            for k in range(widths[i]):
                out[p, offset + k] = mat[row_idx, k]

    return out


# -------------------------------------
# Unfurl Operations
# -------------------------------------

@njit
def unfurl(mat: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Duplicate rows according to counts.

    Args:
        mat: uint8 matrix shape (R, W)
        counts: uint8 vector length R

    Returns:
        out: uint8 matrix shape (sum(counts), W)
        src_idx: int64 vector length sum(counts) - which input row each output came from
    """
    R, W = mat.shape
    total = np.sum(counts)

    out = np.empty((total, W), dtype=np.uint8)
    src_idx = np.empty(total, dtype=np.int64)

    # Pass 2: fill
    p = 0
    for r in range(R):
        count = counts[r]
        for _ in range(count):
            for k in range(W):
                out[p, k] = mat[r, k]
            src_idx[p] = r
            p += 1

    return out, src_idx


@njit
def unfurl_by_spec(mat: np.ndarray, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Unfurl with per-expansion data injection.

    Args:
        mat: uint8 matrix shape (R, W)
        spec: uint8 matrix shape (R, 1 + data_cols)
              - Column 0: dupe count
              - Columns 1+: packed data items

    Returns:
        out: uint8 matrix shape (sum(counts), W + item_width)
        src_idx: int64 vector length sum(counts)

    item_width is derived: data_cols // max_count
    """
    R, W = mat.shape
    spec_cols = spec.shape[1]
    data_cols = spec_cols - 1

    counts = spec[:, 0]
    total = np.sum(counts)
    max_count = np.max(counts)

    if max_count == 0:
        # No expansions - return empty
        item_width = data_cols if data_cols > 0 else 1
        out = np.empty((0, W + item_width), dtype=np.uint8)
        src_idx = np.empty(0, dtype=np.int64)
        return out, src_idx

    item_width = data_cols // max_count

    out = np.empty((total, W + item_width), dtype=np.uint8)
    src_idx = np.empty(total, dtype=np.int64)

    # Pass 2: fill
    p = 0
    for r in range(R):
        count = spec[r, 0]
        for i in range(count):
            # Copy mat row
            for k in range(W):
                out[p, k] = mat[r, k]
            # Copy data item (item i starts at column 1 + i * item_width)
            item_start = 1 + i * item_width
            for k in range(item_width):
                out[p, W + k] = spec[r, item_start + k]
            src_idx[p] = r
            p += 1

    return out, src_idx


@njit
def unfurl_by_spec_sep(mat: np.ndarray, spec: np.ndarray, sep: np.uint8) -> tuple[np.ndarray, np.ndarray]:
    """
    Unfurl with per-expansion data injection and separator.

    Args:
        mat: uint8 matrix shape (R, W)
        spec: uint8 matrix shape (R, 1 + data_cols)
        sep: separator byte

    Returns:
        out: uint8 matrix shape (sum(counts), W + 1 + item_width)
        src_idx: int64 vector length sum(counts)
    """
    R, W = mat.shape
    spec_cols = spec.shape[1]
    data_cols = spec_cols - 1

    counts = spec[:, 0]
    total = np.sum(counts)
    max_count = np.max(counts)

    if max_count == 0:
        item_width = data_cols if data_cols > 0 else 1
        out = np.empty((0, W + 1 + item_width), dtype=np.uint8)
        src_idx = np.empty(0, dtype=np.int64)
        return out, src_idx

    item_width = data_cols // max_count

    out = np.empty((total, W + 1 + item_width), dtype=np.uint8)
    src_idx = np.empty(total, dtype=np.int64)

    # Pass 2: fill
    p = 0
    for r in range(R):
        count = spec[r, 0]
        for i in range(count):
            # Copy mat row
            for k in range(W):
                out[p, k] = mat[r, k]
            # Separator
            out[p, W] = sep
            # Copy data item
            item_start = 1 + i * item_width
            for k in range(item_width):
                out[p, W + 1 + k] = spec[r, item_start + k]
            src_idx[p] = r
            p += 1

    return out, src_idx


@njit
def unfurl_concat(mat: np.ndarray, values: np.ndarray, src_idx: np.ndarray) -> np.ndarray:
    """
    Concatenate mat rows with pre-expanded values using src_idx mapping.

    Args:
        mat: uint8 matrix shape (R, W1)
        values: uint8 matrix shape (N, W2)
        src_idx: int64 vector length N - which row of mat each value row corresponds to

    Returns:
        out: uint8 matrix shape (N, W1 + W2)
    """
    N, W2 = values.shape
    _, W1 = mat.shape
    out = np.empty((N, W1 + W2), dtype=np.uint8)

    for p in range(N):
        r = src_idx[p]
        for k in range(W1):
            out[p, k] = mat[r, k]
        for k in range(W2):
            out[p, W1 + k] = values[p, k]

    return out


@njit
def unfurl_concat_sep(mat: np.ndarray, values: np.ndarray, src_idx: np.ndarray, sep: np.uint8) -> np.ndarray:
    """
    Concatenate mat rows with pre-expanded values, with separator.

    Args:
        mat: uint8 matrix shape (R, W1)
        values: uint8 matrix shape (N, W2)
        src_idx: int64 vector length N
        sep: separator byte

    Returns:
        out: uint8 matrix shape (N, W1 + 1 + W2)
    """
    N, W2 = values.shape
    _, W1 = mat.shape
    out = np.empty((N, W1 + 1 + W2), dtype=np.uint8)

    for p in range(N):
        r = src_idx[p]
        for k in range(W1):
            out[p, k] = mat[r, k]
        out[p, W1] = sep
        for k in range(W2):
            out[p, W1 + 1 + k] = values[p, k]

    return out

# -------------------------------------
# straddle creation
# -------------------------------------

@njit
def nth_occurrence_char(s: str, ch: str, n: int) -> int:
    # n is 1-based
    cnt = 0
    for i, c in enumerate(s):
        if c == ch:
            cnt += 1
            if cnt == n:
                return i
    return -1

@njit
def nth_occurrence(x, v, n):
    idx = np.flatnonzero(x == v)
    if idx.size < n: return -1  # or raise
    return int(idx[n - 1])
@njit
def field(m,s):
    si = nth_occurrence(m[0,:],ord("|"),s)
    ei = nth_occurrence(m[0,:],ord("|"),s+1)
    ss = slice(si+1,ei)
    return m[:,ss]

