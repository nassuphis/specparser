# -------------------------------------
# file reading utilities
# -------------------------------------
import random
import secrets

# Global cache for file contents
_file_cache: dict[str, list[str]] = {}

# Module-level RNG
_RNG = random.Random(secrets.randbits(128))


def set_seed(seed: int | None = None) -> int:
    """
    Set the random seed for file operations.
    If seed is None, reseeds from OS entropy.
    Returns the seed used.
    """
    if seed is None:
        seed = secrets.randbits(128)
    _RNG.seed(seed)
    return seed


def _read_file(fn: str) -> list[str]:
    """
    Read file contents, caching for repeated access.
    Returns list of lines (without trailing newlines).
    """
    if fn not in _file_cache:
        with open(fn, "r", encoding="utf-8") as f:
            _file_cache[fn] = f.read().splitlines()
    return _file_cache[fn]


def clear_cache() -> None:
    """Clear the file cache."""
    _file_cache.clear()


def all_lines(fn: str) -> list[str]:
    """
    Return all lines from file fn.
    Caches file contents across calls.
    """
    return _read_file(fn)


def get_line(fn: str, lno: int) -> str:
    """
    Return line number lno (0-based) from file fn.
    Caches file contents across calls.
    """
    lines = _read_file(fn)
    try:
        return lines[lno]
    except IndexError:
        raise IndexError(f"line number {lno} out of range for file '{fn}'")


def get_lines(fn: str, count: int) -> list[str]:
    """
    Return `count` random lines from file fn.
    """
    lines = _read_file(fn)
    if not lines:
        raise ValueError(f"file '{fn}' is empty")
    return _RNG.choices(lines, k=count)


def get_random_line(fn: str) -> str:
    """
    Return a random line from file fn.
    """
    lines = _read_file(fn)
    if not lines:
        raise ValueError(f"file '{fn}' is empty")
    return _RNG.choice(lines)


def get_lines_paired(fn: str, count: int, delim: str = ":") -> list[str]:
    """
    Return `count` pairs of random lines from file fn, joined by delimiter.
    """
    l1 = get_lines(fn, count)
    l2 = get_lines(fn, count)
    return [f"{a}{delim}{b}" for a, b in zip(l1, l2, strict=True)]


# ============================================================
# CLI
# ============================================================

def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="File reading utilities.")
    p.add_argument("file", help="File to read")
    p.add_argument("--line", "-l", type=int, help="Get specific line number (0-based)")
    p.add_argument("--random", "-r", type=int, help="Get N random lines")
    p.add_argument("--count", "-c", action="store_true", help="Count lines in file")
    args = p.parse_args()

    if args.count:
        lines = _read_file(args.file)
        print(f"{len(lines)} lines")
        return 0

    if args.line is not None:
        print(get_line(args.file, args.line))
        return 0

    if args.random is not None:
        for line in get_lines(args.file, args.random):
            print(line)
        return 0

    # Default: print all lines
    for i, line in enumerate(_read_file(args.file)):
        print(f"{i}: {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
