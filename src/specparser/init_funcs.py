# -------------------------------------
# init-time functions (!{...})
# -------------------------------------
"""
Functions available in !{...} expressions.
These run once at expansion setup time.
"""
from pathlib import Path

from . import expander_state as state


def seed_init(x=None):
    """
    !{seed(123)}      -> deterministic seed
    !{seed()}         -> reseed from OS entropy
    !{seed("auto")}   -> same as seed()
    """
    return state.seed(x)


def set_const_init(name, value):
    """Set a constant in NAMES."""
    state.set_name(str(name), value)
    return value


def set_dict_init(**kwargs):
    """Replace DICT with provided key/value pairs."""
    state.clear_dict()
    for k, v in kwargs.items():
        state.DICT[str(k)] = str(v)
    return len(state.DICT)


def add_dict_init(**kwargs):
    """Update DICT with provided key/value pairs."""
    for k, v in kwargs.items():
        state.DICT[str(k)] = str(v)
    return len(kwargs)


def load_init(path, *, mode="new", start=1, strip=True, skip_empty=False, encoding="utf-8"):
    """
    Load a text file into DICT:
      key = str(line_number)
      value = line (optionally stripped)

    mode:
      - "new": replace DICT
      - "add": update DICT
    """
    p = Path(str(path))
    lines = p.read_text(encoding=encoding).splitlines()

    d = {}
    i = int(start)
    for line in lines:
        s = line.strip() if strip else line
        if skip_empty and s == "":
            i += 1
            continue
        d[str(i)] = s
        i += 1

    if mode == "new":
        state.clear_dict()
    elif mode != "add":
        raise ValueError("mode must be 'new' or 'add'")

    state.DICT.update(d)
    return len(d)


# ============================================================
# Function registry
# ============================================================

INIT_FUNCS: dict[str, object] = {
    "seed": seed_init,
    "const": set_const_init,
    "new": set_dict_init,
    "add": add_dict_init,
    "load": load_init,
}
