# -------------------------------------
# slot management
# -------------------------------------
from pathlib import Path
from collections.abc import Iterable
import re
from typing import Any, Dict

def used_files(schema: str) -> list[str]:
    base = Path(schema)
    dirpath = base.parent
    stem = base.name

    pat = re.compile(rf"^{re.escape(stem)}_(\d+)\.jpg$")
    files: list[str] = []

    for p in dirpath.iterdir():
        if not p.is_file():
            continue
        if pat.match(p.name):
            files.append(str(p))

    return files

def slots2jpegs(schema: str, slots: Iterable[int]) -> list[str]:
    base = Path(schema)
    dirpath = base.parent
    stem = base.name
    out: list[str] = []
    for i in sorted(set(slots)):
        p = dirpath / f"{stem}_{i:05d}.jpg"
        if p.is_file():
            out.append(str(p))
    return out

def slots2specs(schema: str, slots: Iterable[int]) -> list[str]:
    base = Path(schema)
    dirpath = base.parent
    stem = base.name
    out: list[str] = []
    for i in sorted(set(slots)):
        p = dirpath / f"{stem}_{i:05d}.spec"
        if p.is_file():
            out.append(str(p))
    return out

def used_slots(schema: str) -> list[int]:
    base = Path(schema)
    stem = base.name

    pat = re.compile(rf"^{re.escape(stem)}_(\d+)\.jpg$")
    used: set[int] = set()

    for fname in used_files(schema):
        p = Path(fname)
        m = pat.match(p.name)
        if m:
            used.add(int(m.group(1)))

    return list(used)

def max_slot(schema: str) -> int | None:
    used = set(used_slots(schema))
    if not used: return None
    return max(used)

def first_free_slot(schema: str) -> int:
    used = set(used_slots(schema))
    if not used: return 1
    universe = set(range(1,max(used)+2))
    return min(universe - used)

def free_slots(schema: str, required: int) -> list[int]:
    used = set(used_slots(schema))
    if not used: return list(range(1,required+1))
    universe = set(range(1,max(used)+required+1)) # required slot count fits
    free = universe - used
    return sorted(free)[:required]

def slots(context: Dict[str, Any], required: int) -> list[int]:
    return free_slots(context["outschema"], required)

