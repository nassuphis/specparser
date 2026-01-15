# -------------------------------------
# expand-time functions (${...})
# -------------------------------------
"""
Functions available in ${...} expressions.
These run at expansion time and return lists of choices
that participate in the cartesian product.
"""
from pathlib import Path
from typing import List
from collections.abc import Iterable
import re
import subprocess
import numpy as np

from . import expander_state as state
from . import chain as sp
from . import image2spec
from . import slots as slotfuns
from . import dates as datefuns
from . import files as filefuns


# -------------------------------------
# search dict
# -------------------------------------

def search_keys_expand(pat):
    """Return keys from DICT matching regex pattern."""
    rx = re.compile(pat)
    return [k for k in state.DICT.keys() if rx.fullmatch(k)]


def search_values_expand(pat):
    """Return values from DICT where key matches regex pattern."""
    rx = re.compile(pat)
    return [state.DICT[k] for k in state.DICT.keys() if rx.fullmatch(k)]


# -------------------------------------
# random numbers
# -------------------------------------

def rint_expand(N):
    """Return a random integer from 1 to N."""
    return state.RNG.randint(1, N)


def rfloat_expand(a, b):
    """Return a random float between a and b."""
    return state.RNG.uniform(a, b)


# -------------------------------------
# 0 -> 1 steps
# -------------------------------------

def seq(num):
    """Return num evenly spaced values from 0.0 to 1.0."""
    vec = np.linspace(0.0, 1.0, num)
    return [f"{x}" for x in vec]


# -------------------------------------
# read files
# -------------------------------------

def expand_lines(fn: str, lno: int):
    """Return lno random lines from file fn."""
    try:
        return filefuns.get_lines(fn, lno)
    except (FileNotFoundError, ValueError) as e:
        print(f"lines_expand error: {e}")
        return None


def expand_lines2(fn: str, lno: int, delim: str = ":"):
    """Return lno pairs of random lines from file fn, joined by delimiter."""
    try:
        return filefuns.get_lines_paired(fn, lno, delim)
    except (FileNotFoundError, ValueError) as e:
        print(f"lines2_expand error: {e}")
        return None


# -------------------------------------
# spec files
# -------------------------------------

def specfile(specfile: str):
    """Read the whole specfile, return list of spec lines."""
    p = Path(specfile)
    if p.suffix != ".spec":
        p = p.with_suffix(".spec")
    try:
        specs = p.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return specs


def specfile_slots(specfile: str | Path, slots):
    """Read specific slots from specfile."""
    if isinstance(slots, Iterable) and not isinstance(slots, (str, bytes)):
        want = {str(i) for i in slots}
    else:
        want = {str(slots)}

    p = Path(specfile)
    if p.suffix != ".spec":
        p = p.with_suffix(".spec")

    try:
        specs = p.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []

    out = []
    for s in specs:
        d = sp.split_chain(s)
        if "slot" not in d or not d["slot"]:
            continue
        if d["slot"][0] in want:
            out.append(s)
    return out


# -------------------------------------
# slot management
# -------------------------------------

def slots(required: int) -> list[int]:
    """Return required number of free slots based on outschema in DICT."""
    return slotfuns.free_slots(state.DICT["outschema"], required)


# -------------------------------------
# spec modifiers
# -------------------------------------

def spec_rot(spec, rot):
    """Set rotation on a spec."""
    d = sp.split_chain(spec)
    d["rot"] = [str(rot)]
    return sp.concat_chain(d)


def specs_rot(specs: List[str], rot):
    """Set rotation on multiple specs."""
    return [spec_rot(spec, rot) for spec in specs]


def spec_replace(spec: str, key: str, old: str, new: str) -> str:
    """Replace old with new in a specific key of a spec."""
    d = sp.split_chain(spec)
    if key in d:
        d[key] = [v.replace(old, new) for v in d[key]]
    return sp.concat_chain(d)


def specs_replace(specs: str | list[str], key: str, old: str, new: str) -> list[str]:
    """Replace old with new in a specific key of multiple specs."""
    if isinstance(specs, str):
        specs = [specs]
    return [spec_replace(s, key, old, new) for s in specs]


# -------------------------------------
# spec management
# -------------------------------------

def spec2slot(spec: str, slot: int) -> str:
    """Set slot number on a spec."""
    d = sp.split_chain(spec)
    d["slot"] = [str(slot)]
    return sp.concat_chain(d)


def spec2free(spec: str):
    """Assign spec to the first free slot."""
    return spec2slot(spec, slotfuns.first_free_slot(state.DICT["outschema"]))


def specs2free(specs: List[str]):
    """Assign multiple specs to free slots."""
    if isinstance(specs, str):
        specs = [specs]
    new_specs = []
    for spec, slot in zip(specs, slotfuns.free_slots(state.DICT["outschema"], len(specs))):
        new_specs.append(spec2slot(spec, slot))
    return new_specs


# -------------------------------------
# spec files
# -------------------------------------

def specfile2free(specf: str):
    """Read specfile and assign all specs to free slots."""
    specs = specfile(specf)
    if not specs:
        return None
    return specs2free(specs)


def specfile_slots2free(specf: str, slots):
    """Read specific slots from specfile and assign to free slots."""
    specs = specfile_slots(specf, slots)
    if not specs:
        return None
    return specs2free(specs)


# -------------------------------------
# images
# -------------------------------------

def image(imgfile: str):
    """Read spec from image metadata."""
    return image2spec.read_spec_exiftool(imgfile)


def image2free(imgfile: str):
    """Read spec from image and assign to free slot."""
    return spec2free(image(imgfile))


def images(
    schema: str,
    suffices: Iterable[int],
) -> list[str]:
    """Read specs from multiple images."""
    specs: list[str] = []
    for fn in slotfuns.slots2jpegs(schema, suffices):
        specs.append(image(str(fn)))
    return specs


def images2free(
    schema: str,
    suffices: Iterable[int],
) -> list[str]:
    """Read specs from multiple images and assign to free slots."""
    print(f"images2free: {len(suffices)}")
    return specs2free(images(schema, suffices))


# -------------------------------------
# ocr
# -------------------------------------

SCRIPT = Path(__file__).resolve().parent.parent / "lyapunov" / "extract_spec.sh"


def ocr(imagefile):
    """OCR an image file to extract spec."""
    print(f"OCR file:{imagefile}")
    spec = subprocess.check_output(["bash", str(SCRIPT), imagefile], text=True)
    print(f"OCR:{spec}")
    d = sp.split_chain(spec)
    d["source"] = [imagefile]
    return sp.concat_chain(d)


def ocr2free(imagefile):
    """OCR an image and assign to free slot."""
    return spec2free(ocr(imagefile))


def ocrs2free(schema, slots):
    """OCR multiple images and assign to free slots."""
    specs = []
    for jpeg in slotfuns.slots2jpegs(schema, slots):
        specs.append(ocr(jpeg))
    return specs2free(specs)


# ============================================================
# Function registry
# ============================================================

FUNCS: dict[str, object] = {
    "range": range,
    "rint": rint_expand,
    "rfloat": rfloat_expand,
    "key": search_keys_expand,
    "value": search_values_expand,
    # sequence
    "seq": seq,
    # files
    "txt": filefuns.all_lines,
    "lines": expand_lines,
    "lines2": expand_lines2,
    # specs in files
    "specf": specfile_slots2free,
    # specs in images
    "img": images2free,
    # ocr files
    "ocr": ocr2free,
    "ocrs": ocrs2free,
    # individual specs
    "spec": spec2free,
    "rot": specs_rot,
    "swap": specs_replace,
    "free": specs2free,
    # utilities
    "slots": slots,
    "used_slots": slotfuns.used_slots,
    "free_slots": slotfuns.free_slots,
    "slots2jpegs": slotfuns.slots2jpegs,
    "slots2specs": slotfuns.slots2specs,
    "first": slotfuns.first_free_slot,
    # dates
    "bizdays": datefuns.good_days,
}
