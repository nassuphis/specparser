
from __future__ import annotations

import pyvips as vips
import subprocess

# -------------------------------------
# spec -> xml
# -------------------------------------
def _make_xmp_packet(spec: str) -> bytes:
    # Pick any stable URI you like for your namespace
    ns_uri = "https://example.com/lyapunov/1.0/"
    # Minimal XMP packet with a custom namespace + a single property
    xml = f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
 <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
  <rdf:Description xmlns:lyapunov='{ns_uri}'
                   lyapunov:spec='{spec}'/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""
    return xml.encode("utf-8")


# -------------------------------------
# embed value into image
# -------------------------------------
def spec2image(base,spec):
    base1 = base.copy()  # ensure writable metadata
    base1.set_type(vips.GValue.blob_type, "xmp-data", _make_xmp_packet(spec))
    base1.set_type(vips.GValue.gstr_type,"exif-ifd0-UserComment",spec)
    return base1
    

# -------------------------------------
# read value from image
# -------------------------------------
def read_spec_exiftool(path: str) -> str:
    out = subprocess.check_output(
        ["exiftool", "-s3", "-XMP-lyapunov:spec", path],
        text=True,
    )
    if out.strip():
        return out.strip()

    # fallback to EXIF
    out = subprocess.check_output(
        ["exiftool", "-s3", "-UserComment", path],
        text=True,
    )
    return out.strip()
