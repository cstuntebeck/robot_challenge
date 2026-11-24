"""Extract NVIDIA's split asset zip (assets.z01 + assets.z02 + assets.zip)
without merging: presents the parts as one seekable stream and neutralizes
zipfile's multi-disk rejection (offsets in split zips are absolute across the
whole stream, so extraction works once the EOCD disk fields are zeroed).

Usage: python3 extract_split_zip.py <part1> <part2> ... <lastpart> <outdir>
"""

import os
import sys
import time
import zipfile


class SplitFile:
    """Read-only seekable concatenation of multiple files."""

    def __init__(self, paths):
        self.paths = paths
        self.sizes = [os.path.getsize(p) for p in paths]
        self.offsets = []
        off = 0
        for s in self.sizes:
            self.offsets.append(off)
            off += s
        self.total = off
        self.handles = [open(p, "rb") for p in paths]
        self.pos = 0

    def seek(self, pos, whence=0):
        if whence == 0:
            self.pos = pos
        elif whence == 1:
            self.pos += pos
        elif whence == 2:
            self.pos = self.total + pos
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            n = self.total - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.total:
            idx = 0
            for i, off in enumerate(self.offsets):
                if self.pos >= off:
                    idx = i
            local = self.pos - self.offsets[idx]
            h = self.handles[idx]
            h.seek(local)
            chunk = h.read(min(n, self.sizes[idx] - local))
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def close(self):
        for h in self.handles:
            h.close()


# neutralize the multi-disk check: zero the EOCD disk-count fields
_orig_endrec = zipfile._EndRecData


def _patched_endrec(fpin):
    endrec = _orig_endrec(fpin)
    if endrec:
        endrec[zipfile._ECD_DISK_NUMBER] = 0
        endrec[zipfile._ECD_DISK_START] = 0
    return endrec


zipfile._EndRecData = _patched_endrec

parts, outdir = sys.argv[1:-1], sys.argv[-1]
os.makedirs(outdir, exist_ok=True)
sf = SplitFile(parts)
zf = zipfile.ZipFile(sf)
names = zf.namelist()
total = len(names)
print(f"[extract] {total} entries, {sf.total/1e9:.1f} GB compressed stream", flush=True)

t0 = time.time()
for i, name in enumerate(names):
    zf.extract(name, outdir)
    if i % 5000 == 0:
        print(f"[extract] {i}/{total} ({100*i/total:.1f}%) elapsed={time.time()-t0:.0f}s", flush=True)

print(f"[extract] EXTRACT_ALL_DONE {total} entries in {time.time()-t0:.0f}s", flush=True)
