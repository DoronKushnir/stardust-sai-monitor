"""Write the ACE-FTS v5.2 residual-spectrum subset shipped with this repository.

The paper's ACE analyses use the 143 occultations of the curated public
residual release that fall at 20.1-25.0 N in June-August (the sample of
data/ace_floor/sample.csv), at all their tangent heights.  This script copies
exactly those occultation directories from a local mirror of the v5.2
release (FRDR, doi:10.20383/103.01245), rewriting each residual file with
shortest round-trip number formatting (bit-identical values) and gzip
compression, and writes the matching rows of occultationlist.csv.  The
package's readers open the .gz files transparently.

    python fetch/reduce_ace_v52.py /path/to/ace_v52

The residual spectra are ACE-FTS data (Bernath et al., 2005; Boone et al.,
2020); redistribution of this reduced subset is with the ACE team's
knowledge -- see data/PROVENANCE.md.
"""
import csv, gzip, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ace" / "v52_subset"
SAMPLE = ROOT / "data" / "ace_floor" / "sample.csv"


def main(src):
    src = Path(src)
    occs = [r["occultation"] for r in csv.DictReader(SAMPLE.open())]
    (OUT / "residual").mkdir(parents=True, exist_ok=True)
    with open(src / "occultationlist.csv") as f, open(OUT / "occultationlist.csv", "w", newline="") as g:
        rd = csv.DictReader(f); w = csv.DictWriter(g, fieldnames=rd.fieldnames); w.writeheader()
        keep = set(occs); n = 0
        for row in rd:
            if row["occultation_name"] in keep: w.writerow(row); n += 1
    print(f"occultationlist.csv: {n} rows")
    nfiles = 0
    for occ in occs:
        d = src / "residual" / occ; o = OUT / "residual" / occ; o.mkdir(exist_ok=True)
        for fpath in sorted(d.iterdir()):
            dat = np.loadtxt(fpath)
            with gzip.open(o / (fpath.name + ".gz"), "wt") as g:
                for nu, t in dat:
                    g.write(f"{float(nu)!r} {float(t)!r}\n")
            nfiles += 1
    size = sum(p.stat().st_size for p in OUT.rglob("*")) / 1e6
    print(f"wrote {nfiles} residual files for {len(occs)} occultations -> {OUT} ({size:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "Python" / "external" / "ace_v52"))
