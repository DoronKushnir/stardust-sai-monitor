"""Write the ACE Atmospheric Atlas extract shipped with this repository.

Sect. 4.1 uses the tropical 16-20 km bin of the public ACE-FTS Atmospheric
Atlas (Hughes, Bernath and Boone, 2014, JQSRT 148, 18-21; files from
https://ace.scisat.ca) between 950 and 1250 cm^-1.  This script cuts that
window (keeping the file header) out of the two tropical bin files the
analysis knows about and gzips them into data/ace_atlas/.

    python fetch/reduce_ace_atlas.py /path/to/ace_atlas/tro
"""
import gzip, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ace_atlas"
FILES = ["Tropics_016-020km.txt", "Tropics_020-024km.txt"]
NU_MIN, NU_MAX, HEADER = 950.0, 1250.0, 9


def main(src):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        p = Path(src) / name
        if not p.exists(): print(f"skip {name} (not found)"); continue
        with open(p) as f, gzip.open(OUT / (name + ".gz"), "wt") as g:
            for i, line in enumerate(f):
                if i < HEADER: g.write(line); continue
                try: nu = float(line.split()[0])
                except (ValueError, IndexError): continue
                if NU_MIN <= nu <= NU_MAX: g.write(line)
        print(f"wrote {OUT / (name + '.gz')} ({(OUT / (name + '.gz')).stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "Python" / "external" / "ace_atlas" / "tro"))
