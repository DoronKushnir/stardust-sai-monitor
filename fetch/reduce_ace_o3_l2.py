"""Write the ACE-FTS v5.2 Level-2 ozone-profile extract shipped with this repository.

Referee round RC2 (S1) asked for the latitude dependence of the trace-gas floor
to be established from measured ozone.  reproduce/o3_slant_latitude_measured.py
uses the v5.2 Level-2 O3 retrievals (Bernath et al., 2025; FRDR
doi:10.20383/103.01245) of the occultations behind the paper's null tests.
This script reduces the per-occultation ascii Level-2 files of a local mirror
of the release to the columns that analysis needs -- occultation, latitude,
date, month, altitude and O3 number density (VMR x air density, 1-km grid,
filled values dropped) -- for every occultation whose profile reaches from
<= 20 km up to >= 40 km with at least 20 levels (192 of 198 files), and
writes one gzip-compressed CSV.  Values are written with shortest round-trip
formatting, so the extract reproduces the full-release analysis bit for bit.

    python fetch/reduce_ace_o3_l2.py /path/to/ace_v52   # expects l2_asc/*.asc and occultationlist.csv
"""
import csv, glob, gzip, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ace" / "o3_l2_v52" / "o3_profiles_v52.csv.gz"


def read_l2(path):
    lines = path.read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.strip().startswith("z "))
    cols = lines[hdr].replace("P (atm)", "P_atm").split()
    io3, idens = cols.index("O3"), cols.index("dens")
    z, n = [], []
    for l in lines[hdr + 1:]:
        p = l.split()
        if len(p) <= io3:
            continue
        vmr, dens = float(p[io3]), float(p[idens])
        if vmr < 0 or dens < 0:
            continue
        z.append(float(p[0])); n.append(vmr * dens * 1e6)      # cm^-3 -> m^-3
    return z, n


def main(src):
    src = Path(src)
    geo = {r["occultation_name"]: (float(r["latitude"]), int(r["occultation_datetime"][5:7]), r["occultation_datetime"][:10])
           for r in csv.DictReader((src / "occultationlist.csv").open())}
    rows, nocc = [], 0
    for f in sorted(glob.glob(str(src / "l2_asc" / "*.asc"))):
        occ = os.path.basename(f).replace("v5.2.asc", "")
        if occ not in geo:
            continue
        z, n = read_l2(Path(f))
        if len(z) < 20 or min(z) > 20.0 or max(z) < 40.0:
            continue
        nocc += 1
        lat, month, date = geo[occ]
        rows += [(occ, repr(lat), date, month, repr(zi), repr(ni)) for zi, ni in zip(z, n)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt") as fh:
        w = csv.writer(fh)
        w.writerow(["occultation", "latitude_deg", "date", "month", "z_km", "o3_number_density_m3"])
        w.writerows(rows)
    print(f"{nocc} profiles, {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
