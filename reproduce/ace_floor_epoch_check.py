"""
ace_floor_epoch_check.py — Is the 2.0e-3 adjacent-height-difference floor
at the 8.74 um element aerosol variability or measurement systematics?
(Doron's round-13 comment on demonstration.tex: "so why we get only 2e-3
above? Perhaps we are just measuring the aerosol variability?")

Test: split the adjacent-tangent-height difference estimator of
ace_mir_aerosol.py (>= 30 km, robust MAD/sqrt(2)) by epoch.  If the
scatter were real aerosol variability it would track the aerosol loading
-- the post-Hunga-Tonga years carry ~40% more slant OD at this element
near 20 km (0.098 vs 0.070) and an even larger relative enhancement
aloft -- whereas a product-systematics floor is epoch-flat.

Input: outputs/ace_v52/mir_bands.csv (written by ace_mir_aerosol.py).

Result (2026-08-31 run): the floor is epoch-flat, i.e. systematics:
  band 874: all 2.02e-3 | quiet 2004-2019 1.98e-3 | post-HT 2022-24 1.80e-3
  band 400: all 1.44e-3 | quiet 2004-2019 1.51e-3 | post-HT 2022-24 1.14e-3
The post-eruption years show, if anything, a slightly LOWER scatter, and
the 4.0 um element (half the aerosol OD, cleaner gas environment) sits at
0.7x the 8.74 um value -- both inconsistent with an aerosol-variability
origin and consistent with the residual-product systematics reading in
Sect. 4 (ILS knowledge, calibration, pointing).
"""

import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
CSV = _HERE / "outputs" / "ace_v52" / "mir_bands.csv"

rows = list(csv.DictReader(open(CSV)))
occ = np.array([r["occ"] for r in rows])
year = np.array([float(r["year"]) for r in rows])
alt = np.array([float(r["alt"]) for r in rows])
tau = {k: np.array([float(r[f"tau{k}"]) if r[f"tau{k}"] else np.nan
                    for r in rows]) for k in ("874", "400")}


def floor_sigma(band, sel, alt_min=30.0):
    """Robust sigma of adjacent-tangent-height OD differences / sqrt(2)."""
    diffs = []
    for o in set(occ[sel]):
        m = sel & (occ == o) & (alt >= alt_min) & np.isfinite(tau[band])
        if m.sum() >= 2:
            t = tau[band][m][np.argsort(alt[m])]
            diffs.extend(np.diff(t))
    d = np.array(diffs)
    if len(d) < 8:
        return np.nan, len(d)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2), len(d)


SELS = [("all years", np.ones(len(occ), bool)),
        ("quiet 2004-2019", (year >= 2004) & (year <= 2019.99)),
        ("post-HT 2022-2024", (year >= 2022) & (year <= 2024.99))]

print("adjacent-height difference floor (>=30 km), by band and epoch")
print(f"{'band':>5s}  {'selection':20s} {'sigma':>9s} {'N':>6s}")
for band in ("874", "400"):
    for lab, sel in SELS:
        s, n = floor_sigma(band, sel)
        print(f"{band:>5s}  {lab:20s} {s:9.2e} {n:6d}")
