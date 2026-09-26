"""
ace_nulltest_resolution.py -- the ACE null test (Paper 1 Sect. 4) as a
function of the spectral ELEMENT WIDTH, for the 7.8-9.3 um window and the
8-13 um band (round 33, Doron: is the 0.1 um / 0.25 um resolution choice
arbitrary?  can one resolution serve both?).

Same machinery as scripts/ace_fullspectrum_retrieval.py (imported): the 101
ensemble occultations, self-calibrated slant OD near 20.5 km, the full
seven-parameter calibrated background + silica + constant, linearized
around the calibrated background at the occultation's own tangent height.
The residual spectrum and the basis columns are box-averaged into
contiguous elements of width dlam (transmission-weighted for the data,
linear for the basis), MID-INFRARED ALONE (no visible/NIR datum), weighted
with the floor model of scripts/resolution_sensitivity_calibrated.py
(atlas SD at the 0.25-um centers interpolated to the element centers x
the measured width factor of the 8.74-um record).  Elements with <60 %
valid bins (the saturated O3 core) are dropped.

Reported per (range, dlam): number of elements, robust empirical sigma(M)
and M_min(3 sigma) in the slant convention, the median formal sigma, the
ratio, and the shell-convention equivalents (x rho from
outputs/ace_v52/nulltest_convention.json).  Also: the correlation of the
per-element fit residuals between elements as a function of their
separation (at dlam = 0.1 um over the band), i.e. the measured
inter-element correlation length that the per-element-independent floor
model ignores.

Run from the repo root:  python scripts/ace_nulltest_resolution.py
Output: outputs/ace_v52/nulltest_resolution.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ace_fullspectrum_retrieval as F  # noqa: E402
from resolution_sensitivity_calibrated import (  # noqa: E402
    element_centers, floor_model_m1, DLAMS, RANGES, PHASES)

OUT = _ROOT / "outputs" / "ace_v52" / "nulltest_resolution.json"


def element_average(tau, J, wl_um, centers, dlam):
    y, Je, ok = [], [], []
    for c in centers:
        mb = (wl_um >= c - dlam / 2) & (wl_um < c + dlam / 2)
        good = mb & np.isfinite(tau)
        if mb.sum() == 0 or good.sum() < 0.6 * mb.sum():
            y.append(np.nan); Je.append(np.zeros(J.shape[1])); ok.append(False)
            continue
        y.append(float(-np.log(np.mean(np.exp(-tau[good])))))
        Je.append(J[good].mean(axis=0)); ok.append(True)
    return np.array(y), np.array(Je), np.array(ok)


def main():
    nu = F.reference_grid()
    bases = F.build_bases(nu)
    wl = 1e4 / nu
    conv = json.load(open(_ROOT / "outputs" / "ace_v52" / "nulltest_convention.json"))
    rho = conv["rho"]
    keep = set(str(o) for o in np.load(F.SIDE / "matrix_19_22.npz",
                                        allow_pickle=True)["occs"])
    names = F.VARIANTS["full"]
    isil = len(names) - 1
    occs = []
    for d in sorted((F.DATA / "residual").iterdir()):
        if d.name not in keep:
            continue
        got = F.occultation_data(d.name, nu)
        if got is None:
            continue
        h, tau, h_ref = got
        full_rng = (nu >= F.NU_LO) & (nu <= F.NU_HI)
        if (np.isfinite(tau) & full_rng).sum() < 100:
            continue
        occs.append((d.name, h, tau, h_ref,
                     F.basis_columns(h, h_ref, bases, nu, names)))
    print(f"{len(occs)} occultations")

    res = {"rho": rho, "dlams": list(DLAMS), "ranges": {}}
    resid_store = {}
    for rname, (lo, hi) in RANGES.items():
        res["ranges"][rname] = {}
        for dlam in DLAMS:
            per_phase = []
            for ph in PHASES:
                centers = element_centers(lo, hi, dlam, ph)
                floors_od = floor_model_m1(centers, dlam) * F.OD_PER_M1
                Ms, sMs, resids = [], [], []
                for occ, h, tau, h_ref, J in occs:
                    y, Je, ok = element_average(tau, J, wl, centers, dlam)
                    if ok.sum() < Je.shape[1] + 2:
                        continue
                    theta, cov, use, _ = F._solve(Je, y, ok, floors_od, isil,
                                                  clip=False)
                    Ms.append(theta[isil]); sMs.append(np.sqrt(cov[isil, isil]))
                    r = np.full(len(centers), np.nan)
                    r[use] = (y[use] - Je[use] @ theta)
                    resids.append(r)
                if len(Ms) < 30:
                    per_phase.append(dict(phase=ph, n_elements=int(len(centers)),
                                          n_fit=int(len(Ms)), usable=False))
                    continue
                Ms, sMs = np.array(Ms), np.array(sMs)
                med = float(np.median(Ms))
                s_emp = float(1.4826 * np.median(np.abs(Ms - med)))
                s_form = float(np.median(sMs))
                per_phase.append(dict(phase=ph, n_elements=int(len(centers)),
                                      n_fit=int(len(Ms)), usable=True, median_M=med,
                                      sigma_emp=s_emp, sigma_formal=s_form))
                if rname == "band" and abs(dlam - 0.1) < 1e-9 and ph == 0.0:
                    resid_store = dict(centers=centers, R=np.array(resids))
            ok_ph = [r for r in per_phase if r["usable"]]
            if not ok_ph:
                res["ranges"][rname][str(dlam)] = dict(n_elements=per_phase[0]["n_elements"], usable=False)
                print(f"{rname:8s} dlam={dlam:4.2f}: {per_phase[0]['n_elements']:3d} el, "
                      f"too few elements for the 8-parameter fit")
                continue
            se = np.array([r["sigma_emp"] for r in ok_ph]); sf = np.array([r["sigma_formal"] for r in ok_ph])
            row = dict(n_elements=per_phase[0]["n_elements"], usable=True,
                       n_phases=len(ok_ph), median_M=float(np.median([r["median_M"] for r in ok_ph])),
                       sigma_emp=float(np.median(se)), sigma_emp_range=[float(se.min()), float(se.max())],
                       sigma_formal=float(np.median(sf)), ratio=float(np.median(se) / np.median(sf)),
                       mmin_emp_slant=3 * float(np.median(se)),
                       mmin_emp_shell=3 * float(np.median(se)) * rho,
                       mmin_emp_shell_range=[3 * float(se.min()) * rho, 3 * float(se.max()) * rho],
                       mmin_formal_shell=3 * float(np.median(sf)) * rho, per_phase=per_phase)
            res["ranges"][rname][str(dlam)] = row
            print(f"{rname:8s} dlam={dlam:4.2f}: {row['n_elements']:3d} el, "
                  f"sigma_emp={row['sigma_emp']:.4f} [{se.min():.4f}-{se.max():.4f}] "
                  f"(formal {row['sigma_formal']:.4f}, ratio {row['ratio']:.2f}), "
                  f"M_min shell {row['mmin_emp_shell']:.3f} [{row['mmin_emp_shell_range'][0]:.3f}-"
                  f"{row['mmin_emp_shell_range'][1]:.3f}] Tg; median M {row['median_M']:+.3f}")

    # inter-element residual correlation vs separation (band, 0.1 um)
    R = resid_store["R"]; c = resid_store["centers"]
    Rn = R - np.nanmedian(R, axis=0)
    n = len(c)
    corr, sep = [], []
    for i in range(n):
        for j in range(i + 1, n):
            m = np.isfinite(Rn[:, i]) & np.isfinite(Rn[:, j])
            if m.sum() < 30:
                continue
            corr.append(float(np.corrcoef(Rn[m, i], Rn[m, j])[0, 1]))
            sep.append(round(float(c[j] - c[i]), 3))
    corr, sep = np.array(corr), np.array(sep)
    prof = {}
    for s in sorted(set(sep)):
        if s <= 1.5:
            prof[f"{s:.2f}"] = dict(median_corr=float(np.median(corr[sep == s])),
                                    n_pairs=int((sep == s).sum()))
    res["residual_correlation_band_0p1"] = prof
    print("\nband @0.1 um: median residual correlation vs element separation [um]:")
    print({k: round(v["median_corr"], 2) for k, v in prof.items()})
    # a correlation length: first separation where the median drops below 1/e
    ks = sorted(prof, key=float)
    ell = next((float(k) for k in ks if prof[k]["median_corr"] < np.exp(-1)), None)
    res["corr_length_1e_um"] = ell
    print(f"1/e correlation length ~ {ell} um")
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"archived -> {OUT}")


if __name__ == "__main__":
    main()
