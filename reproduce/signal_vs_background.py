# Ported from the working repository's scripts/paper1_signal_vs_background_round62.py (paper 1 rounds 62-63); paths follow the repository layout.
"""
signal_vs_background.py -- Amyad Spector (paper-1 review,
Oct 2026): state early how the detectable signal compares with the natural
background, in particle number (density and column) and in optical depth,
not only in teragrams; NOAA's balloon community thinks in particles per cm^3.

Everything is taken from the paper's own forward model and calibrated
background (calibrated_background_thresholds.py machinery):

  * the 1D silica layer (constant mixing ratio 16 km -> 30 hPa, Segev-2026
    PSD r_med = 268 nm, sigma = 1.31, rho = 2200 kg m^-3) at the baseline
    threshold mass M = 0.101 Tg and at 0.1 Tg: number density at 20 km,
    global-mean column number density, column mass;
  * the calibrated quiet background (GloSSAC 20-25N 2016.5-2018.5 profile,
    SAOD525 = 0.0045, fine mode 60 nm/1.6 at 82 % and coarse mode
    1.5 um/1.8 at 18 % of the 525-nm extinction, member LM65T223): number
    densities at 20 km (total, and r > 0.15 um -- the usual balloon
    optical-particle-counter cut), column number density, stratospheric mass;
  * vertical optical depth above the tropopause (16 km) at 525 nm and at
    the 8.80-um element (0.1 um wide) for both, and their ratios; the same
    for the slant path at a 20-km tangent height.

Run from the repo root:  python reproduce/signal_vs_background.py
Output: outputs/signal_vs_background.json (+ stdout)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calibrated_background_thresholds import (  # noqa: E402
    ALT, iz, RMED_SIL_NM, SIGMA_SIL, CAL_QUIET, TwoComponentBackground,
    glossac_2025N_quiet_profile, member_ri, tangent_to_slant_paths)
from saimon.materials import SilicaRefractiveIndex  # noqa: E402
from saimon.sai import create_silica_sai_layer  # noqa: E402
from saimon.psd import LognormalPSD  # noqa: E402
from saimon.sai_2d import R_EARTH_M  # noqa: E402

M_TG = (0.101, 0.1)
Z_TROP_M = 16_000.0
R_CUT_M = 0.15e-6
OUT = _ROOT / "outputs" / "signal_vs_background.json"
WL = np.array([525e-9, 8.8e-6])
SUB = np.linspace(-0.05, 0.05, 41) * 1e-6      # the 0.1-um element at 8.80 um


def column_number_m2(col):
    n = np.zeros(len(ALT))
    for m in col.modes:
        n += np.asarray(m.n0_m3_profile, float)
    return float(np.trapezoid(n, ALT)), n


def frac_above(rmed_nm, sigma, rcut_m):
    r = np.logspace(-9, -5, 4000)
    p = LognormalPSD(rmed_nm * 1e-9, sigma, 1.0).dn_dr(r)
    return float(np.trapezoid(p[r >= rcut_m], r[r >= rcut_m]) / np.trapezoid(p, r))


def ods(col):
    """vertical OD above the tropopause and slant OD at 20 km tangent, at
    525 nm and at the 8.80-um element."""
    e525 = col.extinction_profile_m1(np.array([525e-9]))[:, 0]
    e88 = col.extinction_profile_m1(8.8e-6 + SUB).mean(axis=1)
    mask = ALT >= Z_TROP_M
    chord = tangent_to_slant_paths(np.array([20_000.0]), ALT)[0]
    return dict(vert_525=float(np.trapezoid(e525[mask], ALT[mask])),
                vert_880=float(np.trapezoid(e88[mask], ALT[mask])),
                slant20_525=float(chord @ e525), slant20_880=float(chord @ e88),
                ext20_525=float(e525[iz]), ext20_880=float(e88[iz]))


def main():
    ri65, wt65, T65 = member_ri("LM65T223")
    ri_kp = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
    area = 4 * np.pi * R_EARTH_M ** 2
    out = {"silica": {}, "background": {}}
    for M in M_TG:
        lay = create_silica_sai_layer(ALT, M, refractive_index=ri_kp,
                                      rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)
        ncol, n = column_number_m2(lay)
        d = ods(lay)
        d.update(mass_tg=M, n20_cm3=float(n[iz] * 1e-6), column_cm2=ncol * 1e-4,
                 column_mass_mg_m2=M * 1e9 / area * 1e6,
                 frac_above_015um=frac_above(RMED_SIL_NM, SIGMA_SIL, R_CUT_M))
        out["silica"][str(M)] = d
    bg = TwoComponentBackground(glossac_2025N_quiet_profile(), CAL_QUIET, ri65, wt65, T65)
    cols = bg.columns()
    tot = {"n20_cm3": 0.0, "n20_cm3_above_015um": 0.0, "column_cm2": 0.0,
           "column_cm2_above_015um": 0.0, "mass_tg": 0.0}
    for (r0, s0, f), c in zip(CAL_QUIET, cols):
        ncol, n = column_number_m2(c)
        fa = frac_above(r0, s0, R_CUT_M)
        d = ods(c)
        d.update(rmed_nm=r0, sigma=s0, share_525=f, n20_cm3=float(n[iz] * 1e-6),
                 column_cm2=ncol * 1e-4, frac_above_015um=fa,
                 mass_tg=float(c.global_mass_kg(alt_min_m=Z_TROP_M) / 1e9))
        out["background"][f"mode_{r0:.0f}nm"] = d
        tot["n20_cm3"] += d["n20_cm3"]; tot["n20_cm3_above_015um"] += d["n20_cm3"] * fa
        tot["column_cm2"] += d["column_cm2"]; tot["column_cm2_above_015um"] += d["column_cm2"] * fa
        tot["mass_tg"] += d["mass_tg"]
    for k in ("vert_525", "vert_880", "slant20_525", "slant20_880", "ext20_525", "ext20_880"):
        tot[k] = sum(out["background"][m][k] for m in out["background"])
    out["background"]["total"] = tot
    s = out["silica"]["0.1"]
    out["ratios_0p1Tg_over_background"] = {
        k: s[k] / tot[k] for k in ("vert_525", "vert_880", "slant20_525", "slant20_880", "ext20_525", "ext20_880")}
    out["ratios_0p1Tg_over_background"]["n20_cm3_over_bg_above_015um"] = s["n20_cm3"] / tot["n20_cm3_above_015um"]
    out["ratios_0p1Tg_over_background"]["column_over_bg_above_015um"] = s["column_cm2"] / tot["column_cm2_above_015um"]
    out["ratios_0p1Tg_over_background"]["mass"] = 0.1 / tot["mass_tg"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
