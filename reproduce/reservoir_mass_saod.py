"""
reservoir_mass_saod.py -- the stratospheric sulfate reservoir in mass, from
the GloSSAC V2.23 SAOD record and the SAOD-to-mass relation of paper1
Appendix A (eq:mass_saod):

    M_H2SO4 = A_Earth * rho_drop * w * (V_mean / sigma_PSD(525 nm)) * SAOD_525

Round 42 (Doron, paper1 Appendix A, calibrated-background subsection):
"provide references for these numbers. Add also the Hunga-Tonga
calculation".  This script

  1. reproduces the two archived reservoir end points (quiet minimum
     ~0.4 Tg H2SO4, post-Pinatubo ~24 Tg) from the GloSSAC-anchored
     presets of saimon.backgrounds (the numbers quoted since round 1), and
     checks that the closed-form eq:mass_saod reproduces the shell-integrated
     burden of those presets;
  2. adds the Hunga Tonga (January 2022) state: the tropical (20S-20N)
     SAOD_525 peak after the eruption from the same record, converted to
     mass with (a) the literature post-eruption single mode used for
     Pinatubo (250 nm, 1.7, 75 wt%) and (b) the post-Hunga Tonga
     microphysics of the ACE anchor fit (fine mode 100 nm, 1.5; 72 wt%),
     which is the paper's own bracket for that epoch;
  3. converts every burden to its SO2-precursor equivalent (x 64.06/98.08)
     and compares Pinatubo with the ~20 Tg SO2 injection of the literature.

Run from the repo root:  python scripts/reservoir_mass_saod.py
Writes outputs/reservoir_mass_saod.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from saimon.backgrounds import get_background_presets, TROPOPAUSE_M  # noqa: E402
from saimon.sulfate_background import (  # noqa: E402
    stratospheric_burden_kg, sulfate_solution_density_kg_m3)
from saimon.refractive_index import sulfuric_acid_at_temperature  # noqa: E402
from saimon.psd import LognormalPSD  # noqa: E402
from saimon.mie import BohrenHuffmanMie  # noqa: E402
from saimon.aerosol_mie import mie_extinction_coefficient  # noqa: E402
from saimon.glossac import GloSSACLoader, compute_tropical_saod  # noqa: E402

R_EARTH = 6.371e6
A_EARTH = 4 * np.pi * R_EARTH**2
SO2_PER_H2SO4 = 64.06 / 98.08
Z_TROP_M = 16_000.0          # tropical integration boundary of the paper
PINATUBO_SO2_TG = 20.0       # literature injection (Bluth et al. 1992; McCormick et al. 1995)


def mass_from_saod(saod, rmed_nm, sigma, wt, T, ri):
    """Closed-form eq:mass_saod [kg]."""
    psd = LognormalPSD(rmed_nm * 1e-9, sigma, n0_m3=1.0)
    rgrid = np.logspace(np.log10(1e-9), np.log10(10e-6), 700)
    m525 = complex(ri(np.array([525e-9]))[0])
    sig = mie_extinction_coefficient(psd, BohrenHuffmanMie(), m525, 525e-9, rgrid)
    vmean = 4 / 3 * np.pi * (rmed_nm * 1e-9) ** 3 * np.exp(4.5 * np.log(sigma) ** 2)
    rho = sulfate_solution_density_kg_m3(wt, T)
    return A_EARTH * rho * (wt / 100) * vmean / sig * saod, sig, vmean, rho


def main():
    out = {}
    alt = np.arange(0.0, 60_000.1, 250.0)
    ri215 = sulfuric_acid_at_temperature(215.0)

    # 1. archived presets (GloSSAC-anchored empirical profiles, 16-km boundary)
    print("Archived reservoir end points (saimon.backgrounds presets, shell integration"
          f" above {Z_TROP_M/1e3:.0f} km):")
    presets = get_background_presets(alt)
    out["presets"] = {}
    for name, col in presets.items():
        m = stratospheric_burden_kg(col, tropopause_m=Z_TROP_M)
        saod = float(col.nadir_od([525e-9], alt_min_m=Z_TROP_M)[0])
        mode = col.modes[0]
        mc, sig, vmean, rho = mass_from_saod(saod, mode.rmed_m * 1e9, mode.sigma,
                                             col.weight_percent_h2so4, 215.0, ri215)
        print(f"  {name:14s}: SAOD525 {saod:.5f}  r_med {mode.rmed_m*1e9:.0f} nm, sigma {mode.sigma}"
              f"  -> {m/1e9:.3f} Tg H2SO4 (closed form {mc/1e9:.3f}); "
              f"{m*SO2_PER_H2SO4/1e9:.3f} Tg SO2-eq")
        out["presets"][name] = dict(saod525=saod, rmed_nm=mode.rmed_m * 1e9, sigma=mode.sigma,
                                    wt=col.weight_percent_h2so4, M_H2SO4_Tg=m / 1e9,
                                    M_H2SO4_closedform_Tg=mc / 1e9,
                                    M_SO2_Tg=m * SO2_PER_H2SO4 / 1e9,
                                    sigma_psd_525_m2=sig, vmean_m3=vmean, rho=rho)

    # 2. the GloSSAC record: tropical mean (computed, as in fig:glossac_saod)
    loader = GloSSACLoader(str(_HERE / "data/glossac/GloSSAC_V2.23_subset.nc"))
    d = loader.load_raw_data(525)
    time, saod_trop, saod_trop_file = compute_tropical_saod(d)
    years = (time // 100) + (time % 100 - 0.5) / 12.0
    od = np.ma.masked_invalid(d["od_file"])
    li = int(np.argmin(np.abs(d["lat"] - 22.5)))
    od_2025 = od[:, li]
    print(f"\nGloSSAC record: {time[0]} .. {time[-1]}")

    def state(label, ym):
        i = int(np.argmin(np.abs(time - ym)))
        return dict(label=label, yyyymm=int(time[i]), saod_trop=float(saod_trop[i]),
                    saod_trop_file=float(saod_trop_file[i]),
                    saod_2025=float(od_2025[i]) if not np.ma.is_masked(od_2025[i]) else None)

    states = {"quiet_min": state("pristine minimum", 200105),
              "modern": state("modern baseline", 202108),
              "pinatubo_peak": state("post-Pinatubo peak", 199112)}
    post_ht = years >= 2022.0
    i_ht = int(np.nanargmax(np.where(post_ht, saod_trop, np.nan)))
    states["hunga_tonga_peak"] = state("post-Hunga Tonga peak", int(time[i_ht]))
    i_ht25 = int(np.ma.argmax(np.ma.masked_where(~post_ht, od_2025)))
    states["hunga_tonga_peak_2025N"] = state("post-Hunga Tonga 20-25N peak", int(time[i_ht25]))
    for k, s in states.items():
        print(f"  {s['label']:32s} {s['yyyymm']}: tropical SAOD525 {s['saod_trop']:.4f} "
              f"(file {s['saod_trop_file']:.4f}), 20-25N {s['saod_2025']}")
    out["states"] = states

    # 3. masses at the record states, closed form
    print("\nEq. mass_saod at the record states:")
    micro = {"quiet (80 nm, 1.6, 75 wt%)": (80.0, 1.6, 75.0, 215.0, ri215),
             "post-eruption (250 nm, 1.7, 75 wt%)": (250.0, 1.7, 75.0, 215.0, ri215),
             "post-HT ACE fit fine mode (100 nm, 1.5, 72 wt%)":
                 (100.0, 1.5, 72.0, 213.0, ri215)}
    out["masses"] = {}
    for sk, s in states.items():
        out["masses"][sk] = {}
        for mk, (r, sg, wt, T, ri) in micro.items():
            m, *_ = mass_from_saod(s["saod_trop"], r, sg, wt, T, ri)
            out["masses"][sk][mk] = dict(M_H2SO4_Tg=m / 1e9, M_SO2_Tg=m * SO2_PER_H2SO4 / 1e9)
            print(f"  {s['label']:32s} x {mk:48s}: {m/1e9:7.3f} Tg H2SO4 = "
                  f"{m*SO2_PER_H2SO4/1e9:6.3f} Tg SO2-eq")
    mp = out["masses"]["pinatubo_peak"]["post-eruption (250 nm, 1.7, 75 wt%)"]
    print(f"\nPinatubo: {mp['M_SO2_Tg']:.1f} Tg SO2-eq resident at the Dec-1991 peak against "
          f"~{PINATUBO_SO2_TG:.0f} Tg SO2 injected (ratio {mp['M_SO2_Tg']/PINATUBO_SO2_TG:.2f}); "
          f"stoichiometric ceiling {PINATUBO_SO2_TG/SO2_PER_H2SO4:.1f} Tg H2SO4")
    out["pinatubo_injection_Tg_SO2"] = PINATUBO_SO2_TG
    (_HERE / "outputs").mkdir(exist_ok=True)
    with open(_HERE / "outputs/reservoir_mass_saod.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("Saved -> outputs/reservoir_mass_saod.json")


if __name__ == "__main__":
    main()
