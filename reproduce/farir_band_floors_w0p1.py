"""farir_band_floors_w0p1.py -- per-element trace-gas-removal (OE) floors for the
far-infrared 5-um bands of Sect. sec:materials at the paper's 0.1-um element
(round 37, Doron: retire the 0.25-um-element floors and the width factor).

The earlier far-IR floors were single numbers per band: the OE budget evaluated
at ONE 0.25-um target element (the band peak) and applied to all 50 elements of
the 5-um band after a width factor.  At a 0.1-um element that single number
becomes hypersensitive to where the element falls relative to the H2O
pure-rotation lines (calcite 28.5 um: 2.0e-7 -> 3.4e-9 m^-1 between 0.25 and
0.1 um), so here every element of every band gets its own floor: for each
element as the predictand (gas OD at that element), with the whole band as the
channel set, the same OE co-retrieval engine (saimon.o3_removal, flown
per-sample precision sigma_lnT = 5e-4 photon-limited, delta_T = 2 K, the
material budgets' gas sets and line-scatter levels) gives sigma(tau) and, by
the onion-peel conversion at dz = 0.5 km, sigma_alpha.  Elements on saturated
line cores get large floors and carry no weight in the thresholds, exactly as
the O3 core is excluded from the design band.

Output: outputs/materials_calibrated/farir_band_floors_w0p1.json
  {material: {centers_um: [...], sigma_tau_od: [...], sigma_alpha_m1: [...],
              tau_gas_od: [...], gases: [...]}}
read by materials_calibrated_thresholds.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from saimon.o3_removal import (GasRemovalSetup, RetrievalGas, O3Band,  # noqa: E402
                              run_gas_removal_analysis)

OUT = _ROOT / "outputs" / "materials_calibrated" / "farir_band_floors_w0p1.json"
ELEMENT_UM = 0.10
BAND_UM = 5.0
TANGENT_KM = 20.0
DZ_M = 500.0
R_EARTH_M = 6.371e6
from saimon.onion_peel import G_ONION  # round 53 (RC1 M1): exact edge-grid gain from saimon.onion_peel


def sigma_alpha_from_od(sigma_od, dz_m=DZ_M):
    return G_ONION * sigma_od / (2.0 * np.sqrt(2.0 * R_EARTH_M * dz_m))


def _gas(name, prior, line, band_sigma=0.05):
    return RetrievalGas(name, prior_sigma_rel=prior,
                        bands=(O3Band(name, band_sigma, (1e-6, 3e-5)),),
                        sigma_line_rel=line)


# band start [um] and co-retrieved gas set, as in the material budgets
# (snr_integration_time_trade.py C band; material_/calcite_trace_gas_removal.py)
BANDS = {
    "silica":   dict(lo=18.0,  gases=(_gas("h2o", 0.30, 0.002), _gas("hno3", 0.30, 0.005))),
    "alumina":  dict(lo=18.25, gases=(_gas("h2o", 0.30, 0.002), _gas("hno3", 0.30, 0.005),
                                      _gas("o3", 0.30, 0.002))),
    "dolomite": dict(lo=22.5,  gases=(_gas("h2o", 0.30, 0.002), _gas("o3", 0.30, 0.002),
                                      _gas("n2o", 0.05, 0.002))),
    "calcite":  dict(lo=26.0,  gases=(_gas("h2o", 0.30, 0.002), _gas("o3", 0.30, 0.002),
                                      _gas("n2o", 0.05, 0.002))),
}


def main():
    only = [a for a in sys.argv[1:] if a in BANDS]
    arch = json.load(open(OUT)) if OUT.exists() else {}
    for mat, spec in BANDS.items():
        if only and mat not in only:
            continue
        n_el = int(round(BAND_UM / ELEMENT_UM))
        centers = np.round(spec["lo"] + ELEMENT_UM / 2 + ELEMENT_UM * np.arange(n_el), 3)
        chans_m = tuple(centers * 1e-6)
        # Voigt-grid pad: chosen per band so that the fetched HITRAN range is
        # 1-1012.8 cm^-1 for every band -- the range already cached offline for
        # H2O, HNO3, O3 and N2O (hitran.org fetches stall in this environment).
        # A pad wider than the physical ~40 cm^-1 need is harmless: the line
        # wings are simply integrated further out.
        nu_pad = 1012.82 - 1e4 / spec["lo"]
        sig_tau, tau_gas = [], []
        t0 = time.time()
        for j, c in enumerate(centers):
            s = GasRemovalSetup(target_wavelength_m=c * 1e-6, channels_m=chans_m,
                                gases=spec["gases"], h_tan_m=TANGENT_KM * 1e3,
                                delta_T_K=2.0, fwhm_um_target=ELEMENT_UM,
                                nu_pad_cm_ir=nu_pad)
            r = run_gas_removal_analysis(s)
            sig_tau.append(float(r["sigma_total"]))
            tau_gas.append(float(r["tau_target_ref"]))
            print(f"{mat:>8} {c:6.2f} um: tau_gas={tau_gas[-1]:.3e} sigma_tau={sig_tau[-1]:.3e} OD "
                  f"-> sigma_alpha={sigma_alpha_from_od(sig_tau[-1]):.3e} 1/m  [{time.time()-t0:.0f} s]",
                  flush=True)
        arch[mat] = dict(element_um=ELEMENT_UM, band_um=[float(spec["lo"]), float(spec["lo"] + BAND_UM)],
                         centers_um=[float(x) for x in centers], sigma_tau_od=sig_tau,
                         sigma_alpha_m1=[float(sigma_alpha_from_od(x)) for x in sig_tau],
                         tau_gas_od=tau_gas,
                         gases=[(g.name, g.prior_sigma_rel, g.sigma_line_rel) for g in spec["gases"]],
                         tangent_km=TANGENT_KM, dz_m=DZ_M, sigma_lnT=5e-4, delta_T_K=2.0,
                         nu_pad_cm_ir=float(nu_pad))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump(arch, open(OUT, "w"), indent=1)
        print(f"{mat}: archived {n_el} elements -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
