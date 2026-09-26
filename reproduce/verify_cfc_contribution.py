"""verify_cfc_contribution.py -- quantify the CFC-11/-12 slant-OD contribution at
the silica B+ channel and the calcite bands, now that BOTH measured HITRAN .xsc
files are present (CCl3F / CCl2F2 in data/trace_gases/cfc_xsec).

Part 1: CFC slant OD at 20 km tangent (0.25 um element) at the key wavelengths,
        vs the silica B+ O3-removal floor (~2e-3 OD) and the calcite signals.
Part 2: re-run the calcite nu2 (11.4 um) and nu3 (6.9 um) OE co-retrieval WITH
        CFC-11/-12 added to the gas set, comparing the floor to the no-CFC run.
"""
import sys
from pathlib import Path
import numpy as np

_P = Path(__file__).resolve().parent.parent
if str(_P) not in sys.path:
    sys.path.insert(0, str(_P))

from saimon.atmosphere import us_standard_atmosphere
from saimon.geometry import tangent_to_slant_paths
from saimon import trace_gases as tg
from saimon.sai import create_calcite_sai_layer
from saimon.o3_removal import (GasRemovalSetup, RetrievalGas, O3Band,
                              run_gas_removal_analysis)

TANGENT_KM = 20.0
ELEMENT_UM = 0.25
R_EARTH_M = 6.371e6
G_ONION = np.sqrt(5.0 - 2.0 * np.sqrt(3.0))
DZ_M = 500.0


def sig_alpha(od):
    return G_ONION * od / (2.0 * np.sqrt(2.0 * R_EARTH_M * DZ_M))


def cfc_slant_od(alt_m, chord, n_prof, wl_um):
    """CFC-11/-12 slant OD at wl_um (band-averaged xsec is T/P-independent)."""
    wl_m = np.asarray(wl_um) * 1e-6
    out = {}
    for key in ("cfc11", "cfc12"):
        sig = tg.gas_xsec_rm(key, wl_m, 216.0, 5000.0, fwhm_um=ELEMENT_UM)  # T,P ignored for CFC
        out[key] = chord @ (sig[None, :] * n_prof[key][:, None])
    return out


def main():
    alt_m = np.arange(14_000.0, 80_001.0, 500.0)
    atm = us_standard_atmosphere(alt_m)
    chord = tangent_to_slant_paths(np.array([TANGENT_KM * 1e3]), alt_m)[0]
    n_prof = tg.number_density_profiles(alt_m, atm.number_density_m3,
                                        gases=["cfc11", "cfc12"])

    print("=== Part 1: CFC slant OD (20 km tangent, 0.25 um element, real .xsc) ===")
    print("    VMR (constant fallback): CFC-11=230 ppt, CFC-12=520 ppt "
          "(overstates above ~20 km)\n")
    wls = [8.80, 8.74, 9.2, 10.9, 11.0, 11.4, 11.8, 6.9, 7.0]
    od = cfc_slant_od(alt_m, chord, n_prof, wls)
    print(f"  {'lam[um]':>8} {'CFC-11':>11} {'CFC-12':>11} {'CFC total':>11} "
          f"{'sig_alpha':>11}")
    for i, l in enumerate(wls):
        tot = od["cfc11"][i] + od["cfc12"][i]
        print(f"  {l:8.2f} {od['cfc11'][i]:11.3e} {od['cfc12'][i]:11.3e} "
              f"{tot:11.3e} {sig_alpha(tot):11.3e}")
    print("\n  reference: silica B+ (8.80um) O3-removal floor ~ 2.0e-3 OD "
          "(sig_alpha ~1.5e-8)")
    print("  reference: calcite nu2 (11.4um) signal 3.8e-2 OD, HNO3 bg 2.4e-1 OD\n")

    # Part 2: OE with and without CFC for the calcite window bands.
    layer = create_calcite_sai_layer(alt_m, target_mass_tg=1.0)

    def gas(name, prior, line):
        return RetrievalGas(name, prior_sigma_rel=prior,
                            bands=(O3Band(name, 0.05, (1e-6, 3e-5)),),
                            sigma_line_rel=line)

    def run(target_um, lo, hi, base_gases, add_cfc):
        chans = tuple(round(x, 3) * 1e-6 for x in np.arange(lo, hi + 1e-9, 0.1))
        gases = list(base_gases)
        if add_cfc:
            # CFC VMR profile above the tangent is uncertain (constant-VMR
            # fallback overstates it); give a loose 15% per-node prior.
            gases += [gas("cfc11", 0.15, 0.02), gas("cfc12", 0.15, 0.02)]
        all_m = np.array(list(chans) + [target_um * 1e-6])
        tau_cal = chord @ layer.extinction_profile_m1(all_m)
        s = GasRemovalSetup(target_wavelength_m=target_um * 1e-6, channels_m=chans,
                            gases=tuple(gases), h_tan_m=TANGENT_KM * 1e3,
                            delta_T_K=2.0, fwhm_um_target=ELEMENT_UM,
                            silica_tau_ref=tau_cal, silica_prior_sigma=1.0e3)
        r = run_gas_removal_analysis(s)
        return r, float(tau_cal[-1])

    print("=== Part 2: calcite OE floor, without vs with CFC-11/-12 co-retrieval ===")
    cases = [
        ("nu2 11.4um", 11.4, 10.5, 12.0,
         (gas("hno3", 0.30, 0.005), gas("h2o", 0.30, 0.002))),
        ("nu3 6.9um", 6.9, 6.3, 7.6,
         (gas("h2o", 0.30, 0.002), gas("ch4", 0.05, 0.002),
          gas("n2o", 0.05, 0.002), gas("o3", 0.30, 0.002))),
    ]
    for label, tgt, lo, hi, base in cases:
        r0, tc = run(tgt, lo, hi, base, add_cfc=False)
        r1, _ = run(tgt, lo, hi, base, add_cfc=True)
        print(f"\n--- {label}  (calcite signal {tc:.3e} OD)")
        for tag, r in (("no CFC", r0), ("+CFC", r1)):
            cfc_od = sum(v for k, v in r["tau_target_by_gas"].items() if k.startswith("cfc"))
            print(f"    {tag:7s}: tau_gas={r['tau_target_ref']:.3e} "
                  f"(CFC={cfc_od:.2e})  sigma_total={r['sigma_total']:.3e}  "
                  f"sigma_cal={r['sigma_silica_od']:.3e}  "
                  f"sig_alpha={sig_alpha(r['sigma_silica_od']):.3e}")


if __name__ == "__main__":
    main()
