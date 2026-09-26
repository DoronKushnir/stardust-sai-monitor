"""
mir_continuum_gas_od.py — Trace-gas slant optical depth across the 3-7 um
sulfuric-acid band region, at the 0.25 um element used for the mid-infrared
channels (Doron's round-10 comments on problem.tex):

  * "The 'best' also combines minimal trace gas removal.  Slightly shorter
    wavelength (~3.8 um by eye, could be better)" -- so the placement of the
    continuum channel must be judged on gas cleanliness as well as on the
    Fisher threshold.  This script gives tau_gas(lambda) on a 0.05 um grid.
  * "the trace gas removal at ~3.8 um (by eye) seems easier than the
    reststrahlen channel" -- checked here against the 8.74 um element, whose
    O3-wing optical depth is ~0.9 (Sect. 3.3).

Result (2026-08-27 run), slant tau_gas per 0.25 um element at a 20 km tangent:
  cleanest in 3-7 um   3.70 um: 0.118      (0.12 x the 8.74 um value)
                       3.80 um: 0.148
                       4.00 um: 0.165
                       3.00 um: 0.173
  reference            8.74 um: 0.975      (O3 9.6 um wing)
  unusable             4.1-4.5 um: 4e3-2e4 (CO2 4.3 um core)
                       5.9-7.0 um: 3-18    (H2O 6.3 um band)
So (i) ~3.7-3.8 um is the right placement, not 4.0 um, and (ii) trace-gas
removal there is ~8x EASIER than at the 8.74 um reststrahlen element, not
harder -- correcting the claim previously made in Sect. 2.3.  The usable
window is narrow, ~3.0-4.0 um.

Machinery as in visnir_channel_gas_od.py / plot_slant_od_trace_gases_wideB.py:
HITRAN line-by-line via saimon.trace_gases, AFGL US-standard profiles, a
20 km tangent, five-node cross-section interpolation, band-averaged to the
0.25 um element.
"""

import sys
from pathlib import Path
import numpy as np

_THIS = Path(__file__).resolve()
_PARENT = _THIS.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from saimon.atmosphere import us_standard_atmosphere
from saimon.geometry import tangent_to_slant_paths
from saimon import trace_gases as tg

TANGENT_KM = 20.0
NODE_KM = [16.0, 20.0, 26.0, 35.0, 55.0]
ELEM_UM = 0.25                       # the mid-infrared design element
GASES = ["h2o", "co2", "o3", "ch4", "n2o", "co"]
# Evaluate on the same wide grid as plot_slant_od_trace_gases_wideB.py
# (2-30 um, 0.01 um step, 0.1 um element, nu_pad 500 cm^-1) so the HITRAN
# line-list cache built for that figure is reused: computing per-placement
# 0.25 um elements directly would open a fresh window per wavelength and
# re-download the line lists.  The 0.25 um design elements are then obtained
# by averaging this curve, which is accurate enough for a placement survey.
GRID_UM = np.round(np.arange(2.0, 30.0001, 0.01), 4)
GRID_FWHM_UM = 0.1
NU_PAD_CM = 500.0
PLACEMENTS = np.round(np.arange(3.0, 7.001, 0.1), 3)
WL_REF = 8.80                        # the reststrahlen channel, for comparison

alt_m = np.arange(14_000.0, 80_001.0, 500.0)
atm = us_standard_atmosphere(alt_m)
chord = tangent_to_slant_paths(np.array([TANGENT_KM * 1e3]), alt_m)[0]
node_m = np.array(NODE_KM) * 1e3
node_atm = us_standard_atmosphere(node_m)
node_idx = [int(np.argmin(np.abs(alt_m - z))) for z in node_m]
n_prof = tg.number_density_profiles(alt_m, atm.number_density_m3, gases=GASES)


def slant_tau_grid():
    """Per-gas slant OD on GRID_UM at GRID_FWHM_UM elements."""
    wl_m = GRID_UM * 1e-6
    out = {}
    for key in GASES:
        s_nodes = np.zeros((len(NODE_KM), len(wl_m)))
        for i, k in enumerate(node_idx):
            T = float(node_atm.temperature_k[i])
            P = float(node_atm.pressure_pa[i])
            vmr = (float(n_prof["h2o"][k] / atm.number_density_m3[k])
                   if key == "h2o" else 0.0)
            s_nodes[i] = tg.gas_xsec_rm(key, wl_m, T, P,
                                        fwhm_um=GRID_FWHM_UM, vmr_h2o=vmr,
                                        nu_pad_cm=NU_PAD_CM)
        sig = np.vstack([np.interp(alt_m, node_m, s_nodes[:, j],
                                   left=s_nodes[0, j], right=s_nodes[-1, j])
                         for j in range(len(wl_m))]).T
        out[key] = chord @ (sig * n_prof[key][:, None])
        print(f"  ... {key} done", flush=True)
    return out


def element_mean(y, centre, width=ELEM_UM):
    m = (GRID_UM >= centre - width / 2) & (GRID_UM <= centre + width / 2)
    return float(np.mean(y[m]))


print(f"Slant tau_gas at a {TANGENT_KM:.0f} km tangent, averaged to "
      f"{ELEM_UM} um elements\n")
tau = slant_tau_grid()
total_grid = sum(tau.values())

print(f"\n{'lambda [um]':>11s} | "
      + " ".join(f"{g:>9s}" for g in GASES)
      + " |     total | dominant")
tot = []
for wl in PLACEMENTS:
    row = [element_mean(tau[g], wl) for g in GASES]
    t = float(np.sum(row))
    tot.append(t)
    print(f"{wl:11.2f} | " + " ".join(f"{v:9.2e}" for v in row)
          + f" | {t:9.3f} | {GASES[int(np.argmax(row))]}")
tot = np.array(tot)

ref_row = [element_mean(tau[g], WL_REF) for g in GASES]
tot_ref = float(np.sum(ref_row))
print(f"\nReference: {WL_REF} um element total tau_gas = {tot_ref:.3f} "
      f"(dominant: {GASES[int(np.argmax(ref_row))]})")

order = np.argsort(tot)
print("\nCleanest five elements in 3-7 um:")
for j in order[:5]:
    print(f"  {PLACEMENTS[j]:.2f} um: tau_gas = {tot[j]:.3f} "
          f"({tot[j] / tot_ref:.2f} x the {WL_REF} um value)")
print("\nDirtiest five:")
for j in order[-5:]:
    print(f"  {PLACEMENTS[j]:.2f} um: tau_gas = {tot[j]:.3f}")
