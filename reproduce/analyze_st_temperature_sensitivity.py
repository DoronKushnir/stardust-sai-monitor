"""
analyze_st_temperature_sensitivity.py -- bound the residual sensitivity of the
band-averaged trace-gas slant OD to the choice of a single representative
temperature.

Motivation
----------
saimon.trace_gases evaluates each gas cross section once at a gas-column-weighted
representative (T, P) (see representative_tp) and scales by the full 16-km-up
column.  Because the cross section is band-averaged to an instrument resolution
element (resolution_matched_xsec: a boxcar over `fwhm_um`), the *line width* --
hence the exact (T, P) chosen -- barely affects the averaged sigma: broadening
only redistributes each line's fixed area within the element.  What DOES carry a
real temperature dependence is the line INTENSITY S(T) (partition function +
Boltzmann factor on the lower-state energy), which HAPI recomputes at whatever
T is requested.

This script isolates that S(T) sensitivity.  For each key detection channel it
recomputes the slant-path gas OD at T_rep and T_rep +/- dT (default 15 K),
HOLDING THE COLUMN (number-density profile + geometry) AND PRESSURE FIXED so
that only the cross section's temperature argument changes.  The fractional
change in OD is then a defensible one-line bound on the "single representative
T" approximation, to cite alongside the HITRAN ierr line-intensity analysis
(analyze_hitran_o3_line_uncertainty.py).

Note: the OE model already carries a co-retrieved uniform T-offset state
(saimon.o3_removal: gas_T_column / (1/sigma) dsigma/dT), so this bound is the
*open-loop* sensitivity before that retrieval degree of freedom absorbs it.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from saimon.atmosphere import us_standard_atmosphere
from saimon.geometry import tangent_to_slant_paths
from saimon import trace_gases as tg

TANGENT_KM = 20.0
# Lighter run: 3 nodes spanning the dominant absorbing layer (was 5).  FWHM-
# averaged sigma is set by the gas-column-weighted region ~16-26 km, so the
# 35/55 km nodes contribute little to the tangent-layer OD.
NODE_KM = [16.0, 20.0, 26.0]
DELTA_T_K = 15.0
NU_PAD_CM = 500.0   # match the MIR-panel pad so band-core line wings are included

# Key detection channels: (label, wavelength_um, fwhm_um, [candidate gases]).
# Trimmed to the two channels whose S(T) sensitivity actually matters, with only
# the dominant interfering gas(es) per band (others were <1% of the band OD).
CHANNELS = [
    ("B+ silica / O3 wing", 8.80, 0.10, ["o3"]),
    ("config C silica", 20.40, 0.10, ["h2o", "hno3"]),
]


def build_context():
    alt_m = np.arange(14_000.0, 80_001.0, 500.0)
    atm = us_standard_atmosphere(alt_m)
    chord = tangent_to_slant_paths(np.array([TANGENT_KM * 1e3]), alt_m)[0]
    node_m = np.array(NODE_KM) * 1e3
    node_atm = us_standard_atmosphere(node_m)
    node_idx = [int(np.argmin(np.abs(alt_m - z))) for z in node_m]
    return alt_m, atm, chord, node_m, node_atm, node_idx


def slant_od(gas, wl_m, fwhm, dT, ctx, n_prof):
    """Slant-path OD of one gas at wavelengths `wl_m`, with every node
    temperature offset by `dT` (pressure and number densities held fixed)."""
    alt_m, atm, chord, node_m, node_atm, node_idx = ctx
    if gas in ("cfc11", "cfc12"):
        # Lab .xsc cross section: no HAPI T argument, so it is T-insensitive by
        # construction.  Reported at dT=0 only.
        sigma_wl = tg.cfc_cross_section(gas, wl_m, fwhm_um=fwhm)
        alpha = sigma_wl[None, :] * n_prof[gas][:, None]
        return chord @ alpha
    s_nodes = np.zeros((len(NODE_KM), len(wl_m)))
    for i, k in enumerate(node_idx):
        Tn = float(node_atm.temperature_k[i]) + dT
        Pn = float(node_atm.pressure_pa[i])
        vmr = float(n_prof["h2o"][k] / atm.number_density_m3[k]) if gas == "h2o" else 0.0
        s_nodes[i] = tg.gas_xsec_rm(gas, wl_m, Tn, Pn, fwhm_um=fwhm,
                                    vmr_h2o=vmr, nu_pad_cm=NU_PAD_CM)
    sigma_prof = np.vstack([
        np.interp(alt_m, node_m, s_nodes[:, j], left=s_nodes[0, j], right=s_nodes[-1, j])
        for j in range(len(wl_m))
    ]).T
    alpha = sigma_prof * n_prof[gas][:, None]
    return chord @ alpha


def main():
    ctx = build_context()
    alt_m, atm, chord, *_ = ctx
    all_gases = sorted({g for _, _, _, gs in CHANNELS for g in gs})
    n_prof = tg.number_density_profiles(alt_m, atm.number_density_m3, gases=all_gases)

    print(f"S(T) sensitivity of band-averaged slant OD  (tangent {TANGENT_KM:.0f} km, "
          f"dT = +/-{DELTA_T_K:.0f} K, P & column fixed)")
    print("=" * 78)
    for label, wl_um, fwhm, gases in CHANNELS:
        wl_m = np.array([wl_um * 1e-6])
        print(f"\n{label}:  {wl_um:.2f} um, element {fwhm*1e3:.0f} nm")
        print(f"  {'gas':6s} {'OD(T_rep)':>12s} {'OD(+dT)':>12s} {'OD(-dT)':>12s} "
              f"{'d(lnOD)/dT [%/K]':>18s}")
        total0 = total_hi = total_lo = 0.0
        for gas in gases:
            od0 = float(slant_od(gas, wl_m, fwhm, 0.0, ctx, n_prof)[0])
            if od0 <= 0:
                continue
            if gas in ("cfc11", "cfc12"):
                print(f"  {gas:6s} {od0:12.4e} {'(lab xsc':>12s} {'T-indep)':>12s} "
                      f"{0.0:18.2f}")
                total0 += od0; total_hi += od0; total_lo += od0
                continue
            od_hi = float(slant_od(gas, wl_m, fwhm, +DELTA_T_K, ctx, n_prof)[0])
            od_lo = float(slant_od(gas, wl_m, fwhm, -DELTA_T_K, ctx, n_prof)[0])
            # symmetric per-K log slope
            slope_pct = 100.0 * (od_hi - od_lo) / (2.0 * DELTA_T_K * od0)
            print(f"  {gas:6s} {od0:12.4e} {od_hi:12.4e} {od_lo:12.4e} {slope_pct:18.2f}")
            total0 += od0; total_hi += od_hi; total_lo += od_lo
        if total0 > 0:
            frac_hi = 100.0 * (total_hi - total0) / total0
            frac_lo = 100.0 * (total_lo - total0) / total0
            print(f"  {'TOTAL':6s} {total0:12.4e} {total_hi:12.4e} {total_lo:12.4e} "
                  f"  ({frac_hi:+.1f}% / {frac_lo:+.1f}% for +/-{DELTA_T_K:.0f} K)")


if __name__ == "__main__":
    main()
