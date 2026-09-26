"""
check_limb_emission_baseline.py — the atmospheric thermal-emission baseline
of the solar-occultation channels (Notes Sec. on the MIR floor; Config C
and the long-wave material channels).

An occultation instrument staring at the sun measures, per element,
    S = < L_sun t(nu) > + < L_emis(nu) > + instrument background,
and ratioing against the exo-atmospheric sun removes neither the emission
nor its tangent-height dependence.  The apparent element transmission is
biased HIGH, so every retrieved slant OD is biased LOW by
    r = <L_emis> / (L_sun <t>).

The Planck ratio B(T_atm)/B(T_sun) rises from 4.6e-4 at 8.74 um (8.80 um since round 43) to 8e-3 at
20.4 um and 1.4e-2 at 29 um, while the long-wave channels sit in the H2O
rotation / HNO3 nu9 forest — so the term must be computed LINE-RESOLVED:
in a saturated line forest the element transmission is carried by the
micro-windows (<e^-tau> >> e^-<tau>), and both <t> and <L_emis> are
radiance averages over the element, not functions of the band-averaged
optical depth.

Method: native Voigt cross sections on a 0.01 cm^-1 grid (tg._voigt_native,
disk-cached; single representative (T,p) at 21 km — the tangent shells
dominate the column) + MT-CKD H2O continuum; path-ordered emission
integral with self-absorption over the limb ray (sun-side + instrument-
side half-chords), including the 1-Tg silica layer's own reststrahlen
emission at the shell temperature; element-boxcar averages of t(nu) and
L_emis(nu) at each report wavelength.

Channels: B+ (8.74 um), C (19-22 um, incl. the alumina 20.7 um W band),
and the carbonate/alumina lattice region (24.5-29.5 um: dolomite 24.9,
calcite 28.5/29.2).  The remaining appendix channels (6.4/6.9, 11.3/11.4,
12.9, 13.7/14.0 um) are assessed in a table mode calibrated against the
line-resolved anchors; the photon-saturated ones (tau_gas > 5) are lower
bounds and those channels are already marginal in their own tables.

Comparison currencies: each channel's gas-removal floor sigma_gas [OD]
from the appendix tables, and the 1-Tg material signal.  The emission
term is forward-modelable from co-retrieved gases + T(z) (~2 K -> ~2.5%);
both raw and modelled-residual ratios are printed.

Caveats: no HITRAN lines below 333 cm^-1 for H2O (>30 um; small here);
no far-IR HNO3 lines in the registry windows — _voigt_native fetches the
continuous range, so HNO3 nu9 (458 cm^-1) IS included where listed.
5772-K solar disk: the true ~20-30 um solar brightness temperature is
~5000 K, making true r ~15-20% larger.

Usage:  python3 scripts/check_limb_emission_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from saimon.atmosphere import us_standard_atmosphere
from saimon.geometry import tangent_to_slant_paths
from saimon import trace_gases as tg
from saimon.sai import create_silica_sai_layer
from saimon.materials import SilicaRefractiveIndex

H, C, KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
T_SUN = 5772.0
RE = 6.371e6

TANGENTS_KM = [15.0, 20.0, 25.0]
NU_STEP = 0.01                  # cm^-1 (lines ~0.004 cm^-1: MC-sampled)
NU_PAD = 40.0                   # cm^-1 beyond the channel edges
Z_REP_M = 21_000.0              # representative (T,p) for the cross sections

# channel: (wl_lo_um, wl_hi_um, element_fwhm_um, gases)
# Round 37 (paper1): `--element 0.10` evaluates the far-IR channels at the
# paper's 0.1-um element (default 0.25 keeps the Notes numbers); with
# `--floors-json <farir_band_floors_w0p1.json>` the comparison floors of the
# far-IR report wavelengths are taken from that per-element OE archive
# (nearest element) instead of the REPORT table.
import sys
FAR_ELEMENT_UM = 0.25
if "--element" in sys.argv:
    FAR_ELEMENT_UM = float(sys.argv[sys.argv.index("--element") + 1])
FLOORS_JSON = (sys.argv[sys.argv.index("--floors-json") + 1]
               if "--floors-json" in sys.argv else None)
# `--floor-min-od 5e-4`: bound the archive floors below (the SNR-2000
# transmission floor, 5e-4 OD <-> 3.9e-9 m^-1, as materials_calibrated_thresholds.py does)
FLOOR_MIN_OD = (float(sys.argv[sys.argv.index("--floor-min-od") + 1])
                if "--floor-min-od" in sys.argv else 0.0)
CHANNELS = {
    "B+ 8.80 um": (8.3, 9.2, 0.10, ["o3", "h2o", "ch4", "n2o", "co2"]),
    "C 19-22 um": (19.0, 22.0, FAR_ELEMENT_UM, ["hno3", "h2o", "o3", "co2", "n2o"]),
    "lattice 24.5-29.5 um": (24.5, 29.5, FAR_ELEMENT_UM,
                             ["hno3", "h2o", "o3", "n2o", "co2"]),
    "carbonate nu3 6.2-7.1 um": (6.2, 7.1, 0.10, ["h2o", "ch4", "n2o"]),
}
# report wavelengths and the sigma_gas floors [OD] they compare against
# (silica: Sec. 05 OE floors; others: appendix C/D/E tables, 20-km tangent)
REPORT = {
    "B+ 8.80 um": [(8.80, 2.0e-3, "silica B+")],
    "C 19-22 um": [(19.8, 3.7e-4, "C co-retrieval edge"),
                   (20.4, 3.7e-4, "silica C"),
                   (20.7, 1.2e-3, "alumina W")],
    "lattice 24.5-29.5 um": [(24.9, 2.0e-3, "dolomite lattice"),
                             (28.5, 2.6e-2, "calcite lattice peak"),
                             (29.2, 2.3e-3, "calcite red wing")],
    "carbonate nu3 6.2-7.1 um": [(6.4, 6.0e-2, "dolomite nu3"),
                                 (6.9, 1.7e-2, "calcite nu3")],
}
# table-mode channels: (material, lam, tau_gas, tau_mat, sigma_gas, note)
# band-averaged ODs from the appendix floor tables (20-km tangent, 1 Tg)
# (the saturated nu4 channels at 13.7/14.0 um are dead in their own floor
# tables (SNR <= 0.004) and are not assessed; the nu3 channels are handled
# line-resolved above)
TABLE_CHANNELS = [
    ("calcite", 11.4,  0.26,  0.038, 1.5e-3, ""),
    ("dolomite", 11.3, 0.30,  0.038, 2.0e-3, ""),
    ("alumina", 12.9,  0.47,  0.321, 2.6e-3, ""),
]
T_EMIS_K = 232.0
MODEL_LEVEL = 0.025             # emission-model accuracy (T to ~2 K + gases)
SIL_TG = 1.0
DZ_BIN_M = 1000.0


def planck(lam_m, T):
    lam_m = np.asarray(lam_m, float)
    return (2.0 * H * C**2 / lam_m**5) / np.expm1(H * C / (lam_m * KB * T))


def _floors_from_json():
    """Override REPORT floors from the per-element far-IR OE archive."""
    import json
    arch = json.load(open(FLOORS_JSON))
    who_mat = {"silica C": "silica", "C co-retrieval edge": "silica",
               "alumina W": "alumina", "dolomite lattice": "dolomite",
               "calcite lattice peak": "calcite", "calcite red wing": "calcite"}
    for label in ("C 19-22 um", "lattice 24.5-29.5 um"):
        new = []
        for lam0, sig_floor, who in REPORT[label]:
            a = arch[who_mat[who]]
            j = int(np.argmin(np.abs(np.asarray(a["centers_um"]) - lam0)))
            new.append((lam0, max(float(a["sigma_tau_od"][j]), FLOOR_MIN_OD), who))
            print(f"  floor override {who} @ {lam0} um: {sig_floor:.2e} -> "
                  f"{new[-1][1]:.2e} OD (element {a['centers_um'][j]} um)")
        REPORT[label] = new


def main() -> None:
    if FLOORS_JSON:
        _floors_from_json()
    alt_m = np.arange(14_000.0, 80_001.0, 500.0)
    atm = us_standard_atmosphere(alt_m)
    chords = tangent_to_slant_paths(np.array(TANGENTS_KM) * 1e3, alt_m)
    rep = us_standard_atmosphere(np.array([Z_REP_M]))
    T_rep = float(rep.temperature_k[0])
    p_rep = float(rep.pressure_pa[0])

    silica = create_silica_sai_layer(
        alt_m, target_mass_tg=SIL_TG, rmed_nm=268.0, sigma=1.31,
        refractive_index=SilicaRefractiveIndex(
            str(_PARENT / "data" / "optics" / "SiO2_KitamuraPopova.yml")))

    l_bin = 2.0 * np.sqrt(2.0 * RE * DZ_BIN_M)
    print(f"cross sections at z = {Z_REP_M/1e3:.0f} km "
          f"(T = {T_rep:.1f} K, p = {p_rep/100:.1f} hPa), "
          f"step {NU_STEP} cm^-1; 1-km bin chord {l_bin/1e3:.0f} km",
          flush=True)

    anchors = []                      # (lam0, tau_eff, r) at 20-km tangent

    for label, (wl_lo, wl_hi, fwhm, gases) in CHANNELS.items():
        nu_lo = max(1.0e4 / wl_hi - NU_PAD, 1.0)
        nu_hi = 1.0e4 / wl_lo + NU_PAD
        nu = np.arange(nu_lo, nu_hi + NU_STEP, NU_STEP)
        sig = {}
        for g in gases:
            nug, coef = tg._voigt_native(
                tg.HITRAN_GASES[g], nu_lo, nu_hi, T_rep,
                p_rep / 101325.0, NU_STEP)
            s = np.interp(nu, nug, coef, left=0.0, right=0.0) * 1e-4
            if g == "h2o":
                s = s + tg.mt_ckd_h2o_continuum(1.0e-2 / nu, T_rep, p_rep,
                                                5.0e-6)
            sig[g] = s
            print(f"  {label}: {g} sigma ready "
                  f"(max {s.max():.2e} m^2)", flush=True)

        lam_fine = (1.0e-2 / nu)[::-1]
        rev = slice(None, None, -1)
        sig_rev = {g: sig[g][rev] for g in gases}

        n_prof = tg.number_density_profiles(alt_m, atm.number_density_m3,
                                            gases=gases)
        # silica on a coarse grid, interpolated (smooth reststrahlen shape)
        lam_c = np.geomspace(lam_fine[0], lam_fine[-1], 120)
        k_sil_c = silica.extinction_profile_m1(lam_c)          # (z, lam_c)
        L_sun = planck(lam_fine, T_SUN)

        print(f"=== {label} (element {fwhm} um; gases: {', '.join(gases)}; "
              f"1-Tg silica along the path) ===", flush=True)
        for it, h_km in enumerate(TANGENTS_KM):
            ell = chords[it]
            idx = np.where(ell > 0)[0]
            order = np.concatenate([idx[::-1], idx])
            # dtau per segment on the fine grid
            k_shell = np.zeros((len(idx), len(lam_fine)))
            for g in gases:
                k_shell += n_prof[g][idx, None] * sig_rev[g][None, :]
            k_shell += np.vstack([
                np.interp(lam_fine, lam_c, k_sil_c[j]) for j in idx])
            k_seg = np.concatenate([k_shell[::-1], k_shell], axis=0)
            w_seg = 0.5 * ell[order]
            dtau = w_seg[:, None] * k_seg
            T_seg = atm.temperature_k[order]
            B_seg = np.array([planck(lam_fine, float(t)) for t in T_seg])
            tau_after = np.cumsum(dtau[::-1], axis=0)[::-1] - dtau
            L_emis = np.sum(B_seg * dtau * np.exp(-tau_after), axis=0)
            t_fine = np.exp(-np.sum(dtau, axis=0))

            for lam0, sig_floor, who in REPORT[label]:
                m = np.abs(lam_fine * 1e6 - lam0) < fwhm / 2.0
                t_bar = float(np.mean(t_fine[m]))
                L_bar = float(np.mean(L_emis[m]))
                Ls = float(np.mean(L_sun[m]))
                r = L_bar / (Ls * t_bar)
                tau_eff = -np.log(t_bar)
                tau_avg = float(-np.mean(np.log(np.maximum(t_fine[m],
                                                           1e-300))))
                if h_km == 20.0:
                    anchors.append((lam0, tau_eff, r))
                print(f"  tan {h_km:4.1f} km, {lam0:5.2f} um ({who:18s}): "
                      f"tau_eff {tau_eff:6.2f} (band-avg {tau_avg:7.2f})  "
                      f"t {t_bar:8.2e}  r {r:8.2e}  r/floor "
                      f"{r/sig_floor:7.2f}  resid/floor "
                      f"{MODEL_LEVEL*r/sig_floor:5.2f}", flush=True)
        print(flush=True)

    # ---------------- table-mode for the remaining channels ----------------
    def r_table(lam_um, tau_tot, t_e=T_EMIS_K):
        lam = np.array([lam_um * 1e-6])
        t = np.exp(-tau_tot)
        return float((1.0 - t) * planck(lam, t_e)[0]
                     / (planck(lam, T_SUN)[0] * t))

    print("=== table-mode channels (band-averaged ODs; saturated rows are "
          "lower bounds) ===")
    fac = [r / r_table(l0, tt) for (l0, tt, r) in anchors if tt < 2.0]
    s_fac = float(np.median(fac)) if fac else 1.3
    print(f"  structure factor from line-resolved anchors (tau<2): "
          f"x{s_fac:.2f}")
    print(f"{'material':9s} {'lam':>5s} {'tau_gas':>8s} {'r(est)':>8s} "
          f"{'r/floor':>8s} {'resid/floor':>11s}  note")
    for mat, lam0, tau_g, tau_m, sfl, note in TABLE_CHANNELS:
        r = s_fac * r_table(lam0, tau_g + tau_m)
        print(f"{mat:9s} {lam0:5.1f} {tau_g:8.2f} {r:8.1e} "
              f"{r/sfl:8.2f} {MODEL_LEVEL*r/sfl:11.2f}  {note}")

    print(f"\nr = <L_emis>/(L_sun <t>): the LOW bias on every retrieved "
          f"slant OD if unmodelled.\nresid/floor: after forward-modelling "
          f"the emission to {MODEL_LEVEL:.1%} (T to ~2 K, co-retrieved "
          f"gases).\n5772-K sun understates r by ~15-20% at 20-30 um "
          f"(solar T_b ~ 5000 K there).")


if __name__ == "__main__":
    main()
