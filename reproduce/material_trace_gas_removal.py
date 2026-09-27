"""material_trace_gas_removal.py -- trace-gas-removal floors for the dolomite
and alumina reststrahlen bands, the analogue of calcite_trace_gas_removal.py
(same OE co-retrieval engine, same 20 km tangent / 0.25 um element / onion-peel
conversion; see that script for the method notes).

Dolomite (CaMg(CO3)2), carbonate bands blue-shifted from calcite:
  nu3  ~6.4  um : CO3 asym stretch      (on the H2O nu2 bending band core)
  nu2  ~11.3 um : CO3 out-of-plane bend (atmospheric window, HNO3/CFCs)
  nu4  ~13.7 um : CO3 in-plane bend     (CO2 15 um shoulder -- expect dead)
  L    ~24.9 um : far-IR lattice mode   (H2O rotation forest, bluer than
                                         calcite's 28.5 um -- easier far-IR)

Alumina (alpha-Al2O3, corundum):
  R    ~12.9 um : main reststrahlen (Frohlich) band -- sits in the gap between
                  the O3 9.6 um and CO2 15 um bands
  W    ~20.7 um : red reststrahlen wing, essentially the silica config-C region

Usage:  python reproduce/material_trace_gas_removal.py [dolomite|alumina|all]
First run is slow (Voigt grids for the new windows); subsequent runs cache.
"""

import sys
import csv
from pathlib import Path
import numpy as np

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from saimon.sai import create_dolomite_sai_layer, create_alumina_sai_layer
from saimon.geometry import tangent_to_slant_paths
from saimon.o3_removal import (GasRemovalSetup, RetrievalGas, O3Band,
                              run_gas_removal_analysis)

TANGENT_KM = 20.0
ELEMENT_UM = 0.25

# Round 37 (paper1): optional element-width override, e.g. `--element 0.10`
# (the paper's single 0.1-um element convention).  The default 0.25 keeps the
# Notes artifacts bit-identical; a non-default width writes a suffixed CSV
# (e.g. *_w0p1.csv) next to the original.
if "--element" in sys.argv:
    ELEMENT_UM = float(sys.argv[sys.argv.index("--element") + 1])
OUT_SUFFIX = "" if abs(ELEMENT_UM - 0.25) < 1e-9 else f"_w{ELEMENT_UM:g}".replace(".", "p")
R_EARTH_M = 6.371e6
from saimon.onion_peel import G_ONION  # round 53 (RC1 M1): exact edge-grid gain from saimon.onion_peel
DZ_M = 500.0                                   # SAGE III/ISS nominal vertical bin


def sigma_alpha_from_od(sigma_od, dz_m=DZ_M):
    """Onion-peel conversion of a slant-OD error to an extinction floor [1/m]."""
    P_kk = 2.0 * np.sqrt(2.0 * R_EARTH_M * dz_m)
    return G_ONION * sigma_od / P_kk


def _gas(name, prior, line, band_sigma=0.05):
    return RetrievalGas(name, prior_sigma_rel=prior,
                        bands=(O3Band(name, band_sigma, (1e-6, 3e-5)),),
                        sigma_line_rel=line)


MATERIALS = {
    "dolomite": dict(
        make_layer=create_dolomite_sai_layer,
        bands=[
            dict(label="nu3 (~6.4um)", target=6.4, lo=5.8, hi=7.2, step=0.1,
                 gases=(_gas("h2o", 0.30, 0.002), _gas("ch4", 0.05, 0.002),
                        _gas("n2o", 0.05, 0.002), _gas("o3", 0.30, 0.002)),
                 note="CO3 asym stretch on the H2O nu2 bending band core"),
            dict(label="nu2 (~11.3um)", target=11.3, lo=10.5, hi=12.0, step=0.1,
                 gases=(_gas("hno3", 0.30, 0.005), _gas("h2o", 0.30, 0.002),
                        _gas("cfc11", 0.15, 0.02), _gas("cfc12", 0.15, 0.02)),
                 note="CO3 out-of-plane bend in the window (HNO3 + CFC-11/12)"),
            dict(label="nu4 (~13.7um)", target=13.7, lo=13.2, hi=14.6, step=0.1,
                 gases=(_gas("co2", 0.005, 0.002), _gas("o3", 0.30, 0.002)),
                 note="CO3 in-plane bend on the CO2 15um shoulder (expect dead)"),
            dict(label="L (~24.9um)", target=24.9, lo=23.5, hi=26.5, step=0.1,
                 gases=(_gas("h2o", 0.30, 0.002), _gas("o3", 0.30, 0.002),
                        _gas("n2o", 0.05, 0.002)),
                 note="far-IR lattice mode on the H2O rotation forest"),
        ]),
    "alumina": dict(
        make_layer=create_alumina_sai_layer,
        bands=[
            dict(label="R (~12.9um)", target=12.9, lo=12.0, hi=13.8, step=0.1,
                 gases=(_gas("co2", 0.005, 0.002), _gas("h2o", 0.30, 0.002),
                        _gas("o3", 0.30, 0.002), _gas("hno3", 0.30, 0.005),
                        _gas("cfc11", 0.15, 0.02)),
                 note="main reststrahlen band between the O3 9.6um and CO2 15um bands"),
            dict(label="W (~20.7um)", target=20.7, lo=19.8, hi=21.6, step=0.1,
                 gases=(_gas("h2o", 0.30, 0.002), _gas("hno3", 0.30, 0.005),
                        _gas("o3", 0.30, 0.002)),
                 note="red reststrahlen wing, the silica config-C region"),
        ]),
}


def material_od_spectrum(layer, chord, wl_m):
    return chord @ layer.extinction_profile_m1(wl_m)


def run_band(band, layer, chord):
    lo, hi, step = band["lo"], band["hi"], band["step"]
    chans_um = np.round(np.arange(lo, hi + 1e-9, step), 3)
    chans_m = tuple(chans_um * 1e-6)
    target_m = band["target"] * 1e-6

    all_m = np.array(list(chans_m) + [target_m])
    tau_mat_all = material_od_spectrum(layer, chord, all_m)

    s = GasRemovalSetup(
        target_wavelength_m=target_m, channels_m=chans_m, gases=band["gases"],
        h_tan_m=TANGENT_KM * 1e3, delta_T_K=2.0, fwhm_um_target=ELEMENT_UM,
        silica_tau_ref=tau_mat_all, silica_prior_sigma=1.0e3)
    r = run_gas_removal_analysis(s)

    tau_mat_target = float(tau_mat_all[-1])
    sig_tot = r["sigma_total"]
    sig_mat = r.get("sigma_silica_od", np.nan)
    sa = sigma_alpha_from_od(sig_tot)
    sa_deg = sigma_alpha_from_od(sig_mat) if np.isfinite(sig_mat) else np.nan
    snr = tau_mat_target / sig_mat if (np.isfinite(sig_mat) and sig_mat > 0) else np.inf
    return dict(
        label=band["label"], note=band["note"], target_um=band["target"],
        n_chan=len(chans_m), tau_mat_target=tau_mat_target,
        tau_gas_target=r["tau_target_ref"],
        tau_gas_by=r["tau_target_by_gas"], sigma_total=sig_tot,
        sigma_material_od=sig_mat, dofs=r["dofs"],
        sigma_alpha=sa, sigma_alpha_deg=sa_deg,
        snr_1tg=snr, min_load_tg=(1.0 / snr if np.isfinite(snr) and snr > 0 else np.inf),
    )


def run_material(mat):
    cfg = MATERIALS[mat]
    alt_m = np.arange(14_000.0, 80_001.0, 500.0)
    chord = tangent_to_slant_paths(np.array([TANGENT_KM * 1e3]), alt_m)[0]
    layer = cfg["make_layer"](alt_m, target_mass_tg=1.0)

    print(f"\n=== {mat.capitalize()} trace-gas-removal floors (1 Tg, "
          f"{TANGENT_KM:.0f} km tangent, {ELEMENT_UM} um element) ===")
    print(f"    ray-arithmetic-mean optics; dz={DZ_M/1e3:.1f} km; "
          f"sigma_alpha via onion-peel\n")

    rows = []
    for band in cfg["bands"]:
        print(f"--- {band['label']}: {band['note']}")
        try:
            res = run_band(band, layer, chord)
        except Exception as e:
            print(f"    FAILED: {e}\n")
            continue
        rows.append(res)
        by = ", ".join(f"{k}={v:.2e}" for k, v in res["tau_gas_by"].items())
        print(f"    tau_{mat}(target) = {res['tau_mat_target']:.3e} OD")
        print(f"    tau_gas(target)     = {res['tau_gas_target']:.3e} OD  ({by})")
        print(f"    sigma_total (removal) = {res['sigma_total']:.3e} OD")
        print(f"    sigma_{mat}_od (deg-aware) = {res['sigma_material_od']:.3e} OD  "
              f"(DOFS {res['dofs']:.2f})")
        print(f"    sigma_alpha = {res['sigma_alpha']:.3e} 1/m  "
              f"(deg-aware {res['sigma_alpha_deg']:.3e})")
        print(f"    SNR(1 Tg) = {res['snr_1tg']:.1f}  ->  "
              f"1-sigma detectable ~ {res['min_load_tg']:.3f} Tg\n")

    print("=== SUMMARY ===")
    print(f"  {'band':>14} {'tau_mat':>9} {'tau_gas':>9} {'sig_tot':>9} "
          f"{'sig_mat':>9} {'sig_alpha':>10} {'SNR/Tg':>7} {'minTg':>7}")
    for r in rows:
        print(f"  {r['label']:>14} {r['tau_mat_target']:9.2e} {r['tau_gas_target']:9.2e} "
              f"{r['sigma_total']:9.2e} {r['sigma_material_od']:9.2e} "
              f"{r['sigma_alpha_deg']:10.2e} {r['snr_1tg']:7.1f} {r['min_load_tg']:7.3f}")

    outdir = _PARENT / "outputs" / mat
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"{mat}_band_floors_{int(TANGENT_KM)}km{OUT_SUFFIX}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band", "note", "target_um", "n_chan", "tau_material_target_OD",
                    "tau_gas_target_OD", "sigma_total_OD", "sigma_material_od_OD",
                    "dofs", "sigma_alpha_1_m", "sigma_alpha_degaware_1_m",
                    "snr_1Tg", "min_detectable_Tg"])
        for r in rows:
            w.writerow([r["label"], r["note"], r["target_um"], r["n_chan"],
                        f"{r['tau_mat_target']:.6e}", f"{r['tau_gas_target']:.6e}",
                        f"{r['sigma_total']:.6e}", f"{r['sigma_material_od']:.6e}",
                        f"{r['dofs']:.4f}", f"{r['sigma_alpha']:.6e}",
                        f"{r['sigma_alpha_deg']:.6e}", f"{r['snr_1tg']:.4f}",
                        f"{r['min_load_tg']:.6f}"])
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--") and not a.replace(".", "").isdigit()]
    which = args[0] if args else "all"
    for mat in (list(MATERIALS) if which == "all" else [which]):
        run_material(mat)
