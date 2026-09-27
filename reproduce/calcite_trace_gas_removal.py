"""calcite_trace_gas_removal.py -- trace-gas-removal floors for the four calcite
reststrahlen bands, the calcite analogue of the silica B+/C study.

For a 1 Tg calcite SAI layer (ray-arithmetic-mean optics, Lederer 2026), at a
20 km tangent, this evaluates each candidate band:

  nu3  ~6.9  um : CO3 asymmetric stretch  (on the H2O nu2 bending band)
  nu2  ~11.4 um : CO3 out-of-plane bend    (on the HNO3 nu2 band -- like config C)
  nu4  ~14.0 um : CO3 in-plane bend         (on the CO2 15 um band -- saturated)
  L    ~28.5 um : far-IR lattice mode       (on the H2O pure-rotation forest)

For each band we run the config-agnostic OE co-retrieval (run_gas_removal_analysis)
over a resolved channel window, co-fitting the calcite amplitude jointly with the
gases (degeneracy-aware).  We report:
  tau_calcite(target)   -- the 1 Tg signal to detect,
  tau_gas(target)       -- the gas background to remove,
  sigma_total           -- gas-removal error floor (OD),
  sigma_calcite_od      -- degeneracy-aware posterior error on calcite OD (OD),
  sigma_alpha           -- extinction floor at dz=0.5 km (1/m),
  SNR(1 Tg) and the 1-sigma detectable loading.

First run is slow (far-IR H2O/HNO3/CO2 Voigt grids); subsequent runs cache.
"""

import sys
import csv
from pathlib import Path
import numpy as np

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from saimon.atmosphere import us_standard_atmosphere
from saimon.geometry import tangent_to_slant_paths
from saimon import trace_gases as tg
from saimon.sai import create_calcite_sai_layer
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
OUTDIR = _PARENT / "outputs" / "calcite"


def sigma_alpha_from_od(sigma_od, dz_m=DZ_M):
    """Onion-peel conversion of a slant-OD error to an extinction floor [1/m],
    the same thin-shell chord formula used for the silica MIR channels."""
    P_kk = 2.0 * np.sqrt(2.0 * R_EARTH_M * dz_m)
    return G_ONION * sigma_od / P_kk


# Each band: (label, target_um, window_lo, window_hi, step_um, gases, note)
def _gas(name, prior, line, band_sigma=0.05):
    return RetrievalGas(name, prior_sigma_rel=prior,
                        bands=(O3Band(name, band_sigma, (1e-6, 3e-5)),),
                        sigma_line_rel=line)


BANDS = [
    dict(label="nu3 (~6.9um)", target=6.9, lo=6.3, hi=7.6, step=0.1,
         gases=(_gas("h2o", 0.30, 0.002), _gas("ch4", 0.05, 0.002),
                _gas("n2o", 0.05, 0.002), _gas("o3", 0.30, 0.002)),
         note="CO3 asym stretch on H2O nu2 bending band"),
    dict(label="nu2 (~11.4um)", target=11.4, lo=10.5, hi=12.0, step=0.1,
         gases=(_gas("hno3", 0.30, 0.005), _gas("h2o", 0.30, 0.002),
                _gas("cfc11", 0.15, 0.02), _gas("cfc12", 0.15, 0.02)),
         note="CO3 out-of-plane bend on HNO3 nu2 (window region; CFC-11/12 co-retrieved)"),
    dict(label="nu4 (~14.0um)", target=14.0, lo=13.2, hi=14.6, step=0.1,
         gases=(_gas("co2", 0.005, 0.002), _gas("o3", 0.30, 0.002)),
         note="CO3 in-plane bend on the CO2 15um band (expect saturated)"),
    dict(label="L (~28.5um)", target=28.5, lo=27.0, hi=30.0, step=0.1,
         gases=(_gas("h2o", 0.30, 0.002), _gas("o3", 0.30, 0.002),
                _gas("n2o", 0.05, 0.002)),
         note="far-IR lattice mode on H2O pure-rotation forest"),
    # Far-IR alternative: place the channel in the 29 um inter-line window.
    dict(label="L-win (~29.2um)", target=29.2, lo=28.0, hi=30.5, step=0.1,
         gases=(_gas("h2o", 0.30, 0.002), _gas("o3", 0.30, 0.002),
                _gas("n2o", 0.05, 0.002)),
         note="far-IR lattice red wing, H2O inter-line window"),
]


def calcite_od_spectrum(layer, chord, wl_m):
    return chord @ layer.extinction_profile_m1(wl_m)


def run_band(band, layer, alt_m, chord):
    lo, hi, step = band["lo"], band["hi"], band["step"]
    chans_um = np.round(np.arange(lo, hi + 1e-9, step), 3)
    chans_m = tuple(chans_um * 1e-6)
    target_m = band["target"] * 1e-6

    # calcite reference OD at [channels..., target]
    all_m = np.array(list(chans_m) + [target_m])
    tau_cal_all = calcite_od_spectrum(layer, chord, all_m)

    s = GasRemovalSetup(
        target_wavelength_m=target_m, channels_m=chans_m, gases=band["gases"],
        h_tan_m=TANGENT_KM * 1e3, delta_T_K=2.0, fwhm_um_target=ELEMENT_UM,
        silica_tau_ref=tau_cal_all, silica_prior_sigma=1.0e3)
    r = run_gas_removal_analysis(s)

    tau_cal_target = float(tau_cal_all[-1])
    sig_tot = r["sigma_total"]
    sig_cal = r.get("sigma_silica_od", np.nan)
    sa = sigma_alpha_from_od(sig_tot)
    sa_deg = sigma_alpha_from_od(sig_cal) if np.isfinite(sig_cal) else np.nan
    snr = tau_cal_target / sig_cal if (np.isfinite(sig_cal) and sig_cal > 0) else np.inf
    return dict(
        label=band["label"], note=band["note"], target_um=band["target"],
        n_chan=len(chans_m), tau_cal_target=tau_cal_target,
        tau_gas_target=r["tau_target_ref"],
        tau_gas_by=r["tau_target_by_gas"], sigma_total=sig_tot,
        sigma_calcite_od=sig_cal, dofs=r["dofs"],
        sigma_alpha=sa, sigma_alpha_deg=sa_deg,
        snr_1tg=snr, min_load_tg=(1.0 / snr if np.isfinite(snr) and snr > 0 else np.inf),
    )


def main():
    alt_m = np.arange(14_000.0, 80_001.0, 500.0)
    chord = tangent_to_slant_paths(np.array([TANGENT_KM * 1e3]), alt_m)[0]
    layer = create_calcite_sai_layer(alt_m, target_mass_tg=1.0)

    print(f"\n=== Calcite trace-gas-removal floors (1 Tg, {TANGENT_KM:.0f} km tangent, "
          f"{ELEMENT_UM} um element) ===")
    print(f"    ray-arithmetic-mean optics; dz={DZ_M/1e3:.1f} km; sigma_alpha via onion-peel\n")

    rows = []
    for band in BANDS:
        print(f"--- {band['label']}: {band['note']}")
        try:
            res = run_band(band, layer, alt_m, chord)
        except Exception as e:
            print(f"    FAILED: {e}\n")
            continue
        rows.append(res)
        by = ", ".join(f"{k}={v:.2e}" for k, v in res["tau_gas_by"].items())
        print(f"    tau_calcite(target) = {res['tau_cal_target']:.3e} OD")
        print(f"    tau_gas(target)     = {res['tau_gas_target']:.3e} OD  ({by})")
        print(f"    sigma_total (removal) = {res['sigma_total']:.3e} OD")
        print(f"    sigma_calcite_od (deg-aware) = {res['sigma_calcite_od']:.3e} OD  "
              f"(DOFS {res['dofs']:.2f})")
        print(f"    sigma_alpha = {res['sigma_alpha']:.3e} 1/m  "
              f"(deg-aware {res['sigma_alpha_deg']:.3e})")
        print(f"    SNR(1 Tg) = {res['snr_1tg']:.1f}  ->  "
              f"1-sigma detectable ~ {res['min_load_tg']:.3f} Tg\n")

    # summary table
    print("=== SUMMARY ===")
    hdr = (f"  {'band':>14} {'tau_cal':>9} {'tau_gas':>9} {'sig_tot':>9} "
           f"{'sig_cal':>9} {'sig_alpha':>10} {'SNR/Tg':>7} {'minTg':>7}")
    print(hdr)
    for r in rows:
        print(f"  {r['label']:>14} {r['tau_cal_target']:9.2e} {r['tau_gas_target']:9.2e} "
              f"{r['sigma_total']:9.2e} {r['sigma_calcite_od']:9.2e} "
              f"{r['sigma_alpha']:10.2e} {r['snr_1tg']:7.1f} {r['min_load_tg']:7.3f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTDIR / f"calcite_band_floors_{int(TANGENT_KM)}km{OUT_SUFFIX}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band", "note", "target_um", "n_chan", "tau_calcite_target_OD",
                    "tau_gas_target_OD", "sigma_total_OD", "sigma_calcite_od_OD",
                    "dofs", "sigma_alpha_1_m", "sigma_alpha_degaware_1_m",
                    "snr_1Tg", "min_detectable_Tg"])
        for r in rows:
            w.writerow([r["label"], r["note"], r["target_um"], r["n_chan"],
                        f"{r['tau_cal_target']:.6e}", f"{r['tau_gas_target']:.6e}",
                        f"{r['sigma_total']:.6e}", f"{r['sigma_calcite_od']:.6e}",
                        f"{r['dofs']:.4f}", f"{r['sigma_alpha']:.6e}",
                        f"{r['sigma_alpha_deg']:.6e}", f"{r['snr_1tg']:.4f}",
                        f"{r['min_load_tg']:.6f}"])
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
